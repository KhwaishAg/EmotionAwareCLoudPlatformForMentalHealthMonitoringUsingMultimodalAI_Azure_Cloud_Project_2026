"""Categorical encoding for questionnaire questions Q1, Q2, and Q25.

This module performs only the Step 8 categorical encoding for the questionnaire
modality. It loads the cleaned questionnaire CSV, encodes Q1 as an ordered
numeric value, Q2 as seven binary multi-select features, and Q25 as one-hot
features, validates the encoded output, saves the encoded CSV, and writes a
report.

It does not modify the original questionnaire columns, create targets, perform
imputation, or calculate stress scores.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_cleaned.csv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_categorical_encoded.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "categorical_encoding_report.txt"

Q1_MAPPING: Dict[str, int] = {
    "1st Year": 1,
    "2nd Year": 2,
    "3rd Year": 3,
    "4th Year": 4,
    "Postgraduate": 5,
}

Q2_CANONICAL_OPTIONS: List[str] = [
    "Regular academic semester",
    "Preparing for examinations",
    "Working on academic projects",
    "Facing assignment/deadline pressure",
    "Preparing for placements & internships",
    "Preparing for coding/technical assessments",
    "Applying for jobs/internships",
]

Q2_FEATURE_COLUMNS: List[str] = [
    "Q2_regular_academic_semester",
    "Q2_preparing_for_examinations",
    "Q2_working_on_academic_projects",
    "Q2_assignment_deadline_pressure",
    "Q2_placements_internships",
    "Q2_coding_technical_assessments",
    "Q2_applying_for_jobs_internships",
]

Q2_CANONICAL_LOOKUP = {option: feature for option, feature in zip(Q2_CANONICAL_OPTIONS, Q2_FEATURE_COLUMNS)}

Q25_CANONICAL_CATEGORIES: List[str] = [
    "Placements/job search",
    "Career uncertainty",
    "Social/personal issues",
    "Examinations",
    "Other",
    "Academic workload",
    "Coding/technical assessments",
    "Financial concerns",
    "Family expectations",
]

Q25_FEATURE_COLUMNS: List[str] = [
    "Q25_Placements_job_search",
    "Q25_Career_uncertainty",
    "Q25_Social_personal_issues",
    "Q25_Examinations",
    "Q25_Other",
    "Q25_Academic_workload",
    "Q25_Coding_technical_assessments",
    "Q25_Financial_concerns",
    "Q25_Family_expectations",
]

Q25_CANONICAL_LOOKUP = {category: feature for category, feature in zip(Q25_CANONICAL_CATEGORIES, Q25_FEATURE_COLUMNS)}


def load_input_data(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the cleaned questionnaire CSV without modifying it."""

    resolved_path = Path(csv_path) if csv_path is not None else INPUT_CSV_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {resolved_path}")

    return pd.read_csv(resolved_path, encoding="utf-8")


def _normalize_whitespace(value: str) -> str:
    """Normalize repeated internal whitespace in a string."""

    if not isinstance(value, str):
        return value

    return " ".join(value.strip().split())


def _is_missing(value: Any) -> bool:
    """Return True when a value should be treated as missing."""

    return pd.isna(value) or (isinstance(value, str) and value.strip() == "")


def _new_nullable_int_series(index: pd.Index) -> pd.Series:
    """Create a Series that can store 0, 1, or missing values."""

    return pd.Series(pd.NA, index=index, dtype="Int64")


def encode_q1(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Encode Q1 as an ordered numeric feature using the required mapping."""

    if "Q1" not in df.columns:
        raise ValueError("The input dataset is missing the Q1 column.")

    encoded_df = df.copy()
    encoded_df["Q1_year_encoded"] = _new_nullable_int_series(encoded_df.index)

    q1_category_counts: Counter[str] = Counter()
    q1_unexpected_values: Dict[str, int] = {}

    for row_index, response_value in df["Q1"].items():
        if _is_missing(response_value):
            continue

        normalized_value = _normalize_whitespace(str(response_value))
        if normalized_value in Q1_MAPPING:
            encoded_df.at[row_index, "Q1_year_encoded"] = Q1_MAPPING[normalized_value]
            q1_category_counts[normalized_value] += 1
        else:
            q1_unexpected_values[normalized_value] = q1_unexpected_values.get(normalized_value, 0) + 1

    metadata = {
        "q1_category_counts": dict(sorted(q1_category_counts.items())),
        "q1_unexpected_values": dict(sorted(q1_unexpected_values.items())),
    }

    return encoded_df, metadata


def parse_q2_response(response_value: Any) -> Dict[str, Any]:
    """Split, normalize, and validate a semicolon-delimited Q2 response."""

    result = {
        "selected_options": set(),
        "contains_all_of_them": False,
        "unexpected_options": {},
    }

    if _is_missing(response_value):
        return result

    if not isinstance(response_value, str):
        return result

    parts = []
    for raw_part in response_value.split(";"):
        clean_part = _normalize_whitespace(raw_part)
        if clean_part == "":
            continue
        parts.append(clean_part)

    if not parts:
        return result

    if any(part.lower() == "all of them" for part in parts):
        result["contains_all_of_them"] = True
        result["selected_options"] = set(Q2_CANONICAL_OPTIONS)

    for part in parts:
        if part.lower() == "all of them":
            continue

        if part in Q2_CANONICAL_LOOKUP:
            result["selected_options"].add(part)
        else:
            result["unexpected_options"][part] = result["unexpected_options"].get(part, 0) + 1

    return result


def encode_q2(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Encode Q2 into seven binary features while preserving missingness."""

    if "Q2" not in df.columns:
        raise ValueError("The input dataset is missing the Q2 column.")

    encoded_df = df.copy()
    for feature_name in Q2_FEATURE_COLUMNS:
        encoded_df[feature_name] = _new_nullable_int_series(encoded_df.index)

    q2_option_counts: Counter[str] = Counter()
    q2_missing_count = 0
    q2_all_of_them_count = 0
    q2_unexpected_options: Dict[str, int] = {}

    for row_index, response_value in df["Q2"].items():
        if _is_missing(response_value):
            q2_missing_count += 1
            continue

        parsed_response = parse_q2_response(response_value)

        if parsed_response["contains_all_of_them"]:
            q2_all_of_them_count += 1
            for option in Q2_CANONICAL_OPTIONS:
                q2_option_counts[option] += 1
                encoded_df.at[row_index, Q2_CANONICAL_LOOKUP[option]] = 1
            for feature_name in Q2_FEATURE_COLUMNS:
                if feature_name not in Q2_CANONICAL_LOOKUP.values():
                    continue

            for option in parsed_response["unexpected_options"]:
                q2_unexpected_options[option] = q2_unexpected_options.get(option, 0) + parsed_response["unexpected_options"][option]

            continue

        valid_options: set[str] = parsed_response["selected_options"]
        for option in valid_options:
            q2_option_counts[option] += 1
            encoded_df.at[row_index, Q2_CANONICAL_LOOKUP[option]] = 1

        for feature_name in Q2_FEATURE_COLUMNS:
            if feature_name not in encoded_df.columns:
                continue
            if pd.isna(encoded_df.at[row_index, feature_name]):
                encoded_df.at[row_index, feature_name] = 0

        for option, count in parsed_response["unexpected_options"].items():
            q2_unexpected_options[option] = q2_unexpected_options.get(option, 0) + count

    metadata = {
        "q2_option_counts": dict(sorted(q2_option_counts.items())),
        "q2_missing_count": q2_missing_count,
        "q2_all_of_them_count": q2_all_of_them_count,
        "q2_unexpected_options": dict(sorted(q2_unexpected_options.items())),
    }

    return encoded_df, metadata


def encode_q25(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Encode Q25 as one-hot features while preserving missingness."""

    if "Q25" not in df.columns:
        raise ValueError("The input dataset is missing the Q25 column.")

    encoded_df = df.copy()
    for feature_name in Q25_FEATURE_COLUMNS:
        encoded_df[feature_name] = _new_nullable_int_series(encoded_df.index)

    q25_category_counts: Counter[str] = Counter()
    q25_missing_count = 0
    q25_unexpected_categories: Dict[str, int] = {}

    for row_index, response_value in df["Q25"].items():
        if _is_missing(response_value):
            q25_missing_count += 1
            continue

        normalized_value = _normalize_whitespace(str(response_value))
        if normalized_value in Q25_CANONICAL_LOOKUP:
            for feature_name in Q25_FEATURE_COLUMNS:
                encoded_df.at[row_index, feature_name] = 0
            encoded_df.at[row_index, Q25_CANONICAL_LOOKUP[normalized_value]] = 1
            q25_category_counts[normalized_value] += 1
        else:
            q25_unexpected_categories[normalized_value] = q25_unexpected_categories.get(normalized_value, 0) + 1

    metadata = {
        "q25_category_counts": dict(sorted(q25_category_counts.items())),
        "q25_missing_count": q25_missing_count,
        "q25_unexpected_categories": dict(sorted(q25_unexpected_categories.items())),
    }

    return encoded_df, metadata


def validate_encoded_data(
    encoded_df: pd.DataFrame,
    original_df: pd.DataFrame,
    q2_feature_columns: List[str],
    q25_feature_columns: List[str],
    expected_rows: int,
    q2_unexpected_options: Dict[str, int],
    q25_unexpected_categories: Dict[str, int],
) -> bool:
    """Validate the categorical encoding output against the Step 8 rules."""

    if len(encoded_df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(encoded_df)}.")

    if len(encoded_df) != len(original_df):
        raise ValueError("Row count changed during encoding; rows were dropped or added.")

    if not encoded_df.loc[:, original_df.columns].equals(original_df):
        raise ValueError("Original questionnaire columns were modified.")

    if len(encoded_df.columns) != len(original_df.columns) + 1 + len(q2_feature_columns) + len(q25_feature_columns):
        raise ValueError("Unexpected number of encoded columns were added.")

    q1_encoded_series = encoded_df["Q1_year_encoded"]
    if not q1_encoded_series.dropna().isin(set(Q1_MAPPING.values())).all():
        raise ValueError("Q1_year_encoded contains unexpected mapped values.")

    for feature_name in q2_feature_columns:
        series = encoded_df[feature_name]
        observed_non_missing = series.dropna()
        if not observed_non_missing.empty and not observed_non_missing.isin([0, 1]).all():
            raise ValueError(f"Feature {feature_name} contains values outside of {{0, 1, NaN}}.")

    for feature_name in q25_feature_columns:
        series = encoded_df[feature_name]
        observed_non_missing = series.dropna()
        if not observed_non_missing.empty and not observed_non_missing.isin([0, 1]).all():
            raise ValueError(f"Feature {feature_name} contains values outside of {{0, 1, NaN}}.")

    q2_missing_mask = original_df["Q2"].apply(lambda value: _is_missing(value))
    for feature_name in q2_feature_columns:
        if not encoded_df.loc[q2_missing_mask, feature_name].isna().all():
            raise ValueError(f"Missing Q2 rows should have NaN across all Q2 features: {feature_name}")

    q25_missing_mask = original_df["Q25"].apply(lambda value: _is_missing(value))
    for feature_name in q25_feature_columns:
        if not encoded_df.loc[q25_missing_mask, feature_name].isna().all():
            raise ValueError(f"Missing Q25 rows should have NaN across all Q25 features: {feature_name}")

    if q2_unexpected_options:
        raise ValueError(f"Unexpected Q2 options found: {q2_unexpected_options}")

    if q25_unexpected_categories:
        raise ValueError(f"Unexpected Q25 categories found: {q25_unexpected_categories}")

    all_q2_features_ones = (encoded_df[q2_feature_columns].fillna(0) == 1).all(axis=1)
    q2_all_of_them_rows = encoded_df["Q2"].apply(lambda value: str(value).strip() == "All of them")
    if not all_q2_features_ones[q2_all_of_them_rows].all():
        raise ValueError("Rows with 'All of them' do not have all seven Q2 features set to 1.")

    return True


def generate_report(
    encoded_df: pd.DataFrame,
    q1_metadata: Dict[str, Any],
    q2_metadata: Dict[str, Any],
    q25_metadata: Dict[str, Any],
    validation_result: str,
    report_path: Optional[str | Path] = None,
) -> str:
    """Create a human-readable report for the categorical encoding output."""

    resolved_report_path = Path(report_path) if report_path is not None else REPORT_PATH
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        "Categorical Encoding Report",
        "==========================",
        f"Input rows: {len(encoded_df)}",
        f"Output rows: {len(encoded_df)}",
        "",
        "Q1 category counts:",
    ]

    q1_category_counts = q1_metadata["q1_category_counts"]
    if q1_category_counts:
        for category, count in q1_category_counts.items():
            lines.append(f"- {category}: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "Q1 mapping:"])
    for category, mapped_value in Q1_MAPPING.items():
        lines.append(f"- {category} -> {mapped_value}")

    lines.extend(["", "Q2 canonical options:"])
    for option in Q2_CANONICAL_OPTIONS:
        lines.append(f"- {option}")

    lines.extend(["", "Q2 option frequencies:"])
    q2_counts = q2_metadata["q2_option_counts"]
    if q2_counts:
        for option, count in q2_counts.items():
            lines.append(f"- {option}: {count}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            f"Q2 missing count: {q2_metadata['q2_missing_count']}",
            f"Q2 unexpected options: {q2_metadata['q2_unexpected_options'] or 'None'}",
            f"Q2 All of them count: {q2_metadata['q2_all_of_them_count']}",
            "",
            "Q25 category frequencies:",
        ]
    )

    q25_counts = q25_metadata["q25_category_counts"]
    if q25_counts:
        for category, count in q25_counts.items():
            lines.append(f"- {category}: {count}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            f"Q25 missing count: {q25_metadata['q25_missing_count']}",
            f"Q25 unexpected categories: {q25_metadata['q25_unexpected_categories'] or 'None'}",
            "",
            "Validation result: " + validation_result,
            "",
            "Final feature counts:",
            f"- Q1 encoded features: 1",
            f"- Q2 encoded features: {len(Q2_FEATURE_COLUMNS)}",
            f"- Q25 encoded features: {len(Q25_FEATURE_COLUMNS)}",
        ]
    )

    report_content = "\n".join(lines)
    resolved_report_path.write_text(report_content, encoding="utf-8")
    return report_content


def save_encoded_data(encoded_df: pd.DataFrame, output_path: Optional[str | Path] = None) -> Path:
    """Save the encoded dataframe to the processed output CSV."""

    resolved_output_path = Path(output_path) if output_path is not None else OUTPUT_CSV_PATH
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_df.to_csv(resolved_output_path, index=False, encoding="utf-8")
    return resolved_output_path


def run_categorical_encoding(
    input_csv_path: Optional[str | Path] = None,
    output_csv_path: Optional[str | Path] = None,
    report_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, str]:
    """Run the full Q1/Q2/Q25 categorical encoding pipeline."""

    input_path = Path(input_csv_path) if input_csv_path is not None else INPUT_CSV_PATH
    output_path = Path(output_csv_path) if output_csv_path is not None else OUTPUT_CSV_PATH
    report_file_path = Path(report_path) if report_path is not None else REPORT_PATH

    input_df = load_input_data(input_path)
    original_df = input_df.copy()

    encoded_df, q1_metadata = encode_q1(input_df)
    encoded_df, q2_metadata = encode_q2(encoded_df)
    encoded_df, q25_metadata = encode_q25(encoded_df)

    validation_result = validate_encoded_data(
        encoded_df=encoded_df,
        original_df=original_df,
        q2_feature_columns=Q2_FEATURE_COLUMNS,
        q25_feature_columns=Q25_FEATURE_COLUMNS,
        expected_rows=52,
        q2_unexpected_options=q2_metadata["q2_unexpected_options"],
        q25_unexpected_categories=q25_metadata["q25_unexpected_categories"],
    )

    save_encoded_data(encoded_df, output_path)

    report_text = generate_report(
        encoded_df=encoded_df,
        q1_metadata=q1_metadata,
        q2_metadata=q2_metadata,
        q25_metadata=q25_metadata,
        validation_result="PASS" if validation_result else "FAIL",
        report_path=report_file_path,
    )

    return encoded_df, report_text


def main() -> None:
    """Run the categorical encoding pipeline and print a short summary."""

    encoded_df, report_text = run_categorical_encoding()

    print("CATEGORICAL ENCODING COMPLETE")
    print(f"Rows: {len(encoded_df)}")
    print(f"Q1 encoded feature count: 1")
    print(f"Q2 encoded feature count: {len(Q2_FEATURE_COLUMNS)}")
    print(f"Q25 encoded feature count: {len(Q25_FEATURE_COLUMNS)}")
    print(f"Output: {OUTPUT_CSV_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Validation result from report: {report_text.split('Validation result: ', 1)[1].splitlines()[0]}")


if __name__ == "__main__":
    main()
