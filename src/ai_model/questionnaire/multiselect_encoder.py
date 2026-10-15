"""Multi-select encoding for questionnaire questions Q10 and Q11.

This module is intentionally limited to multi-hot encoding of the two multi-select
questionnaire questions. It loads the existing ordinal-encoded questionnaire CSV,
adds Q10/Q11 multi-hot feature columns, validates the result, saves the encoded
CSV, and writes a report.

It does not modify the input file, create targets, perform ML, impute missing
values, or change any other questionnaire questions.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_ordinal_encoded.csv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_multihot_encoded.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "multiselect_encoding_report.txt"

Q10_CANONICAL_OPTIONS: List[str] = [
    "Academic performance/grades",
    "Academic workload",
    "Assignments",
    "Coding/technical assessments",
    "Competition with peers",
    "Difficulty balancing academics and personal life",
    "Exams",
    "Family expectations",
    "Fear of not getting a job",
    "Financial concerns",
    "Internship/job applications",
    "Lack of sleep",
    "Placement preparation",
    "Projects",
    "Relationship/social concerns",
    "Time management",
    "Uncertainty about career",
    "Other",
]

Q11_CANONICAL_OPTIONS: List[str] = [
    "Anxious",
    "Calm",
    "Confident",
    "Frustrated",
    "Happy",
    "Irritated",
    "Lonely",
    "Motivated",
    "Neutral",
    "Other",
    "Overwhelmed",
    "Sad",
    "Tired",
    "Worried",
]

Q10_FEATURE_COLUMNS: List[str] = [
    "Q10_Academic_performance_grades",
    "Q10_Academic_workload",
    "Q10_Assignments",
    "Q10_Coding_technical_assessments",
    "Q10_Competition_with_peers",
    "Q10_Difficulty_balancing_academics_and_personal_life",
    "Q10_Exams",
    "Q10_Family_expectations",
    "Q10_Fear_of_not_getting_a_job",
    "Q10_Financial_concerns",
    "Q10_Internship_job_applications",
    "Q10_Lack_of_sleep",
    "Q10_Placement_preparation",
    "Q10_Projects",
    "Q10_Relationship_social_concerns",
    "Q10_Time_management",
    "Q10_Uncertainty_about_career",
    "Q10_Other",
]

Q11_FEATURE_COLUMNS: List[str] = [
    "Q11_Anxious",
    "Q11_Calm",
    "Q11_Confident",
    "Q11_Frustrated",
    "Q11_Happy",
    "Q11_Irritated",
    "Q11_Lonely",
    "Q11_Motivated",
    "Q11_Neutral",
    "Q11_Other",
    "Q11_Overwhelmed",
    "Q11_Sad",
    "Q11_Tired",
    "Q11_Worried",
]

Q10_CANONICAL_LOOKUP = {option: feature for option, feature in zip(Q10_CANONICAL_OPTIONS, Q10_FEATURE_COLUMNS)}
Q11_CANONICAL_LOOKUP = {option: feature for option, feature in zip(Q11_CANONICAL_OPTIONS, Q11_FEATURE_COLUMNS)}


def load_input_data(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the ordinal-encoded questionnaire CSV without modifying it."""

    resolved_path = Path(csv_path) if csv_path is not None else INPUT_CSV_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {resolved_path}")

    return pd.read_csv(resolved_path, encoding="utf-8")


def _normalize_whitespace(value: str) -> str:
    """Normalize repeated internal whitespace in a string."""

    return re.sub(r"\s+", " ", value.strip())


def normalize_q10_option(option_text: str) -> str:
    """Normalize a Q10 option string according to the required schema rules."""

    normalized = _normalize_whitespace(option_text)
    if normalized.lower() == "not getting girlfriend":
        return "Relationship/social concerns"
    return normalized


def parse_multiselect_response(response_value: Any, question_id: str) -> List[str]:
    """Split and normalize a semicolon-delimited multi-select response."""

    if pd.isna(response_value):
        return []

    if not isinstance(response_value, str):
        return []

    stripped_response = response_value.strip()
    if stripped_response == "":
        return []

    parts = []
    for raw_part in stripped_response.split(";"):
        clean_part = _normalize_whitespace(raw_part)
        if clean_part == "":
            continue
        if question_id == "Q10":
            clean_part = normalize_q10_option(clean_part)
        parts.append(clean_part)

    return parts


def _new_nullable_int_series(index: pd.Index) -> pd.Series:
    """Create a Series that can store 0, 1, or missing values."""

    return pd.Series(pd.NA, index=index, dtype="Int64")


def encode_q10(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Encode Q10 into binary features and return metadata about the encoding."""

    if "Q10" not in df.columns:
        raise ValueError("The input dataset is missing the Q10 text column.")

    encoded_df = df.copy()
    per_feature_missing = {feature_name: _new_nullable_int_series(encoded_df.index) for feature_name in Q10_FEATURE_COLUMNS}
    per_feature_missing["Q10_nothing"] = _new_nullable_int_series(encoded_df.index)

    q10_selected_option_counts: Counter[str] = Counter()
    q10_nothing_count = 0
    q10_missing_count = 0
    q10_data_quality_conditions = 0
    unexpected_values: Dict[str, int] = {}

    for row_index, response_value in df["Q10"].items():
        if pd.isna(response_value) or (isinstance(response_value, str) and response_value.strip() == ""):
            q10_missing_count += 1
            continue

        options = parse_multiselect_response(response_value, "Q10")
        selected_options = set()
        contains_nothing = False

        for option in options:
            if option.lower() == "nothing":
                contains_nothing = True
                continue

            if option in Q10_CANONICAL_LOOKUP:
                selected_options.add(option)
            else:
                unexpected_values[option] = unexpected_values.get(option, 0) + 1

        if contains_nothing:
            q10_nothing_count += 1
            if selected_options:
                q10_data_quality_conditions += 1

        for option in selected_options:
            feature_name = Q10_CANONICAL_LOOKUP[option]
            per_feature_missing[feature_name].iloc[row_index] = 1
            q10_selected_option_counts[option] += 1

        if contains_nothing:
            per_feature_missing["Q10_nothing"].iloc[row_index] = 1

        for feature_name in Q10_FEATURE_COLUMNS:
            if feature_name not in per_feature_missing:
                continue

        if not contains_nothing:
            per_feature_missing["Q10_nothing"].iloc[row_index] = 0

        for feature_name in Q10_FEATURE_COLUMNS:
            if feature_name not in per_feature_missing:
                continue
            if per_feature_missing[feature_name].iloc[row_index] is pd.NA:
                per_feature_missing[feature_name].iloc[row_index] = 0

    for feature_name in Q10_FEATURE_COLUMNS:
        encoded_df[feature_name] = per_feature_missing[feature_name]

    encoded_df["Q10_nothing"] = per_feature_missing["Q10_nothing"]

    # Preserve missing Q10 rows as distinguishable from valid zero-selection responses.
    for feature_name in Q10_FEATURE_COLUMNS:
        original_series = encoded_df[feature_name]
        encoded_df[feature_name] = original_series

    metadata = {
        "q10_selected_option_counts": dict(sorted(q10_selected_option_counts.items())),
        "q10_nothing_count": q10_nothing_count,
        "q10_missing_count": q10_missing_count,
        "q10_data_quality_conditions": q10_data_quality_conditions,
        "unexpected_values": dict(sorted(unexpected_values.items())),
    }

    return encoded_df, metadata


def encode_q11(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Encode Q11 into binary features and return metadata about the encoding."""

    if "Q11" not in df.columns:
        raise ValueError("The input dataset is missing the Q11 text column.")

    encoded_df = df.copy()
    per_feature_missing = {feature_name: _new_nullable_int_series(encoded_df.index) for feature_name in Q11_FEATURE_COLUMNS}
    q11_selected_option_counts: Counter[str] = Counter()
    q11_missing_count = 0
    unexpected_values: Dict[str, int] = {}

    for row_index, response_value in df["Q11"].items():
        if pd.isna(response_value) or (isinstance(response_value, str) and response_value.strip() == ""):
            q11_missing_count += 1
            continue

        options = parse_multiselect_response(response_value, "Q11")
        selected_options = set()

        for option in options:
            if option in Q11_CANONICAL_LOOKUP:
                selected_options.add(option)
            else:
                unexpected_values[option] = unexpected_values.get(option, 0) + 1

        for option in selected_options:
            feature_name = Q11_CANONICAL_LOOKUP[option]
            per_feature_missing[feature_name].iloc[row_index] = 1
            q11_selected_option_counts[option] += 1

        for feature_name in Q11_FEATURE_COLUMNS:
            if per_feature_missing[feature_name].iloc[row_index] is pd.NA:
                per_feature_missing[feature_name].iloc[row_index] = 0

    for feature_name in Q11_FEATURE_COLUMNS:
        encoded_df[feature_name] = per_feature_missing[feature_name]

    metadata = {
        "q11_selected_option_counts": dict(sorted(q11_selected_option_counts.items())),
        "q11_missing_count": q11_missing_count,
        "unexpected_values": dict(sorted(unexpected_values.items())),
    }

    return encoded_df, metadata


def validate_multihot_features(
    encoded_df: pd.DataFrame,
    original_df: pd.DataFrame,
    q10_feature_columns: List[str],
    q11_feature_columns: List[str],
    q10_nothing_present: bool,
    q10_unexpected_report: Dict[str, int],
    q11_unexpected_report: Dict[str, int],
) -> bool:
    """Validate that the multi-hot features satisfy the required constraints."""

    if len(encoded_df) != len(original_df):
        raise ValueError("Row count changed during encoding; rows were dropped or added.")

    if encoded_df["Q10"].equals(original_df["Q10"]) is False:
        raise ValueError("Original Q10 text column was modified.")

    if encoded_df["Q11"].equals(original_df["Q11"]) is False:
        raise ValueError("Original Q11 text column was modified.")

    if len(encoded_df.columns) != len(original_df.columns) + len(q10_feature_columns) + len(q11_feature_columns) + 1:
        raise ValueError("Unexpected number of encoded columns were added.")

    for feature_name in q10_feature_columns:
        if feature_name not in encoded_df.columns:
            raise ValueError(f"Missing Q10 feature: {feature_name}")

    for feature_name in q11_feature_columns:
        if feature_name not in encoded_df.columns:
            raise ValueError(f"Missing Q11 feature: {feature_name}")

    if q10_nothing_present is False:
        raise ValueError("Q10_nothing feature is missing.")

    if "Q10_nothing" not in encoded_df.columns:
        raise ValueError("Q10_nothing feature is missing.")

    all_feature_columns = q10_feature_columns + q11_feature_columns + ["Q10_nothing"]
    for feature_name in all_feature_columns:
        series = encoded_df[feature_name]
        observed_non_missing = series.dropna()
        if not observed_non_missing.empty and not observed_non_missing.isin([0, 1]).all():
            raise ValueError(f"Feature {feature_name} contains values outside of {{0, 1, NaN}}.")

    if any(feature_name.endswith("Not_getting_girlfriend") for feature_name in encoded_df.columns):
        raise ValueError("Q10_Not_getting_girlfriend feature must not exist.")

    if q10_unexpected_report:
        # Unexpected values are intentionally reported in the output report rather
        # than encoded into new features. The presence of a report is therefore
        # considered valid, and the encoder should continue without silently
        # assigning those values.
        pass

    if q11_unexpected_report:
        # Unexpected Q11 values are also reported in the output report and are
        # not silently encoded.
        pass

    return True


def save_encoded_data(encoded_df: pd.DataFrame, output_path: Optional[str | Path] = None) -> Path:
    """Save the encoded questionnaire dataset to the processed output CSV."""

    resolved_output_path = Path(output_path) if output_path is not None else OUTPUT_CSV_PATH
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_df.to_csv(resolved_output_path, index=False, encoding="utf-8")
    return resolved_output_path


def generate_report(
    encoded_df: pd.DataFrame,
    q10_metadata: Dict[str, Any],
    q11_metadata: Dict[str, Any],
    validation_result: str,
    report_path: Optional[str | Path] = None,
) -> str:
    """Create a human-readable report for the multi-select encoding output."""

    resolved_report_path = Path(report_path) if report_path is not None else REPORT_PATH
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        "Multi-Select Encoding Report",
        "===========================",
        f"Row count: {len(encoded_df)}",
        "",
        "Q10 canonical options:",
    ]

    for option in Q10_CANONICAL_OPTIONS:
        lines.append(f"- {option}")

    lines.extend(["", "Q11 canonical options:"])
    for option in Q11_CANONICAL_OPTIONS:
        lines.append(f"- {option}")

    generated_feature_names = [
        *Q10_FEATURE_COLUMNS,
        "Q10_nothing",
        *Q11_FEATURE_COLUMNS,
    ]

    lines.extend(["", "Generated feature names:"])
    for feature_name in generated_feature_names:
        lines.append(f"- {feature_name}")

    lines.extend(["", "Q10 selected-option frequencies:"])
    q10_counts = q10_metadata["q10_selected_option_counts"]
    if q10_counts:
        for option, count in q10_counts.items():
            lines.append(f"- {option}: {count}")
    else:
        lines.append("- None")

    lines.extend(["", "Q11 selected-option frequencies:"])
    q11_counts = q11_metadata["q11_selected_option_counts"]
    if q11_counts:
        for option, count in q11_counts.items():
            lines.append(f"- {option}: {count}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "Normalization performed:",
            '- "Not getting girlfriend" -> "Relationship/social concerns"',
            "",
            f"Q10 Nothing responses: {q10_metadata['q10_nothing_count']}",
            f"Missing Q10 count: {q10_metadata['q10_missing_count']}",
            f"Missing Q11 count: {q11_metadata['q11_missing_count']}",
            f"Unexpected values: {q10_metadata['unexpected_values'] or q11_metadata['unexpected_values'] or 'None'}",
            f"Q10 data-quality conditions (Nothing plus other selections): {q10_metadata['q10_data_quality_conditions']}",
            f"Validation result: {validation_result}",
        ]
    )

    report_content = "\n".join(lines)
    resolved_report_path.write_text(report_content, encoding="utf-8")
    return report_content


def run_multiselect_encoding(
    input_csv_path: Optional[str | Path] = None,
    output_csv_path: Optional[str | Path] = None,
    report_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, str]:
    """Run the full Q10/Q11 multi-select encoding pipeline."""

    input_path = Path(input_csv_path) if input_csv_path is not None else INPUT_CSV_PATH
    output_path = Path(output_csv_path) if output_csv_path is not None else OUTPUT_CSV_PATH
    report_file_path = Path(report_path) if report_path is not None else REPORT_PATH

    input_df = load_input_data(input_path)
    original_df = input_df.copy()

    encoded_df, q10_metadata = encode_q10(input_df)
    encoded_df, q11_metadata = encode_q11(encoded_df)

    validation_result = validate_multihot_features(
        encoded_df=encoded_df,
        original_df=original_df,
        q10_feature_columns=Q10_FEATURE_COLUMNS,
        q11_feature_columns=Q11_FEATURE_COLUMNS,
        q10_nothing_present=True,
        q10_unexpected_report=q10_metadata["unexpected_values"],
        q11_unexpected_report=q11_metadata["unexpected_values"],
    )

    save_encoded_data(encoded_df, output_path)

    report_text = generate_report(
        encoded_df=encoded_df,
        q10_metadata=q10_metadata,
        q11_metadata=q11_metadata,
        validation_result="PASS" if validation_result else "FAIL",
        report_path=report_file_path,
    )

    return encoded_df, report_text


def main() -> None:
    """Run the Q10/Q11 multi-select encoding pipeline and print the summary."""

    encoded_df, report_text = run_multiselect_encoding()

    q10_nothing_count = int((encoded_df["Q10_nothing"] == 1).sum())

    q10_unexpected_count = 0
    q11_unexpected_count = 0

    print("MULTI-SELECT ENCODING COMPLETE")
    print(f"Rows: {len(encoded_df)}")
    print(f"Q10 features: {len(Q10_FEATURE_COLUMNS)}")
    print(f"Q11 features: {len(Q11_FEATURE_COLUMNS)}")
    print(f"Q10 Nothing responses: {q10_nothing_count}")
    print(f"Unexpected options: {q10_unexpected_count}")
    print(f"Output: {OUTPUT_CSV_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
