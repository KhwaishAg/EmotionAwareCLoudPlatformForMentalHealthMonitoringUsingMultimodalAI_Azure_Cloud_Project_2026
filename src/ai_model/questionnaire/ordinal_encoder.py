"""Semantic ordinal encoding for the questionnaire modality.

This module performs ordinal encoding only for the questionnaire questions that
require a deterministic semantic mapping. It does not create targets, stress
scores, or ML models. It also does not modify the raw CSV or the cleaned CSV.

The module creates a new encoded dataset in the processed folder, along with a
text report describing the mappings, missing values, special-case handling, and
any unexpected categories found during validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from .column_mapping import get_column_for_question, validate_column_mapping
except ImportError:  # pragma: no cover - fallback for direct script execution
    from column_mapping import get_column_for_question, validate_column_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLEANED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_cleaned.csv"
DEFAULT_RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "questionnaire_responses.csv"
DEFAULT_OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_ordinal_encoded.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "ordinal_encoding_report.txt"

ENCODED_QUESTION_IDS: Tuple[str, ...] = (
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "Q7",
    "Q8",
    "Q9",
    "Q12",
    "Q13",
    "Q14",
    "Q15",
    "Q16",
    "Q17",
    "Q18",
    "Q19",
    "Q20",
)

MULTI_SELECT_EXCLUDED_QUESTIONS: Tuple[str, ...] = ("Q10", "Q11")

QUESTION_CATEGORY_MAPPINGS: Dict[str, Dict[str, int]] = {
    "Q3": {
        "Less than 2 hours": 0,
        "2–4 hours": 1,
        "4–6 hours": 2,
        "6–8 hours": 3,
        "More than 8 hours": 4,
    },
    "Q4": {
        "1 — Not at all": 1,
        "2 — Slightly": 2,
        "3 — Moderately": 3,
        "4 — Very": 4,
        "5 — Extremely": 5,
    },
    "Q5": {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3,
        "Always": 4,
    },
    "Q6": {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3,
        "Always": 4,
    },
    "Q7": {
        "Not difficult at all": 0,
        "Slightly difficult": 1,
        "Moderately difficult": 2,
        "Very difficult": 3,
        "Extremely difficult": 4,
    },
    "Q8": {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3,
        "Always": 4,
    },
    "Q9": {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3,
        "Always": 4,
    },
    "Q12": {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
    },
    "Q13": {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
    },
    "Q14": {
        "Less than 5 minutes": 0,
        "5–15 minutes": 1,
        "15–30 minutes": 2,
        "30–60 minutes": 3,
        "1–3 hours": 4,
        "More than 3 hours": 5,
        "Most of the day": 6,
    },
    "Q15": {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3,
        "Very often": 4,
    },
    "Q16": {
        "Within a few minutes": 0,
        "Within 15–30 minutes": 1,
        "Within an hour": 2,
        "Within a few hours": 3,
        "By the end of the day": 4,
        "It usually takes longer than a day": 5,
    },
    "Q17": {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3,
        "Very often": 4,
    },
    "Q18": {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3,
        "Very often": 4,
    },
    "Q19": {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3,
        "Very often": 4,
    },
    "Q20": {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3,
        "Very often": 4,
    },
}

SPECIAL_RESPONSE_TEXT = "It varies considerably"


def load_cleaned_data(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the cleaned questionnaire dataset without modifying it.

    Args:
        csv_path: Optional path to the cleaned CSV file.

    Returns:
        A DataFrame copy of the cleaned questionnaire dataset.

    Raises:
        FileNotFoundError: If the cleaned CSV does not exist.
    """

    resolved_path = Path(csv_path) if csv_path is not None else DEFAULT_CLEANED_CSV_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Cleaned CSV file not found: {resolved_path}")

    validate_column_mapping(DEFAULT_RAW_CSV_PATH)

    cleaned_df = pd.read_csv(resolved_path, encoding="utf-8", dtype=str)
    return cleaned_df.copy()


def _get_question_column(question_id: str, cleaned_df: Optional[pd.DataFrame] = None) -> str:
    """Return the logical question column name used by the cleaned dataset.

    The module uses the existing column-mapping helper to validate the raw schema,
    then resolves the logical Q-ID in the cleaned dataset. This keeps the encoder
    aligned with the project-defined logical question IDs without relying on the
    long raw column names from the original CSV.
    """

    if question_id not in QUESTION_CATEGORY_MAPPINGS:
        raise ValueError(f"Unsupported question ID for ordinal encoding: {question_id}")

    # Validate that the question exists in the project-defined mapping and raw CSV.
    get_column_for_question(question_id, DEFAULT_RAW_CSV_PATH)

    if cleaned_df is None:
        return question_id

    if question_id in cleaned_df.columns:
        return question_id

    raise ValueError(
        f"Question {question_id} is not present in the cleaned dataset columns."
    )


def encode_ordinal_features(
    cleaned_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create semantic ordinal-encoded columns for the selected questionnaire questions.

    Args:
        cleaned_df: The cleaned questionnaire DataFrame.

    Returns:
        A tuple containing the encoded DataFrame and a metadata dictionary.
    """

    encoded_df = cleaned_df.copy()
    unexpected_categories: List[Dict[str, Any]] = []
    missing_encoded_values: Dict[str, int] = {}

    for question_id in ENCODED_QUESTION_IDS:
        source_column = _get_question_column(question_id, encoded_df)
        encoded_column = f"{question_id}_encoded"
        encoded_series = pd.Series([None] * len(encoded_df), index=encoded_df.index, dtype="float64")

        if question_id in {"Q14", "Q16"}:
            varies_column = f"{question_id}_varies"
            varies_series = pd.Series(0, index=encoded_df.index, dtype="int64")

        for row_index, response in enumerate(encoded_df[source_column]):
            if pd.isna(response):
                continue

            response_text = str(response)

            if question_id in {"Q14", "Q16"} and response_text == SPECIAL_RESPONSE_TEXT:
                encoded_series.iloc[row_index] = pd.NA
                varies_series.iloc[row_index] = 1
                continue

            mapping = QUESTION_CATEGORY_MAPPINGS[question_id]
            if response_text in mapping:
                encoded_series.iloc[row_index] = mapping[response_text]
            else:
                unexpected_categories.append(
                    {
                        "question_id": question_id,
                        "row_index": row_index,
                        "response": response_text,
                        "actual_column": source_column,
                    }
                )

        encoded_df[encoded_column] = encoded_series

        if question_id in {"Q14", "Q16"}:
            encoded_df[varies_column] = varies_series

        missing_encoded_values[encoded_column] = int(encoded_series.isna().sum())

    return encoded_df, {
        "unexpected_categories": unexpected_categories,
        "missing_encoded_values": missing_encoded_values,
        "q14_varies_count": int((encoded_df["Q14_varies"] == 1).sum()),
        "q16_varies_count": int((encoded_df["Q16_varies"] == 1).sum()),
        "encoded_questions": list(ENCODED_QUESTION_IDS),
        "excluded_questions": list(MULTI_SELECT_EXCLUDED_QUESTIONS),
    }


def validate_encoded_features(encoded_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate the encoded dataset and report any unexpected categories.

    Args:
        encoded_df: The encoded DataFrame.

    Returns:
        A dictionary containing validation results and summary information.
    """

    validation_results: Dict[str, Any] = {
        "unexpected_categories": [],
        "missing_encoded_values": {},
        "encoded_questions": list(ENCODED_QUESTION_IDS),
        "excluded_questions": list(MULTI_SELECT_EXCLUDED_QUESTIONS),
    }

    for question_id in ENCODED_QUESTION_IDS:
        encoded_column = f"{question_id}_encoded"
        if encoded_column not in encoded_df.columns:
            raise ValueError(f"Missing encoded column expected by validation: {encoded_column}")

        validation_results["missing_encoded_values"][encoded_column] = int(
            encoded_df[encoded_column].isna().sum()
        )

    if "Q14_varies" not in encoded_df.columns or "Q16_varies" not in encoded_df.columns:
        raise ValueError("Missing Q14_varies or Q16_varies columns after encoding.")

    validation_results["q14_varies_count"] = int((encoded_df["Q14_varies"] == 1).sum())
    validation_results["q16_varies_count"] = int((encoded_df["Q16_varies"] == 1).sum())

    # Re-scan the source questionnaire responses to ensure only explicitly mapped
    # responses are assigned an ordinal value. Any other non-missing response is
    # reported as unexpected instead of being silently coerced.
    for question_id in ENCODED_QUESTION_IDS:
        source_column = _get_question_column(question_id, encoded_df)

        for row_index, response in enumerate(encoded_df[source_column]):
            if pd.isna(response):
                continue

            response_text = str(response)
            if question_id in {"Q14", "Q16"} and response_text == SPECIAL_RESPONSE_TEXT:
                continue

            if response_text not in QUESTION_CATEGORY_MAPPINGS[question_id]:
                validation_results["unexpected_categories"].append(
                    {
                        "question_id": question_id,
                        "row_index": row_index,
                        "response": response_text,
                        "actual_column": source_column,
                    }
                )

    return validation_results


def save_encoded_data(
    encoded_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Save the encoded DataFrame to the processed output CSV."""

    resolved_output_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_CSV_PATH
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_df.to_csv(resolved_output_path, index=False, encoding="utf-8")
    return resolved_output_path


def generate_ordinal_encoding_report(
    encoded_df: pd.DataFrame,
    validation_results: Dict[str, Any],
    report_path: Optional[str | Path] = None,
) -> str:
    """Generate the text report describing the ordinal encoding results."""

    resolved_report_path = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

    encoded_questions = validation_results["encoded_questions"]
    excluded_questions = validation_results["excluded_questions"]
    q14_varies_count = validation_results["q14_varies_count"]
    q16_varies_count = validation_results["q16_varies_count"]
    unexpected_categories = validation_results["unexpected_categories"]

    lines: List[str] = [
        "Ordinal Encoding Report",
        "======================",
        f"Number of rows: {len(encoded_df)}",
        f"Encoded questions: {len(encoded_questions)}",
        "",
        "Excluded multi-select questions (handled later):",
    ]

    for question_id in excluded_questions:
        lines.append(f"- {question_id}")

    lines.extend([
        "",
        "Category mappings:",
    ])

    for question_id in encoded_questions:
        lines.append(f"- {question_id}: {QUESTION_CATEGORY_MAPPINGS[question_id]}")

    lines.extend([
        "",
        "Missing encoded values:",
    ])

    for question_id in encoded_questions:
        encoded_column = f"{question_id}_encoded"
        count = validation_results["missing_encoded_values"].get(encoded_column, 0)
        lines.append(f"- {encoded_column}: {count}")

    lines.extend([
        "",
        f"Q14 \"{SPECIAL_RESPONSE_TEXT}\" count: {q14_varies_count}",
        f"Q16 \"{SPECIAL_RESPONSE_TEXT}\" count: {q16_varies_count}",
        "",
        "Unexpected categories:",
    ])

    if unexpected_categories:
        for item in unexpected_categories:
            lines.append(
                f"- Row {item['row_index']} | Question {item['question_id']} | "
                f"Column {item['actual_column']} | Response: {item['response']}"
            )
    else:
        lines.append("None")

    report_text = "\n".join(lines) + "\n"
    resolved_report_path.write_text(report_text, encoding="utf-8")
    return report_text


def run_ordinal_encoding(
    cleaned_csv_path: Optional[str | Path] = None,
    output_csv_path: Optional[str | Path] = None,
    report_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """Run the semantic ordinal encoding pipeline and save its outputs.

    Args:
        cleaned_csv_path: Optional path to the cleaned CSV input.
        output_csv_path: Optional path to the encoded CSV output.
        report_path: Optional path to the encoding report text file.

    Returns:
        A tuple containing the encoded DataFrame, validation metadata, and report text.
    """

    resolved_cleaned_csv_path = Path(cleaned_csv_path) if cleaned_csv_path is not None else DEFAULT_CLEANED_CSV_PATH
    resolved_output_csv_path = Path(output_csv_path) if output_csv_path is not None else DEFAULT_OUTPUT_CSV_PATH
    resolved_report_path = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH

    cleaned_df = load_cleaned_data(resolved_cleaned_csv_path)
    encoded_df, encode_metadata = encode_ordinal_features(cleaned_df)
    validation_results = validate_encoded_features(encoded_df)

    save_encoded_data(encoded_df, resolved_output_csv_path)
    report_text = generate_ordinal_encoding_report(
        encoded_df,
        validation_results,
        resolved_report_path,
    )

    return encoded_df, validation_results, report_text


def main() -> None:
    """Run the ordinal encoding pipeline and print a concise summary."""

    encoded_df, validation_results, report_text = run_ordinal_encoding()
    unexpected_categories_count = len(validation_results["unexpected_categories"])

    print("ORDINAL ENCODING COMPLETE")
    print(f"Rows: {len(encoded_df)}")
    print(f"Encoded questions: {len(validation_results['encoded_questions'])}")
    print(f"Unexpected categories: {unexpected_categories_count}")
    print(f"Output: {DEFAULT_OUTPUT_CSV_PATH}")
    print(f"Report: {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()
