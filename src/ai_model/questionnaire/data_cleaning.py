"""Data cleaning pipeline for the questionnaire modality.

This module is intentionally limited to data cleaning only. It loads the raw
questionnaire CSV, validates the current header against the existing schema,
normalizes values in memory, writes a cleaned CSV to the processed folder, and
creates a cleaning report.

It does not perform ML, encoding, target creation, or any modification of the
raw CSV file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from .column_mapping import validate_column_mapping
except ImportError:
    from column_mapping import validate_column_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "questionnaire_responses.csv"
DEFAULT_OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_cleaned.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaning_report.txt"

MULTI_SELECT_COLUMNS = {"Q2", "Q10", "Q11"}
TEXT_LIKE_COLUMNS = {"Q22", "Q23"}


def load_raw_data(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the raw questionnaire CSV as a DataFrame.

    The CSV is loaded without modifying the original file. The header is also
    validated via the existing column mapping module before processing begins.

    Args:
        csv_path: Optional path to the raw questionnaire CSV.

    Returns:
        A pandas DataFrame containing the raw questionnaire responses.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV header is not valid for this project.
    """

    resolved_path = Path(csv_path) if csv_path is not None else DEFAULT_RAW_CSV_PATH

    # Validation is performed before any data is transformed.
    validate_column_mapping(resolved_path)

    raw_df = pd.read_csv(resolved_path, dtype=str, encoding="utf-8-sig")
    return raw_df.copy()


def _count_blank_or_null_values(df: pd.DataFrame) -> int:
    """Count cells that are missing or effectively blank strings."""

    count = 0
    for row in df.itertuples(index=False, name=None):
        for value in row:
            if pd.isna(value):
                count += 1
            elif isinstance(value, str) and value.strip() == "":
                count += 1
    return count


def _normalize_text_value(value: Any) -> Any:
    """Normalize surrounding whitespace and repeated internal whitespace for strings."""

    if pd.isna(value):
        return value

    if not isinstance(value, str):
        return value

    trimmed = value.strip()
    normalized = re.sub(r"\s+", " ", trimmed)
    return normalized


def _normalize_multiselect_value(value: Any) -> Any:
    """Normalize multi-select answers while preserving individual selected options."""

    if pd.isna(value):
        return value

    if not isinstance(value, str):
        return value

    parts = [part.strip() for part in value.split(";")]
    cleaned_parts = [re.sub(r"\s+", " ", part).strip() for part in parts if part.strip()]
    return "; ".join(cleaned_parts) if cleaned_parts else ""


def _normalize_google_forms_timestamp(value: Any) -> Any:
    """Normalize Google Forms GMT offsets like GMT+5:30 to GMT+05:30.

    This keeps the rest of the cleaning pipeline unchanged while allowing pandas
    to parse the timestamp using an explicit format.
    """

    if pd.isna(value):
        return value

    if not isinstance(value, str):
        return value

    return re.sub(
        r"GMT([+-])(\d):(\d{2})$",
        lambda match: f"GMT{match.group(1)}0{match.group(2)}:{match.group(3)}",
        value,
    )


def clean_dataframe(
    df: pd.DataFrame,
    csv_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean a questionnaire DataFrame without modifying the raw CSV.

    The returned DataFrame uses logical question IDs (Q1..Q25, Timestamp) as
    columns. The raw CSV itself is never rewritten.

    Args:
        df: Raw questionnaire DataFrame loaded from the CSV.
        csv_path: Optional path to the raw CSV, used only for validation.

    Returns:
        A tuple containing the cleaned DataFrame and a summary dictionary.
    """

    validated_mapping = validate_column_mapping(csv_path)
    reverse_mapping = {actual_column: logical_id for logical_id, actual_column in validated_mapping.items()}

    original_df = df.copy()
    working_df = df.copy()

    # Rename in-memory columns to logical IDs for consistency with schema.
    working_df = working_df.rename(columns=reverse_mapping)

    # Preserve the logical column order expected by the schema.
    ordered_columns = ["Timestamp", *[f"Q{i}" for i in range(1, 26)]]
    working_df = working_df.reindex(columns=ordered_columns)

    # Parse Timestamp explicitly and track any failures.
    original_timestamp = original_df[validated_mapping["Timestamp"]].copy()
    normalized_timestamp = original_timestamp.map(_normalize_google_forms_timestamp)
    parsed_timestamp = pd.to_datetime(
        normalized_timestamp,
        format="%Y/%m/%d %I:%M:%S %p GMT%z",
        errors="coerce",
    )
    timestamp_failed_parsing = int((original_timestamp.notna() & parsed_timestamp.isna()).sum())
    working_df["Timestamp"] = parsed_timestamp

    whitespace_normalized_cells = 0

    # Clean all non-Timestamp columns.
    for question_id in [f"Q{i}" for i in range(1, 26)]:
        series = working_df[question_id].copy()

        if question_id in MULTI_SELECT_COLUMNS:
            cleaned_series = series.map(_normalize_multiselect_value)
        else:
            cleaned_series = series.map(_normalize_text_value)

        # Count cells where text was changed.
        for index, (original_value, cleaned_value) in enumerate(zip(series, cleaned_series)):
            if isinstance(original_value, str) and isinstance(cleaned_value, str):
                if original_value != cleaned_value:
                    whitespace_normalized_cells += 1

        working_df[question_id] = cleaned_series

    # Preserve missing values and report malformed rows rather than deleting them.
    malformed_rows = []
    for index, row in original_df.iterrows():
        if row.isna().all():
            malformed_rows.append(int(index))

    original_row_count = len(original_df)
    final_row_count = len(working_df)
    original_column_count = len(original_df.columns)
    final_column_count = len(working_df.columns)

    report: Dict[str, Any] = {
        "original_row_count": original_row_count,
        "final_row_count": final_row_count,
        "original_column_count": original_column_count,
        "final_column_count": final_column_count,
        "timestamp_failed_parsing": timestamp_failed_parsing,
        "duplicate_rows_detected": int(original_df.duplicated().sum()),
        "blank_or_null_before_cleaning": _count_blank_or_null_values(original_df),
        "blank_or_null_after_cleaning": _count_blank_or_null_values(working_df),
        "string_cells_whitespace_normalized": whitespace_normalized_cells,
        "malformed_rows": malformed_rows,
        "raw_csv_path": str(DEFAULT_RAW_CSV_PATH),
        "processed_csv_path": str(DEFAULT_OUTPUT_CSV_PATH),
        "report_path": str(DEFAULT_REPORT_PATH),
    }

    return working_df, report


def save_cleaned_data(
    cleaned_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Save the cleaned DataFrame to the processed output CSV."""

    resolved_output_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_CSV_PATH
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(resolved_output_path, index=False, encoding="utf-8")
    return resolved_output_path


def generate_cleaning_report(report: Dict[str, Any], report_path: Optional[str | Path] = None) -> str:
    """Generate a human-readable cleaning summary report."""

    resolved_report_path = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Questionnaire Cleaning Report",
        "=" * 28,
        f"Original row count: {report['original_row_count']}",
        f"Final row count: {report['final_row_count']}",
        f"Original column count: {report['original_column_count']}",
        f"Final column count: {report['final_column_count']}",
        f"Timestamps that failed parsing: {report['timestamp_failed_parsing']}",
        f"Duplicate rows detected: {report['duplicate_rows_detected']}",
        f"Blank/null values before cleaning: {report['blank_or_null_before_cleaning']}",
        f"Blank/null values after cleaning: {report['blank_or_null_after_cleaning']}",
        f"String cells whose whitespace was normalized: {report['string_cells_whitespace_normalized']}",
    ]

    if report.get("malformed_rows"):
        lines.append(f"Malformed rows detected: {report['malformed_rows']}")
    else:
        lines.append("Malformed rows detected: []")

    report_text = "\n".join(lines) + "\n"
    resolved_report_path.write_text(report_text, encoding="utf-8")
    return report_text


def run_cleaning_pipeline(
    csv_path: Optional[str | Path] = None,
    output_csv_path: Optional[str | Path] = None,
    report_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """Run the complete questionnaire cleaning pipeline.

    Args:
        csv_path: Optional path to the raw questionnaire CSV.
        output_csv_path: Optional path for the cleaned output CSV.
        report_path: Optional path for the text report.

    Returns:
        A tuple of (cleaned_dataframe, report_dictionary, report_text).
    """

    resolved_csv_path = Path(csv_path) if csv_path is not None else DEFAULT_RAW_CSV_PATH
    resolved_output = Path(output_csv_path) if output_csv_path is not None else DEFAULT_OUTPUT_CSV_PATH
    resolved_report = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH

    raw_df = load_raw_data(resolved_csv_path)
    cleaned_df, report = clean_dataframe(raw_df, resolved_csv_path)

    save_cleaned_data(cleaned_df, resolved_output)
    report_text = generate_cleaning_report(report, resolved_report)

    return cleaned_df, report, report_text


def main() -> None:
    """Run the cleaning pipeline and print a concise summary."""

    cleaned_df, report, report_text = run_cleaning_pipeline()

    print("DATA CLEANING COMPLETE")
    print(
        f"Rows: {report['original_row_count']} -> {report['final_row_count']}"
    )
    print(
        f"Columns: {report['original_column_count']} -> {report['final_column_count']}"
    )
    print(f"Output: {report['processed_csv_path']}")
    print(f"Report: {report['report_path']}")


if __name__ == "__main__":
    main()
