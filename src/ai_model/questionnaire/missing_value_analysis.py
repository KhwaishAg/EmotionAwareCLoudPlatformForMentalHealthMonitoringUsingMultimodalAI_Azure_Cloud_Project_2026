"""Missing-value analysis for the questionnaire modality.

This module performs a strict analysis-only pass over the processed questionnaire
CSV. It loads the cleaned dataset, quantifies missing values for every column,
identifies which questions contain missing responses, ranks columns by missingness,
finds completely empty columns, and surfaces rows with unusually high missingness.

It does not modify the cleaned dataset, fill missing values, delete rows or
columns, encode values, or create any downstream ML artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from .column_mapping import validate_column_mapping
    from .questionnaire_schema import QUESTIONNAIRE_SCHEMA
except ImportError:
    from column_mapping import validate_column_mapping
    from questionnaire_schema import QUESTIONNAIRE_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLEANED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_cleaned.csv"
DEFAULT_RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "questionnaire_responses.csv"
DEFAULT_REPORT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "missing_value_report.csv"
DEFAULT_REPORT_TEXT_PATH = PROJECT_ROOT / "data" / "processed" / "missing_value_report.txt"


def load_cleaned_data(
    csv_path: Optional[str | Path] = None,
    raw_csv_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load the cleaned questionnaire dataset without modifying it.

    Args:
        csv_path: Optional path to the cleaned CSV file.
        raw_csv_path: Optional path to the raw questionnaire CSV used for schema validation.

    Returns:
        A DataFrame containing the processed questionnaire responses.

    Raises:
        FileNotFoundError: If the cleaned CSV is missing.
        ValueError: If the raw CSV schema is invalid.
    """

    resolved_path = Path(csv_path) if csv_path is not None else DEFAULT_CLEANED_CSV_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Cleaned CSV file not found: {resolved_path}")

    resolved_raw_path = Path(raw_csv_path) if raw_csv_path is not None else DEFAULT_RAW_CSV_PATH

    # Reuse the existing column-mapping validation so the analysis starts from a
    # verified schema state rather than assuming the processed file is already valid.
    validate_column_mapping(resolved_raw_path)

    cleaned_df = pd.read_csv(resolved_path, encoding="utf-8")
    return cleaned_df.copy()


def calculate_missing_statistics(
    df: pd.DataFrame,
    raw_csv_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Calculate missing-value statistics for every column in the cleaned dataset.

    Args:
        df: The cleaned questionnaire DataFrame.
        raw_csv_path: Optional path to the raw questionnaire CSV used for schema validation.

    Returns:
        A DataFrame containing per-column missing-value statistics.
    """

    resolved_raw_path = Path(raw_csv_path) if raw_csv_path is not None else DEFAULT_RAW_CSV_PATH
    validate_column_mapping(resolved_raw_path)

    total_rows = len(df)
    rows = []

    question_ids = {question_id for question_id in QUESTIONNAIRE_SCHEMA if question_id != "Timestamp"}

    for column_name in df.columns:
        series = df[column_name]
        missing_count = int(series.isna().sum())
        non_missing_count = int(total_rows - missing_count)
        missing_percentage = (missing_count / total_rows * 100.0) if total_rows else 0.0

        schema_entry = QUESTIONNAIRE_SCHEMA.get(column_name)
        logical_question_id = schema_entry.question_id if schema_entry is not None else None
        question_group = schema_entry.question_group if schema_entry is not None else None

        if logical_question_id == "Timestamp":
            column_category = "Timestamp"
        elif logical_question_id in question_ids:
            column_category = "Q1-Q25"
        else:
            column_category = "Other"

        rows.append(
            {
                "column_name": column_name,
                "logical_question_id": logical_question_id,
                "question_group": question_group,
                "column_category": column_category,
                "total_rows": total_rows,
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
                "non_missing_count": non_missing_count,
                "has_missing_values": missing_count > 0,
                "completely_empty": missing_count == total_rows,
            }
        )

    statistics_df = pd.DataFrame(rows)
    statistics_df = statistics_df.sort_values(
        ["missing_percentage", "missing_count", "column_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return statistics_df


def identify_high_missing_rows(
    df: pd.DataFrame,
    high_missingness_ratio: float = 0.25,
) -> pd.DataFrame:
    """Identify rows whose missingness exceeds a configurable threshold.

    Args:
        df: The cleaned questionnaire DataFrame.
        high_missingness_ratio: Fraction of columns that must be missing before a
            row is treated as having unusually high missingness.

    Returns:
        A DataFrame containing row-level missingness details.
    """

    total_columns = len(df.columns)
    high_missing_count_threshold = max(1, int(total_columns * high_missingness_ratio))

    missing_counts = df.isna().sum(axis=1)
    high_rows = missing_counts[missing_counts >= high_missing_count_threshold]

    if high_rows.empty:
        return pd.DataFrame(
            columns=[
                "row_index",
                "missing_count",
                "missing_percentage",
                "missing_columns",
            ]
        )

    rows = []
    for row_index, missing_count in high_rows.items():
        missing_columns = [column for column in df.columns if pd.isna(df.at[row_index, column])]
        rows.append(
            {
                "row_index": row_index,
                "missing_count": int(missing_count),
                "missing_percentage": (missing_count / total_columns) * 100.0,
                "missing_columns": missing_columns,
            }
        )

    high_rows_df = pd.DataFrame(rows)
    high_rows_df = high_rows_df.sort_values(
        ["missing_count", "row_index"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return high_rows_df


def generate_missing_value_report(
    cleaned_df: Optional[pd.DataFrame] = None,
    raw_csv_path: Optional[str | Path] = None,
    report_csv_path: Optional[str | Path] = None,
    report_text_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Generate the CSV and text missing-value reports.

    Args:
        cleaned_df: Optional cleaned DataFrame. If omitted, the default processed CSV is loaded.
        raw_csv_path: Optional path to the raw CSV used for validation.
        report_csv_path: Optional path to the CSV report output.
        report_text_path: Optional path to the text report output.

    Returns:
        A dictionary containing the statistics table, high-missingness rows, and output paths.
    """

    resolved_cleaned_path = Path(cleaned_df) if isinstance(cleaned_df, (str, Path)) else None
    if cleaned_df is None:
        cleaned_df = load_cleaned_data()

    resolved_raw_path = Path(raw_csv_path) if raw_csv_path is not None else DEFAULT_RAW_CSV_PATH
    resolved_report_csv_path = Path(report_csv_path) if report_csv_path is not None else DEFAULT_REPORT_CSV_PATH
    resolved_report_text_path = Path(report_text_path) if report_text_path is not None else DEFAULT_REPORT_TEXT_PATH

    if resolved_cleaned_path is not None and not resolved_cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned CSV file not found: {resolved_cleaned_path}")

    statistics_df = calculate_missing_statistics(cleaned_df, raw_csv_path=resolved_raw_path)
    high_missing_rows_df = identify_high_missing_rows(cleaned_df)

    total_rows = len(cleaned_df)
    total_columns = len(cleaned_df.columns)
    total_missing_cells = int(statistics_df["missing_count"].sum())
    overall_missing_cell_percentage = (
        (total_missing_cells / (total_rows * total_columns)) * 100.0 if (total_rows * total_columns) else 0.0
    )

    statistics_for_output = statistics_df.copy()
    statistics_for_output["missing_percentage"] = statistics_for_output["missing_percentage"].round(2)

    resolved_report_csv_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_for_output.to_csv(resolved_report_csv_path, index=False, encoding="utf-8")

    timestamp_stats = statistics_df[statistics_df["column_category"] == "Timestamp"]
    question_stats = statistics_df[statistics_df["column_category"] == "Q1-Q25"]

    questions_with_missing_values = question_stats[question_stats["has_missing_values"]]["logical_question_id"].tolist()
    questions_without_missing_values = question_stats[~question_stats["has_missing_values"]]["logical_question_id"].tolist()
    completely_empty_columns = statistics_df[statistics_df["completely_empty"]]["column_name"].tolist()

    report_lines: List[str] = [
        "Missing Value Analysis Report",
        "============================",
        f"Total rows: {total_rows}",
        f"Total columns: {total_columns}",
        f"Total missing cells: {total_missing_cells}",
        f"Overall missing-cell percentage: {overall_missing_cell_percentage:.2f}%",
        "",
        "Timestamp column statistics:",
    ]

    if timestamp_stats.empty:
        report_lines.append("No Timestamp column found.")
    else:
        for _, row in timestamp_stats.iterrows():
            report_lines.append(
                f"- {row['column_name']}: missing {row['missing_count']} of {row['total_rows']} rows "
                f"({row['missing_percentage']:.2f}%); non-missing: {row['non_missing_count']}"
            )

    report_lines.extend(["", "Per-question missing counts and percentages:"])
    if question_stats.empty:
        report_lines.append("No questionnaire questions found.")
    else:
        for _, row in question_stats.iterrows():
            report_lines.append(
                f"- {row['column_name']} ({row['logical_question_id']}): missing {row['missing_count']} of "
                f"{row['total_rows']} rows ({row['missing_percentage']:.2f}%); non-missing: {row['non_missing_count']}"
            )

    report_lines.extend(["", "Questions with no missing values:"])
    if questions_without_missing_values:
        report_lines.extend(f"- {question}" for question in questions_without_missing_values)
    else:
        report_lines.append("None")

    report_lines.extend(["", "Questions with missing values:"])
    if questions_with_missing_values:
        report_lines.extend(f"- {question}" for question in questions_with_missing_values)
    else:
        report_lines.append("None")

    report_lines.extend(["", "Completely empty columns:"])
    if completely_empty_columns:
        report_lines.extend(f"- {column_name}" for column_name in completely_empty_columns)
    else:
        report_lines.append("None")

    report_lines.extend(["", "Rows with unusually high missingness:"])
    if high_missing_rows_df.empty:
        report_lines.append("None")
    else:
        report_lines.append(
            f"Rows with at least {max(1, int(len(cleaned_df.columns) * 0.25))} missing answers are flagged as high-missingness rows."
        )
        for _, row in high_missing_rows_df.iterrows():
            missing_columns_text = "; ".join(row["missing_columns"])
            report_lines.append(
                f"- Row {row['row_index']}: {row['missing_count']} missing answers "
                f"({row['missing_percentage']:.2f}%); missing columns: {missing_columns_text}"
            )

    resolved_report_text_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(report_lines) + "\n"
    resolved_report_text_path.write_text(report_text, encoding="utf-8")

    return {
        "cleaned_df": cleaned_df,
        "statistics_df": statistics_df,
        "high_missing_rows_df": high_missing_rows_df,
        "report_csv_path": str(resolved_report_csv_path),
        "report_text_path": str(resolved_report_text_path),
        "total_rows": total_rows,
        "total_columns": total_columns,
        "total_missing_cells": total_missing_cells,
        "overall_missing_cell_percentage": overall_missing_cell_percentage,
        "report_text": report_text,
    }


def run_missing_value_analysis(
    cleaned_csv_path: Optional[str | Path] = None,
    report_csv_path: Optional[str | Path] = None,
    report_text_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run the complete missing-value analysis pipeline.

    Args:
        cleaned_csv_path: Optional path to the cleaned questionnaire CSV.
        report_csv_path: Optional path to the output CSV report.
        report_text_path: Optional path to the output text report.

    Returns:
        A tuple containing the cleaned DataFrame, the statistics DataFrame, and the
        generated report metadata dictionary.
    """

    resolved_cleaned_csv_path = Path(cleaned_csv_path) if cleaned_csv_path is not None else DEFAULT_CLEANED_CSV_PATH
    resolved_report_csv_path = Path(report_csv_path) if report_csv_path is not None else DEFAULT_REPORT_CSV_PATH
    resolved_report_text_path = Path(report_text_path) if report_text_path is not None else DEFAULT_REPORT_TEXT_PATH

    cleaned_df = load_cleaned_data(resolved_cleaned_csv_path)
    report_data = generate_missing_value_report(
        cleaned_df=cleaned_df,
        report_csv_path=resolved_report_csv_path,
        report_text_path=resolved_report_text_path,
    )

    statistics_df = report_data["statistics_df"]
    high_missing_rows_df = report_data["high_missing_rows_df"]

    return cleaned_df, statistics_df, {**report_data, "high_missing_rows_df": high_missing_rows_df}


def main() -> None:
    """Execute the analysis pipeline for the default processed dataset."""

    cleaned_df, statistics_df, report_data = run_missing_value_analysis()

    print("MISSING VALUE ANALYSIS COMPLETE")
    print(f"Total missing cells: {report_data['total_missing_cells']}")
    print(f"Report CSV: {report_data['report_csv_path']}")
    print(f"Report Text: {report_data['report_text_path']}")


if __name__ == "__main__":
    main()
