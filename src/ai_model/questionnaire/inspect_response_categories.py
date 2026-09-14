"""Inspection utility for questionnaire response categories.

This script is intentionally analysis-only. It loads the cleaned questionnaire CSV,
resolves Q3 through Q20 to their actual CSV column names via the existing
column-mapping module, and prints each unique non-missing response together with
its frequency count.

It does not encode, impute, modify, or otherwise transform the dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    from .column_mapping import get_column_for_question
except ImportError:  # pragma: no cover - fallback for direct script execution
    from column_mapping import get_column_for_question


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLEANED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_cleaned.csv"
QUESTION_START = 3
QUESTION_END = 20


def load_cleaned_data(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the cleaned questionnaire CSV without modifying it."""

    resolved_path = Path(csv_path) if csv_path is not None else DEFAULT_CLEANED_CSV_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Cleaned CSV file not found: {resolved_path}")

    return pd.read_csv(resolved_path, encoding="utf-8")


def _format_response_counts(series: pd.Series) -> pd.DataFrame:
    """Return unique response values and their counts for display."""

    non_missing_series = series.dropna()
    if non_missing_series.empty:
        return pd.DataFrame(columns=["response", "count"])

    counts = non_missing_series.value_counts(dropna=True)
    counts_df = counts.reset_index()
    counts_df.columns = ["response", "count"]
    counts_df["response"] = counts_df["response"].astype(str)
    counts_df = counts_df.sort_values(["response", "count"], ascending=[True, False], ignore_index=True)
    return counts_df


def inspect_questionnaire_responses(
    cleaned_csv_path: Optional[str | Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Inspect every response category for questions Q3 through Q20.

    Args:
        cleaned_csv_path: Optional path to the cleaned CSV file.

    Returns:
        A dictionary keyed by question ID, each containing a DataFrame with
        unique non-missing response values and their counts.
    """

    df = load_cleaned_data(cleaned_csv_path)

    results: Dict[str, pd.DataFrame] = {}
    for question_number in range(QUESTION_START, QUESTION_END + 1):
        question_id = f"Q{question_number}"
        actual_column_name = get_column_for_question(question_id, DEFAULT_CLEANED_CSV_PATH)
        series = df[actual_column_name]
        results[question_id] = _format_response_counts(series)

    return results


def _print_question_summary(question_id: str, actual_column_name: str, response_counts: pd.DataFrame) -> None:
    """Print one question's inspection summary in a readable format."""

    print(f"\n{'=' * 80}")
    print(f"{question_id} | Column: {actual_column_name}")
    print(f"{'=' * 80}")

    if response_counts.empty:
        print("No non-missing responses found.")
        return

    for _, row in response_counts.iterrows():
        response_text = str(row["response"])
        count = int(row["count"])
        print(f"- {response_text}: {count}")


def main() -> None:
    """Run the inspection utility and print response categories for Q3 through Q20."""

    df = load_cleaned_data(DEFAULT_CLEANED_CSV_PATH)

    print("QUESTIONNAIRE RESPONSE CATEGORY INSPECTION")
    print("=" * 80)
    print(f"Loaded cleaned dataset: {DEFAULT_CLEANED_CSV_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    for question_number in range(QUESTION_START, QUESTION_END + 1):
        question_id = f"Q{question_number}"
        actual_column_name = get_column_for_question(question_id, DEFAULT_CLEANED_CSV_PATH)
        response_counts = _format_response_counts(df[actual_column_name])
        _print_question_summary(question_id, actual_column_name, response_counts)


if __name__ == "__main__":
    main()
