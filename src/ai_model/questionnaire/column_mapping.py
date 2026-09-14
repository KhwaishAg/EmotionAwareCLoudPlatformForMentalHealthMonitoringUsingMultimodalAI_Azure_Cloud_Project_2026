"""Column mapping and validation utilities for the questionnaire modality.

This module loads the current questionnaire CSV header, infers the logical
question IDs from the column names, validates them against the formal schema,
and provides reusable helpers for later preprocessing or feature-engineering
steps.

This file intentionally does not modify the raw CSV, does not create target
variables, and does not perform preprocessing or encoding.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .questionnaire_schema import QUESTIONNAIRE_SCHEMA
except ImportError:
    from questionnaire_schema import QUESTIONNAIRE_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "questionnaire_responses.csv"
EXPECTED_QUESTION_IDS = [f"Q{i}" for i in range(1, 26)]
EXPECTED_LOGICAL_IDS = {"Timestamp", *EXPECTED_QUESTION_IDS}
QUESTION_ID_PATTERN = re.compile(r"^(Q\d+)\b")


def load_column_names(csv_path: Optional[str | Path] = None) -> List[str]:
    """Load the raw CSV header without modifying the source file.

    Args:
        csv_path: Optional path to the questionnaire CSV file.

    Returns:
        The raw column names in the CSV header.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """

    resolved_path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    with resolved_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {resolved_path}") from exc

    return header


def build_column_mapping(csv_path: Optional[str | Path] = None) -> Dict[str, str]:
    """Build a logical_question_id -> actual_csv_column_name mapping.

    The mapping is derived from the actual CSV header. Logical IDs are inferred
    from the beginning of each column name, for example:

    - "Q1. What is your current year of study?" -> "Q1"
    - "Q25. Which type of situation most strongly affects your stress?" -> "Q25"

    The Timestamp column is included as a separate mapping entry.

    Args:
        csv_path: Optional path to the questionnaire CSV file.

    Returns:
        A dictionary mapping logical question IDs to their raw CSV column names.

    Raises:
        ValueError: If duplicate logical IDs are found.
    """

    column_names = load_column_names(csv_path)
    mapping: Dict[str, str] = {}
    duplicates: Dict[str, List[str]] = {}

    for column_name in column_names:
        logical_id = _infer_logical_question_id(column_name)
        if logical_id is None:
            continue

        if logical_id in mapping:
            duplicates.setdefault(logical_id, [mapping[logical_id]])
            duplicates[logical_id].append(column_name)
            continue

        mapping[logical_id] = column_name

    if duplicates:
        duplicate_details = "; ".join(
            f"{logical_id} -> {', '.join(columns)}" for logical_id, columns in duplicates.items()
        )
        raise ValueError(f"Duplicate logical question IDs detected in CSV header: {duplicate_details}")

    return mapping


def validate_column_mapping(csv_path: Optional[str | Path] = None) -> Dict[str, str]:
    """Validate the discovered CSV-to-question mapping against the schema.

    This function checks that:
      - Timestamp exists exactly once
      - Q1 through Q25 each exist exactly once
      - no missing logical question IDs exist
      - no duplicate logical IDs exist
      - every mapped CSV column name actually exists in the CSV
      - the number of mapped questionnaire questions is exactly 25
      - the discovered logical IDs match QUESTIONNAIRE_SCHEMA

    Args:
        csv_path: Optional path to the questionnaire CSV file.

    Returns:
        The validated mapping dictionary.

    Raises:
        ValueError: If the CSV header is missing, malformed, or inconsistent
            with the schema.
    """

    column_names = load_column_names(csv_path)
    mapping = build_column_mapping(csv_path)

    header_set = set(column_names)

    # Validate that Timestamp exists exactly once and maps correctly.
    if "Timestamp" not in header_set:
        raise ValueError("Validation failed: Timestamp column is missing from the CSV header.")

    if column_names.count("Timestamp") != 1:
        raise ValueError("Validation failed: Timestamp appears more than once in the CSV header.")

    # Validate number of questionnaire question mappings.
    discovered_question_ids = {question_id for question_id in mapping.keys() if question_id != "Timestamp"}

    if len(discovered_question_ids) != 25:
        raise ValueError(
            f"Validation failed: expected exactly 25 questionnaire questions, found {len(discovered_question_ids)}."
        )

    expected_question_ids = set(EXPECTED_QUESTION_IDS)
    if discovered_question_ids != expected_question_ids:
        missing_ids = sorted(expected_question_ids - discovered_question_ids)
        unexpected_ids = sorted(discovered_question_ids - expected_question_ids)
        details = []
        if missing_ids:
            details.append(f"missing logical IDs: {', '.join(missing_ids)}")
        if unexpected_ids:
            details.append(f"unexpected logical IDs: {', '.join(unexpected_ids)}")
        raise ValueError(f"Validation failed: logical question IDs do not match the expected Q1-Q25 set ({'; '.join(details)}).")

    # Validate schema agreement.
    schema_ids = set(QUESTIONNAIRE_SCHEMA.keys())
    if schema_ids != set(mapping.keys()):
        schema_only = sorted(schema_ids - set(mapping.keys()))
        csv_only = sorted(set(mapping.keys()) - schema_ids)
        details = []
        if schema_only:
            details.append(f"schema-only IDs: {', '.join(schema_only)}")
        if csv_only:
            details.append(f"CSV-only IDs: {', '.join(csv_only)}")
        raise ValueError(
            "Validation failed: QUESTIONNAIRE_SCHEMA and CSV mapping do not agree "
            f"({'; '.join(details)})."
        )

    # Validate that every mapped CSV column actually exists in the header.
    for logical_id, actual_column_name in mapping.items():
        if actual_column_name not in header_set:
            raise ValueError(
                f"Validation failed: mapped CSV column '{actual_column_name}' for logical ID '{logical_id}' does not exist in the CSV header."
            )

    # Validate that all required Q1-Q25 entries are represented exactly once.
    for question_id in EXPECTED_QUESTION_IDS:
        if question_id not in mapping:
            raise ValueError(f"Validation failed: missing questionnaire question {question_id} in the CSV mapping.")

    return mapping


def get_column_for_question(question_id: str, csv_path: Optional[str | Path] = None) -> str:
    """Return the actual CSV column name for a logical question ID.

    Args:
        question_id: Logical question ID such as "Q1" or "Timestamp".
        csv_path: Optional path to the questionnaire CSV file.

    Returns:
        The actual CSV column name.

    Raises:
        ValueError: If the mapping is invalid or the question is not present.
    """

    mapping = validate_column_mapping(csv_path)

    if question_id not in mapping:
        raise ValueError(f"Question ID '{question_id}' was not found in the CSV mapping.")

    return mapping[question_id]


def _infer_logical_question_id(column_name: str) -> Optional[str]:
    """Infer the logical question ID from a CSV column name."""

    stripped_name = column_name.strip()

    if stripped_name == "Timestamp":
        return "Timestamp"

    match = QUESTION_ID_PATTERN.match(stripped_name)
    if match:
        return match.group(1)

    return None


def _build_report(mapping: Dict[str, str], csv_path: Optional[str | Path] = None) -> str:
    """Create a readable validation report for terminal display."""

    resolved_path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    header = load_column_names(resolved_path)

    lines = [
        "Questionnaire Column Mapping Report",
        "=" * 36,
        f"CSV path: {resolved_path}",
        f"Total CSV columns: {len(header)}",
        "",
        f"Timestamp -> {mapping.get('Timestamp', '<missing>')}",
    ]

    for question_id in EXPECTED_QUESTION_IDS:
        lines.append(f"{question_id} -> {mapping.get(question_id, '<missing>')}")

    lines.append("")
    lines.append("Validation status: PASS")
    return "\n".join(lines)


def main() -> None:
    """Run a simple validation report for the current questionnaire CSV."""

    try:
        mapping = validate_column_mapping()
        report = _build_report(mapping)
        print(report)
    except Exception as exc:
        print("Questionnaire Column Mapping Report")
        print("=" * 36)
        print(f"CSV path: {DEFAULT_CSV_PATH}")
        print(f"Total CSV columns: {len(load_column_names())}")
        print("")
        print(f"Validation status: FAIL")
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
