"""Inspection utility for Q10 and Q11 multi-select questionnaire options.

This module is intentionally analysis-only. It loads the cleaned questionnaire
CSV, resolves Q10 and Q11 through the existing column-mapping helpers, splits
multi-select responses on semicolons, normalizes spacing only for comparison,
and reports the distinct observed options, formatting variants, unexpected
options, and any canonical options that were never selected.

It does not encode, impute, delete, create targets, create stress scores, or
train any model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from .column_mapping import get_column_for_question, validate_column_mapping
    from .questionnaire_schema import QUESTIONNAIRE_SCHEMA
except ImportError:  # pragma: no cover - fallback for direct script execution
    from column_mapping import get_column_for_question, validate_column_mapping
    from questionnaire_schema import QUESTIONNAIRE_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLEANED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_cleaned.csv"
DEFAULT_RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "questionnaire_responses.csv"

MULTI_SELECT_QUESTION_IDS: Tuple[str, ...] = ("Q10", "Q11")


def load_cleaned_data(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the cleaned questionnaire dataset without modifying it."""

    resolved_path = Path(csv_path) if csv_path is not None else DEFAULT_CLEANED_CSV_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Cleaned CSV file not found: {resolved_path}")

    validate_column_mapping(DEFAULT_RAW_CSV_PATH)

    cleaned_df = pd.read_csv(resolved_path, encoding="utf-8")
    return cleaned_df.copy()


def _normalize_for_comparison(option_text: str) -> str:
    """Collapse repeated internal whitespace for comparison only."""

    return " ".join(option_text.strip().split())


def _split_multiselect_response(response_value: Any) -> List[str]:
    """Split a multi-select response string into trimmed option fragments."""

    if pd.isna(response_value):
        return []

    if not isinstance(response_value, str):
        return []

    raw_parts = response_value.split(";")
    return [part.strip() for part in raw_parts if part.strip()]


def _get_schema_canonical_options(question_id: str) -> List[str]:
    """Return canonical options for a question from the schema when available."""

    schema_entry = QUESTIONNAIRE_SCHEMA.get(question_id)
    if schema_entry is None:
        return []

    return list(schema_entry.canonical_options or [])


def inspect_multiselect_question(
    question_id: str,
    cleaned_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Inspect one multi-select question and return its observed option statistics."""

    if question_id not in MULTI_SELECT_QUESTION_IDS:
        raise ValueError(f"Unsupported question ID for multiselect inspection: {question_id}")

    actual_column_name = get_column_for_question(question_id, DEFAULT_RAW_CSV_PATH)

    if question_id not in cleaned_df.columns:
        raise ValueError(
            f"Question {question_id} is not present in the cleaned dataset columns."
        )

    schema_entry = QUESTIONNAIRE_SCHEMA.get(question_id)
    canonical_options = _get_schema_canonical_options(question_id)
    canonical_lookup = {
        _normalize_for_comparison(option): option for option in canonical_options
    }

    observed_counter: Counter[str] = Counter()
    normalized_to_raw_variants: Dict[str, set[str]] = defaultdict(set)
    raw_variants_found: set[str] = set()

    for response_value in cleaned_df[question_id].tolist():
        for option_text in _split_multiselect_response(response_value):
            normalized_option = _normalize_for_comparison(option_text)
            observed_counter[normalized_option] += 1
            normalized_to_raw_variants[normalized_option].add(option_text)
            raw_variants_found.add(option_text)

    formatting_variants: List[Dict[str, Any]] = []
    for normalized_option in sorted(normalized_to_raw_variants):
        variants = sorted(normalized_to_raw_variants[normalized_option])
        if len(variants) > 1:
            formatting_variants.append(
                {
                    "normalized_option": normalized_option,
                    "canonical_option": canonical_lookup.get(normalized_option, normalized_option),
                    "observed_variants": variants,
                    "count": int(observed_counter[normalized_option]),
                }
            )

    unexpected_options = []
    for normalized_option in sorted(observed_counter):
        if normalized_option not in canonical_lookup:
            unexpected_options.append(
                {
                    "normalized_option": normalized_option,
                    "count": int(observed_counter[normalized_option]),
                    "observed_variants": sorted(normalized_to_raw_variants[normalized_option]),
                }
            )

    zero_selection_options = [
        option
        for option in canonical_options
        if _normalize_for_comparison(option) not in observed_counter
    ]

    return {
        "question_id": question_id,
        "column_name": actual_column_name,
        "description": schema_entry.description if schema_entry is not None else "",
        "data_type": schema_entry.data_type if schema_entry is not None else "",
        "canonical_options": canonical_options,
        "observed_options": sorted(
            set(
                canonical_lookup.get(normalized_option, normalized_option)
                for normalized_option in observed_counter
            )
        ),
        "observed_option_counts": dict(sorted(observed_counter.items())),
        "formatting_variants": formatting_variants,
        "unexpected_options": unexpected_options,
        "options_with_zero_selections": zero_selection_options,
        "total_selected_options": int(sum(observed_counter.values())),
        "raw_variants_found": sorted(raw_variants_found),
    }


def generate_multiselect_inspection_report(
    inspection_results: Dict[str, Dict[str, Any]],
) -> str:
    """Build a readable text report summarizing the Q10/Q11 multiselect inspection."""

    lines: List[str] = [
        "Multi-Select Inspection Report",
        "============================",
        "",
    ]

    for question_id in MULTI_SELECT_QUESTION_IDS:
        result = inspection_results[question_id]
        lines.extend(
            [
                f"Q{question_id[1:]}: {result['description']}",
                f"Actual column name: {result['column_name']}",
                f"Data type: {result['data_type']}",
                "",
                "Canonical options:",
            ]
        )

        if result["canonical_options"]:
            for option in result["canonical_options"]:
                lines.append(f"- {option}")
        else:
            lines.append("None")

        lines.extend(["", "Observed options and frequencies:"])
        observed_counts = result["observed_option_counts"]
        if observed_counts:
            for normalized_option, count in observed_counts.items():
                lines.append(f"- {normalized_option}: {count}")
        else:
            lines.append("None")

        lines.extend(["", "Formatting variants:"])
        if result["formatting_variants"]:
            for variant in result["formatting_variants"]:
                lines.append(
                    f"- {variant['canonical_option']} -> observed variants: {', '.join(variant['observed_variants'])}"
                )
        else:
            lines.append("None")

        lines.extend(["", "Unexpected options:"])
        if result["unexpected_options"]:
            for option in result["unexpected_options"]:
                lines.append(
                    f"- {option['normalized_option']} (count={option['count']}; observed variants: {', '.join(option['observed_variants'])})"
                )
        else:
            lines.append("None")

        lines.extend(["", "Options with zero selections:"])
        if result["options_with_zero_selections"]:
            for option in result["options_with_zero_selections"]:
                lines.append(f"- {option}")
        else:
            lines.append("None")

        lines.extend(
            [
                "",
                f"Total selected options: {result['total_selected_options']}",
                "",
                "=" * 28,
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    """Run the multiselect inspection utility and print the findings."""

    cleaned_df = load_cleaned_data(DEFAULT_CLEANED_CSV_PATH)

    inspection_results: Dict[str, Dict[str, Any]] = {}
    for question_id in MULTI_SELECT_QUESTION_IDS:
        inspection_results[question_id] = inspect_multiselect_question(question_id, cleaned_df)

    print("MULTI-SELECT INSPECTION COMPLETE")
    print(generate_multiselect_inspection_report(inspection_results))


if __name__ == "__main__":
    main()
