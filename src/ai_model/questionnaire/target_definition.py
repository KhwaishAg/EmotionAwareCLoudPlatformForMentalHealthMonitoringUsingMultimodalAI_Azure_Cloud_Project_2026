"""Prediction target definition for the questionnaire modality.

This module defines the project-defined stress target from Q4_encoded only,
creates the explicit target encoding, validates leakage prevention rules, and
writes the target-enriched dataset plus the associated report and metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_features.csv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_with_target.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "target_definition_report.txt"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_target_metadata.json"

EXPECTED_ROWS = 52

TARGET_CLASSES = ["low", "moderate", "high"]
TARGET_ENCODING = {"low": 0, "moderate": 1, "high": 2}

TARGET_MAPPING = {
    1: "low",
    2: "low",
    3: "moderate",
    4: "high",
    5: "high",
}

EXCLUDED_PREDICTOR_COLUMNS = ["Q4", "Q4_encoded", "stress_level", "stress_level_encoded"]


def load_features(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the feature-engineered questionnaire dataset."""

    resolved_path = Path(csv_path) if csv_path is not None else INPUT_CSV_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input feature CSV file not found: {resolved_path}")

    return pd.read_csv(resolved_path, encoding="utf-8")


def define_stress_level(df: pd.DataFrame) -> pd.DataFrame:
    """Create stress_level directly from Q4_encoded using the project-defined mapping."""

    if "Q4_encoded" not in df.columns:
        raise ValueError("Q4_encoded is required to define the stress target.")

    df = df.copy()
    stress_levels = []

    for value in df["Q4_encoded"]:
        if pd.isna(value):
            raise ValueError("Q4_encoded contains missing values; target definition requires a complete source column.")

        if value in TARGET_MAPPING:
            stress_levels.append(TARGET_MAPPING[int(value)])
        else:
            raise ValueError(f"Unexpected Q4_encoded value encountered: {value}")

    df["stress_level"] = pd.Series(stress_levels, index=df.index, dtype="object")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create stress_level_encoded using the explicit low/moderate/high encoding."""

    if "stress_level" not in df.columns:
        raise ValueError("stress_level must exist before encoding the target.")

    df = df.copy()
    encoded_values = []

    for value in df["stress_level"]:
        if pd.isna(value):
            raise ValueError("stress_level contains missing values; target encoding requires complete classes.")

        if value not in TARGET_ENCODING:
            raise ValueError(f"Unexpected stress_level value encountered: {value}")

        encoded_values.append(TARGET_ENCODING[value])

    df["stress_level_encoded"] = pd.Series(encoded_values, index=df.index, dtype="int64")
    return df


def calculate_class_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate class counts, percentages, and imbalance information from the target column."""

    total_rows = len(df)
    class_counts = df["stress_level"].value_counts(dropna=False)

    distribution = {
        "total_rows": total_rows,
        "count_low": int(class_counts.get("low", 0)),
        "count_moderate": int(class_counts.get("moderate", 0)),
        "count_high": int(class_counts.get("high", 0)),
        "percentage_low": (class_counts.get("low", 0) / total_rows * 100) if total_rows else 0.0,
        "percentage_moderate": (class_counts.get("moderate", 0) / total_rows * 100) if total_rows else 0.0,
        "percentage_high": (class_counts.get("high", 0) / total_rows * 100) if total_rows else 0.0,
        "missing_q4_count": int(df["Q4_encoded"].isna().sum()),
        "missing_target_count": int(df["stress_level"].isna().sum()),
        "min_class_size": int(min(class_counts.get(label, 0) for label in TARGET_CLASSES)),
        "max_class_size": int(max(class_counts.get(label, 0) for label in TARGET_CLASSES)),
    }

    smallest = distribution["min_class_size"]
    largest = distribution["max_class_size"]
    distribution["class_imbalance_ratio"] = (largest / smallest) if smallest else 0.0

    return distribution


def validate_target(
    df: pd.DataFrame,
    expected_rows: int = EXPECTED_ROWS,
    future_predictor_columns: Optional[List[str]] = None,
) -> bool:
    """Validate target definition, target mapping, and leakage prevention rules."""

    if len(df) != expected_rows:
        raise ValueError(f"Expected exactly {expected_rows} rows, found {len(df)}.")

    if df["Q4"].isna().any():
        raise ValueError("Q4 contains missing values; target definition requires a complete source column.")

    if df["Q4_encoded"].isna().any():
        raise ValueError("Q4_encoded contains missing values; target definition requires a complete source column.")

    if df["stress_level"].isna().any():
        raise ValueError("stress_level contains missing values.")

    if not df["stress_level"].isin(TARGET_CLASSES).all():
        raise ValueError("stress_level contains values other than low, moderate, or high.")

    if not df["stress_level_encoded"].isin([0, 1, 2]).all():
        raise ValueError("stress_level_encoded contains values other than 0, 1, or 2.")

    q4_expected_map = {
        1: "low",
        2: "low",
        3: "moderate",
        4: "high",
        5: "high",
    }

    for row_index, value in df["Q4_encoded"].items():
        if pd.isna(value):
            raise ValueError(f"Missing Q4_encoded value found at row {row_index}.")

        mapped_class = q4_expected_map.get(int(value))
        if mapped_class is None:
            raise ValueError(f"Unexpected Q4_encoded value {value} found at row {row_index}.")

        if df.at[row_index, "stress_level"] != mapped_class:
            raise ValueError(
                f"Mapping mismatch at row {row_index}: Q4_encoded={value} should map to {mapped_class}, "
                f"but stress_level={df.at[row_index, 'stress_level']}"
            )

        if df.at[row_index, "stress_level_encoded"] != TARGET_ENCODING[mapped_class]:
            raise ValueError(
                f"Encoded target mismatch at row {row_index}: stress_level={mapped_class} should encode to "
                f"{TARGET_ENCODING[mapped_class]}, but found {df.at[row_index, 'stress_level_encoded']}"
            )

    validate_no_target_leakage(future_predictor_columns)

    return True


def validate_no_target_leakage(future_predictor_columns: Optional[List[str]] = None) -> bool:
    """Validate that excluded target-related columns are not included in future predictor lists."""

    if future_predictor_columns is None:
        return True

    unexpected_columns = [column for column in EXCLUDED_PREDICTOR_COLUMNS if column in future_predictor_columns]
    if unexpected_columns:
        raise ValueError(
            "Future predictor list contains target leakage columns: "
            + ", ".join(unexpected_columns)
        )

    return True


def generate_report(
    df: pd.DataFrame,
    distribution: Dict[str, Any],
    validation_result: str,
    report_path: Optional[str | Path] = None,
) -> str:
    """Generate the target-definition report."""

    resolved_report_path = Path(report_path) if report_path is not None else REPORT_PATH
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Target Definition Report",
        "=======================",
        "Target definition:",
        "- Source column: Q4_encoded",
        "- Project-defined mapping: Q4 1 or 2 -> low; Q4 3 -> moderate; Q4 4 or 5 -> high",
        "- Encoded target mapping: low -> 0, moderate -> 1, high -> 2",
        "",
        "Mapping table:",
        "- Q4 1 -> low -> 0",
        "- Q4 2 -> low -> 0",
        "- Q4 3 -> moderate -> 1",
        "- Q4 4 -> high -> 2",
        "- Q4 5 -> high -> 2",
        "",
        "Class distribution:",
        f"- low count: {distribution['count_low']}",
        f"- moderate count: {distribution['count_moderate']}",
        f"- high count: {distribution['count_high']}",
        f"- low percentage: {distribution['percentage_low']:.2f}%",
        f"- moderate percentage: {distribution['percentage_moderate']:.2f}%",
        f"- high percentage: {distribution['percentage_high']:.2f}%",
        f"- total rows: {distribution['total_rows']}",
        f"- missing Q4 count: {distribution['missing_q4_count']}",
        f"- missing target count: {distribution['missing_target_count']}",
        f"- minimum class size: {distribution['min_class_size']}",
        f"- maximum class size: {distribution['max_class_size']}",
        f"- class imbalance ratio: {distribution['class_imbalance_ratio']:.2f}",
        "",
        "Leakage prevention rule:",
        "- Q4_encoded is the source of the target and must not be used as a predictor feature.",
        f"- Excluded predictor columns: {', '.join(EXCLUDED_PREDICTOR_COLUMNS)}",
        "",
        f"Validation result: {validation_result}",
    ]

    report_content = "\n".join(lines)
    resolved_report_path.write_text(report_content, encoding="utf-8")
    return report_content


def save_outputs(
    df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    report_path: Optional[str | Path] = None,
) -> tuple[Path, Path]:
    """Save the target-enriched dataset and the report."""

    resolved_output_path = Path(output_path) if output_path is not None else OUTPUT_CSV_PATH
    resolved_report_path = Path(report_path) if report_path is not None else REPORT_PATH

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(resolved_output_path, index=False, encoding="utf-8")
    return resolved_output_path, resolved_report_path


def save_metadata(
    metadata_path: Optional[str | Path] = None,
) -> Path:
    """Save the machine-readable target metadata as JSON."""

    resolved_metadata_path = Path(metadata_path) if metadata_path is not None else METADATA_PATH
    resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "target_column": "stress_level",
        "target_encoded_column": "stress_level_encoded",
        "target_source": "Q4_encoded",
        "target_type": "three_class",
        "classes": ["low", "moderate", "high"],
        "encoding": {"low": 0, "moderate": 1, "high": 2},
        "excluded_predictor_columns": EXCLUDED_PREDICTOR_COLUMNS,
        "clinical_diagnosis": False,
    }

    resolved_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return resolved_metadata_path


def run_target_definition(
    input_csv_path: Optional[str | Path] = None,
    output_csv_path: Optional[str | Path] = None,
    report_path: Optional[str | Path] = None,
    metadata_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, str, Path]:
    """Run the full target-definition pipeline."""

    input_path = Path(input_csv_path) if input_csv_path is not None else INPUT_CSV_PATH
    output_path = Path(output_csv_path) if output_csv_path is not None else OUTPUT_CSV_PATH
    report_file_path = Path(report_path) if report_path is not None else REPORT_PATH
    metadata_file_path = Path(metadata_path) if metadata_path is not None else METADATA_PATH

    df = load_features(input_path)

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected exactly {EXPECTED_ROWS} rows, found {len(df)}.")

    df = define_stress_level(df)
    df = encode_target(df)

    future_predictor_columns = [column for column in df.columns if column not in EXCLUDED_PREDICTOR_COLUMNS]
    validate_target(df, expected_rows=EXPECTED_ROWS, future_predictor_columns=future_predictor_columns)

    distribution = calculate_class_distribution(df)
    save_outputs(df, output_path, report_file_path)
    save_metadata(metadata_file_path)

    report_text = generate_report(
        df=df,
        distribution=distribution,
        validation_result="PASS",
        report_path=report_file_path,
    )

    return df, report_text, metadata_file_path


def main() -> None:
    """Run the target-definition pipeline and print a concise summary."""

    df, report_text, metadata_path = run_target_definition()
    distribution = calculate_class_distribution(df)

    print("TARGET DEFINITION COMPLETE")
    print(f"Rows: {len(df)}")
    print(f"Class distribution: low={distribution['count_low']}, moderate={distribution['count_moderate']}, high={distribution['count_high']}")
    print(f"Output: {OUTPUT_CSV_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Metadata: {metadata_path}")
    print(f"Validation result from report: {report_text.split('Validation result: ', 1)[1].splitlines()[0]}")


if __name__ == "__main__":
    main()
