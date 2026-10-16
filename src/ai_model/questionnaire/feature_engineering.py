"""Feature engineering for the questionnaire modality.

This module builds a unified feature dataset from the existing processed
questionnaire outputs without creating targets, imputing missing values, or
modifying prior processing stages.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ORDINAL_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_ordinal_encoded.csv"
MULTIHOT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_multihot_encoded.csv"
CATEGORICAL_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_categorical_encoded.csv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_features.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineering_report.txt"

EXPECTED_ROWS = 52

CORE_STRESS_FEATURES = [
    "Q4_encoded",
    "Q5_encoded",
    "Q6_encoded",
    "Q7_encoded",
    "Q8_encoded",
    "Q9_encoded",
    "Q12_encoded",
    "Q13_stress_aligned",
    "Q14_encoded",
    "Q15_encoded",
    "Q16_encoded",
    "Q17_encoded",
    "Q18_encoded",
    "Q19_encoded",
    "Q20_encoded",
]

DERIVED_DIMENSIONS = [
    "overwhelm_dimension",
    "stress_impact_dimension",
    "emotional_strain_dimension",
    "Q13_stress_aligned",
    "stress_persistence_dimension",
    "recovery_difficulty_dimension",
    "stress_behavior_dimension",
    "academic_preparation_load",
]

CONTEXT_FEATURES = [
    "Q1_year_encoded",
    "Q2_regular_academic_semester",
    "Q2_preparing_for_examinations",
    "Q2_working_on_academic_projects",
    "Q2_assignment_deadline_pressure",
    "Q2_placements_internships",
    "Q2_coding_technical_assessments",
    "Q2_applying_for_jobs_internships",
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
    "Q10_nothing",
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

SPECIAL_INDICATORS = ["Q14_varies", "Q16_varies"]


def load_processed_files(
    ordinal_path: Optional[str | Path] = None,
    multihot_path: Optional[str | Path] = None,
    categorical_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three processed questionnaire files."""

    ordinal_file = Path(ordinal_path) if ordinal_path is not None else ORDINAL_CSV_PATH
    multihot_file = Path(multihot_path) if multihot_path is not None else MULTIHOT_CSV_PATH
    categorical_file = Path(categorical_path) if categorical_path is not None else CATEGORICAL_CSV_PATH

    for path in [ordinal_file, multihot_file, categorical_file]:
        if not path.exists():
            raise FileNotFoundError(f"Processed file not found: {path}")

    ordinal_df = pd.read_csv(ordinal_file, encoding="utf-8")
    multihot_df = pd.read_csv(multihot_file, encoding="utf-8")
    categorical_df = pd.read_csv(categorical_file, encoding="utf-8")

    return ordinal_df, multihot_df, categorical_df


def validate_source_alignment(
    ordinal_df: pd.DataFrame,
    multihot_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
) -> tuple[bool, Dict[str, Any]]:
    """Validate row count equality and Timestamp alignment across the source files."""

    details: Dict[str, Any] = {
        "ordinal_rows": len(ordinal_df),
        "multihot_rows": len(multihot_df),
        "categorical_rows": len(categorical_df),
        "row_counts_match": False,
        "timestamp_alignment": False,
        "duplicate_rows": False,
    }

    if not (len(ordinal_df) == len(multihot_df) == len(categorical_df) == EXPECTED_ROWS):
        raise ValueError(
            "Source files do not have exactly 52 rows. "
            f"Found ordinal={len(ordinal_df)}, multihot={len(multihot_df)}, categorical={len(categorical_df)}."
        )

    details["row_counts_match"] = True

    if ordinal_df.duplicated(subset=["Timestamp"]).any() or multihot_df.duplicated(subset=["Timestamp"]).any() or categorical_df.duplicated(subset=["Timestamp"]).any():
        details["duplicate_rows"] = True
        raise ValueError("Duplicate Timestamp rows detected in one or more source files.")

    if not ordinal_df["Timestamp"].equals(multihot_df["Timestamp"]) or not multihot_df["Timestamp"].equals(categorical_df["Timestamp"]):
        raise ValueError("Timestamp values are not aligned across the three source files.")

    if not ordinal_df["Timestamp"].equals(categorical_df["Timestamp"]):
        raise ValueError("Timestamp values do not align row-by-row between ordinal and categorical sources.")

    details["timestamp_alignment"] = True
    details["duplicate_rows"] = False
    return True, details


def _row_mean_ignoring_missing(values: pd.Series) -> float:
    """Compute a row-wise mean over available numeric values without imputing."""

    valid_values = values.dropna()
    if valid_values.empty:
        return np.nan
    return float(valid_values.mean())


def merge_processed_features(
    ordinal_df: pd.DataFrame,
    multihot_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the processed datasets while preserving original questionnaire columns and encoded features."""

    validation_result, _ = validate_source_alignment(ordinal_df, multihot_df, categorical_df)
    if not validation_result:
        raise ValueError("Source validation failed before merging.")

    merged_df = ordinal_df.copy()

    for source_df in [multihot_df, categorical_df]:
        for column in source_df.columns:
            if column in merged_df.columns:
                continue
            merged_df[column] = source_df[column]

    # Ensure no duplicate rows or duplicate column names were introduced.
    if merged_df.duplicated().any():
        raise ValueError("Duplicate rows were introduced during feature merge.")

    return merged_df


def create_overwhelm_dimension(df: pd.DataFrame) -> pd.DataFrame:
    """Create the overwhelm_dimension from Q5_encoded and Q6_encoded."""

    if "Q5_encoded" not in df.columns or "Q6_encoded" not in df.columns:
        raise ValueError("Q5_encoded and Q6_encoded are required for overwhelm_dimension.")

    df = df.copy()
    df["overwhelm_dimension"] = df[["Q5_encoded", "Q6_encoded"]].apply(
        lambda row: _row_mean_ignoring_missing(row), axis=1
    )
    return df


def create_stress_impact_dimension(df: pd.DataFrame) -> pd.DataFrame:
    """Create the stress_impact_dimension from Q7_encoded, Q8_encoded, and Q9_encoded."""

    required_columns = ["Q7_encoded", "Q8_encoded", "Q9_encoded"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column missing for stress_impact_dimension: {column}")

    df = df.copy()
    df["stress_impact_dimension"] = df[required_columns].apply(
        lambda row: _row_mean_ignoring_missing(row), axis=1
    )
    return df


def create_emotional_strain_dimension(df: pd.DataFrame) -> pd.DataFrame:
    """Create emotional_strain_dimension from Q12_encoded."""

    if "Q12_encoded" not in df.columns:
        raise ValueError("Q12_encoded is required for emotional_strain_dimension.")

    df = df.copy()
    df["emotional_strain_dimension"] = df["Q12_encoded"]
    return df


def create_coping_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Q13_stress_aligned while preserving the original Q13_encoded column."""

    if "Q13_encoded" not in df.columns:
        raise ValueError("Q13_encoded is required for Q13_stress_aligned.")

    df = df.copy()
    df["Q13_stress_aligned"] = df["Q13_encoded"].apply(
        lambda value: pd.NA if pd.isna(value) else 6 - value
    )
    return df


def create_persistence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stress_persistence_dimension using Q14_encoded while preserving Q14_varies."""

    required_columns = ["Q14_encoded", "Q14_varies"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column missing for stress_persistence_dimension: {column}")

    df = df.copy()
    df["stress_persistence_dimension"] = df["Q14_encoded"].where(df["Q14_varies"].fillna(0) == 0, other=pd.NA)
    return df


def create_recovery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create recovery_difficulty_dimension using Q16_encoded while preserving Q16_varies."""

    required_columns = ["Q16_encoded", "Q16_varies"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column missing for recovery_difficulty_dimension: {column}")

    df = df.copy()
    df["recovery_difficulty_dimension"] = df["Q16_encoded"].where(df["Q16_varies"].fillna(0) == 0, other=pd.NA)
    return df


def create_behavior_dimension(df: pd.DataFrame) -> pd.DataFrame:
    """Create stress_behavior_dimension from Q17_encoded, Q18_encoded, Q19_encoded, and Q20_encoded."""

    required_columns = ["Q17_encoded", "Q18_encoded", "Q19_encoded", "Q20_encoded"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column missing for stress_behavior_dimension: {column}")

    df = df.copy()
    df["stress_behavior_dimension"] = df[required_columns].apply(
        lambda row: _row_mean_ignoring_missing(row), axis=1
    )
    return df


def create_academic_load_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Create academic_preparation_load from Q3_encoded."""

    if "Q3_encoded" not in df.columns:
        raise ValueError("Q3_encoded is required for academic_preparation_load.")

    df = df.copy()
    df["academic_preparation_load"] = df["Q3_encoded"]
    return df


def validate_features(
    df: pd.DataFrame,
    original_columns: List[str],
    expected_rows: int,
) -> bool:
    """Validate the merged and engineered feature dataset."""

    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(df)}.")

    if df.duplicated().any():
        raise ValueError("Duplicate rows detected after feature engineering.")

    for column in original_columns:
        if column not in df.columns:
            raise ValueError(f"Original questionnaire column missing after feature engineering: {column}")

    forbidden_columns = ["target", "label", "risk_level", "risk_score"]
    for column in forbidden_columns:
        if column in df.columns:
            raise ValueError(f"Forbidden target/risk column detected: {column}")

    for feature in ["Q14_varies", "Q16_varies"]:
        if feature not in df.columns:
            raise ValueError(f"Required special indicator missing: {feature}")

    for feature_name in [
        "overwhelm_dimension",
        "stress_impact_dimension",
        "emotional_strain_dimension",
        "Q13_stress_aligned",
        "stress_persistence_dimension",
        "recovery_difficulty_dimension",
        "stress_behavior_dimension",
        "academic_preparation_load",
    ]:
        if feature_name not in df.columns:
            raise ValueError(f"Missing derived feature: {feature_name}")

    binary_like_columns = [
        col for col in df.columns if col.startswith("Q2_") or col.startswith("Q10_") or col.startswith("Q11_") or col.startswith("Q25_")
    ]
    for column in binary_like_columns:
        observed_non_missing = df[column].dropna()
        if not observed_non_missing.empty and not observed_non_missing.isin([0, 1]).all():
            raise ValueError(f"Binary-like feature {column} contains values outside of {{0, 1, NaN}}.")

    # Preserve original encoded ordinal numeric values.
    for column in ["Q3_encoded", "Q4_encoded", "Q5_encoded", "Q6_encoded", "Q7_encoded", "Q8_encoded", "Q9_encoded", "Q12_encoded", "Q13_encoded", "Q14_encoded", "Q15_encoded", "Q16_encoded", "Q17_encoded", "Q18_encoded", "Q19_encoded", "Q20_encoded"]:
        if column in df.columns:
            observed_non_missing = df[column].dropna()
            if not observed_non_missing.empty and not pd.api.types.is_numeric_dtype(observed_non_missing):
                raise ValueError(f"Ordinal encoded column {column} is not numeric.")

    # Q13_stress_aligned should be bounded where available.
    if "Q13_stress_aligned" in df.columns:
        observed_non_missing = df["Q13_stress_aligned"].dropna()
        if not observed_non_missing.empty and not observed_non_missing.between(1, 5).all():
            raise ValueError("Q13_stress_aligned contains values outside of the expected range [1, 5].")

    # Derived dimensions must be numeric where available.
    for feature_name in [
        "overwhelm_dimension",
        "stress_impact_dimension",
        "emotional_strain_dimension",
        "stress_persistence_dimension",
        "recovery_difficulty_dimension",
        "stress_behavior_dimension",
        "academic_preparation_load",
    ]:
        if feature_name in df.columns:
            observed_non_missing = df[feature_name].dropna()
            if not observed_non_missing.empty and not pd.api.types.is_numeric_dtype(observed_non_missing):
                raise ValueError(f"Derived feature {feature_name} is not numeric.")

    return True


def generate_report(
    df: pd.DataFrame,
    source_details: Dict[str, Any],
    validation_result: str,
    report_path: Optional[str | Path] = None,
) -> str:
    """Generate a feature engineering report."""

    resolved_report_path = Path(report_path) if report_path is not None else REPORT_PATH
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

    derived_feat_names = [
        "overwhelm_dimension",
        "stress_impact_dimension",
        "emotional_strain_dimension",
        "Q13_stress_aligned",
        "stress_persistence_dimension",
        "recovery_difficulty_dimension",
        "stress_behavior_dimension",
        "academic_preparation_load",
    ]

    missing_counts = {}
    for feature_name in derived_feat_names:
        if feature_name in df.columns:
            missing_counts[feature_name] = int(df[feature_name].isna().sum())

    feature_list = list(df.columns)

    lines: List[str] = [
        "Feature Engineering Report",
        "========================",
        f"Input files: {source_details['input_files']}",
        f"Row counts: ordinal={source_details['ordinal_rows']}, multihot={source_details['multihot_rows']}, categorical={source_details['categorical_rows']}",
        f"Timestamp alignment result: {source_details['timestamp_alignment']}",
        f"Final row count: {len(df)}",
        f"Original column count: {source_details['original_column_count']}",
        f"Encoded feature count: {source_details['encoded_feature_count']}",
        f"Derived feature count: {source_details['derived_feature_count']}",
        "",
        "Feature groups:",
        "Core stress features:",
        "- Q4_encoded",
        "- Q5_encoded",
        "- Q6_encoded",
        "- Q7_encoded",
        "- Q8_encoded",
        "- Q9_encoded",
        "- Q12_encoded",
        "- Q13_stress_aligned",
        "- Q14_encoded",
        "- Q15_encoded",
        "- Q16_encoded",
        "- Q17_encoded",
        "- Q18_encoded",
        "- Q19_encoded",
        "- Q20_encoded",
        "",
        "Derived dimensions:",
        "- overwhelm_dimension",
        "- stress_impact_dimension",
        "- emotional_strain_dimension",
        "- Q13_stress_aligned",
        "- stress_persistence_dimension",
        "- recovery_difficulty_dimension",
        "- stress_behavior_dimension",
        "- academic_preparation_load",
        "",
        "Context features:",
        "- Q1_year_encoded",
        "- Q2_*",
        "- Q10_*",
        "- Q11_*",
        "- Q25_*",
        "",
        "Special indicators:",
        "- Q14_varies",
        "- Q16_varies",
        "",
        "Complete final feature list:",
    ]

    for feature_name in feature_list:
        lines.append(f"- {feature_name}")

    lines.extend(["", "Missing-value counts for derived features:"])
    for feature_name in derived_feat_names:
        if feature_name in missing_counts:
            lines.append(f"- {feature_name}: {missing_counts[feature_name]}")

    lines.extend(
        [
            "",
            f"Validation result: {validation_result}",
            "No target column created.",
            "No risk_level column created.",
            "No risk_score column created.",
            "No imputation performed in Step 9.",
        ]
    )

    report_content = "\n".join(lines)
    resolved_report_path.write_text(report_content, encoding="utf-8")
    return report_content


def save_features(df: pd.DataFrame, output_path: Optional[str | Path] = None) -> Path:
    """Save the engineered feature dataset to disk."""

    resolved_output_path = Path(output_path) if output_path is not None else OUTPUT_CSV_PATH
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(resolved_output_path, index=False, encoding="utf-8")
    return resolved_output_path


def run_feature_engineering(
    ordinal_path: Optional[str | Path] = None,
    multihot_path: Optional[str | Path] = None,
    categorical_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    report_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, str]:
    """Run the full questionnaire feature engineering pipeline."""

    ordinal_df, multihot_df, categorical_df = load_processed_files(ordinal_path, multihot_path, categorical_path)
    validate_source_alignment(ordinal_df, multihot_df, categorical_df)

    merged_df = merge_processed_features(ordinal_df, multihot_df, categorical_df)

    # Preserve original questionnaire columns and only add new engineered features.
    original_columns = list(ordinal_df.columns)

    feature_df = merged_df.copy()

    required_presence = [
        "Q4_encoded",
        "Q5_encoded",
        "Q6_encoded",
        "Q7_encoded",
        "Q8_encoded",
        "Q9_encoded",
        "Q12_encoded",
        "Q13_encoded",
        "Q14_encoded",
        "Q15_encoded",
        "Q16_encoded",
        "Q17_encoded",
        "Q18_encoded",
        "Q19_encoded",
        "Q20_encoded",
        "Q14_varies",
        "Q16_varies",
    ]
    for column in required_presence:
        if column not in feature_df.columns:
            raise ValueError(f"Required feature missing from processed input: {column}")

    feature_df = create_overwhelm_dimension(feature_df)
    feature_df = create_stress_impact_dimension(feature_df)
    feature_df = create_emotional_strain_dimension(feature_df)
    feature_df = create_coping_features(feature_df)
    feature_df = create_persistence_features(feature_df)
    feature_df = create_recovery_features(feature_df)
    feature_df = create_behavior_dimension(feature_df)
    feature_df = create_academic_load_feature(feature_df)

    validate_features(feature_df, original_columns, EXPECTED_ROWS)

    encoded_feature_count = len([column for column in feature_df.columns if column not in original_columns])
    derived_feature_count = len(
        [
            "overwhelm_dimension",
            "stress_impact_dimension",
            "emotional_strain_dimension",
            "Q13_stress_aligned",
            "stress_persistence_dimension",
            "recovery_difficulty_dimension",
            "stress_behavior_dimension",
            "academic_preparation_load",
        ]
    )

    source_details = {
        "input_files": [
            str(ordinal_path or ORDINAL_CSV_PATH),
            str(multihot_path or MULTIHOT_CSV_PATH),
            str(categorical_path or CATEGORICAL_CSV_PATH),
        ],
        "ordinal_rows": len(ordinal_df),
        "multihot_rows": len(multihot_df),
        "categorical_rows": len(categorical_df),
        "timestamp_alignment": True,
        "original_column_count": len(original_columns),
        "encoded_feature_count": encoded_feature_count,
        "derived_feature_count": derived_feature_count,
    }

    save_features(feature_df, output_path)

    report_text = generate_report(
        df=feature_df,
        source_details=source_details,
        validation_result="PASS",
        report_path=report_path,
    )

    return feature_df, report_text


def main() -> None:
    """Run the feature engineering pipeline and print a concise summary."""

    df, report_text = run_feature_engineering()

    print("FEATURE ENGINEERING COMPLETE")
    print(f"Rows: {len(df)}")
    print(f"Original columns: {len([col for col in df.columns if col in ['Timestamp', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25']])}")
    print(f"Derived features: {len(['overwhelm_dimension', 'stress_impact_dimension', 'emotional_strain_dimension', 'Q13_stress_aligned', 'stress_persistence_dimension', 'recovery_difficulty_dimension', 'stress_behavior_dimension', 'academic_preparation_load'])}")
    print(f"Output: {OUTPUT_CSV_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Validation: {report_text.split('Validation result: ', 1)[1].splitlines()[0]}")


if __name__ == "__main__":
    main()
