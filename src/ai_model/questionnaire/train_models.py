"""Train the questionnaire classification models for Step 11.

This module loads the target-defined questionnaire dataset, validates the allowed
predictor feature space, trains a Logistic Regression pipeline and a Random
Forest pipeline, saves model artifacts, and writes a training report.

It does not perform evaluation or cross-validation; those are handled in Step 12.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "questionnaire_with_target.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "questionnaire"
RESULTS_DIR = PROJECT_ROOT / "results" / "questionnaire"

EXPECTED_ROWS = 52
TARGET_COLUMN = "stress_level"
TARGET_ENCODED_COLUMN = "stress_level_encoded"
RANDOM_STATE = 42

EXCLUDED_COLUMNS = [
    "Timestamp",
    "Q4",
    "Q4_encoded",
    "stress_level",
    "stress_level_encoded",
    "Q21",
    "Q22",
    "Q23",
    "Q24",
    "Q1",
    "Q2",
    "Q3",
    "Q5",
    "Q6",
    "Q7",
    "Q8",
    "Q9",
    "Q10",
    "Q11",
    "Q12",
    "Q13",
    "Q14",
    "Q15",
    "Q16",
    "Q17",
    "Q18",
    "Q19",
    "Q20",
    "Q25",
]

REQUIRED_MODEL_FEATURES: List[str] = [
    "Q3_encoded",
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
    "Q1_year_encoded",
    "Q2_regular_academic_semester",
    "Q2_preparing_for_examinations",
    "Q2_working_on_academic_projects",
    "Q2_assignment_deadline_pressure",
    "Q2_placements_internships",
    "Q2_coding_technical_assessments",
    "Q2_applying_for_jobs_internships",
    "Q25_Placements_job_search",
    "Q25_Career_uncertainty",
    "Q25_Social_personal_issues",
    "Q25_Examinations",
    "Q25_Other",
    "Q25_Academic_workload",
    "Q25_Coding_technical_assessments",
    "Q25_Financial_concerns",
    "Q25_Family_expectations",
    "overwhelm_dimension",
    "stress_impact_dimension",
    "emotional_strain_dimension",
    "Q13_stress_aligned",
    "stress_persistence_dimension",
    "recovery_difficulty_dimension",
    "stress_behavior_dimension",
    "academic_preparation_load",
]


def load_dataset(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the target-defined dataset."""

    resolved_path = Path(csv_path) if csv_path is not None else INPUT_CSV_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {resolved_path}")

    return pd.read_csv(resolved_path, encoding="utf-8")


def define_target(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the input dataset with the target confirmed as stress_level."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")

    if TARGET_ENCODED_COLUMN not in df.columns:
        raise ValueError(f"Encoded target column missing: {TARGET_ENCODED_COLUMN}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    if not df[TARGET_COLUMN].isin(["low", "moderate", "high"]).all():
        raise ValueError("Target column contains values other than low, moderate, or high.")

    return df.copy()


def define_predictor_features(df: pd.DataFrame) -> List[str]:
    """Define the allowed numeric predictor feature list and validate required features exist."""

    missing_required = [feature for feature in REQUIRED_MODEL_FEATURES if feature not in df.columns]
    if missing_required:
        raise ValueError(
            "Required model features are missing from the dataset: " + ", ".join(missing_required)
        )

    feature_names = [feature for feature in REQUIRED_MODEL_FEATURES if feature in df.columns]

    if len(feature_names) != len(REQUIRED_MODEL_FEATURES):
        raise ValueError("Feature-list mismatch detected after validation.")

    return feature_names


def validate_features(
    df: pd.DataFrame,
    feature_names: List[str],
    target_column: str,
    excluded_columns: List[str],
) -> None:
    """Validate the dataset, target, leakage rules, and predictor numerics."""

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected exactly {EXPECTED_ROWS} rows, found {len(df)}.")

    if df[target_column].isna().any():
        raise ValueError(f"Target column {target_column} contains missing values.")

    if not df[target_column].isin(["low", "moderate", "high"]).all():
        raise ValueError("Target column contains unexpected classes.")

    for column in excluded_columns:
        if column in df.columns and column in feature_names:
            raise ValueError(f"Leakage detected: excluded column '{column}' is present in predictor features.")

    for column in ["Q4_encoded", "Q4", "Timestamp", TARGET_COLUMN, TARGET_ENCODED_COLUMN]:
        if column in feature_names:
            raise ValueError(f"Leakage detected: forbidden column '{column}' is included in model predictors.")

    raw_text_columns = [
        "Q1",
        "Q2",
        "Q3",
        "Q5",
        "Q6",
        "Q7",
        "Q8",
        "Q9",
        "Q10",
        "Q11",
        "Q12",
        "Q13",
        "Q14",
        "Q15",
        "Q16",
        "Q17",
        "Q18",
        "Q19",
        "Q20",
        "Q25",
    ]

    for column in raw_text_columns:
        if column in feature_names:
            raise ValueError(f"Leakage detected: raw questionnaire column '{column}' is included in model predictors.")

    for feature_name in feature_names:
        if not pd.api.types.is_numeric_dtype(df[feature_name]):
            raise ValueError(f"Predictor feature '{feature_name}' is not numeric.")

    if not df[TARGET_ENCODED_COLUMN].isin([0, 1, 2]).all():
        raise ValueError("Encoded target column contains unexpected values.")


def build_logistic_pipeline() -> Pipeline:
    """Build the Logistic Regression training pipeline with imputation and scaling."""

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return pipeline


def build_random_forest_pipeline() -> Pipeline:
    """Build the Random Forest training pipeline with imputation only."""

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=4,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def validate_pipelines(
    logistic_pipeline: Pipeline,
    random_forest_pipeline: Pipeline,
) -> None:
    """Validate the pipelines contain required preprocessing and support predict_proba()."""

    if "imputer" not in logistic_pipeline.named_steps:
        raise ValueError("Logistic Regression pipeline is missing the imputer step.")

    if "scaler" not in logistic_pipeline.named_steps:
        raise ValueError("Logistic Regression pipeline is missing the scaler step.")

    if "imputer" not in random_forest_pipeline.named_steps:
        raise ValueError("Random Forest pipeline is missing the imputer step.")

    if not hasattr(logistic_pipeline.named_steps["model"], "predict_proba"):
        raise ValueError("Logistic Regression model does not support predict_proba().")

    if not hasattr(random_forest_pipeline.named_steps["model"], "predict_proba"):
        raise ValueError("Random Forest model does not support predict_proba().")


def save_models(
    logistic_pipeline: Pipeline,
    random_forest_pipeline: Pipeline,
    feature_names: List[str],
    excluded_columns: List[str],
) -> tuple[Path, Path, Path]:
    """Save trained model artifacts and metadata files."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logistic_path = MODELS_DIR / "logistic_regression.joblib"
    random_forest_path = MODELS_DIR / "random_forest.joblib"
    feature_names_path = MODELS_DIR / "questionnaire_feature_names.json"

    joblib.dump(logistic_pipeline, logistic_path)
    joblib.dump(random_forest_pipeline, random_forest_path)

    feature_names_metadata = {
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "excluded_columns": excluded_columns,
        "target_column": TARGET_COLUMN,
    }

    feature_names_path.write_text(json.dumps(feature_names_metadata, indent=2), encoding="utf-8")

    return logistic_path, random_forest_path, feature_names_path


def save_training_metadata(
    df: pd.DataFrame,
    feature_names: List[str],
    excluded_columns: List[str],
) -> Path:
    """Save training metadata JSON."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    target_counts = df[TARGET_COLUMN].value_counts().sort_index()
    class_counts = {label: int(target_counts.get(label, 0)) for label in ["low", "moderate", "high"]}

    metadata = {
        "dataset_rows": int(len(df)),
        "feature_count": len(feature_names),
        "target_column": TARGET_COLUMN,
        "target_classes": ["low", "moderate", "high"],
        "class_counts": class_counts,
        "excluded_columns": excluded_columns,
        "random_state": RANDOM_STATE,
        "logistic_regression_configuration": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 2000,
            "solver": "lbfgs",
            "random_state": RANDOM_STATE,
        },
        "random_forest_configuration": {
            "n_estimators": 300,
            "max_depth": 4,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "imputation_strategy": "median",
        "scaling_used": True,
        "class_weight_strategy": "balanced",
        "synthetic_data_used": False,
        "training_accuracy_not_used_for_model_claims": True,
    }

    metadata_path = MODELS_DIR / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def generate_training_report(
    df: pd.DataFrame,
    feature_names: List[str],
    excluded_columns: List[str],
    logistic_pipeline: Pipeline,
    random_forest_pipeline: Pipeline,
    logistic_path: Path,
    random_forest_path: Path,
    feature_names_path: Path,
    metadata_path: Path,
) -> str:
    """Generate a training report documenting the artifact creation and pipeline setup."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "training_report.txt"

    target_counts = df[TARGET_COLUMN].value_counts().sort_index()
    class_distribution_lines = [
        f"- low: {int(target_counts.get('low', 0))}",
        f"- moderate: {int(target_counts.get('moderate', 0))}",
        f"- high: {int(target_counts.get('high', 0))}",
    ]

    lines: List[str] = [
        "Questionnaire Model Training Report",
        "===================================",
        f"Dataset rows: {len(df)}",
        f"Predictor feature count: {len(feature_names)}",
        "",
        "Predictor feature list:",
    ]

    for feature_name in feature_names:
        lines.append(f"- {feature_name}")

    lines.extend(
        [
            "",
            "Excluded columns:",
        ]
    )

    for excluded_column in excluded_columns:
        lines.append(f"- {excluded_column}")

    lines.extend(
        [
            "",
            "Target distribution:",
        ]
    )
    lines.extend(class_distribution_lines)

    lines.extend(
        [
            "",
            "Logistic Regression configuration:",
            f"- C: {logistic_pipeline.named_steps['model'].C}",
            f"- class_weight: {logistic_pipeline.named_steps['model'].class_weight}",
            f"- max_iter: {logistic_pipeline.named_steps['model'].max_iter}",
            f"- solver: {logistic_pipeline.named_steps['model'].solver}",
            f"- random_state: {logistic_pipeline.named_steps['model'].random_state}",
            "",
            "Random Forest configuration:",
            f"- n_estimators: {random_forest_pipeline.named_steps['model'].n_estimators}",
            f"- max_depth: {random_forest_pipeline.named_steps['model'].max_depth}",
            f"- min_samples_leaf: {random_forest_pipeline.named_steps['model'].min_samples_leaf}",
            f"- class_weight: {random_forest_pipeline.named_steps['model'].class_weight}",
            f"- random_state: {random_forest_pipeline.named_steps['model'].random_state}",
            f"- n_jobs: {random_forest_pipeline.named_steps['model'].n_jobs}",
            "",
            "Preprocessing pipeline:",
            "- Logistic Regression: SimpleImputer(median) -> StandardScaler -> LogisticRegression",
            "- Random Forest: SimpleImputer(median) -> RandomForestClassifier",
            "",
            "Missing-value strategy:",
            "- Imputation is performed inside each sklearn Pipeline using SimpleImputer(strategy='median').",
            "- No global imputation is applied outside the pipeline.",
            "",
            "Class balancing strategy:",
            "- Logistic Regression: class_weight='balanced'",
            "- Random Forest: class_weight='balanced'",
            "",
            "Leakage validation:",
            "- Target leakage check: PASS",
            "- Raw questionnaire text columns excluded from predictors: PASS",
            "- Q4_encoded excluded from predictors: PASS",
            "",
            "Artifact paths:",
            f"- Logistic Regression model: {logistic_path}",
            f"- Random Forest model: {random_forest_path}",
            f"- Feature metadata: {feature_names_path}",
            f"- Training metadata: {metadata_path}",
            "",
            "Training-set performance is not used as an estimate of generalization performance. Cross-validation is performed separately in Step 12.",
        ]
    )

    report_content = "\n".join(lines)
    report_path.write_text(report_content, encoding="utf-8")
    return report_content


def run_training(
    input_csv_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run the complete training workflow."""

    input_path = Path(input_csv_path) if input_csv_path is not None else INPUT_CSV_PATH
    df = load_dataset(input_path)
    df = define_target(df)

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected exactly {EXPECTED_ROWS} rows, found {len(df)}.")

    feature_names = define_predictor_features(df)
    validate_features(df, feature_names, TARGET_COLUMN, EXCLUDED_COLUMNS)

    X = df[feature_names]
    y = df[TARGET_COLUMN]

    if len(X) != len(y):
        raise ValueError("X and y do not have matching row counts.")

    logistic_pipeline = build_logistic_pipeline()
    random_forest_pipeline = build_random_forest_pipeline()

    validate_pipelines(logistic_pipeline, random_forest_pipeline)

    logistic_pipeline.fit(X, y)
    random_forest_pipeline.fit(X, y)

    logistic_path, random_forest_path, feature_names_path = save_models(
        logistic_pipeline, random_forest_pipeline, feature_names, EXCLUDED_COLUMNS
    )
    metadata_path = save_training_metadata(df, feature_names, EXCLUDED_COLUMNS)

    training_report = generate_training_report(
        df=df,
        feature_names=feature_names,
        excluded_columns=EXCLUDED_COLUMNS,
        logistic_pipeline=logistic_pipeline,
        random_forest_pipeline=random_forest_pipeline,
        logistic_path=logistic_path,
        random_forest_path=random_forest_path,
        feature_names_path=feature_names_path,
        metadata_path=metadata_path,
    )

    return {
        "dataset_rows": len(df),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "target_column": TARGET_COLUMN,
        "excluded_columns": EXCLUDED_COLUMNS,
        "logistic_path": logistic_path,
        "random_forest_path": random_forest_path,
        "feature_names_path": feature_names_path,
        "metadata_path": metadata_path,
        "report_path": RESULTS_DIR / "training_report.txt",
        "report_text": training_report,
    }


def main() -> None:
    """Run the Step 11 training workflow and print a concise summary."""

    results = run_training()

    class_counts = pd.read_csv(INPUT_CSV_PATH)["stress_level"].value_counts(dropna=False)

    print("QUESTIONNAIRE MODEL TRAINING COMPLETE")
    print(f"Rows: {results['dataset_rows']}")
    print(f"Predictor features: {results['feature_count']}")
    print("Target classes:")
    print(f"low: {int(class_counts.get('low', 0))}")
    print(f"moderate: {int(class_counts.get('moderate', 0))}")
    print(f"high: {int(class_counts.get('high', 0))}")
    print("Logistic Regression: trained")
    print("Random Forest: trained")
    print("Target leakage check: PASS")
    print("Pipeline validation: PASS")
    print(f"Artifacts: {results['logistic_path']}, {results['random_forest_path']}, {results['feature_names_path']}, {results['metadata_path']}")


if __name__ == "__main__":
    main()
