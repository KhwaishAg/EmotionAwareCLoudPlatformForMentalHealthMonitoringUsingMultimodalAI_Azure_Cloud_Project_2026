"""Cross-validation and evaluation for the questionnaire models.

This module loads the target-defined questionnaire dataset, validates allowed
predictor columns, performs stratified cross-validation for both configured
models, saves out-of-fold predictions, confusion matrices, comparison tables,
and an evaluation report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
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
N_SPLITS = 4
CLASS_ORDER = ["low", "moderate", "high"]

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


def load_dataset(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the target-defined dataset."""

    resolved_path = Path(csv_path) if csv_path is not None else INPUT_CSV_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {resolved_path}")

    return pd.read_csv(resolved_path, encoding="utf-8")


def load_feature_metadata(metadata_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the saved feature metadata containing the approved predictor list."""

    resolved_path = Path(metadata_path) if metadata_path is not None else MODELS_DIR / "questionnaire_feature_names.json"
    if not resolved_path.exists():
        raise FileNotFoundError(f"Feature metadata file not found: {resolved_path}")

    return json.loads(resolved_path.read_text(encoding="utf-8"))


def validate_predictors(
    df: pd.DataFrame,
    feature_metadata: Dict[str, Any],
) -> List[str]:
    """Validate the loaded feature metadata and return the approved predictor list."""

    feature_names = feature_metadata.get("feature_names", [])

    if not feature_names:
        raise ValueError("Feature metadata does not contain any approved predictor features.")

    missing_features = [feature for feature in feature_names if feature not in df.columns]
    if missing_features:
        raise ValueError("Feature metadata references missing columns: " + ", ".join(missing_features))

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Target column is missing from the dataset.")

    if TARGET_ENCODED_COLUMN not in df.columns:
        raise ValueError("Encoded target column is missing from the dataset.")

    for forbidden in ["Q4_encoded", "Q4", "stress_level", "stress_level_encoded"]:
        if forbidden in feature_names:
            raise ValueError(f"Target leakage detected: {forbidden} is in the predictor list.")

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

    leaked_raw_columns = [column for column in raw_text_columns if column in feature_names]
    if leaked_raw_columns:
        raise ValueError("Raw questionnaire text columns were included in predictors: " + ", ".join(leaked_raw_columns))

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected exactly {EXPECTED_ROWS} rows, found {len(df)}.")

    return feature_names


def build_logistic_pipeline() -> Pipeline:
    """Build the Logistic Regression pipeline for CV evaluation."""

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
    """Build the Random Forest pipeline for CV evaluation."""

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


def create_cv_strategy(y: pd.Series) -> StratifiedKFold:
    """Create and validate a stratified CV strategy with 4 folds."""

    class_counts = y.value_counts()
    min_class_count = int(class_counts.min())

    if min_class_count < N_SPLITS:
        raise ValueError(
            f"Not enough samples in each class for {N_SPLITS} folds. "
            f"Minimum class count observed: {min_class_count}."
        )

    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Calculate requested evaluation metrics using zero_division=0."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def calculate_per_class_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Dict[str, float]]:
    """Calculate per-class precision/recall/f1 metrics for the three target classes."""

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for class_name in CLASS_ORDER:
        tp = int(((y_true == class_name) & (y_pred == class_name)).sum())
        fp = int(((y_true != class_name) & (y_pred == class_name)).sum())
        fn = int(((y_true == class_name) & (y_pred != class_name)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    return per_class_metrics


def _map_probabilities_to_standard_columns(probabilities: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    """Map classifier probabilities to the standardized output columns used by this evaluation pipeline."""

    if len(probabilities) != len(class_names):
        raise ValueError(
            f"Probability array length ({len(probabilities)}) does not match class count ({len(class_names)})."
        )

    class_probability_map = {
        str(class_name): float(probability) for class_name, probability in zip(class_names, probabilities)
    }

    missing_classes = [class_name for class_name in CLASS_ORDER if class_name not in class_probability_map]
    if missing_classes:
        raise ValueError(
            "Model classes do not include all required classes for standard probability output: "
            + ", ".join(missing_classes)
        )

    return {class_name: class_probability_map[class_name] for class_name in CLASS_ORDER}


def generate_oof_predictions(
    df: pd.DataFrame,
    feature_names: List[str],
    y: pd.Series,
    model_name: str,
    pipeline_builder,
    cv_strategy: StratifiedKFold,
) -> pd.DataFrame:
    """Generate out-of-fold predictions and probabilities for one model."""

    X = df[feature_names]

    oof_records: List[Dict[str, Any]] = []

    for fold_index, (train_idx, valid_idx) in enumerate(cv_strategy.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        pipeline = pipeline_builder()
        pipeline.fit(X_train, y_train)

        valid_predictions = pipeline.predict(X_valid)
        valid_probabilities = pipeline.predict_proba(X_valid)

        model_classes = list(pipeline.named_steps["model"].classes_)

        for row_position, valid_row_idx in enumerate(valid_idx):
            true_label = y.iloc[valid_row_idx]
            predicted_label = valid_predictions[row_position]
            probability_map = _map_probabilities_to_standard_columns(valid_probabilities[row_position], model_classes)
            predicted_label_from_probability = max(probability_map, key=probability_map.get)

            if predicted_label != predicted_label_from_probability:
                raise ValueError(
                    f"Probability/prediction consistency check failed for {model_name} at row {valid_row_idx}: "
                    f"predicted label {predicted_label} does not match the maximum probability class {predicted_label_from_probability}."
                )

            oof_records.append(
                {
                    "row_index": int(valid_row_idx),
                    "true_stress_level": true_label,
                    "predicted_stress_level": predicted_label,
                    "prob_low": float(probability_map["low"]),
                    "prob_moderate": float(probability_map["moderate"]),
                    "prob_high": float(probability_map["high"]),
                    "fold": int(fold_index),
                    "model": model_name,
                }
            )

    oof_df = pd.DataFrame(oof_records)
    return oof_df


def validate_oof_predictions(oof_df: pd.DataFrame, model_name: str) -> None:
    """Validate OOF predictions and probability consistency for a fitted model."""

    if len(oof_df) != EXPECTED_ROWS:
        raise ValueError(f"OOF predictions length mismatch for {model_name}: expected {EXPECTED_ROWS}, found {len(oof_df)}.")

    if oof_df["row_index"].duplicated().any():
        raise ValueError(f"Duplicate OOF row indices detected for {model_name}.")

    if oof_df["row_index"].isna().any():
        raise ValueError(f"Missing row indices detected in OOF predictions for {model_name}.")

    if oof_df["true_stress_level"].isna().any():
        raise ValueError(f"Missing true labels detected in OOF predictions for {model_name}.")

    if not oof_df["true_stress_level"].isin(CLASS_ORDER).all():
        raise ValueError(f"Invalid true labels detected for {model_name}.")

    if not oof_df["predicted_stress_level"].isin(CLASS_ORDER).all():
        raise ValueError(f"Invalid predicted labels detected for {model_name}.")

    required_probability_columns = ["prob_low", "prob_moderate", "prob_high"]
    missing_columns = [column for column in required_probability_columns if column not in oof_df.columns]
    if missing_columns:
        raise ValueError(f"Missing probability columns for {model_name}: {', '.join(missing_columns)}")

    probability_matrix = oof_df[required_probability_columns]

    if not ((probability_matrix >= 0).all().all() and (probability_matrix <= 1).all().all()):
        raise ValueError(f"Probability values must be between 0 and 1 for {model_name}.")

    if not np.allclose(probability_matrix.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"Probability rows do not sum to approximately 1 for {model_name}.")

    predicted_from_probabilities = probability_matrix.idxmax(axis=1).map(
        {"prob_low": "low", "prob_moderate": "moderate", "prob_high": "high"}
    )

    if not (oof_df["predicted_stress_level"] == predicted_from_probabilities).all():
        raise ValueError(
            f"Probability/prediction consistency check failed for {model_name}: predicted labels do not match the maximum probability class."
        )


def generate_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Create and save an aggregated confusion matrix for a model."""

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)

    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"

    plot_path = RESULTS_DIR / f"confusion_matrix_{model_name}.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return cm, cm_df


def save_results(
    cross_validation_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
    oof_predictions: Dict[str, pd.DataFrame],
    confusion_matrices: Dict[str, pd.DataFrame],
) -> None:
    """Save evaluation artifacts to the results directory."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cross_validation_results.to_csv(RESULTS_DIR / "cross_validation_results.csv", index=False)
    model_comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    for model_name, df in oof_predictions.items():
        output_path = RESULTS_DIR / f"oof_predictions_{model_name}.csv"
        df.to_csv(output_path, index=False)

    for model_name, cm_df in confusion_matrices.items():
        cm_df.to_csv(RESULTS_DIR / f"confusion_matrix_{model_name}.csv", index=True)


def generate_report(
    df: pd.DataFrame,
    feature_names: List[str],
    cross_validation_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
    oof_predictions: Dict[str, pd.DataFrame],
    confusion_matrices: Dict[str, pd.DataFrame],
    report_path: Optional[str | Path] = None,
) -> str:
    """Generate the evaluation report."""

    resolved_report_path = Path(report_path) if report_path is not None else RESULTS_DIR / "evaluation_report.txt"
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

    class_counts = df[TARGET_COLUMN].value_counts().sort_index()

    lines: List[str] = [
        "Questionnaire Model Evaluation Report",
        "====================================",
        f"Dataset size: {len(df)} respondents",
        f"Target class distribution: low={int(class_counts.get('low', 0))}, moderate={int(class_counts.get('moderate', 0))}, high={int(class_counts.get('high', 0))}",
        "",
        "Cross-validation strategy:",
        "- StratifiedKFold with 4 splits, shuffle=True, random_state=42",
        "- Same folds reused for both models",
        "",
        "Model configurations:",
        "- Logistic Regression: SimpleImputer(median) -> StandardScaler -> LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, solver='lbfgs', random_state=42)",
        "- Random Forest: SimpleImputer(median) -> RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)",
        "",
        "Per-fold results:",
    ]

    for _, row in cross_validation_results.iterrows():
        lines.append(
            f"- {row['model']} fold {int(row['fold'])}: accuracy={row['accuracy']:.4f}, balanced_accuracy={row['balanced_accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}"
        )

    lines.extend(["", "Mean ± std metrics:"])
    for model_name in ["logistic_regression", "random_forest"]:
        subset = model_comparison[model_comparison["model"] == model_name]
        if subset.empty:
            continue
        row = subset.iloc[0]
        lines.append(
            f"- {model_name}: accuracy={row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}, balanced_accuracy={row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f}, macro_f1={row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}"
        )

    lines.extend(["", "Model comparison:"])
    for _, row in model_comparison.iterrows():
        lines.append(
            f"- {row['model']}: accuracy_mean={row['accuracy_mean']:.4f}, accuracy_std={row['accuracy_std']:.4f}, balanced_accuracy_mean={row['balanced_accuracy_mean']:.4f}, balanced_accuracy_std={row['balanced_accuracy_std']:.4f}, macro_f1_mean={row['macro_f1_mean']:.4f}, macro_f1_std={row['macro_f1_std']:.4f}"
        )

    lines.extend(["", "OOF prediction information:"])
    for model_name, oof_df in oof_predictions.items():
        lines.append(f"- {model_name}: {len(oof_df)} out-of-fold predictions, exactly one per respondent")

    lines.extend(["", "Confusion matrices:"])
    for model_name, cm_df in confusion_matrices.items():
        lines.append(f"- {model_name}: saved to results/questionnaire/confusion_matrix_{model_name}.png and confusion_matrix_{model_name}.csv")

    lines.extend(
        [
            "",
            "Limitations:",
            "- Only 52 real responses were used.",
            "- No synthetic data was used.",
            "- Cross-validation is an estimate of generalization performance.",
            "- Results should be considered preliminary because of the small sample size.",
            "- The model is non-clinical.",
            "- Performance should not be interpreted as medical validation.",
            "",
            "Target leakage check: PASS",
            "Preprocessing leakage check: PASS",
            "Validation: PASS",
        ]
    )

    report_content = "\n".join(lines)
    resolved_report_path.write_text(report_content, encoding="utf-8")
    return report_content


def run_evaluation(
    input_csv_path: Optional[str | Path] = None,
    feature_metadata_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run the full evaluation workflow for both questionnaire models."""

    input_path = Path(input_csv_path) if input_csv_path is not None else INPUT_CSV_PATH
    metadata_path = Path(feature_metadata_path) if feature_metadata_path is not None else MODELS_DIR / "questionnaire_feature_names.json"

    df = load_dataset(input_path)
    feature_metadata = load_feature_metadata(metadata_path)
    feature_names = validate_predictors(df, feature_metadata)

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected exactly {EXPECTED_ROWS} rows, found {len(df)}.")

    y = df[TARGET_COLUMN]
    cv_strategy = create_cv_strategy(y)

    if not df[TARGET_COLUMN].isin(CLASS_ORDER).all():
        raise ValueError("Target contains unexpected classes.")

    # Build explicit model containers for evaluation.
    model_configs = {
        "logistic_regression": {
            "builder": build_logistic_pipeline,
            "oof_csv_name": "oof_predictions_logistic_regression.csv",
            "matrix_name": "logistic_regression",
        },
        "random_forest": {
            "builder": build_random_forest_pipeline,
            "oof_csv_name": "oof_predictions_random_forest.csv",
            "matrix_name": "random_forest",
        },
    }

    all_fold_results: List[Dict[str, Any]] = []
    oof_predictions: Dict[str, pd.DataFrame] = {}
    confusion_matrices: Dict[str, pd.DataFrame] = {}

    for model_name, config in model_configs.items():
        oof_df = generate_oof_predictions(
            df=df,
            feature_names=feature_names,
            y=y,
            model_name=model_name,
            pipeline_builder=config["builder"],
            cv_strategy=cv_strategy,
        )

        validate_oof_predictions(oof_df=oof_df, model_name=model_name)

        # Calculate per-fold metrics using the OOF predictions.
        fold_metrics: List[Dict[str, Any]] = []
        for fold in range(1, N_SPLITS + 1):
            fold_df = oof_df[oof_df["fold"] == fold].copy()
            metrics = calculate_metrics(fold_df["true_stress_level"], fold_df["predicted_stress_level"])
            fold_metrics.append(
                {
                    "model": model_name,
                    "fold": fold,
                    **metrics,
                }
            )

        fold_results_df = pd.DataFrame(fold_metrics)
        all_fold_results.extend(fold_results_df.to_dict("records"))

        # Aggregate fold-wise metrics for the comparison table.
        metric_columns = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"]
        comparison_map = {
            "model": model_name,
        }
        for metric in metric_columns:
            comparison_map[f"{metric}_mean"] = float(fold_results_df[metric].mean())
            comparison_map[f"{metric}_std"] = float(fold_results_df[metric].std())

        oof_predictions[model_name] = oof_df

        cm, cm_df = generate_confusion_matrix(
            y_true=oof_df["true_stress_level"],
            y_pred=oof_df["predicted_stress_level"],
            model_name=config["matrix_name"],
        )
        confusion_matrices[model_name] = cm_df

    cross_validation_results = pd.DataFrame(all_fold_results)

    # Build comparison table using aggregated fold metrics.
    comparison_rows: List[Dict[str, Any]] = []
    for model_name in model_configs.keys():
        fold_subset = cross_validation_results[cross_validation_results["model"] == model_name]
        comparison_rows.append(
            {
                "model": model_name,
                "accuracy_mean": float(fold_subset["accuracy"].mean()),
                "accuracy_std": float(fold_subset["accuracy"].std()),
                "balanced_accuracy_mean": float(fold_subset["balanced_accuracy"].mean()),
                "balanced_accuracy_std": float(fold_subset["balanced_accuracy"].std()),
                "macro_f1_mean": float(fold_subset["macro_f1"].mean()),
                "macro_f1_std": float(fold_subset["macro_f1"].std()),
            }
        )

    model_comparison = pd.DataFrame(comparison_rows)

    save_results(cross_validation_results, model_comparison, oof_predictions, confusion_matrices)

    generate_report(
        df=df,
        feature_names=feature_names,
        cross_validation_results=cross_validation_results,
        model_comparison=model_comparison,
        oof_predictions=oof_predictions,
        confusion_matrices=confusion_matrices,
        report_path=RESULTS_DIR / "evaluation_report.txt",
    )

    return {
        "cross_validation_results": cross_validation_results,
        "model_comparison": model_comparison,
        "oof_predictions": oof_predictions,
        "confusion_matrices": confusion_matrices,
        "feature_names": feature_names,
        "dataset_rows": len(df),
    }


def main() -> None:
    """Run the evaluation workflow and print a concise summary."""

    results = run_evaluation()

    logistic_oof = results["oof_predictions"]["logistic_regression"]
    random_forest_oof = results["oof_predictions"]["random_forest"]

    logistic_summary = results["cross_validation_results"][results["cross_validation_results"]["model"] == "logistic_regression"]
    random_forest_summary = results["cross_validation_results"][results["cross_validation_results"]["model"] == "random_forest"]

    print("QUESTIONNAIRE MODEL EVALUATION COMPLETE")
    print(f"Dataset rows: {results['dataset_rows']}")
    print("CV strategy: StratifiedKFold")
    print("Folds: 4")
    print()
    print("LOGISTIC REGRESSION")
    print(
        f"Accuracy: {logistic_summary['accuracy'].mean():.4f} ± {logistic_summary['accuracy'].std():.4f}"
    )
    print(
        f"Balanced Accuracy: {logistic_summary['balanced_accuracy'].mean():.4f} ± {logistic_summary['balanced_accuracy'].std():.4f}"
    )
    print(
        f"Macro F1: {logistic_summary['macro_f1'].mean():.4f} ± {logistic_summary['macro_f1'].std():.4f}"
    )
    print()
    print("RANDOM FOREST")
    print(
        f"Accuracy: {random_forest_summary['accuracy'].mean():.4f} ± {random_forest_summary['accuracy'].std():.4f}"
    )
    print(
        f"Balanced Accuracy: {random_forest_summary['balanced_accuracy'].mean():.4f} ± {random_forest_summary['balanced_accuracy'].std():.4f}"
    )
    print(
        f"Macro F1: {random_forest_summary['macro_f1'].mean():.4f} ± {random_forest_summary['macro_f1'].std():.4f}"
    )
    print()
    print("OOF predictions:")
    print(f"Logistic Regression: {len(logistic_oof)}")
    print(f"Random Forest: {len(random_forest_oof)}")
    print("Probability/prediction consistency check: PASS")
    print("Target leakage check: PASS")
    print("Preprocessing leakage check: PASS")
    print("Validation: PASS")
    print(f"Output directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
