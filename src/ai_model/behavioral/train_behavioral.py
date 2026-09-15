"""Train the behavioral stress-risk model (Logistic Regression, deliberately simple).

Trains on synthetic self-reportable behavioral data (see generate_synthetic_data.py).
Saves the fitted pipeline and supporting metadata under models/behavioral/,
mirroring the artifact layout Member 1 used for the questionnaire modality.

Run:
    python generate_synthetic_data.py
    python train_behavioral.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .behavioral_schema import REQUIRED_BEHAVIORAL_FIELDS
except ImportError:  # pragma: no cover - direct execution support
    from behavioral_schema import REQUIRED_BEHAVIORAL_FIELDS

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "behavioral_synthetic.csv"
MODELS_DIR = PROJECT_ROOT / "models" / "behavioral"
RESULTS_DIR = PROJECT_ROOT / "results" / "behavioral"
RANDOM_SEED = 42


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run generate_synthetic_data.py first."
        )
    df = pd.read_csv(DATA_PATH)
    feature_names = list(REQUIRED_BEHAVIORAL_FIELDS)
    X = df[feature_names]
    y = df["stress_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred, labels=["low", "moderate", "high"])
    accuracy = float(report["accuracy"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODELS_DIR / "logistic_regression.joblib")

    (MODELS_DIR / "behavioral_feature_names.json").write_text(
        json.dumps({"feature_names": feature_names, "feature_count": len(feature_names)}, indent=2),
        encoding="utf-8",
    )

    # Feature importance via absolute standardized logistic-regression coefficients,
    # averaged across the one-vs-rest classes. This is a model-association measure,
    # not a causal explanation.
    clf: LogisticRegression = pipeline.named_steps["clf"]
    coefficients = np.abs(clf.coef_).mean(axis=0)
    importance_order = np.argsort(coefficients)[::-1]
    top_features = [
        {"feature": feature_names[i], "importance": float(coefficients[i])}
        for i in importance_order
    ]
    (MODELS_DIR / "explainability_metadata.json").write_text(
        json.dumps({"top_features": top_features}, indent=2), encoding="utf-8"
    )

    (MODELS_DIR / "training_metadata.json").write_text(
        json.dumps(
            {
                "synthetic_data": True,
                "notice": "Trained on synthetic self-reported behavioral data. "
                          "Near-perfect scores reflect clean synthetic class separation, "
                          "not real-world predictive performance.",
                "model": "logistic_regression",
                "n_train": len(X_train),
                "n_test": len(X_test),
                "test_accuracy": accuracy,
                "random_seed": RANDOM_SEED,
                "classes": list(clf.classes_),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (RESULTS_DIR / "evaluation_report.txt").write_text(
        "BEHAVIORAL MODEL EVALUATION (SYNTHETIC DATA)\n"
        "==============================================\n"
        "NOTICE: Trained and evaluated on synthetic self-reported behavioral\n"
        "data. Scores reflect clean synthetic class separation, not real-world\n"
        "performance. Do not present as validated clinical or research results.\n\n"
        f"Test accuracy: {accuracy:.4f}\n\n"
        f"Classification report:\n{classification_report(y_test, y_pred)}\n"
        f"Confusion matrix (rows=true, cols=pred, order=[low, moderate, high]):\n{matrix}\n",
        encoding="utf-8",
    )

    print("BEHAVIORAL MODEL TRAINING COMPLETE (SYNTHETIC DATA)")
    print(f"Test accuracy: {accuracy:.4f}")
    print("NOTE: synthetic data -> near-perfect scores are expected and not meaningful")
    print(f"\nSaved model: {MODELS_DIR / 'logistic_regression.joblib'}")
    print(f"Saved metadata: {MODELS_DIR / 'behavioral_feature_names.json'}")
    print(f"Saved explainability: {MODELS_DIR / 'explainability_metadata.json'}")


if __name__ == "__main__":
    main()
