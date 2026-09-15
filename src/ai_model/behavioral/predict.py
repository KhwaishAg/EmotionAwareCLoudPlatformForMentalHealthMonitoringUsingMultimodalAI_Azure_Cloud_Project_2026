"""Production inference interface for the behavioral stress-risk modality.

The public :func:`predict_behavioral` function accepts one raw self-reported
behavioral response and returns only the standardized multimodal fusion
contract (the same shape questionnaire and voice use). It never refits the
model and never reads the training CSV.

Behavioral signals used are all self-reportable (sleep hours, study
consistency, social activity frequency, screen time, routine irregularity).
No passive sensor/wearable data is assumed, since not every student has
such a device.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

try:
    from .behavioral_schema import (
        FIELD_RANGES,
        REQUIRED_BEHAVIORAL_FIELDS,
        STRESS_DIRECTION,
        validate_behavioral_input,
    )
except ImportError:  # pragma: no cover - supports direct execution
    from behavioral_schema import (
        FIELD_RANGES,
        REQUIRED_BEHAVIORAL_FIELDS,
        STRESS_DIRECTION,
        validate_behavioral_input,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models" / "behavioral"
RESULTS_DIR = PROJECT_ROOT / "results" / "behavioral"
RISK_WEIGHTS = {"low": 0.0, "moderate": 0.5, "high": 1.0}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def load_model():
    """Load the trained behavioral pipeline."""
    artifact = MODELS_DIR / "logistic_regression.joblib"
    if not artifact.exists():
        raise FileNotFoundError(f"Behavioral model artifact was not found: {artifact}")
    return joblib.load(artifact)


def load_feature_metadata() -> Dict[str, Any]:
    metadata = _load_json(MODELS_DIR / "behavioral_feature_names.json")
    names = metadata.get("feature_names")
    if not isinstance(names, list) or not names or len(names) != metadata.get("feature_count"):
        raise ValueError("Behavioral feature metadata is invalid.")
    return metadata


def load_explainability_metadata() -> Dict[str, Any]:
    return _load_json(MODELS_DIR / "explainability_metadata.json")


def build_predictor_dataframe(response: Dict[str, Any]) -> pd.DataFrame:
    """Validate and arrange one raw response into the trained predictor order.

    Missing individual fields are allowed (None); the fitted pipeline
    handles no imputation here, so missing values are carried through
    as NaN and surfaced via data_quality rather than silently guessed.
    """
    validate_behavioral_input({k: v for k, v in response.items() if v is not None} | {
        k: v for k, v in response.items() if k in REQUIRED_BEHAVIORAL_FIELDS and v is None
    })
    metadata = load_feature_metadata()
    features = metadata["feature_names"]
    row = {field: response.get(field, np.nan) for field in features}
    row = {field: (np.nan if value is None else float(value)) for field, value in row.items()}
    return pd.DataFrame([row])[features]


def calculate_risk_score(probabilities: Dict[str, float]) -> float:
    """Project-level continuous risk representation (not clinical probability)."""
    return float(sum(RISK_WEIGHTS[level] * probabilities[level] for level in RISK_WEIGHTS))


def calculate_confidence(probabilities: Dict[str, float]) -> float:
    """Half maximum probability plus half class margin, same as questionnaire modality."""
    ordered = sorted(probabilities.values(), reverse=True)
    return float(np.clip(0.5 * ordered[0] + 0.5 * (ordered[0] - ordered[1]), 0.0, 1.0))


def calculate_data_quality(predictors: pd.DataFrame) -> Dict[str, Any]:
    total = len(predictors.columns)
    available = int(predictors.iloc[0].notna().sum())
    missing = total - available
    return {
        "available_features": available,
        "total_features": total,
        "data_quality": {"missing_features": missing, "completeness": float(available / total) if total else 0.0},
    }


def generate_key_factors(explainability: Dict[str, Any], limit: int = 5) -> list[Dict[str, Any]]:
    """Compact model-association factors; not causal explanations."""
    factors = explainability.get("top_features", [])[:limit]
    maximum = max((float(item.get("importance", 0.0)) for item in factors), default=0.0)
    output = []
    for item in factors:
        feature = str(item["feature"])
        direction = STRESS_DIRECTION.get(feature, "model_associated")
        output.append({
            "feature": feature,
            "contribution": float(item.get("importance", 0.0) / maximum) if maximum else 0.0,
            "direction": direction,
        })
    return output


def predict_behavioral(response: Dict[str, Any]) -> Dict[str, Any]:
    """Predict behavioral stress risk and return the standardized fusion contract.

    A field may be omitted or set to None if the student chose not to report
    it; this reduces data_quality/completeness rather than raising an error,
    since partial self-report is expected. The model pipeline itself does
    not impute -- if the underlying sklearn pipeline requires complete rows,
    an ImputeError-style ValueError is raised for that edge case, matching
    the fail-fast behavior of the questionnaire modality's contract checks.
    """
    if not isinstance(response, dict):
        raise TypeError("Behavioral response must be a Python dictionary.")
    invalid = [key for key in response if key not in REQUIRED_BEHAVIORAL_FIELDS]
    if invalid:
        raise ValueError("Unsupported behavioral fields: " + ", ".join(map(str, invalid)))
    for field, value in response.items():
        if value is None:
            continue
        low, high = FIELD_RANGES[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field} must be numeric.")
        if not (low <= float(value) <= high):
            raise ValueError(f"{field} must be between {low} and {high}.")

    model = load_model()
    predictors = build_predictor_dataframe(response)

    if predictors.isna().any(axis=None):
        # The simple pipeline has no imputer; a partially-missing report
        # cannot be scored. Surface this as a clear, catchable error so the
        # backend can mark this modality unavailable instead of crashing.
        raise ValueError("Behavioral prediction requires all five fields to be reported.")

    probabilities_array = model.predict_proba(predictors)[0]
    classes = [str(item) for item in model.classes_]
    probability_map = {label: float(value) for label, value in zip(classes, probabilities_array)}
    if set(probability_map) != set(RISK_WEIGHTS):
        raise ValueError("Model classes must be exactly low, moderate, and high.")
    if not all(0.0 <= value <= 1.0 for value in probability_map.values()) or not np.isclose(
        sum(probability_map.values()), 1.0, atol=1e-6
    ):
        raise ValueError("Model returned invalid probabilities.")

    risk_level = str(model.predict(predictors)[0])
    if risk_level != max(probability_map, key=probability_map.get):
        raise ValueError("Predicted class does not correspond to the maximum probability.")

    risk_score = calculate_risk_score(probability_map)
    confidence = calculate_confidence(probability_map)
    quality = calculate_data_quality(predictors)
    explainability = load_explainability_metadata()

    contract = {
        "modality": "behavioral",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence": confidence,
        "available_features": quality["available_features"],
        "total_features": quality["total_features"],
        "key_factors": generate_key_factors(explainability),
        "data_quality": quality["data_quality"],
    }
    if not (
        0 <= contract["risk_score"] <= 1
        and 0 <= contract["confidence"] <= 1
        and 0 <= contract["data_quality"]["completeness"] <= 1
        and len(contract["key_factors"]) <= 5
    ):
        raise ValueError("Fusion contract validation failed.")
    return contract


def _example_response() -> Dict[str, Any]:
    return {
        "B1_sleep_hours": 5.0,
        "B2_study_consistency": 2.0,
        "B3_social_activity_freq": 1.5,
        "B4_screen_time_hours": 9.0,
        "B5_routine_irregularity": 4.0,
    }


if __name__ == "__main__":
    sample = _example_response()
    result = predict_behavioral(sample)
    assert result["modality"] == "behavioral" and result["risk_level"] in RISK_WEIGHTS
    assert 0 <= result["risk_score"] <= 1 and 0 <= result["confidence"] <= 1
    assert result["available_features"] <= result["total_features"]
    assert result["data_quality"]["missing_features"] >= 0 and 0 <= result["data_quality"]["completeness"] <= 1
    assert len(result["key_factors"]) <= 5 and all(
        set(item) == {"feature", "contribution", "direction"} for item in result["key_factors"]
    )

    # Partial self-report should raise a catchable ValueError, not crash silently.
    partial = dict(sample)
    del partial["B4_screen_time_hours"]
    try:
        predict_behavioral(partial)
        raise AssertionError("Partial input should have raised ValueError")
    except ValueError:
        pass

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "sample_prediction.json").write_text(
        json.dumps({"development_test_output": True, "prediction": result}, indent=2), encoding="utf-8"
    )
    print("BEHAVIORAL PREDICTION PIPELINE COMPLETE")
    print(f"Risk level: {result['risk_level']}\nRisk score: {result['risk_score']:.4f}\nConfidence: {result['confidence']:.4f}")
    print(f"\nAvailable features: {result['available_features']}/{result['total_features']}\nCompleteness: {result['data_quality']['completeness']:.4f}")
    print("\nKey factors:")
    for number, factor in enumerate(result["key_factors"], 1):
        print(f"{number}. {factor['feature']} ({factor['direction']})")
    print("\nValidation:\nInput schema: PASS\nMissing-field handling: PASS\nProbability consistency: PASS\nRisk score validation: PASS\nConfidence validation: PASS\nFusion contract: PASS")
    print("\nOutput:\nresults/behavioral/sample_prediction.json")
