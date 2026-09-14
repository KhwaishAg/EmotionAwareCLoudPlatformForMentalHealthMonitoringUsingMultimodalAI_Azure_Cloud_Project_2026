"""Production inference interface for the questionnaire stress-risk modality.

The public :func:`predict_questionnaire` function accepts one raw response and
returns only the standardized multimodal fusion contract.  It never fits
preprocessing or reads the questionnaire training CSV.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from .ordinal_encoder import QUESTION_CATEGORY_MAPPINGS, SPECIAL_RESPONSE_TEXT
    from .multiselect_encoder import Q10_FEATURE_COLUMNS, encode_q10, encode_q11
    from .categorical_encoder import encode_q1, encode_q2, encode_q25
    from .feature_engineering import (
        create_academic_load_feature, create_behavior_dimension, create_coping_features,
        create_emotional_strain_dimension, create_overwhelm_dimension,
        create_persistence_features, create_recovery_features, create_stress_impact_dimension,
    )
except ImportError:  # pragma: no cover - supports direct execution
    from ordinal_encoder import QUESTION_CATEGORY_MAPPINGS, SPECIAL_RESPONSE_TEXT
    from multiselect_encoder import Q10_FEATURE_COLUMNS, encode_q10, encode_q11
    from categorical_encoder import encode_q1, encode_q2, encode_q25
    from feature_engineering import (
        create_academic_load_feature, create_behavior_dimension, create_coping_features,
        create_emotional_strain_dimension, create_overwhelm_dimension,
        create_persistence_features, create_recovery_features, create_stress_impact_dimension,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models" / "questionnaire"
RESULTS_DIR = PROJECT_ROOT / "results" / "questionnaire"
REQUIRED_QUESTION_FIELDS = tuple(f"Q{i}" for i in range(1, 21))
OPTIONAL_QUESTION_FIELDS = ("Q21", "Q22", "Q23", "Q24", "Q25", "Timestamp")
RISK_WEIGHTS = {"low": 0.0, "moderate": 0.5, "high": 1.0}
STRESS_ALIGNED = {
    "overwhelm_dimension", "stress_impact_dimension", "emotional_strain_dimension",
    "stress_persistence_dimension", "recovery_difficulty_dimension",
    "stress_behavior_dimension", "Q13_stress_aligned",
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def load_selected_model() -> Tuple[Any, str]:
    """Load the configured model pipeline and its dynamically selected name."""
    selected = _load_json(MODELS_DIR / "selected_model.json")
    model_name = selected.get("selected_model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("selected_model.json does not contain a valid selected_model.")
    artifact = MODELS_DIR / f"{model_name}.joblib"
    if not artifact.exists():
        raise FileNotFoundError(f"Selected model artifact was not found: {artifact}")
    return joblib.load(artifact), model_name


def load_feature_metadata() -> Dict[str, Any]:
    """Load and validate the exact training predictor list."""
    metadata = _load_json(MODELS_DIR / "questionnaire_feature_names.json")
    names = metadata.get("feature_names")
    if not isinstance(names, list) or not names or len(names) != metadata.get("feature_count"):
        raise ValueError("Questionnaire feature metadata is invalid.")
    forbidden = {"Q4", "Q4_encoded", "stress_level", "stress_level_encoded"}
    leaked = forbidden.intersection(names)
    if leaked:
        raise ValueError(f"Target leakage in predictor metadata: {sorted(leaked)}")
    return metadata


def load_explainability_metadata() -> Dict[str, Any]:
    """Load selected-model explainability and confidence metadata."""
    return _load_json(MODELS_DIR / "explainability_metadata.json")


def validate_questionnaire_input(response: dict) -> None:
    """Validate the API input shape and required raw questionnaire fields."""
    if not isinstance(response, dict):
        raise TypeError("Questionnaire response must be a Python dictionary.")
    missing = [field for field in REQUIRED_QUESTION_FIELDS if field not in response]
    if missing:
        raise ValueError("Missing required questionnaire fields: " + ", ".join(missing))
    invalid = [key for key in response if key not in REQUIRED_QUESTION_FIELDS + OPTIONAL_QUESTION_FIELDS]
    if invalid:
        raise ValueError("Unsupported questionnaire fields: " + ", ".join(map(str, invalid)))


def _normalise_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return " ".join(value.strip().replace("â€“", "–").replace("â€”", "—").split())


def _ordinal_value(question: str, value: Any) -> float:
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return np.nan
    normalized_mapping = {_normalise_text(key): number for key, number in QUESTION_CATEGORY_MAPPINGS[question].items()}
    return float(normalized_mapping.get(_normalise_text(value), np.nan))


def preprocess_single_response(response: dict) -> pd.DataFrame:
    """Transform one raw response using the same project encoding logic as training."""
    validate_questionnaire_input(response)
    raw = {field: response.get(field, np.nan) for field in REQUIRED_QUESTION_FIELDS + OPTIONAL_QUESTION_FIELDS}
    df = pd.DataFrame([raw])
    for question in QUESTION_CATEGORY_MAPPINGS:
        value = df.at[0, question]
        is_varies = question in {"Q14", "Q16"} and _normalise_text(value) == _normalise_text(SPECIAL_RESPONSE_TEXT)
        df[f"{question}_encoded"] = np.nan if is_varies else _ordinal_value(question, value)
        if question in {"Q14", "Q16"}:
            df[f"{question}_varies"] = 1 if is_varies else 0

    # Reuse the batch-safe encoders on this one-row frame; missing values remain missing.
    df, _ = encode_q10(df)
    # "Nothing" is an explicit no-stressor response, even if an inconsistent
    # multi-select payload also contains another option.
    q10_parts = [part.strip().lower() for part in str(df.at[0, "Q10"]).split(";")]
    if "nothing" in q10_parts:
        df.loc[:, Q10_FEATURE_COLUMNS] = 0
        df.loc[:, "Q10_nothing"] = 1
    df, _ = encode_q11(df)
    df, _ = encode_q1(df)
    df, _ = encode_q2(df)
    df, _ = encode_q25(df)
    for builder in (
        create_overwhelm_dimension, create_stress_impact_dimension,
        create_emotional_strain_dimension, create_coping_features,
        create_persistence_features, create_recovery_features,
        create_behavior_dimension, create_academic_load_feature,
    ):
        df = builder(df)
    return df


def build_predictor_dataframe(processed: pd.DataFrame, feature_metadata: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Select training-order predictors and preserve pre-imputation missingness."""
    metadata = feature_metadata or load_feature_metadata()
    features = metadata["feature_names"]
    missing = [feature for feature in features if feature not in processed.columns]
    if missing:
        raise ValueError("Preprocessing did not produce predictor columns: " + ", ".join(missing))
    predictors = processed.loc[:, features].copy().apply(pd.to_numeric, errors="coerce")
    if len(predictors.columns) != len(features) or list(predictors.columns) != features:
        raise ValueError("Predictor columns do not match training metadata.")
    return predictors


def calculate_risk_score(probabilities: Dict[str, float]) -> float:
    """Calculate the project-level continuous risk representation (not clinical probability)."""
    return float(sum(RISK_WEIGHTS[level] * probabilities[level] for level in RISK_WEIGHTS))


def calculate_confidence(probabilities: Dict[str, float]) -> float:
    """Step 13 confidence: half maximum probability plus half class margin."""
    ordered = sorted(probabilities.values(), reverse=True)
    return float(np.clip(0.5 * ordered[0] + 0.5 * (ordered[0] - ordered[1]), 0.0, 1.0))


def calculate_data_quality(predictors: pd.DataFrame) -> Dict[str, Any]:
    """Measure observed predictor availability before the pipeline imputes values."""
    total = len(predictors.columns)
    available = int(predictors.iloc[0].notna().sum())
    missing = total - available
    return {"available_features": available, "total_features": total,
            "data_quality": {"missing_features": missing, "completeness": float(available / total)}}


def generate_key_factors(explainability: Dict[str, Any], limit: int = 5) -> list[Dict[str, Any]]:
    """Return compact model-association factors; they are not causal explanations."""
    factors = explainability.get("top_features", [])[:limit]
    maximum = max((float(item.get("importance", 0.0)) for item in factors), default=0.0)
    output = []
    for item in factors:
        feature = str(item["feature"])
        direction = "increases_risk" if feature in STRESS_ALIGNED else ("decreases_risk" if feature == "Q13_encoded" else "model_associated")
        output.append({"feature": feature, "contribution": float(item.get("importance", 0.0) / maximum) if maximum else 0.0,
                       "direction": direction})
    return output


def predict_questionnaire(response: dict) -> dict:
    """Predict questionnaire stress risk and return the standardized fusion contract."""
    model, _ = load_selected_model()
    metadata = load_feature_metadata()
    explainability = load_explainability_metadata()
    predictors = build_predictor_dataframe(preprocess_single_response(response), metadata)
    probabilities_array = model.predict_proba(predictors)[0]
    classes = [str(item) for item in model.classes_]
    probability_map = {label: float(value) for label, value in zip(classes, probabilities_array)}
    if set(probability_map) != set(RISK_WEIGHTS):
        raise ValueError("Model classes must be exactly low, moderate, and high.")
    if not all(0.0 <= value <= 1.0 for value in probability_map.values()) or not np.isclose(sum(probability_map.values()), 1.0, atol=1e-6):
        raise ValueError("Model returned invalid probabilities.")
    risk_level = str(model.predict(predictors)[0])
    if risk_level != max(probability_map, key=probability_map.get):
        raise ValueError("Predicted class does not correspond to the maximum probability.")
    risk_score = calculate_risk_score(probability_map)
    confidence = calculate_confidence(probability_map)
    quality = calculate_data_quality(predictors)
    contract = {"modality": "questionnaire", "risk_score": risk_score, "risk_level": risk_level,
                "confidence": confidence, "available_features": quality["available_features"],
                "total_features": quality["total_features"], "key_factors": generate_key_factors(explainability),
                "data_quality": quality["data_quality"]}
    if not (0 <= contract["risk_score"] <= 1 and 0 <= contract["confidence"] <= 1 and
            0 <= contract["data_quality"]["completeness"] <= 1 and len(contract["key_factors"]) <= 5):
        raise ValueError("Fusion contract validation failed.")
    return contract


def _example_response() -> dict:
    return {"Q1": "4th Year", "Q2": "Regular academic semester; Working on academic projects", "Q3": "4–6 hours",
            "Q4": "4 — Very", "Q5": "Often", "Q6": "Sometimes", "Q7": "Very difficult", "Q8": "Often",
            "Q9": "Sometimes", "Q10": "Academic workload; Projects", "Q11": "Anxious; Tired", "Q12": "4",
            "Q13": "3", "Q14": "15–30 minutes", "Q15": "Sometimes", "Q16": "Within an hour", "Q17": "Often",
            "Q18": "Sometimes", "Q19": "Rarely", "Q20": "Sometimes", "Q21": "Academic workload", "Q22": "Overwhelmed",
            "Q23": "Rest", "Q24": "Yes", "Q25": "Academic workload"}


if __name__ == "__main__":
    sample = _example_response()
    result = predict_questionnaire(sample)
    optional_missing = {key: value for key, value in sample.items() if key not in {"Q21", "Q22", "Q23", "Q25"}}
    assert predict_questionnaire(optional_missing)["modality"] == "questionnaire"
    special = dict(sample, Q10="Nothing", Q2="All of them", Q14="It varies considerably", Q16="It varies considerably")
    special_processed = preprocess_single_response(special)
    assert special_processed.at[0, "Q14_varies"] == 1 and pd.isna(special_processed.at[0, "Q14_encoded"])
    assert special_processed.at[0, "Q16_varies"] == 1 and pd.isna(special_processed.at[0, "Q16_encoded"])
    assert special_processed.loc[0, Q10_FEATURE_COLUMNS].eq(0).all() and special_processed.at[0, "Q10_nothing"] == 1
    assert special_processed.loc[0, [name for name in special_processed if name.startswith("Q2_")]].eq(1).all()
    assert result["modality"] == "questionnaire" and result["risk_level"] in RISK_WEIGHTS
    assert 0 <= result["risk_score"] <= 1 and 0 <= result["confidence"] <= 1
    assert result["available_features"] <= result["total_features"] and "Q4" not in load_feature_metadata()["feature_names"]
    assert result["data_quality"]["missing_features"] >= 0 and 0 <= result["data_quality"]["completeness"] <= 1
    assert len(result["key_factors"]) <= 5 and all(set(item) == {"feature", "contribution", "direction"} for item in result["key_factors"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "sample_prediction.json").write_text(json.dumps({"development_test_output": True, "prediction": result}, indent=2), encoding="utf-8")
    print("QUESTIONNAIRE PREDICTION PIPELINE COMPLETE")
    print(f"Selected model: {load_selected_model()[1]}")
    print(f"Risk level: {result['risk_level']}\nRisk score: {result['risk_score']:.4f}\nConfidence: {result['confidence']:.4f}")
    print(f"\nAvailable features: {result['available_features']}/{result['total_features']}\nCompleteness: {result['data_quality']['completeness']:.4f}")
    print("\nKey factors:")
    for number, factor in enumerate(result["key_factors"], 1): print(f"{number}. {factor['feature']}")
    print("\nValidation:\nInput schema: PASS\nTarget leakage: PASS\nProbability consistency: PASS\nRisk score validation: PASS\nConfidence validation: PASS\nFusion contract: PASS")
    print("\nOutput:\nresults/questionnaire/sample_prediction.json\nresults/questionnaire/prediction_contract.md")
