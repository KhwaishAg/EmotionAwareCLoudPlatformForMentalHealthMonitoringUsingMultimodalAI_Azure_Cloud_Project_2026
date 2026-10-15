"""Confidence- and data-quality-aware fusion of questionnaire, voice, and behavioral outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:  # supports `python src/.../fusion_engine.py`
    sys.path.insert(0, str(PROJECT_ROOT))

ALLOWED_MODALITIES = ("questionnaire", "voice", "behavioral")
RISK_LEVELS = ("low", "moderate", "high")
REQUIRED_FIELDS = {"modality", "risk_score", "risk_level", "confidence", "available_features", "total_features", "key_factors", "data_quality"}
RESULTS_PATH = PROJECT_ROOT / "results" / "fusion" / "fusion_test_results.json"


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{field} must be a finite numeric value.")
    return float(value)


def validate_modality_result(result: Mapping[str, Any]) -> None:
    """Raise ValueError unless one standardized modality contract is valid."""
    if not isinstance(result, Mapping):
        raise ValueError("Modality result must be a dictionary.")
    missing = REQUIRED_FIELDS.difference(result)
    if missing:
        raise ValueError("Malformed modality result; missing fields: " + ", ".join(sorted(missing)))
    if result["modality"] not in ALLOWED_MODALITIES:
        raise ValueError(f"Unknown modality: {result['modality']!r}. Allowed: {', '.join(ALLOWED_MODALITIES)}.")
    for field in ("risk_score", "confidence"):
        if not 0 <= _number(result[field], field) <= 1:
            raise ValueError(f"{field} must be between 0 and 1.")
    if result["risk_level"] not in RISK_LEVELS:
        raise ValueError("risk_level must be low, moderate, or high.")
    available = _number(result["available_features"], "available_features")
    total = _number(result["total_features"], "total_features")
    if available < 0 or total < 0 or available > total or available != int(available) or total != int(total):
        raise ValueError("Feature counts must be non-negative integers with available_features <= total_features.")
    quality = result["data_quality"]
    if not isinstance(quality, Mapping) or {"missing_features", "completeness"}.difference(quality):
        raise ValueError("data_quality must include missing_features and completeness.")
    missing_features = _number(quality["missing_features"], "missing_features")
    completeness = _number(quality["completeness"], "completeness")
    if missing_features < 0 or missing_features != int(missing_features) or not 0 <= completeness <= 1:
        raise ValueError("data_quality values are invalid.")
    if not isinstance(result["key_factors"], list):
        raise ValueError("key_factors must be a list.")
    for factor in result["key_factors"]:
        if not isinstance(factor, Mapping) or {"feature", "contribution", "direction"}.difference(factor):
            raise ValueError("Each key factor requires feature, contribution, and direction.")
        if not isinstance(factor["feature"], str) or not isinstance(factor["direction"], str):
            raise ValueError("Key-factor feature and direction must be strings.")
        if not 0 <= _number(factor["contribution"], "key factor contribution") <= 1:
            raise ValueError("Key-factor contribution must be between 0 and 1.")


def calculate_modality_weight(result: Mapping[str, Any]) -> float:
    return float(result["confidence"] * result["data_quality"]["completeness"])


def normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        return {}
    normalized = {name: float(weight / total) for name, weight in weights.items()}
    if not np.isclose(sum(normalized.values()), 1.0, atol=1e-9):
        raise ValueError("Normalized modality weights do not sum to 1.")
    return normalized


def calculate_fused_risk(results: Mapping[str, Mapping[str, Any]], weights: Mapping[str, float]) -> float:
    return float(np.clip(sum(results[name]["risk_score"] * weights[name] for name in results), 0.0, 1.0))


def calculate_agreement(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    scores = [result["risk_score"] for result in results.values()]
    score = 1.0 if len(scores) == 1 else float(np.clip(1.0 - (max(scores) - min(scores)), 0.0, 1.0))
    level = "high" if score >= .75 else "moderate" if score >= .50 else "low"
    return {"score": score, "level": level}


def calculate_fused_confidence(results: Mapping[str, Mapping[str, Any]], weights: Mapping[str, float], agreement_score: float) -> float:
    weighted_confidence = sum(results[name]["confidence"] * weights[name] for name in results)
    return float(np.clip(weighted_confidence * (.5 + .5 * agreement_score), 0.0, 1.0))


def aggregate_key_factors(results: Mapping[str, Mapping[str, Any]], weights: Mapping[str, float]) -> list[Dict[str, Any]]:
    """Aggregate model-associated factors; importance is not causal evidence."""
    aggregated: Dict[str, Dict[str, Any]] = {}
    for modality, result in results.items():
        for factor in result["key_factors"]:
            feature = factor["feature"]
            weighted = float(factor["contribution"] * weights[modality])
            if feature not in aggregated:
                aggregated[feature] = {"modality": modality, "feature": feature, "contribution": 0.0,
                                       "direction": factor["direction"], "_modalities": [modality]}
            item = aggregated[feature]
            item["contribution"] += weighted
            if modality not in item["_modalities"]:
                item["_modalities"].append(modality)
    output = []
    for item in aggregated.values():
        item["modality"] = ",".join(item.pop("_modalities"))
        item["contribution"] = float(np.clip(item["contribution"], 0.0, 1.0))
        output.append(item)
    return sorted(output, key=lambda item: item["contribution"], reverse=True)[:5]


def calculate_data_quality(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"available_features": 0, "total_features": 0, "data_quality": {"missing_features": 0, "completeness": 0.0}}
    available = int(sum(item["available_features"] for item in results.values()))
    total = int(sum(item["total_features"] for item in results.values()))
    return {"available_features": available, "total_features": total,
            "data_quality": {"missing_features": total - available, "completeness": float(available / total)}}


def _level(score: float) -> str:
    return "low" if score < .34 else "moderate" if score < .67 else "high"


def fuse_modalities(questionnaire=None, voice=None, behavioral=None) -> dict:
    """Fuse any usable standardized outputs; unavailable modalities are ignored, never zero-filled."""
    supplied = {"questionnaire": questionnaire, "voice": voice, "behavioral": behavioral}
    usable: Dict[str, Mapping[str, Any]] = {}
    for expected_name, result in supplied.items():
        if result is None:
            continue
        validate_modality_result(result)
        if result["modality"] != expected_name:
            raise ValueError(f"Result supplied as {expected_name} has modality {result['modality']!r}.")
        if result["total_features"] > 0 and result["available_features"] > 0 and result["data_quality"]["completeness"] > 0:
            usable[expected_name] = result
    if not usable:
        quality = calculate_data_quality({})
        return {"modality": "multimodal_fusion", "risk_score": None, "risk_level": None, "confidence": 0.0,
                **quality, "key_factors": [], "modalities_used": [], "modality_scores": {},
                "modality_confidences": {}, "modality_weights": {}, "normalized_weights": {},
                "agreement": {"score": 0.0, "level": "low"}, "status": "fusion_unavailable"}
    raw_weights = {name: calculate_modality_weight(item) for name, item in usable.items()}
    normalized = normalize_weights(raw_weights)
    if not normalized:
        # Valid but zero-confidence inputs cannot support a meaningful weighted risk.
        quality = calculate_data_quality(usable)
        return {"modality": "multimodal_fusion", "risk_score": None, "risk_level": None, "confidence": 0.0,
                **quality, "key_factors": [], "modalities_used": list(usable), "modality_scores": {name: item["risk_score"] for name, item in usable.items()},
                "modality_confidences": {name: item["confidence"] for name, item in usable.items()}, "modality_weights": raw_weights,
                "normalized_weights": {}, "agreement": {"score": 0.0, "level": "low"}, "status": "fusion_unavailable_zero_weight"}
    risk = calculate_fused_risk(usable, normalized)
    agreement = calculate_agreement(usable)
    confidence = calculate_fused_confidence(usable, normalized, agreement["score"])
    quality = calculate_data_quality(usable)
    return {"modality": "multimodal_fusion", "risk_score": risk, "risk_level": _level(risk), "confidence": confidence,
            **quality, "key_factors": aggregate_key_factors(usable, normalized), "modalities_used": list(usable),
            "modality_scores": {name: item["risk_score"] for name, item in usable.items()},
            "modality_confidences": {name: item["confidence"] for name, item in usable.items()}, "modality_weights": raw_weights,
            "normalized_weights": normalized, "agreement": agreement, "status": "available"}


def _mock(modality: str, score: float, confidence: float, feature: str) -> dict:
    """TEST ONLY — NOT REAL MODEL OUTPUT."""
    return {"modality": modality, "risk_score": score, "risk_level": _level(score), "confidence": confidence,
            "available_features": 8, "total_features": 10, "key_factors": [{"feature": feature, "contribution": .8, "direction": "model_associated"}],
            "data_quality": {"missing_features": 2, "completeness": .8}}


def _questionnaire_example() -> dict:
    from src.ai_model.questionnaire.predict import predict_questionnaire
    response = {"Q1": "4th Year", "Q2": "Regular academic semester; Working on academic projects", "Q3": "4–6 hours", "Q4": "4 — Very", "Q5": "Often", "Q6": "Sometimes", "Q7": "Very difficult", "Q8": "Often", "Q9": "Sometimes", "Q10": "Academic workload; Projects", "Q11": "Anxious; Tired", "Q12": "4", "Q13": "3", "Q14": "15–30 minutes", "Q15": "Sometimes", "Q16": "Within an hour", "Q17": "Often", "Q18": "Sometimes", "Q19": "Rarely", "Q20": "Sometimes"}
    return predict_questionnaire(response)


def _assert_contract(result: dict) -> None:
    assert result["risk_score"] is None or 0 <= result["risk_score"] <= 1
    assert result["risk_level"] is None or result["risk_level"] in RISK_LEVELS
    assert 0 <= result["confidence"] <= 1 and 0 <= result["agreement"]["score"] <= 1
    assert len(result["key_factors"]) <= 5 and 0 <= result["data_quality"]["completeness"] <= 1
    if result["normalized_weights"]: assert np.isclose(sum(result["normalized_weights"].values()), 1.0)


if __name__ == "__main__":
    questionnaire = _questionnaire_example()
    mock_voice_result = _mock("voice", .72, .80, "stress_related_prosody")  # TEST ONLY — NOT REAL MODEL OUTPUT
    mock_behavioral_result = _mock("behavioral", .58, .70, "stress_behavior_dimension")  # TEST ONLY — NOT REAL MODEL OUTPUT
    scenarios = [
        ("Questionnaire only", dict(questionnaire=questionnaire)), ("Questionnaire + Voice", dict(questionnaire=questionnaire, voice=mock_voice_result)),
        ("Questionnaire + Behavior", dict(questionnaire=questionnaire, behavioral=mock_behavioral_result)), ("All modalities", dict(questionnaire=questionnaire, voice=mock_voice_result, behavioral=mock_behavioral_result)),
        ("Voice only", dict(voice=mock_voice_result)), ("Behavior only", dict(behavioral=mock_behavioral_result)), ("No modalities", {}),
        ("High disagreement", dict(voice=_mock("voice", .05, .9, "a"), behavioral=_mock("behavioral", .95, .9, "b"))),
        ("High agreement", dict(voice=_mock("voice", .60, .9, "a"), behavioral=_mock("behavioral", .62, .9, "b"))),
    ]
    records = []
    for name, arguments in scenarios:
        result = fuse_modalities(**arguments); _assert_contract(result); records.append({"test": name, "status": "PASS", "result": result})
    try:
        fuse_modalities(voice={"modality": "voice"})
        raise AssertionError("Malformed input was accepted")
    except ValueError:
        records.insert(7, {"test": "Malformed input", "status": "PASS"})
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"integration_test_only": True, "notice": "Mock voice and behavioral results are test fixtures, not model outputs or research results.", "tests": records}, indent=2), encoding="utf-8")
    print("MULTIMODAL FUSION ENGINE COMPLETE\n")
    for index, record in enumerate(records, 1): print(f"Test {index} - {record['test']}: {record['status']}")
    print("\nValidation:\nWeight validation: PASS\nRisk score validation: PASS\nConfidence validation: PASS\nAgreement validation: PASS\nMissing modality handling: PASS\nFusion contract: PASS\n\nOutput:\nresults/fusion/fusion_test_results.json")
