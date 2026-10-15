"""Non-mutating final validation for the Member 1 questionnaire and fusion components."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_model.questionnaire.predict import (build_predictor_dataframe, load_feature_metadata,
                                                  predict_questionnaire, preprocess_single_response)
from src.ai_model.fusion.fusion_engine import fuse_modalities

MODELS = PROJECT_ROOT / "models" / "questionnaire"
RESULTS = PROJECT_ROOT / "results" / "questionnaire"
REPORT = PROJECT_ROOT / "results" / "final" / "member1_final_validation_report.txt"
LEVELS = {"low", "moderate", "high"}


def _require(paths: list[Path]) -> None:
    absent = [str(path.relative_to(PROJECT_ROOT)) for path in paths if not path.exists()]
    if absent:
        raise FileNotFoundError("Required artifacts are missing: " + ", ".join(absent))


def example_response() -> dict:
    return {"Q1": "4th Year", "Q2": "Regular academic semester; Working on academic projects", "Q3": "4–6 hours", "Q4": "4 — Very", "Q5": "Often", "Q6": "Sometimes", "Q7": "Very difficult", "Q8": "Often", "Q9": "Sometimes", "Q10": "Academic workload; Projects", "Q11": "Anxious; Tired", "Q12": "4", "Q13": "3", "Q14": "15–30 minutes", "Q15": "Sometimes", "Q16": "Within an hour", "Q17": "Often", "Q18": "Sometimes", "Q19": "Rarely", "Q20": "Sometimes", "Q21": "Academic workload", "Q22": "Overwhelmed", "Q23": "Rest", "Q24": "Yes", "Q25": "Academic workload"}


def validate_dataset() -> None:
    _require([PROJECT_ROOT / "data/raw/questionnaire_responses.csv", PROJECT_ROOT / "data/processed/questionnaire_cleaned.csv", PROJECT_ROOT / "data/processed/questionnaire_features.csv", PROJECT_ROOT / "data/processed/questionnaire_with_target.csv"])
    target = pd.read_csv(PROJECT_ROOT / "data/processed/questionnaire_with_target.csv")
    assert len(target) == 52
    assert target["stress_level"].value_counts().to_dict() == {"high": 24, "moderate": 20, "low": 8}


def validate_models() -> None:
    _require([MODELS / name for name in ["logistic_regression.joblib", "random_forest.joblib", "questionnaire_feature_names.json", "training_metadata.json", "selected_model.json", "explainability_metadata.json"]])
    import joblib
    selected = json.loads((MODELS / "selected_model.json").read_text(encoding="utf-8"))["selected_model"]
    assert selected in {"logistic_regression", "random_forest"}
    model = joblib.load(MODELS / f"{selected}.joblib")
    assert callable(getattr(model, "predict", None)) and callable(getattr(model, "predict_proba", None))
    assert len(load_feature_metadata()["feature_names"]) == load_feature_metadata()["feature_count"]


def validate_evaluation() -> None:
    _require([RESULTS / name for name in ["cross_validation_results.csv", "model_comparison.csv", "oof_predictions_logistic_regression.csv", "oof_predictions_random_forest.csv", "random_forest_feature_importance.csv", "logistic_regression_coefficients.csv", "oof_confidence_analysis.csv", "evaluation_report.txt", "explainability_report.txt"]])
    for name in ("logistic_regression", "random_forest"):
        oof = pd.read_csv(RESULTS / f"oof_predictions_{name}.csv")
        assert len(oof) == 52 and not oof["row_index"].duplicated().any()
        probabilities = oof[["prob_low", "prob_moderate", "prob_high"]]
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        assert oof["true_stress_level"].isin(LEVELS).all() and oof["predicted_stress_level"].isin(LEVELS).all()
        predicted = probabilities.idxmax(axis=1).map({"prob_low": "low", "prob_moderate": "moderate", "prob_high": "high"})
        assert (predicted == oof["predicted_stress_level"]).all()


def validate_prediction_and_edges() -> dict:
    response = example_response()
    result = predict_questionnaire(response)
    assert set(result) == {"modality", "risk_score", "risk_level", "confidence", "available_features", "total_features", "key_factors", "data_quality"}
    assert result["modality"] == "questionnaire" and 0 <= result["risk_score"] <= 1 and result["risk_level"] in LEVELS
    assert 0 <= result["confidence"] <= 1 and len(result["key_factors"]) <= 5 and 0 <= result["data_quality"]["completeness"] <= 1
    # Development-only inference cases; no dataset files are changed.
    cases = [
        {key: value for key, value in response.items() if key not in {"Q21", "Q22", "Q23", "Q25"}},
        dict(response, Q10="Nothing"), dict(response, Q2="All of them"),
        dict(response, Q14="It varies considerably"), dict(response, Q16="It varies considerably"),
        {key: value for key, value in response.items() if key not in {"Q21", "Q22", "Q23", "Q24", "Q25"}},
    ]
    for case in cases:
        assert predict_questionnaire(case)["risk_level"] in LEVELS
    processed = preprocess_single_response(dict(response, Q10="Nothing", Q2="All of them", Q14="It varies considerably", Q16="It varies considerably"))
    assert processed.at[0, "Q10_nothing"] == 1 and processed.at[0, "Q14_varies"] == 1 and processed.at[0, "Q16_varies"] == 1
    predictors = build_predictor_dataframe(processed)
    leaked = {"Q4", "Q4_encoded", "stress_level", "stress_level_encoded"}.intersection(predictors.columns)
    assert not leaked
    return result


def _mock(modality: str, risk: float) -> dict:
    return {"modality": modality, "risk_score": risk, "risk_level": "high" if risk >= .67 else "moderate", "confidence": .8, "available_features": 8, "total_features": 10, "key_factors": [{"feature": f"{modality}_test_indicator", "contribution": .8, "direction": "model_associated"}], "data_quality": {"missing_features": 2, "completeness": .8}}


def validate_fusion(questionnaire: dict) -> None:
    voice, behavioral = _mock("voice", .72), _mock("behavioral", .58)  # integration fixtures only
    scenarios = [{"questionnaire": questionnaire}, {"questionnaire": questionnaire, "voice": voice}, {"questionnaire": questionnaire, "behavioral": behavioral}, {"questionnaire": questionnaire, "voice": voice, "behavioral": behavioral}, {"voice": voice}, {"behavioral": behavioral}, {}]
    for arguments in scenarios:
        fused = fuse_modalities(**arguments)
        assert fused["risk_score"] is None or 0 <= fused["risk_score"] <= 1
        assert fused["risk_level"] is None or fused["risk_level"] in LEVELS
        assert 0 <= fused["confidence"] <= 1 and 0 <= fused["agreement"]["score"] <= 1 and len(fused["key_factors"]) <= 5
        if fused["normalized_weights"]:
            assert np.isclose(sum(fused["normalized_weights"].values()), 1.0)
    assert fuse_modalities(questionnaire=questionnaire, voice=None)["risk_score"] == fuse_modalities(questionnaire=questionnaire)["risk_score"]


def validate_reproducibility() -> None:
    metadata = json.loads((MODELS / "training_metadata.json").read_text(encoding="utf-8"))
    assert metadata["random_state"] == 42 and metadata["synthetic_data_used"] is False
    assert metadata["logistic_regression_configuration"]["random_state"] == 42
    assert metadata["random_forest_configuration"]["random_state"] == 42


def main() -> None:
    validate_dataset(); validate_models(); validate_evaluation(); prediction = validate_prediction_and_edges(); validate_fusion(prediction); validate_reproducibility()
    _require([PROJECT_ROOT / "requirements.txt", PROJECT_ROOT / "src/ai_model/questionnaire/README.md", PROJECT_ROOT / "src/ai_model/fusion/README.md", PROJECT_ROOT / "docs/member1_questionnaire_implementation_report.md"])
    lines = ["Member 1 Final Validation Report", "================================", "", "Dataset validation: PASS", "Model artifacts: PASS", "Evaluation artifacts: PASS", "Prediction pipeline: PASS", "Target leakage: PASS", "Probability consistency: PASS", "Edge cases: PASS", "Fusion integration: PASS", "Reproducibility: PASS", "Documentation: PASS", "", "Warnings:", "- Small dataset: 52 respondents.", "- Model performance is preliminary.", "- System is non-clinical."]
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("MEMBER 1 FINAL VALIDATION COMPLETE\n")
    for line in lines[3:14]: print(line)
    print("\nOverall status: PASS\n\nWarnings:\n- Small dataset: 52 respondents\n- Preliminary model performance\n- Non-clinical system")


if __name__ == "__main__":
    main()
