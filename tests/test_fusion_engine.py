"""Fusion engine: contract validation, weighting, missing-modality policy."""
import copy

import pytest

from src.ai_model.fusion.fusion_engine import fuse_modalities, validate_modality_result


def make_result(modality, score, confidence=0.8, available=10, total=10, factor="f"):
    level = "low" if score < 0.34 else "moderate" if score < 0.67 else "high"
    return {
        "modality": modality,
        "risk_score": score,
        "risk_level": level,
        "confidence": confidence,
        "available_features": available,
        "total_features": total,
        "key_factors": [{"feature": factor, "contribution": 0.5, "direction": "model_associated"}],
        "data_quality": {"missing_features": total - available, "completeness": available / total if total else 0.0},
    }


class TestNormalFusion:
    def test_single_modality(self):
        fused = fuse_modalities(questionnaire=make_result("questionnaire", 0.2))
        assert fused["status"] == "available"
        assert fused["modalities_used"] == ["questionnaire"]
        assert fused["risk_score"] == pytest.approx(0.2)
        assert fused["risk_level"] == "low"
        assert fused["agreement"]["level"] == "not_applicable"  # one source cannot agree or disagree
        assert fused["agreement"]["sources"] == 1

    def test_three_modalities_weights_sum_to_one(self):
        fused = fuse_modalities(
            questionnaire=make_result("questionnaire", 0.6, 0.9),
            voice=make_result("voice", 0.5, 0.4),
            behavioral=make_result("behavioral", 0.7, 0.7),
        )
        assert sorted(fused["modalities_used"]) == ["behavioral", "questionnaire", "voice"]
        assert sum(fused["normalized_weights"].values()) == pytest.approx(1.0)
        assert 0.5 <= fused["risk_score"] <= 0.7
        # higher-confidence modality gets the larger weight
        assert fused["normalized_weights"]["questionnaire"] > fused["normalized_weights"]["voice"]

    def test_disagreement_reduces_confidence(self):
        agree = fuse_modalities(voice=make_result("voice", 0.6), behavioral=make_result("behavioral", 0.62))
        clash = fuse_modalities(voice=make_result("voice", 0.05), behavioral=make_result("behavioral", 0.95))
        assert clash["agreement"]["level"] == "low"
        assert clash["confidence"] < agree["confidence"]

    def test_inputs_are_not_mutated(self):
        original = make_result("voice", 0.4)
        snapshot = copy.deepcopy(original)
        fuse_modalities(voice=original)
        assert original == snapshot

    def test_data_quality_aggregates(self):
        fused = fuse_modalities(
            questionnaire=make_result("questionnaire", 0.4, available=8, total=10),
            behavioral=make_result("behavioral", 0.4, available=5, total=5),
        )
        assert fused["available_features"] == 13 and fused["total_features"] == 15


class TestMissingModalityPolicy:
    def test_no_modalities_is_explicitly_unavailable(self):
        fused = fuse_modalities()
        assert fused["status"] == "fusion_unavailable"
        assert fused["risk_score"] is None and fused["risk_level"] is None
        assert fused["modalities_used"] == []

    def test_missing_modality_is_not_zero_filled(self):
        only_q = fuse_modalities(questionnaire=make_result("questionnaire", 0.9))
        assert only_q["risk_score"] == pytest.approx(0.9)
        assert "voice" not in only_q["modality_scores"]

    def test_empty_modality_data_is_excluded(self):
        empty = make_result("voice", 0.5, available=0, total=10)
        fused = fuse_modalities(voice=empty, behavioral=make_result("behavioral", 0.3))
        assert fused["modalities_used"] == ["behavioral"]

    def test_zero_confidence_inputs_yield_no_risk(self):
        fused = fuse_modalities(voice=make_result("voice", 0.5, confidence=0.0))
        assert fused["status"] == "fusion_unavailable_zero_weight"
        assert fused["risk_score"] is None


class TestMalformedInputs:
    @pytest.mark.parametrize("missing", ["risk_score", "confidence", "data_quality", "key_factors", "risk_level"])
    def test_missing_field_rejected(self, missing):
        result = make_result("voice", 0.5)
        del result[missing]
        with pytest.raises(ValueError, match="missing fields"):
            fuse_modalities(voice=result)

    def test_not_a_dict_rejected(self):
        with pytest.raises(ValueError):
            validate_modality_result(["voice"])

    def test_unknown_modality_rejected(self):
        with pytest.raises(ValueError, match="Unknown modality"):
            validate_modality_result(make_result("face", 0.5))

    def test_modality_passed_in_wrong_slot_rejected(self):
        with pytest.raises(ValueError, match="has modality"):
            fuse_modalities(voice=make_result("behavioral", 0.5))

    @pytest.mark.parametrize("field,value", [("risk_score", 1.5), ("confidence", -0.1), ("risk_score", float("nan")),
                                             ("risk_score", True), ("risk_score", "0.5")])
    def test_out_of_range_or_non_numeric_rejected(self, field, value):
        result = make_result("voice", 0.5)
        result[field] = value
        with pytest.raises(ValueError):
            validate_modality_result(result)

    def test_bad_risk_level_rejected(self):
        result = make_result("voice", 0.5)
        result["risk_level"] = "critical"
        with pytest.raises(ValueError, match="risk_level"):
            validate_modality_result(result)

    def test_available_greater_than_total_rejected(self):
        with pytest.raises(ValueError, match="Feature counts"):
            validate_modality_result(make_result("voice", 0.5, available=11, total=10))

    def test_bad_key_factor_rejected(self):
        result = make_result("voice", 0.5)
        result["key_factors"] = [{"feature": "x"}]
        with pytest.raises(ValueError, match="key factor"):
            validate_modality_result(result)
