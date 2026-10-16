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


class TestAgreementAndConsistency:
    def test_single_source_confidence_is_not_boosted_or_penalised(self):
        fused = fuse_modalities(questionnaire=make_result("questionnaire", 0.2, confidence=0.7))
        assert fused["confidence"] == pytest.approx(0.7)

    def test_single_source_keeps_numeric_contract_but_is_not_applicable(self):
        fused = fuse_modalities(questionnaire=make_result("questionnaire", 0.2))
        assert fused["agreement"] == {"score": 1.0, "level": "not_applicable", "sources": 1}
        assert fused["consistency"]["applicable"] is False and fused["consistency"]["status"] == "single_source"
        assert fused["consistency"]["pattern"] == "none"

    def test_unavailable_results_report_agreement_not_applicable(self):
        empty = fuse_modalities()
        assert empty["agreement"] == {"score": 0.0, "level": "not_applicable", "sources": 0}
        assert empty["consistency"]["status"] == "no_sources"
        zero = fuse_modalities(voice=make_result("voice", 0.5, confidence=0.0))
        assert zero["agreement"]["level"] == "not_applicable" and zero["agreement"]["sources"] == 1

    def test_two_sources_are_compared(self):
        close = fuse_modalities(questionnaire=make_result("questionnaire", 0.50), behavioral=make_result("behavioral", 0.55))
        assert close["agreement"]["level"] == "high" and close["consistency"]["status"] == "consistent"
        assert close["consistency"]["applicable"] is True and close["consistency"]["pattern"] == "none"

    @pytest.mark.parametrize("q,b,status", [(0.50, 0.60, "consistent"), (0.40, 0.65, "moderate_difference"), (0.10, 0.80, "conflict")])
    def test_consistency_status_bands(self, q, b, status):
        fused = fuse_modalities(questionnaire=make_result("questionnaire", q), behavioral=make_result("behavioral", b))
        assert fused["consistency"]["status"] == status
        assert fused["consistency"]["gap"] == pytest.approx(b - q)

    def test_masked_stress_needs_two_sources_and_low_self_report(self):
        masked = fuse_modalities(questionnaire=make_result("questionnaire", 0.15), behavioral=make_result("behavioral", 0.80))
        assert masked["consistency"]["pattern"] == "masked_stress_possible"
        # the same low questionnaire alone can never show it
        assert fuse_modalities(questionnaire=make_result("questionnaire", 0.15))["consistency"]["pattern"] == "none"
        # a high self-report is not "masked"
        high = fuse_modalities(questionnaire=make_result("questionnaire", 0.75), behavioral=make_result("behavioral", 0.95))
        assert high["consistency"]["pattern"] == "none"

    def test_masked_pattern_uses_voice_too_and_ignores_voice_vs_behavioral(self):
        with_voice = fuse_modalities(questionnaire=make_result("questionnaire", 0.2), voice=make_result("voice", 0.7))
        assert with_voice["consistency"]["pattern"] == "masked_stress_possible"
        no_self_report = fuse_modalities(voice=make_result("voice", 0.1), behavioral=make_result("behavioral", 0.9))
        assert no_self_report["consistency"]["status"] == "conflict" and no_self_report["consistency"]["pattern"] == "none"

    def test_self_report_higher_pattern(self):
        fused = fuse_modalities(questionnaire=make_result("questionnaire", 0.85), behavioral=make_result("behavioral", 0.25))
        assert fused["consistency"]["pattern"] == "self_report_higher"

    def test_disagreement_lowers_confidence_but_single_source_is_unchanged(self):
        agree = fuse_modalities(questionnaire=make_result("questionnaire", 0.5, 0.8), behavioral=make_result("behavioral", 0.52, 0.8))
        clash = fuse_modalities(questionnaire=make_result("questionnaire", 0.1, 0.8), behavioral=make_result("behavioral", 0.9, 0.8))
        assert clash["confidence"] < agree["confidence"]
