"""Voice -> standardized fusion contract (predict_voice_from_features and audio path).

Uses a test-only model trained on random synthetic features. These tests verify
wiring, validation and error behavior, not predictive quality.
"""
import pytest

from src.ai_model.fusion.fusion_engine import fuse_modalities, validate_modality_result
from src.ai_model.voice import predict as voice_predict
from src.ai_model.voice.features import COMBINED_FEATURE_NAMES
from src.ai_model.voice.predict import (
    PredictionAudioError,
    VoiceModelNotTrainedError,
    extract_voice_features_from_audio,
    predict_voice_contract_from_audio,
    predict_voice_from_features,
)
from tests.voice_fixtures import full_feature_dict, make_test_voice_model, write_test_wav


@pytest.fixture(scope="module")
def model():
    return make_test_voice_model()


@pytest.fixture
def empty_models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_predict, "MODELS_DIR", tmp_path)
    return tmp_path


class TestVoiceContractFromFeatures:
    def test_contract_passes_fusion_validation(self, model):
        result = predict_voice_from_features(full_feature_dict(), model=model)
        validate_modality_result(result)  # raises on any contract violation
        assert result["modality"] == "voice"
        assert result["risk_level"] in ("low", "moderate", "high")
        assert 0.0 <= result["risk_score"] <= 1.0
        assert result["available_features"] == result["total_features"] == len(COMBINED_FEATURE_NAMES)
        assert result["data_quality"]["completeness"] == 1.0

    def test_voice_result_is_consumed_by_fusion_engine(self, model):
        result = predict_voice_from_features(full_feature_dict(), model=model)
        fused = fuse_modalities(voice=result)
        assert fused["status"] == "available"
        assert fused["modalities_used"] == ["voice"]
        assert fused["normalized_weights"] == {"voice": pytest.approx(1.0)}

    def test_incomplete_features_rejected_not_imputed(self, model):
        features = full_feature_dict()
        del features[COMBINED_FEATURE_NAMES[0]]
        with pytest.raises(ValueError, match="incomplete"):
            predict_voice_from_features(features, model=model)

    def test_none_values_count_as_missing(self, model):
        features = full_feature_dict()
        features[COMBINED_FEATURE_NAMES[3]] = None
        with pytest.raises(ValueError, match="incomplete"):
            predict_voice_from_features(features, model=model)

    @pytest.mark.parametrize(
        "bad, exc",
        [
            ("not a dict", TypeError),
            ({"not_a_feature": 1.0}, ValueError),
            ({COMBINED_FEATURE_NAMES[0]: "loud"}, ValueError),
            ({COMBINED_FEATURE_NAMES[0]: float("nan")}, ValueError),
            ({COMBINED_FEATURE_NAMES[0]: True}, ValueError),
            ({}, ValueError),
        ],
    )
    def test_malformed_features_rejected(self, model, bad, exc):
        with pytest.raises(exc):
            predict_voice_from_features(bad, model=model)

    def test_non_risk_class_model_rejected(self):
        wrong = make_test_voice_model(classes=["calm", "stressed", "neutral"])
        with pytest.raises(voice_predict.PredictionModelError):
            predict_voice_from_features(full_feature_dict(), model=wrong)


class TestModelAvailability:
    def test_untrained_raises_specific_error(self, empty_models_dir):
        with pytest.raises(VoiceModelNotTrainedError):
            predict_voice_from_features(full_feature_dict())

    def test_default_artifact_is_discovered(self, empty_models_dir, model):
        model.save(empty_models_dir / "voice_model.joblib")
        loaded, name = voice_predict.load_selected_voice_model()
        assert name == "voice_model"
        validate_modality_result(predict_voice_from_features(full_feature_dict()))

    def test_selected_model_json_points_to_missing_artifact(self, empty_models_dir):
        (empty_models_dir / "selected_model.json").write_text('{"selected_model": "ghost"}', encoding="utf-8")
        with pytest.raises(VoiceModelNotTrainedError):
            voice_predict.load_selected_voice_model()


class TestAudioPath:
    def test_extracts_all_combined_features(self, tmp_path):
        wav = write_test_wav(tmp_path / "speech.wav")
        features = extract_voice_features_from_audio(wav)
        assert set(features) == set(COMBINED_FEATURE_NAMES)

    def test_audio_to_contract(self, tmp_path, model):
        wav = write_test_wav(tmp_path / "speech.wav")
        result = predict_voice_contract_from_audio(wav, model=model)
        validate_modality_result(result)

    def test_missing_audio_file(self, tmp_path):
        with pytest.raises(PredictionAudioError):
            extract_voice_features_from_audio(tmp_path / "nope.wav")

    def test_corrupt_audio_file(self, tmp_path):
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"this is not a wav file")
        with pytest.raises(PredictionAudioError):
            extract_voice_features_from_audio(bad)

    def test_untrained_fails_before_audio_processing(self, empty_models_dir, tmp_path):
        with pytest.raises(VoiceModelNotTrainedError):
            predict_voice_contract_from_audio(tmp_path / "does_not_matter.wav")
