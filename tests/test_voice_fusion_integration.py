"""
Integration Tests for Voice-to-Multimodal Fusion Interface (Commit 24).

Covers:
    - VoiceFusionPayload construction, serialization, and validation
    - Conversion of VoicePredictionResult to VoiceFusionPayload via .to_fusion_payload()
    - Full prepare_voice_fusion_payload() pipeline from audio file to fusion input
    - Arbitrary target labels (string, categorical, integer) without hardcoded assumptions
    - Exact preservation of class probabilities and predicted class
    - Rejection of malformed/missing payloads with typed InvalidFusionPayloadError
    - Mock/stub late decision-level fusion consumer integration
    - Regression protection: existing predict_voice() behavior unchanged

All tests use synthetic audio and fitted/mocked models.
No private participant recordings, no fabricated real labels, no cloud calls.
"""

import math
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

from src.ai_model.voice.config import AUDIO_CONFIG, FEATURE_CONFIG, AudioConfig, FeatureConfig
from src.ai_model.voice.model import (
    BaseVoiceClassifier,
    VoiceLogisticRegression,
    VoiceRandomForest,
)
from src.ai_model.voice.predict import (
    PredictionAudioError,
    PredictionError,
    VoicePredictionResult,
    predict_voice,
)
from src.ai_model.voice.fusion_interface import (
    MODALITY_NAME,
    InvalidFusionPayloadError,
    VoiceFusionIntegrationError,
    VoiceFusionPayload,
    prepare_voice_fusion_payload,
    validate_voice_fusion_payload,
)


# ===========================================================================
# Helpers for Synthetic Test Generation
# ===========================================================================

def generate_test_wav(filepath: Path, duration_seconds: float = 5.0, sample_rate: int = 16000) -> Path:
    """Generate a clean synthetic PCM WAV file."""
    t = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    waveform = (0.6 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    scaled = np.clip(waveform * 32767.0, -32768.0, 32767.0).astype(np.int16)
    wavfile.write(str(filepath), sample_rate, scaled)
    return filepath


def create_fitted_voice_model(
    model_class=VoiceRandomForest,
    classes: Optional[list] = None,
    n_features: int = 41,
) -> BaseVoiceClassifier:
    """Create a fitted model with synthetic features."""
    if classes is None:
        classes = ["calm", "stressed", "neutral"]

    rng = np.random.RandomState(42)
    n_samples = 60
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = np.array(classes * (n_samples // len(classes)))[:n_samples]

    model = model_class(random_state=42)
    model.fit(X, y)
    return model


# ===========================================================================
# 1. VoiceFusionPayload Structure and Serialization
# ===========================================================================

class TestVoiceFusionPayload:
    """Tests for VoiceFusionPayload data structure and serialization."""

    def test_basic_construction_and_validation(self):
        """Verify well-formed VoiceFusionPayload constructs and validates cleanly."""
        payload = VoiceFusionPayload(
            modality=MODALITY_NAME,
            predicted_class="calm",
            class_probabilities={"calm": 0.7, "stressed": 0.2, "neutral": 0.1},
            confidence=0.7,
            model_type="VoiceRandomForest",
            feature_count=41,
            audio_duration_seconds=5.0,
            model_path="/models/voice.joblib",
            metadata={"source": "test"},
        )
        # Should not raise
        payload.validate()
        assert payload.modality == "voice"
        assert payload.predicted_class == "calm"
        assert payload.confidence == 0.7

    def test_arbitrary_labels_support(self):
        """Verify VoiceFusionPayload works with arbitrary user-defined labels."""
        # Non-standard categorical labels
        payload_arbitrary = VoiceFusionPayload(
            modality="voice",
            predicted_class="tier_b",
            class_probabilities={"tier_a": 0.1, "tier_b": 0.6, "tier_c": 0.3},
            confidence=0.6,
            model_type="VoiceLogisticRegression",
            feature_count=41,
            audio_duration_seconds=5.0,
            model_path="/models/voice.joblib",
        )
        payload_arbitrary.validate()
        assert payload_arbitrary.predicted_class == "tier_b"

        # Integer class labels
        payload_int = VoiceFusionPayload(
            modality="voice",
            predicted_class=1,
            class_probabilities={"0": 0.3, "1": 0.7},
            confidence=0.7,
            model_type="VoiceLogisticRegression",
            feature_count=41,
            audio_duration_seconds=5.0,
            model_path="/models/voice.joblib",
        )
        payload_int.validate()
        assert payload_int.predicted_class == 1

    def test_to_dict_and_from_dict_roundtrip(self):
        """Verify dictionary serialization preserves all fields."""
        payload = VoiceFusionPayload(
            modality=MODALITY_NAME,
            predicted_class="stressed",
            class_probabilities={"calm": 0.3, "stressed": 0.7},
            confidence=0.7,
            model_type="VoiceRandomForest",
            feature_count=41,
            audio_duration_seconds=10.2,
            model_path="/path/to/model.joblib",
            metadata={"trained_samples": 120},
        )
        data = payload.to_dict()

        assert data["modality"] == "voice"
        assert data["predicted_class"] == "stressed"
        assert data["class_probabilities"] == {"calm": 0.3, "stressed": 0.7}
        assert data["confidence"] == 0.7
        assert data["model_type"] == "VoiceRandomForest"
        assert data["feature_count"] == 41
        assert data["metadata"] == {"trained_samples": 120}

        reconstructed = VoiceFusionPayload.from_dict(data)
        assert reconstructed.modality == payload.modality
        assert reconstructed.predicted_class == payload.predicted_class
        assert reconstructed.class_probabilities == payload.class_probabilities
        assert reconstructed.confidence == payload.confidence

    def test_validate_voice_fusion_payload_helper(self):
        """Verify validate_voice_fusion_payload accepts both payload object and dict."""
        raw_dict = {
            "modality": "voice",
            "predicted_class": "mild",
            "class_probabilities": {"mild": 0.8, "severe": 0.2},
            "confidence": 0.8,
            "model_type": "VoiceRandomForest",
            "feature_count": 41,
            "audio_duration_seconds": 6.0,
            "model_path": "/m.joblib",
        }
        validated = validate_voice_fusion_payload(raw_dict)
        assert isinstance(validated, VoiceFusionPayload)
        assert validated.predicted_class == "mild"

        # Re-validating the object itself
        again = validate_voice_fusion_payload(validated)
        assert again is validated


# ===========================================================================
# 2. Rejection of Malformed Payloads
# ===========================================================================

class TestFusionPayloadValidationRejection:
    """Tests that malformed or incomplete payloads are strictly rejected."""

    def test_wrong_modality_rejected(self):
        """Verify non-voice modality identifier is rejected."""
        payload = VoiceFusionPayload(
            modality="behavioral",
            predicted_class="calm",
            class_probabilities={"calm": 1.0},
            confidence=1.0,
            model_type="Model",
            feature_count=41,
            audio_duration_seconds=5.0,
            model_path="/m.joblib",
        )
        with pytest.raises(InvalidFusionPayloadError, match="Invalid modality identifier"):
            payload.validate()

    def test_none_predicted_class_rejected(self):
        """Verify None predicted_class is rejected."""
        payload = VoiceFusionPayload(
            modality="voice",
            predicted_class=None,
            class_probabilities={"calm": 1.0},
            confidence=1.0,
            model_type="Model",
            feature_count=41,
            audio_duration_seconds=5.0,
            model_path="/m.joblib",
        )
        with pytest.raises(InvalidFusionPayloadError, match="Predicted class cannot be None"):
            payload.validate()

    def test_predicted_class_not_in_probabilities_rejected(self):
        """Verify predicted_class missing from probability keys is rejected."""
        payload = VoiceFusionPayload(
            modality="voice",
            predicted_class="absent_class",
            class_probabilities={"calm": 0.6, "stressed": 0.4},
            confidence=0.6,
            model_type="Model",
            feature_count=41,
            audio_duration_seconds=5.0,
            model_path="/m.joblib",
        )
        with pytest.raises(InvalidFusionPayloadError, match="not present in class_probabilities"):
            payload.validate()

    def test_empty_probabilities_rejected(self):
        """Verify empty probability dictionary is rejected."""
        payload = VoiceFusionPayload(
            modality="voice",
            predicted_class="calm",
            class_probabilities={},
            confidence=0.0,
            model_type="Model",
            feature_count=41,
            audio_duration_seconds=5.0,
            model_path="/m.joblib",
        )
        with pytest.raises(InvalidFusionPayloadError, match="non-empty dictionary"):
            payload.validate()

    def test_probabilities_sum_violates_axiom_rejected(self):
        """Verify probabilities that do not sum to ~1.0 are rejected."""
        payload = VoiceFusionPayload(
            modality="voice",
            predicted_class="calm",
            class_probabilities={"calm": 0.3, "stressed": 0.2},  # sums to 0.5
            confidence=0.3,
            model_type="Model",
            feature_count=41,
            audio_duration_seconds=5.0,
            model_path="/m.joblib",
        )
        with pytest.raises(InvalidFusionPayloadError, match="must sum to 1.0"):
            payload.validate()

    def test_negative_probability_rejected(self):
        """Verify negative probabilities are rejected."""
        payload = VoiceFusionPayload(
            modality="voice",
            predicted_class="calm",
            class_probabilities={"calm": 1.2, "stressed": -0.2},
            confidence=1.2,
            model_type="Model",
            feature_count=41,
            audio_duration_seconds=5.0,
            model_path="/m.joblib",
        )
        with pytest.raises(InvalidFusionPayloadError, match="must be in"):
            payload.validate()

    def test_nan_probability_rejected(self):
        """Verify NaN or Inf probability values are rejected."""
        payload = VoiceFusionPayload(
            modality="voice",
            predicted_class="calm",
            class_probabilities={"calm": float("nan"), "stressed": 0.5},
            confidence=0.5,
            model_type="Model",
            feature_count=41,
            audio_duration_seconds=5.0,
            model_path="/m.joblib",
        )
        with pytest.raises(InvalidFusionPayloadError, match="not finite"):
            payload.validate()

    def test_missing_keys_in_from_dict_rejected(self):
        """Verify from_dict checks for all mandatory keys."""
        incomplete_dict = {
            "modality": "voice",
            "predicted_class": "calm",
            # missing class_probabilities, confidence, etc.
        }
        with pytest.raises(InvalidFusionPayloadError, match="Missing required fields"):
            VoiceFusionPayload.from_dict(incomplete_dict)

    def test_from_prediction_result_type_check(self):
        """Verify from_prediction_result rejects non-VoicePredictionResult objects."""
        with pytest.raises(InvalidFusionPayloadError, match="Expected VoicePredictionResult"):
            VoiceFusionPayload.from_prediction_result("not a result")  # type: ignore


# ===========================================================================
# 3. Conversion from VoicePredictionResult
# ===========================================================================

class TestVoicePredictionToFusionConversion:
    """Tests converting VoicePredictionResult into VoiceFusionPayload."""

    def test_conversion_preserves_all_attributes(self):
        """Verify .to_fusion_payload() accurately converts prediction output."""
        pred_result = VoicePredictionResult(
            predicted_class="stressed",
            class_probabilities={"calm": 0.25, "stressed": 0.65, "neutral": 0.10},
            model_type="VoiceRandomForest",
            feature_count=41,
            audio_duration_seconds=8.5,
            model_path="/models/rf.joblib",
            metadata={"feature_names": ["mfcc_1_mean"]},
        )

        fusion_payload = pred_result.to_fusion_payload()

        assert isinstance(fusion_payload, VoiceFusionPayload)
        assert fusion_payload.modality == "voice"
        assert fusion_payload.predicted_class == "stressed"
        assert fusion_payload.class_probabilities == {
            "calm": 0.25,
            "stressed": 0.65,
            "neutral": 0.10,
        }
        assert math.isclose(fusion_payload.confidence, 0.65, abs_tol=1e-4)
        assert fusion_payload.model_type == "VoiceRandomForest"
        assert fusion_payload.feature_count == 41
        assert fusion_payload.audio_duration_seconds == 8.5
        assert fusion_payload.model_path == "/models/rf.joblib"
        assert fusion_payload.metadata == {"feature_names": ["mfcc_1_mean"]}


# ===========================================================================
# 4. End-to-End Pipeline: prepare_voice_fusion_payload
# ===========================================================================

class TestPrepareVoiceFusionPayloadEndToEnd:
    """Tests the full inference-to-fusion pipeline on audio files."""

    def test_prepare_voice_fusion_payload_with_random_forest(self, tmp_path: Path):
        """Verify full inference and fusion packaging with fitted Random Forest."""
        wav_file = tmp_path / "sample.wav"
        generate_test_wav(wav_file, duration_seconds=5.0)

        classes = ["calm", "stressed"]
        model = create_fitted_voice_model(VoiceRandomForest, classes=classes)

        payload = prepare_voice_fusion_payload(
            audio_file_path=wav_file,
            model=model,
        )

        assert isinstance(payload, VoiceFusionPayload)
        assert payload.modality == "voice"
        assert payload.predicted_class in classes
        assert set(payload.class_probabilities.keys()) == set(classes)
        assert math.isclose(sum(payload.class_probabilities.values()), 1.0, abs_tol=1e-4)
        assert payload.feature_count == 41
        assert payload.model_type == "VoiceRandomForest"
        assert payload.confidence >= 0.5  # For binary classification

    def test_prepare_voice_fusion_payload_with_logistic_regression(self, tmp_path: Path):
        """Verify full inference and fusion packaging with fitted Logistic Regression."""
        wav_file = tmp_path / "sample.wav"
        generate_test_wav(wav_file, duration_seconds=5.0)

        classes = ["class_0", "class_1", "class_2"]
        model = create_fitted_voice_model(VoiceLogisticRegression, classes=classes)

        payload = prepare_voice_fusion_payload(
            audio_file_path=wav_file,
            model=model,
        )

        assert payload.modality == "voice"
        assert payload.predicted_class in classes
        assert set(payload.class_probabilities.keys()) == set(classes)
        assert payload.model_type == "VoiceLogisticRegression"

    def test_prepare_voice_fusion_payload_audio_error_propagation(self, tmp_path: Path):
        """Verify prediction audio errors (e.g. missing file) propagate cleanly."""
        missing_wav = tmp_path / "nonexistent.wav"
        model = create_fitted_voice_model()

        with pytest.raises(PredictionAudioError):
            prepare_voice_fusion_payload(
                audio_file_path=missing_wav,
                model=model,
            )


# ===========================================================================
# 5. Mock / Stub Multimodal Late-Fusion Consumer Integration
# ===========================================================================

class TestLateFusionConsumerMockIntegration:
    """
    Demonstrates and tests integration with a mock late-fusion layer consumer.
    Ensures VoiceFusionPayload provides everything late decision fusion needs
    alongside Questionnaire and Behavioral modalities.
    """

    class MockLateFusionConsumer:
        """Stub consumer representing Netal's planned Late Decision Fusion Layer."""

        def __init__(self, weights: Optional[Dict[str, float]] = None):
            # Default equal weights across the 3 modalities
            self.weights = weights or {"questionnaire": 0.4, "voice": 0.3, "behavioral": 0.3}

        def fuse(self, modality_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
            """
            Perform late decision-level fusion by computing weighted class probabilities.
            """
            fused_probabilities: Dict[str, float] = {}
            total_weight = 0.0

            for payload_dict in modality_payloads:
                # Validate using our helper
                if payload_dict.get("modality") == "voice":
                    validated = validate_voice_fusion_payload(payload_dict)
                    mod = validated.modality
                    probs = validated.class_probabilities
                else:
                    mod = payload_dict["modality"]
                    probs = payload_dict["class_probabilities"]

                w = self.weights.get(mod, 0.0)
                total_weight += w

                for cls_name, p in probs.items():
                    fused_probabilities[cls_name] = fused_probabilities.get(cls_name, 0.0) + (w * p)

            # Normalize
            if total_weight > 0:
                for cls_name in fused_probabilities:
                    fused_probabilities[cls_name] /= total_weight

            fused_predicted = max(fused_probabilities, key=fused_probabilities.get) if fused_probabilities else None

            return {
                "fused_predicted_class": fused_predicted,
                "fused_class_probabilities": fused_probabilities,
                "contributing_modalities": [p["modality"] for p in modality_payloads],
            }

    def test_voice_payload_integrates_seamlessly_with_fusion_consumer(self, tmp_path: Path):
        """Verify VoiceFusionPayload feeds into multimodal late-fusion consumer."""
        wav_file = tmp_path / "voice_sample.wav"
        generate_test_wav(wav_file, duration_seconds=5.0)

        classes = ["low_strain", "high_strain"]
        voice_model = create_fitted_voice_model(VoiceRandomForest, classes=classes)

        # 1. Obtain voice payload from pipeline
        voice_payload = prepare_voice_fusion_payload(wav_file, model=voice_model)

        # 2. Mock outputs from the other two modalities (Questionnaire & Behavioral)
        questionnaire_output = {
            "modality": "questionnaire",
            "predicted_class": "high_strain",
            "class_probabilities": {"low_strain": 0.2, "high_strain": 0.8},
            "confidence": 0.8,
        }
        behavioral_output = {
            "modality": "behavioral",
            "predicted_class": "high_strain",
            "class_probabilities": {"low_strain": 0.3, "high_strain": 0.7},
            "confidence": 0.7,
        }

        # 3. Feed all 3 modalities into the late fusion consumer
        fusion_engine = self.MockLateFusionConsumer()
        fused_result = fusion_engine.fuse([
            questionnaire_output,
            voice_payload.to_dict(),
            behavioral_output,
        ])

        # 4. Verify fused output incorporates voice correctly
        assert "voice" in fused_result["contributing_modalities"]
        assert len(fused_result["contributing_modalities"]) == 3
        assert set(fused_result["fused_class_probabilities"].keys()) == set(classes)
        assert math.isclose(sum(fused_result["fused_class_probabilities"].values()), 1.0, abs_tol=1e-4)


# ===========================================================================
# 6. Existing Voice Prediction Unchanged
# ===========================================================================

class TestExistingVoicePredictionUnchanged:
    """Verify that existing predict_voice functionality is preserved."""

    def test_predict_voice_returns_voice_prediction_result(self, tmp_path: Path):
        """Verify predict_voice() still directly returns VoicePredictionResult."""
        wav_file = tmp_path / "test.wav"
        generate_test_wav(wav_file, duration_seconds=5.0)
        model = create_fitted_voice_model()

        result = predict_voice(wav_file, model=model)
        assert isinstance(result, VoicePredictionResult)
        assert not hasattr(result, "fused_score")  # No invented fusion scores in voice prediction
        dict_output = result.to_dict()
        assert "predicted_class" in dict_output
        assert "class_probabilities" in dict_output
