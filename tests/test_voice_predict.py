"""
Tests for Voice Prediction Interface (Commit 22).

Covers:
    - End-to-end predict_voice() pipeline using synthetic audio and mocked model
    - VoicePredictionResult structure, to_dict(), and arbitrary labels
    - Audio validation errors (missing file, invalid format)
    - Feature extraction integration with synthetic audio
    - Model loading errors (missing file, corrupt model)
    - Feature incompatibility detection
    - Pre-loaded model passthrough
    - Prediction exception hierarchy
    - Non-clinical: no hardcoded labels or invented scores

All tests use synthetic/generated audio and mocked model loading.
No private recordings, real Azure calls, or real trained models required.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from src.ai_model.voice.config import AUDIO_CONFIG, FEATURE_CONFIG, PATH_CONFIG
from src.ai_model.voice.features import COMBINED_FEATURE_NAMES
from src.ai_model.voice.model import (
    BaseVoiceClassifier,
    ModelCorruptError,
    ModelNotFoundError,
    VoiceLogisticRegression,
    VoiceRandomForest,
    load_voice_model,
    save_voice_model,
)
from src.ai_model.voice.predict import (
    PredictionAudioError,
    PredictionError,
    PredictionFeatureError,
    PredictionModelError,
    VoicePredictionResult,
    predict_voice,
)
from src.ai_model.voice.recording import AudioError


# ===========================================================================
# Helpers: Synthetic Audio Generation
# ===========================================================================

def generate_synthetic_wav(
    filepath: Path,
    duration_seconds: float = 5.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> Path:
    """Generate a synthetic sine-wave WAV file for testing."""
    t = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    sf.write(str(filepath), waveform, sample_rate, format="WAV", subtype="FLOAT")
    return filepath


def create_fitted_model(
    model_class=VoiceRandomForest,
    classes: Optional[list] = None,
    n_features: int = 41,
) -> BaseVoiceClassifier:
    """Create a fitted voice model with synthetic data for testing."""
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
# VoicePredictionResult Tests
# ===========================================================================

class TestVoicePredictionResult:
    """Tests for the VoicePredictionResult dataclass."""

    def test_basic_construction(self):
        """Verify VoicePredictionResult can be constructed with all fields."""
        result = VoicePredictionResult(
            predicted_class="stressed",
            class_probabilities={"calm": 0.2, "stressed": 0.7, "neutral": 0.1},
            model_type="VoiceRandomForest",
            feature_count=41,
            audio_duration_seconds=10.5,
            model_path="models/voice/model.joblib",
        )

        assert result.predicted_class == "stressed"
        assert result.class_probabilities["stressed"] == 0.7
        assert result.model_type == "VoiceRandomForest"
        assert result.feature_count == 41
        assert result.audio_duration_seconds == 10.5
        assert result.metadata == {}

    def test_arbitrary_labels(self):
        """Verify labels are not hardcoded — any string/int labels work."""
        result = VoicePredictionResult(
            predicted_class=2,
            class_probabilities={0: 0.1, 1: 0.3, 2: 0.6},
            model_type="VoiceLogisticRegression",
            feature_count=41,
            audio_duration_seconds=8.0,
            model_path="model.joblib",
        )

        assert result.predicted_class == 2
        assert result.class_probabilities[2] == 0.6

    def test_to_dict(self):
        """Verify to_dict() returns a JSON-serializable dictionary."""
        result = VoicePredictionResult(
            predicted_class="calm",
            class_probabilities={"calm": 0.85, "stressed": 0.15},
            model_type="VoiceLogisticRegression",
            feature_count=41,
            audio_duration_seconds=12.0,
            model_path="path/to/model.joblib",
            metadata={"source": "test"},
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["predicted_class"] == "calm"
        assert d["class_probabilities"]["calm"] == 0.85
        assert d["model_type"] == "VoiceLogisticRegression"
        assert d["feature_count"] == 41
        assert d["metadata"]["source"] == "test"

    def test_to_dict_string_keys(self):
        """Verify to_dict() converts class probability keys to strings."""
        result = VoicePredictionResult(
            predicted_class=1,
            class_probabilities={0: 0.4, 1: 0.6},
            model_type="VoiceRandomForest",
            feature_count=41,
            audio_duration_seconds=5.0,
            model_path="model.joblib",
        )

        d = result.to_dict()
        assert all(isinstance(k, str) for k in d["class_probabilities"])


# ===========================================================================
# Exception Hierarchy Tests
# ===========================================================================

class TestPredictionExceptions:
    """Tests for the prediction exception hierarchy."""

    def test_prediction_error_is_audio_error(self):
        """Verify PredictionError inherits from AudioError."""
        assert issubclass(PredictionError, AudioError)

    def test_prediction_audio_error_hierarchy(self):
        """Verify PredictionAudioError inherits from PredictionError."""
        assert issubclass(PredictionAudioError, PredictionError)

    def test_prediction_feature_error_hierarchy(self):
        """Verify PredictionFeatureError inherits from PredictionError."""
        assert issubclass(PredictionFeatureError, PredictionError)

    def test_prediction_model_error_hierarchy(self):
        """Verify PredictionModelError inherits from PredictionError."""
        assert issubclass(PredictionModelError, PredictionError)


# ===========================================================================
# Audio Validation Error Tests
# ===========================================================================

class TestPredictVoiceAudioErrors:
    """Tests for audio validation and preprocessing error handling."""

    def test_missing_audio_file(self, tmp_path):
        """Verify PredictionAudioError is raised for missing audio file."""
        missing_path = tmp_path / "nonexistent.wav"

        with pytest.raises(PredictionAudioError, match="Audio validation failed"):
            predict_voice(
                audio_file_path=missing_path,
                model_path=tmp_path / "model.joblib",
            )

    def test_invalid_audio_format(self, tmp_path):
        """Verify PredictionAudioError is raised for unsupported audio format."""
        txt_file = tmp_path / "not_audio.txt"
        txt_file.write_text("this is not audio")

        with pytest.raises(PredictionAudioError, match="Audio validation failed"):
            predict_voice(
                audio_file_path=txt_file,
                model_path=tmp_path / "model.joblib",
            )

    def test_invalid_extension(self, tmp_path):
        """Verify PredictionAudioError for unsupported file extension."""
        mp3_file = tmp_path / "audio.mp3"
        mp3_file.write_bytes(b"\x00" * 100)

        with pytest.raises(PredictionAudioError, match="Audio validation"):
            predict_voice(
                audio_file_path=mp3_file,
                model_path=tmp_path / "model.joblib",
            )


# ===========================================================================
# Model Loading Error Tests
# ===========================================================================

class TestPredictVoiceModelErrors:
    """Tests for model loading error handling."""

    def test_missing_model_file(self, tmp_path):
        """Verify PredictionModelError when model file does not exist."""
        wav_path = generate_synthetic_wav(tmp_path / "test.wav")
        missing_model = tmp_path / "missing_model.joblib"

        with pytest.raises(PredictionModelError, match="Voice model not found"):
            predict_voice(
                audio_file_path=wav_path,
                model_path=missing_model,
            )

    def test_corrupt_model_file(self, tmp_path):
        """Verify PredictionModelError when model file is corrupt."""
        wav_path = generate_synthetic_wav(tmp_path / "test.wav")
        corrupt_model = tmp_path / "corrupt.joblib"
        corrupt_model.write_bytes(b"not a valid joblib file")

        with pytest.raises(PredictionModelError, match="corrupt|invalid|Failed"):
            predict_voice(
                audio_file_path=wav_path,
                model_path=corrupt_model,
            )

    def test_empty_model_file(self, tmp_path):
        """Verify PredictionModelError when model file is empty."""
        wav_path = generate_synthetic_wav(tmp_path / "test.wav")
        empty_model = tmp_path / "empty.joblib"
        empty_model.write_bytes(b"")

        with pytest.raises(PredictionModelError, match="corrupt|invalid|empty"):
            predict_voice(
                audio_file_path=wav_path,
                model_path=empty_model,
            )

    def test_unfitted_model_raises_error(self, tmp_path):
        """Verify PredictionModelError when a pre-loaded model is not fitted."""
        wav_path = generate_synthetic_wav(tmp_path / "test.wav")
        unfitted_model = VoiceRandomForest()

        with pytest.raises(PredictionModelError, match="not in a fitted state"):
            predict_voice(
                audio_file_path=wav_path,
                model=unfitted_model,
            )

    def test_non_classifier_model_raises_error(self, tmp_path):
        """Verify PredictionModelError when model is not a BaseVoiceClassifier."""
        wav_path = generate_synthetic_wav(tmp_path / "test.wav")
        fake_model = "not_a_model"

        with pytest.raises(PredictionModelError, match="not a BaseVoiceClassifier"):
            predict_voice(
                audio_file_path=wav_path,
                model=fake_model,
            )


# ===========================================================================
# End-to-End Prediction Tests (with pre-loaded model)
# ===========================================================================

class TestPredictVoiceEndToEnd:
    """End-to-end prediction tests using synthetic audio and fitted models."""

    def test_predict_voice_with_preloaded_random_forest(self, tmp_path):
        """Verify full prediction pipeline with a pre-loaded RandomForest model."""
        wav_path = generate_synthetic_wav(tmp_path / "test.wav", duration_seconds=5.0)
        model = create_fitted_model(VoiceRandomForest, classes=["calm", "stressed", "neutral"])

        result = predict_voice(
            audio_file_path=wav_path,
            model=model,
        )

        assert isinstance(result, VoicePredictionResult)
        assert result.predicted_class in ["calm", "stressed", "neutral"]
        assert len(result.class_probabilities) == 3
        assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-6
        assert result.model_type == "VoiceRandomForest"
        assert result.feature_count == 41
        assert result.audio_duration_seconds > 0

    def test_predict_voice_with_preloaded_logistic_regression(self, tmp_path):
        """Verify full prediction pipeline with a pre-loaded LogisticRegression model."""
        wav_path = generate_synthetic_wav(tmp_path / "test.wav", duration_seconds=6.0)
        model = create_fitted_model(VoiceLogisticRegression, classes=["anxious", "relaxed"])

        result = predict_voice(
            audio_file_path=wav_path,
            model=model,
        )

        assert isinstance(result, VoicePredictionResult)
        assert result.predicted_class in ["anxious", "relaxed"]
        assert len(result.class_probabilities) == 2
        assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-6
        assert result.model_type == "VoiceLogisticRegression"

    def test_predict_voice_arbitrary_integer_labels(self, tmp_path):
        """Verify prediction works with arbitrary integer labels (not hardcoded strings)."""
        wav_path = generate_synthetic_wav(tmp_path / "test.wav", duration_seconds=5.0)
        model = create_fitted_model(VoiceRandomForest, classes=[0, 1, 2, 3])

        result = predict_voice(
            audio_file_path=wav_path,
            model=model,
        )

        assert result.predicted_class in [0, 1, 2, 3]
        assert len(result.class_probabilities) == 4

    def test_predict_voice_result_metadata(self, tmp_path):
        """Verify metadata is passed through to the result."""
        wav_path = generate_synthetic_wav(tmp_path / "test.wav")
        model = create_fitted_model(VoiceRandomForest)

        result = predict_voice(
            audio_file_path=wav_path,
            model=model,
            metadata={"session_id": "S001", "modality": "voice"},
        )

        assert result.metadata["session_id"] == "S001"
        assert result.metadata["modality"] == "voice"
        # Built-in metadata is also present
        assert "feature_names" in result.metadata
        assert "model_classes" in result.metadata

    def test_predict_voice_to_dict_serializable(self, tmp_path):
        """Verify the result's to_dict() produces a JSON-serializable structure."""
        import json

        wav_path = generate_synthetic_wav(tmp_path / "test.wav")
        model = create_fitted_model(VoiceRandomForest)

        result = predict_voice(
            audio_file_path=wav_path,
            model=model,
        )

        d = result.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["predicted_class"] in ["calm", "stressed", "neutral"]


# ===========================================================================
# Prediction with Serialized Model (save/load roundtrip)
# ===========================================================================

class TestPredictVoiceWithSerializedModel:
    """Tests that predict_voice works with model files loaded from disk."""

    def test_predict_voice_from_saved_model(self, tmp_path):
        """Verify prediction with a model that was saved and loaded from disk."""
        wav_path = generate_synthetic_wav(tmp_path / "test.wav", duration_seconds=5.0)

        # Create and save a fitted model
        model = create_fitted_model(VoiceRandomForest, classes=["happy", "sad"])
        model_file = tmp_path / "test_model.joblib"
        save_voice_model(model, model_file)

        # Predict using the saved model file
        result = predict_voice(
            audio_file_path=wav_path,
            model_path=model_file,
        )

        assert isinstance(result, VoicePredictionResult)
        assert result.predicted_class in ["happy", "sad"]
        assert len(result.class_probabilities) == 2
        assert result.model_path == str(model_file)

    def test_predict_voice_logistic_regression_from_saved(self, tmp_path):
        """Verify prediction with a saved LogisticRegression model."""
        wav_path = generate_synthetic_wav(tmp_path / "test.wav", duration_seconds=5.0)

        model = create_fitted_model(VoiceLogisticRegression, classes=["low", "high"])
        model_file = tmp_path / "lr_model.joblib"
        save_voice_model(model, model_file)

        result = predict_voice(
            audio_file_path=wav_path,
            model_path=model_file,
        )

        assert result.model_type == "VoiceLogisticRegression"
        assert result.predicted_class in ["low", "high"]


# ===========================================================================
# Feature Incompatibility Tests
# ===========================================================================

class TestPredictVoiceFeatureIncompatibility:
    """Tests for feature count mismatch between extracted features and model."""

    def test_feature_count_mismatch_raises_error(self, tmp_path):
        """Verify PredictionFeatureError when model expects different feature count."""
        wav_path = generate_synthetic_wav(tmp_path / "test.wav", duration_seconds=5.0)

        # Create a model trained on 10 features (not 41)
        model = create_fitted_model(VoiceRandomForest, n_features=10)

        with pytest.raises(PredictionFeatureError, match="Number of features|incompatib"):
            predict_voice(
                audio_file_path=wav_path,
                model=model,
            )


# ===========================================================================
# Validation Skip Tests
# ===========================================================================

class TestPredictVoiceValidationSkip:
    """Tests for the validate_audio flag."""

    def test_skip_validation(self, tmp_path):
        """Verify prediction works when validate_audio=False."""
        wav_path = generate_synthetic_wav(tmp_path / "test.wav", duration_seconds=5.0)
        model = create_fitted_model(VoiceRandomForest)

        result = predict_voice(
            audio_file_path=wav_path,
            model=model,
            validate_audio=False,
        )

        assert isinstance(result, VoicePredictionResult)
        assert result.predicted_class in ["calm", "stressed", "neutral"]


# ===========================================================================
# Package Export Tests
# ===========================================================================

class TestPredictPackageExports:
    """Tests for predict module exports from the voice package."""

    def test_predict_symbols_exported(self):
        """Verify all prediction symbols are accessible from the voice package."""
        import src.ai_model.voice as voice

        assert hasattr(voice, "predict_voice")
        assert hasattr(voice, "VoicePredictionResult")
        assert hasattr(voice, "PredictionError")
        assert hasattr(voice, "PredictionAudioError")
        assert hasattr(voice, "PredictionFeatureError")
        assert hasattr(voice, "PredictionModelError")
