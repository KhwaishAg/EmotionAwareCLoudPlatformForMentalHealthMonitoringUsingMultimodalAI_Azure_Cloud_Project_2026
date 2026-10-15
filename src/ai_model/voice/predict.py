"""
Voice Modality — Prediction Interface

Provides the predict_voice() inference interface for the trained voice model.
Returns predicted class labels and class probabilities for consumption
by the multimodal fusion component.

Pipeline:
    1. Validate the audio file using the existing Commit 4–7 pipeline.
    2. Preprocess the audio (mono, 16 kHz, silence trimming, normalization).
    3. Extract combined acoustic features (41D default: MFCC + pitch + energy + spectral).
    4. Load the serialized voice model safely using Commit 18 utilities.
    5. Run prediction and return a structured VoicePredictionResult.

Guarantees:
    - Uses only existing validated preprocessing, feature extraction, and model code.
    - Target labels are arbitrary/configurable — no hardcoded LOW/MODERATE/HIGH.
    - No invented risk score mapping; returns raw model output only.
    - Non-clinical: no diagnosis or medical claims.
    - Clear exceptions for missing model, invalid audio, incompatible features, prediction errors.

Note:
    This module produces voice-level assessments (predicted classes, class probabilities)
    that are consumed by the future multimodal fusion component (Commit 24).
    It is NOT a clinical diagnostic tool.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from .config import AUDIO_CONFIG, FEATURE_CONFIG, PATH_CONFIG, AudioConfig, FeatureConfig
from .recording import (
    AudioError,
    AudioNotFoundError,
    AudioFormatError,
    validate_audio_file,
)
from .preprocessing import PreprocessedAudio, preprocess_audio
from .features import (
    COMBINED_FEATURE_NAMES,
    FeatureExtractionError,
    extract_combined_features,
)
from .model import (
    BaseVoiceClassifier,
    ModelCorruptError,
    ModelError,
    ModelInputError,
    ModelNotFittedError,
    ModelNotFoundError,
    load_voice_model,
)


# ===========================================================================
# Prediction Exceptions
# ===========================================================================

class PredictionError(AudioError):
    """Base exception for all voice prediction pipeline errors."""
    pass


class PredictionAudioError(PredictionError):
    """Raised when audio validation or preprocessing fails during prediction."""
    pass


class PredictionFeatureError(PredictionError):
    """Raised when feature extraction fails or produces incompatible features."""
    pass


class PredictionModelError(PredictionError):
    """Raised when model loading or inference fails during prediction."""
    pass


# ===========================================================================
# Prediction Result
# ===========================================================================

@dataclass
class VoicePredictionResult:
    """
    Structured result from the voice prediction pipeline.

    Attributes:
        predicted_class: The predicted class label from the voice model.
        class_probabilities: Dictionary mapping each class label to its
            predicted probability (values sum to 1.0).
        model_type: Name of the model class used for prediction
            (e.g. 'VoiceRandomForest', 'VoiceLogisticRegression').
        feature_count: Number of acoustic features extracted.
        audio_duration_seconds: Duration of the preprocessed audio in seconds.
        model_path: Path to the serialized model file that was loaded.
        metadata: Optional additional metadata dictionary.
    """

    predicted_class: Any
    class_probabilities: Dict[Any, float]
    model_type: str
    feature_count: int
    audio_duration_seconds: float
    model_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a plain dictionary suitable for JSON serialization
        and downstream multimodal fusion consumption.
        """
        return {
            "predicted_class": self.predicted_class,
            "class_probabilities": {
                str(k): float(v) for k, v in self.class_probabilities.items()
            },
            "model_type": self.model_type,
            "feature_count": self.feature_count,
            "audio_duration_seconds": self.audio_duration_seconds,
            "model_path": self.model_path,
            "metadata": self.metadata,
        }

    def to_fusion_payload(self) -> Any:
        """
        Convert this prediction result into a standardized VoiceFusionPayload
        for consumption by the multimodal fusion layer.
        """
        from .fusion_interface import VoiceFusionPayload
        return VoiceFusionPayload.from_prediction_result(self)


# ===========================================================================
# Core Prediction Interface
# ===========================================================================

def predict_voice(
    audio_file_path: Union[str, Path],
    model_path: Union[str, Path, None] = None,
    model: Optional[BaseVoiceClassifier] = None,
    expected_model_type: Optional[Type[BaseVoiceClassifier]] = None,
    audio_config: AudioConfig = AUDIO_CONFIG,
    feature_config: FeatureConfig = FEATURE_CONFIG,
    validate_audio: bool = True,
    strict_validation: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> VoicePredictionResult:
    """
    End-to-end voice prediction pipeline: audio file → predicted class + probabilities.

    Steps:
        1. Validate the audio file (format, existence, integrity).
        2. Preprocess the audio (mono, 16 kHz, silence trimming, normalization).
        3. Extract combined acoustic features (41D default).
        4. Load the serialized voice model (or use pre-loaded model).
        5. Run prediction and return a structured VoicePredictionResult.

    Parameters:
        audio_file_path: Path to the audio WAV file to predict on.
        model_path: Path to the serialized voice model file (.joblib).
            If None, uses the default path from PathConfig.
        model: Optional pre-loaded BaseVoiceClassifier instance.
            If provided, model_path is ignored for loading but still
            recorded in the result for traceability.
        expected_model_type: Optional expected model class type to verify.
        audio_config: Centralized AudioConfig settings.
        feature_config: Centralized FeatureConfig settings.
        validate_audio: Whether to run full audio file validation before
            preprocessing (default: True).
        strict_validation: If True, audio validation raises exceptions
            on any issues rather than returning a ValidationResult.
        metadata: Optional additional metadata to include in the result.

    Returns:
        VoicePredictionResult with predicted class, probabilities, and metadata.

    Raises:
        PredictionAudioError: If the audio file is missing, corrupt, or invalid.
        PredictionFeatureError: If feature extraction fails or produces
            incompatible features for the loaded model.
        PredictionModelError: If model loading, fitting state, or inference fails.
    """
    audio_path = Path(audio_file_path)

    # -----------------------------------------------------------------------
    # Step 1: Validate audio file
    # -----------------------------------------------------------------------
    if validate_audio:
        try:
            validation = validate_audio_file(
                file_path=audio_path,
                config=audio_config,
                strict=strict_validation,
            )
            if not validation.is_valid:
                errors_str = "; ".join(validation.errors) if validation.errors else "Unknown validation failure"
                raise PredictionAudioError(
                    f"Audio validation failed for '{audio_path}': {errors_str}"
                )
        except PredictionAudioError:
            raise
        except AudioError as e:
            raise PredictionAudioError(
                f"Audio validation error for '{audio_path}': {e}"
            ) from e

    # -----------------------------------------------------------------------
    # Step 2: Preprocess the audio
    # -----------------------------------------------------------------------
    try:
        preprocessed: PreprocessedAudio = preprocess_audio(
            audio=audio_path,
            config=audio_config,
        )
    except AudioError as e:
        raise PredictionAudioError(
            f"Audio preprocessing failed for '{audio_path}': {e}"
        ) from e
    except Exception as e:
        raise PredictionAudioError(
            f"Unexpected error preprocessing '{audio_path}': {e}"
        ) from e

    # -----------------------------------------------------------------------
    # Step 3: Extract combined acoustic features
    # -----------------------------------------------------------------------
    try:
        features: np.ndarray = extract_combined_features(
            audio=preprocessed,
            config=feature_config,
        )
    except FeatureExtractionError as e:
        raise PredictionFeatureError(
            f"Feature extraction failed for '{audio_path}': {e}"
        ) from e
    except AudioError as e:
        raise PredictionFeatureError(
            f"Audio error during feature extraction for '{audio_path}': {e}"
        ) from e
    except Exception as e:
        raise PredictionFeatureError(
            f"Unexpected error during feature extraction for '{audio_path}': {e}"
        ) from e

    # Validate feature shape
    if features.ndim != 1 or features.size == 0:
        raise PredictionFeatureError(
            f"Feature extraction produced invalid shape {features.shape}. "
            "Expected a non-empty 1D feature vector."
        )

    # -----------------------------------------------------------------------
    # Step 4: Load the voice model
    # -----------------------------------------------------------------------
    resolved_model_path = str(model_path or Path(PATH_CONFIG.models_dir) / "voice_model.joblib")

    if model is not None:
        # Use pre-loaded model; skip file loading
        loaded_model = model
        if not isinstance(loaded_model, BaseVoiceClassifier):
            raise PredictionModelError(
                f"Provided model is not a BaseVoiceClassifier instance "
                f"(got {type(loaded_model).__name__})."
            )
    else:
        try:
            loaded_model = load_voice_model(
                filepath=resolved_model_path,
                expected_type=expected_model_type,
            )
        except ModelNotFoundError as e:
            raise PredictionModelError(
                f"Voice model not found at '{resolved_model_path}': {e}"
            ) from e
        except ModelCorruptError as e:
            raise PredictionModelError(
                f"Voice model is corrupt or invalid at '{resolved_model_path}': {e}"
            ) from e
        except ModelError as e:
            raise PredictionModelError(
                f"Failed to load voice model from '{resolved_model_path}': {e}"
            ) from e

    # Verify the model is fitted
    if not loaded_model.is_fitted:
        raise PredictionModelError(
            f"Loaded model '{loaded_model.__class__.__name__}' is not in a fitted state. "
            "Cannot run prediction on an unfitted model."
        )

    # -----------------------------------------------------------------------
    # Step 5: Run prediction
    # -----------------------------------------------------------------------
    try:
        # Reshape features to 2D (1 sample, N features)
        features_2d = features.reshape(1, -1)

        predicted_classes = loaded_model.predict(features_2d)
        predicted_class = predicted_classes[0]

        proba_dicts = loaded_model.predict_proba_dict(features_2d)
        class_probabilities = proba_dicts[0]

    except ModelInputError as e:
        raise PredictionFeatureError(
            f"Feature incompatibility with loaded model: {e}"
        ) from e
    except ModelNotFittedError as e:
        raise PredictionModelError(
            f"Model is not fitted: {e}"
        ) from e
    except ModelError as e:
        raise PredictionModelError(
            f"Model prediction failed: {e}"
        ) from e
    except Exception as e:
        raise PredictionModelError(
            f"Unexpected prediction error: {e}"
        ) from e

    # -----------------------------------------------------------------------
    # Build and return result
    # -----------------------------------------------------------------------
    result_metadata = {
        "feature_names": list(COMBINED_FEATURE_NAMES),
        "model_classes": [
            str(c) for c in (
                loaded_model.classes_ if loaded_model.classes_ is not None else []
            )
        ],
    }
    if metadata:
        result_metadata.update(metadata)

    return VoicePredictionResult(
        predicted_class=predicted_class,
        class_probabilities=class_probabilities,
        model_type=loaded_model.__class__.__name__,
        feature_count=int(features.size),
        audio_duration_seconds=float(preprocessed.duration_seconds),
        model_path=resolved_model_path,
        metadata=result_metadata,
    )


# ===========================================================================
# Feature-dict entry point (standardized fusion contract)
# ===========================================================================
# Added when merging feature/Khwaish: the backend/fusion layer works with
# already-extracted features and the standardized modality contract
# (risk_score / risk_level / confidence / key_factors / data_quality), while
# predict_voice() above takes an audio file and returns raw class
# probabilities. Both entry points share the same model artifacts.

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models" / "voice"
RISK_WEIGHTS = {"low": 0.0, "moderate": 0.5, "high": 1.0}


class VoiceModelNotTrainedError(ModelNotFoundError):
    """Raised when no trained voice model is available yet.

    Subclasses ModelNotFoundError so callers already handling voice model
    errors catch this too. The backend/API layer should catch it specifically
    and mark the voice modality unavailable rather than treating it as an
    unexpected failure.
    """


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def load_selected_voice_model():
    """Load the trained voice model named in models/voice/selected_model.json.

    Raises VoiceModelNotTrainedError until a real model is trained on a
    labeled voice dataset and saved to models/voice/.
    """
    selector = MODELS_DIR / "selected_model.json"
    if not selector.exists():
        raise VoiceModelNotTrainedError(
            "No trained voice model is available yet. Voice model training is "
            "pending a suitable real labeled voice dataset; this modality is "
            "currently optional/unavailable."
        )
    selected = _load_json(selector)
    model_name = selected.get("selected_model")
    artifact = MODELS_DIR / f"{model_name}.joblib"
    if not artifact.exists():
        raise VoiceModelNotTrainedError(f"Selected voice model artifact was not found: {artifact}")
    return load_voice_model(artifact), model_name


def predict_voice_from_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Predict voice-derived stress risk from extracted features.

    Parameters
    ----------
    features:
        Dict of the combined acoustic features (COMBINED_FEATURE_NAMES), as
        produced by src.ai_model.voice.features.extract_combined_dict().

    Returns
    -------
    The standardized fusion contract dict (modality, risk_score, risk_level,
    confidence, available_features, total_features, key_factors, data_quality).

    Raises
    ------
    VoiceModelNotTrainedError
        Until a trained model artifact exists at models/voice/. Callers must
        treat voice as an unavailable modality, never a crash.
    """
    model, _model_name = load_selected_voice_model()

    ordered = {name: features.get(name, np.nan) for name in COMBINED_FEATURE_NAMES}
    probability_map = model.predict_proba_dict([ordered])[0]
    if set(probability_map) != set(RISK_WEIGHTS):
        raise ValueError("Voice model classes must be exactly low, moderate, and high.")
    risk_level = max(probability_map, key=probability_map.get)
    risk_score = float(sum(RISK_WEIGHTS[level] * probability_map[level] for level in RISK_WEIGHTS))
    ordered_probs = sorted(probability_map.values(), reverse=True)
    confidence = float(np.clip(0.5 * ordered_probs[0] + 0.5 * (ordered_probs[0] - ordered_probs[1]), 0.0, 1.0))

    total = len(COMBINED_FEATURE_NAMES)
    available = sum(
        1 for value in ordered.values()
        if value is not None and not (isinstance(value, float) and np.isnan(value))
    )

    explainability_path = MODELS_DIR / "voice_explainability_metadata.json"
    key_factors = []
    if explainability_path.exists():
        explainability = _load_json(explainability_path)
        top = explainability.get("top_features", [])[:5]
        maximum = max((float(item.get("importance", 0.0)) for item in top), default=0.0)
        key_factors = [
            {
                "feature": str(item["feature"]),
                "contribution": float(item.get("importance", 0.0) / maximum) if maximum else 0.0,
                "direction": "model_associated",
            }
            for item in top
        ]

    return {
        "modality": "voice",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence": confidence,
        "available_features": available,
        "total_features": total,
        "key_factors": key_factors,
        "data_quality": {
            "missing_features": total - available,
            "completeness": float(available / total) if total else 0.0,
        },
    }
