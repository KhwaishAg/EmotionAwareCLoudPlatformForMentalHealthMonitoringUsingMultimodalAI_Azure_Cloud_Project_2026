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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd

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


class VoiceModelLoadError(PredictionModelError):
    """Raised when a voice model artifact exists but cannot be used.

    Covers corrupt/unreadable files, a feature set that does not match the
    current extraction pipeline, and demo models that are not allowed to act
    as real predictors.
    """


TRAINING_REPORT_FILENAME = "voice_training_report.json"
#: Setting this environment variable to ``1`` allows models trained on
#: ``synthetic_demo`` / ``public_acted`` data to be loaded (demonstrations/tests only).
ALLOW_DEMO_MODEL_ENV = "VOICE_ALLOW_DEMO_MODEL"
#: ``dataset_kind`` values (written by train_voice.py) that are never served as
#: stress-risk predictors unless explicitly overridden: synthetic data, and acted
#: emotional speech, which is not defensible ground truth for student stress.
DEMO_DATASET_KINDS = ("synthetic_demo", "public_acted")
#: The only model task this stress-risk loader serves.
STRESS_TASK = "stress_risk"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def read_voice_training_report() -> Optional[Dict[str, Any]]:
    """Return the training report written next to the model, or ``None``.

    Unreadable reports are treated as absent rather than raising.
    """
    path = MODELS_DIR / TRAINING_REPORT_FILENAME
    if not path.is_file():
        return None
    try:
        report = _load_json(path)
    except (OSError, ValueError):
        return None
    return report if isinstance(report, dict) else None


def _check_model_allowed() -> None:
    report = read_voice_training_report()
    if not report:
        return
    if report.get("task", STRESS_TASK) != STRESS_TASK:
        raise VoiceModelLoadError("This model was not trained for stress-risk prediction and cannot be used for it.")
    if report.get("dataset_kind") in DEMO_DATASET_KINDS and os.environ.get(ALLOW_DEMO_MODEL_ENV) != "1":
        raise VoiceModelLoadError(
            "The voice model was trained on synthetic or acted-emotion data, which is not defensible stress "
            "ground truth, and is not served as a real stress predictor."
        )


def load_selected_voice_model():
    """Load the trained voice model, returning ``(model, model_name)``.

    Resolution order:
        1. ``models/voice/selected_model.json`` naming ``<selected_model>.joblib``
           (the layout used by the questionnaire and behavioral modalities).
        2. ``models/voice/voice_model.joblib`` (the default artifact written by
           the voice training/serialization utilities).

    Raises:
        VoiceModelNotTrainedError: if neither artifact exists (no trained model).
        VoiceModelLoadError: if an artifact exists but is corrupt, was trained on
            a different feature set, or is a synthetic demo model.
    """
    selector = MODELS_DIR / "selected_model.json"
    default_artifact = MODELS_DIR / "voice_model.joblib"
    if selector.exists():
        try:
            model_name = _load_json(selector).get("selected_model")
        except (OSError, ValueError, AttributeError) as error:
            raise VoiceModelLoadError("selected_model.json could not be read.") from error
        artifact = MODELS_DIR / f"{model_name}.joblib"
        if not isinstance(model_name, str) or not model_name or Path(model_name).name != model_name:
            raise VoiceModelLoadError("selected_model.json does not name a valid model.")
        if not artifact.exists():
            raise VoiceModelNotTrainedError(f"Selected voice model artifact was not found: {artifact.name}")
    elif default_artifact.exists():
        model_name, artifact = default_artifact.stem, default_artifact
    else:
        raise VoiceModelNotTrainedError(
            "No trained voice model is available yet. Voice model training is "
            "pending a suitable real labeled voice dataset; this modality is "
            "currently optional/unavailable."
        )

    _check_model_allowed()
    try:
        model = load_voice_model(artifact)
    except ModelError as error:
        raise VoiceModelLoadError(f"Voice model artifact '{artifact.name}' could not be loaded.") from error
    except Exception as error:  # joblib/pickle/sklearn version problems
        raise VoiceModelLoadError(f"Voice model artifact '{artifact.name}' is unreadable.") from error
    if list(model.feature_names_ or []) != list(COMBINED_FEATURE_NAMES):
        raise VoiceModelLoadError(
            "Voice model was trained on a different feature set than the current extraction pipeline."
        )
    return model, str(model_name)


def _validate_feature_dict(features: Any) -> Dict[str, float]:
    """Validate a raw acoustic-feature dict and return only the known, non-null values.

    ``None`` values are dropped (and later reported as missing); unknown keys
    and non-finite / non-numeric values are rejected.
    """
    if not isinstance(features, dict):
        raise TypeError("Voice features must be a dictionary.")
    unknown = [key for key in features if key not in COMBINED_FEATURE_NAMES]
    if unknown:
        raise ValueError("Unsupported voice features: " + ", ".join(sorted(map(str, unknown))))
    cleaned: Dict[str, float] = {}
    for name, value in features.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"Voice feature '{name}' must be numeric.")
        if not np.isfinite(float(value)):
            raise ValueError(f"Voice feature '{name}' must be finite.")
        cleaned[name] = float(value)
    return cleaned


def _build_fusion_contract(model: BaseVoiceClassifier, features: Dict[str, float]) -> Dict[str, Any]:
    """Run ``model`` on validated features and build the standardized contract."""
    total = len(COMBINED_FEATURE_NAMES)
    missing = [name for name in COMBINED_FEATURE_NAMES if name not in features]
    if missing:
        # The fitted voice classifiers have no imputer; guessing values would
        # silently distort the result. A recording always yields every feature.
        raise ValueError(f"Voice features are incomplete: {len(missing)} of {total} missing.")
    row = pd.DataFrame([{name: features[name] for name in COMBINED_FEATURE_NAMES}])[list(COMBINED_FEATURE_NAMES)]
    available = total

    try:
        probability_map = model.predict_proba_dict(row)[0]
    except Exception as error:  # model errors must never leak as raw exceptions
        raise PredictionModelError("Voice model inference failed.") from error
    if set(probability_map) != set(RISK_WEIGHTS):
        raise PredictionModelError("Voice model classes must be exactly low, moderate, and high.")
    risk_level = max(probability_map, key=probability_map.get)
    risk_score = float(sum(RISK_WEIGHTS[level] * probability_map[level] for level in RISK_WEIGHTS))
    ordered_probs = sorted(probability_map.values(), reverse=True)
    confidence = float(np.clip(0.5 * ordered_probs[0] + 0.5 * (ordered_probs[0] - ordered_probs[1]), 0.0, 1.0))

    key_factors: List[Dict[str, Any]] = []
    explainability_path = MODELS_DIR / "voice_explainability_metadata.json"
    if explainability_path.exists():
        top = _load_json(explainability_path).get("top_features", [])[:5]
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
        "available_features": int(available),
        "total_features": total,
        "key_factors": key_factors,
        "data_quality": {
            "missing_features": total - int(available),
            "completeness": float(available / total) if total else 0.0,
        },
    }


def predict_voice_from_features(
    features: Dict[str, Any],
    model: Optional[BaseVoiceClassifier] = None,
) -> Dict[str, Any]:
    """Predict voice-derived stress risk from extracted acoustic features.

    Parameters
    ----------
    features:
        Dict keyed by COMBINED_FEATURE_NAMES, as produced by
        :func:`extract_voice_features_from_audio`.
    model:
        Optional pre-loaded fitted classifier (mainly for tests); when omitted
        the trained artifact in ``models/voice/`` is loaded.

    Returns
    -------
    The standardized fusion contract dict (modality, risk_score, risk_level,
    confidence, available_features, total_features, key_factors, data_quality).

    Raises
    ------
    VoiceModelNotTrainedError
        If no trained model artifact exists. Callers must treat voice as an
        unavailable modality, never a crash.
    TypeError / ValueError
        For malformed feature input.
    """
    cleaned = _validate_feature_dict(features)
    if model is None:
        model, _ = load_selected_voice_model()
    return _build_fusion_contract(model, cleaned)


def extract_voice_features_from_audio(
    audio_file_path: Union[str, Path],
    audio_config: AudioConfig = AUDIO_CONFIG,
    feature_config: FeatureConfig = FEATURE_CONFIG,
) -> Dict[str, float]:
    """Validate, preprocess and extract the combined acoustic feature dict.

    Raises:
        PredictionAudioError: invalid, missing or corrupt audio.
        PredictionFeatureError: feature extraction failed.
    """
    from .features import extract_combined_dict

    path = Path(audio_file_path)
    try:
        validation = validate_audio_file(file_path=path, config=audio_config, strict=False)
        if not validation.is_valid:
            errors = "; ".join(validation.errors) if validation.errors else "Unknown validation failure"
            raise PredictionAudioError(f"Audio validation failed: {errors}")
        preprocessed = preprocess_audio(audio=path, config=audio_config)
    except PredictionAudioError:
        raise
    except AudioError as error:
        raise PredictionAudioError(f"Audio could not be processed: {error}") from error
    except Exception as error:
        raise PredictionAudioError(f"Unexpected audio processing failure: {error}") from error
    try:
        combined = extract_combined_dict(audio=preprocessed, config=feature_config)
        return {name: float(value) for name, value in combined.items()}
    except FeatureExtractionError as error:
        raise PredictionFeatureError(f"Feature extraction failed: {error}") from error
    except Exception as error:
        raise PredictionFeatureError(f"Unexpected feature extraction failure: {error}") from error


def predict_voice_contract_from_audio(
    audio_file_path: Union[str, Path],
    model: Optional[BaseVoiceClassifier] = None,
) -> Dict[str, Any]:
    """Audio file in, standardized fusion contract out (voice end to end)."""
    if model is None:
        model, _ = load_selected_voice_model()  # fail fast before heavy audio work
    features = extract_voice_features_from_audio(audio_file_path)
    return predict_voice_from_features(features, model=model)
