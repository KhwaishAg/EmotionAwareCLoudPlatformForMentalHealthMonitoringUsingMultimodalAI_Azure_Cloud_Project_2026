"""
Voice Modality — Multimodal Fusion Integration Interface

Provides the integration boundary connecting the Voice prediction pipeline
to the downstream multimodal fusion layer.

The project encompasses three modalities:
    1. Questionnaire (subjective self-report)
    2. Voice (objective acoustic features)
    3. Behavioral (academic and lifestyle indicators)

Responsibilities of this module:
    - Package VoicePredictionResult into a standardized, validated VoiceFusionPayload.
    - Expose modality identity ("voice") and class posterior probabilities.
    - Ensure compatibility with arbitrary, user-defined target labels (binary or multiclass).
    - Provide explicit validation of payload completeness, probability axioms, and label integrity.
    - Preserve clean separation between voice prediction, late fusion, and overall assessment.
    - Strictly non-clinical: no diagnostic claims, no invented risk-score formulas.

Guarantees:
    - Directly reuses predict_voice() from src.ai_model.voice.predict without duplicating inference.
    - Fully serializable to plain Python dicts / JSON for API and inter-service transport.
    - Rejects malformed, incomplete, or corrupted voice predictions with typed exceptions.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from .config import AUDIO_CONFIG, FEATURE_CONFIG, AudioConfig, FeatureConfig
from .model import BaseVoiceClassifier
from .predict import (
    PredictionError,
    VoicePredictionResult,
    predict_voice,
)


# ===========================================================================
# Modality Identifier
# ===========================================================================

MODALITY_NAME: str = "voice"


# ===========================================================================
# Exceptions
# ===========================================================================

class VoiceFusionIntegrationError(PredictionError):
    """Base exception for errors occurring during voice-to-fusion integration."""
    pass


class InvalidFusionPayloadError(VoiceFusionIntegrationError):
    """Raised when a voice payload does not conform to the required fusion contract."""
    pass


# ===========================================================================
# Standard Voice Fusion Payload
# ===========================================================================

@dataclass
class VoiceFusionPayload:
    """
    Standardized payload format produced by the Voice modality for consumption
    by the multimodal fusion layer.

    Attributes:
        modality: Identifier string for the modality (always "voice").
        predicted_class: Predicted class label from the voice model (arbitrary type/string).
        class_probabilities: Mapping of each class label to its predicted probability in [0.0, 1.0].
        confidence: Posterior probability of the predicted class P(predicted_class|x).
        model_type: Name of the underlying classifier class (e.g. 'VoiceRandomForest').
        feature_count: Total acoustic features extracted (e.g. 41).
        audio_duration_seconds: Preprocessed audio duration in seconds.
        model_path: Path of the serialized model used for inference.
        metadata: Additional metadata dictionary (feature names, model classes, etc.).
    """

    modality: str
    predicted_class: Any
    class_probabilities: Dict[str, float]
    confidence: float
    model_type: str
    feature_count: int
    audio_duration_seconds: float
    model_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate that the payload conforms strictly to the fusion contract.

        Raises:
            InvalidFusionPayloadError: If any field is invalid, missing, or violates probability axioms.
        """
        if self.modality != MODALITY_NAME:
            raise InvalidFusionPayloadError(
                f"Invalid modality identifier '{self.modality}'. Expected '{MODALITY_NAME}'."
            )

        if self.predicted_class is None:
            raise InvalidFusionPayloadError("Predicted class cannot be None.")

        if not isinstance(self.class_probabilities, dict) or len(self.class_probabilities) == 0:
            raise InvalidFusionPayloadError(
                "class_probabilities must be a non-empty dictionary mapping class labels to probabilities."
            )

        pred_str = str(self.predicted_class)
        # Check if predicted_class exists in class_probabilities (either directly or as string)
        if self.predicted_class not in self.class_probabilities and pred_str not in self.class_probabilities:
            raise InvalidFusionPayloadError(
                f"Predicted class '{self.predicted_class}' is not present in class_probabilities keys: "
                f"{list(self.class_probabilities.keys())}."
            )

        total_prob = 0.0
        for cls_name, prob in self.class_probabilities.items():
            if not isinstance(prob, (int, float, np.floating)):
                raise InvalidFusionPayloadError(
                    f"Probability for class '{cls_name}' must be numeric, got {type(prob).__name__}."
                )
            if not math.isfinite(prob):
                raise InvalidFusionPayloadError(
                    f"Probability for class '{cls_name}' is not finite ({prob})."
                )
            if not (0.0 <= prob <= 1.0):
                raise InvalidFusionPayloadError(
                    f"Probability for class '{cls_name}' must be in [0.0, 1.0], got {prob}."
                )
            total_prob += float(prob)

        if not math.isclose(total_prob, 1.0, abs_tol=0.05):
            raise InvalidFusionPayloadError(
                f"Class probabilities must sum to 1.0 (got {total_prob:.4f})."
            )

        if not (0.0 <= self.confidence <= 1.0) or not math.isfinite(self.confidence):
            raise InvalidFusionPayloadError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}."
            )

        if self.feature_count <= 0:
            raise InvalidFusionPayloadError(
                f"feature_count must be a positive integer, got {self.feature_count}."
            )

        if self.audio_duration_seconds <= 0.0:
            raise InvalidFusionPayloadError(
                f"audio_duration_seconds must be positive, got {self.audio_duration_seconds}."
            )

    @classmethod
    def from_prediction_result(cls, result: VoicePredictionResult) -> "VoiceFusionPayload":
        """
        Construct and validate a VoiceFusionPayload from a VoicePredictionResult.
        """
        if not isinstance(result, VoicePredictionResult):
            raise InvalidFusionPayloadError(
                f"Expected VoicePredictionResult instance, got {type(result).__name__}."
            )

        # Standardize probabilities with string keys
        prob_dict = {str(k): float(v) for k, v in result.class_probabilities.items()}

        # Confidence is model's probability for the predicted class
        pred_key = result.predicted_class
        pred_key_str = str(pred_key)
        if pred_key in result.class_probabilities:
            conf = float(result.class_probabilities[pred_key])
        elif pred_key_str in prob_dict:
            conf = float(prob_dict[pred_key_str])
        else:
            conf = max(prob_dict.values()) if prob_dict else 0.0

        payload = cls(
            modality=MODALITY_NAME,
            predicted_class=result.predicted_class,
            class_probabilities=prob_dict,
            confidence=conf,
            model_type=result.model_type,
            feature_count=result.feature_count,
            audio_duration_seconds=result.audio_duration_seconds,
            model_path=result.model_path,
            metadata=dict(result.metadata),
        )
        payload.validate()
        return payload

    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to plain dictionary for JSON/serialization transport."""
        return {
            "modality": self.modality,
            "predicted_class": self.predicted_class,
            "class_probabilities": dict(self.class_probabilities),
            "confidence": self.confidence,
            "model_type": self.model_type,
            "feature_count": self.feature_count,
            "audio_duration_seconds": self.audio_duration_seconds,
            "model_path": self.model_path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceFusionPayload":
        """
        Reconstruct and validate a VoiceFusionPayload from a dictionary.
        """
        if not isinstance(data, dict):
            raise InvalidFusionPayloadError(f"Expected dict, got {type(data).__name__}.")

        required_keys = {
            "modality",
            "predicted_class",
            "class_probabilities",
            "confidence",
            "model_type",
            "feature_count",
            "audio_duration_seconds",
            "model_path",
        }
        missing = required_keys - set(data.keys())
        if missing:
            raise InvalidFusionPayloadError(f"Missing required fields in fusion payload: {missing}")

        payload = cls(
            modality=data["modality"],
            predicted_class=data["predicted_class"],
            class_probabilities={str(k): float(v) for k, v in data["class_probabilities"].items()},
            confidence=float(data["confidence"]),
            model_type=str(data["model_type"]),
            feature_count=int(data["feature_count"]),
            audio_duration_seconds=float(data["audio_duration_seconds"]),
            model_path=str(data["model_path"]),
            metadata=dict(data.get("metadata", {})),
        )
        payload.validate()
        return payload


# ===========================================================================
# Validation Helper
# ===========================================================================

def validate_voice_fusion_payload(payload: Union[VoiceFusionPayload, Dict[str, Any]]) -> VoiceFusionPayload:
    """
    Validate a voice fusion payload or dictionary against the fusion contract.

    Parameters:
        payload: VoiceFusionPayload instance or dictionary representing one.

    Returns:
        Validated VoiceFusionPayload instance.

    Raises:
        InvalidFusionPayloadError: If the payload is invalid.
    """
    if isinstance(payload, VoiceFusionPayload):
        payload.validate()
        return payload
    elif isinstance(payload, dict):
        return VoiceFusionPayload.from_dict(payload)
    else:
        raise InvalidFusionPayloadError(
            f"Expected VoiceFusionPayload or dict, got {type(payload).__name__}."
        )


# ===========================================================================
# Main Integration Function
# ===========================================================================

def prepare_voice_fusion_payload(
    audio_file_path: Union[str, Path],
    model_path: Union[str, Path, None] = None,
    model: Optional[BaseVoiceClassifier] = None,
    expected_model_type: Optional[Type[BaseVoiceClassifier]] = None,
    audio_config: AudioConfig = AUDIO_CONFIG,
    feature_config: FeatureConfig = FEATURE_CONFIG,
    validate_audio: bool = True,
    strict_validation: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> VoiceFusionPayload:
    """
    Execute voice inference pipeline on an audio file and package the result
    into a standardized VoiceFusionPayload for the multimodal fusion layer.

    Parameters:
        audio_file_path: Path to the WAV audio file.
        model_path: Path to the serialized voice model file (.joblib).
        model: Optional pre-loaded BaseVoiceClassifier instance.
        expected_model_type: Optional expected model class type to verify.
        audio_config: Centralized AudioConfig settings.
        feature_config: Centralized FeatureConfig settings.
        validate_audio: Whether to run audio file validation before preprocessing.
        strict_validation: If True, validation raises exceptions on errors.
        metadata: Optional additional metadata to include in the payload.

    Returns:
        Validated VoiceFusionPayload ready for late decision-level fusion.

    Raises:
        PredictionAudioError: If the audio file is missing or invalid.
        PredictionFeatureError: If feature extraction fails.
        PredictionModelError: If model loading or prediction fails.
        InvalidFusionPayloadError: If the generated payload fails validation.
    """
    result = predict_voice(
        audio_file_path=audio_file_path,
        model_path=model_path,
        model=model,
        expected_model_type=expected_model_type,
        audio_config=audio_config,
        feature_config=feature_config,
        validate_audio=validate_audio,
        strict_validation=strict_validation,
        metadata=metadata,
    )
    return VoiceFusionPayload.from_prediction_result(result)
