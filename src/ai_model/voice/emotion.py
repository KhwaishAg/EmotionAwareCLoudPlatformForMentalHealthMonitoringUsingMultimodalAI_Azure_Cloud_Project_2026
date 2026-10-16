"""Experimental vocal-EMOTION recognition, kept strictly separate from stress-risk.

This model recognises six ACTED emotion categories from speech. It is trained on
CREMA-D (acted sentences by professional actors) and exists to exercise and
validate the audio preprocessing / feature-extraction pipeline.

What it is NOT
    * It is not a stress or mental-health measure and its classes are never
      mapped to the low / moderate / high stress-risk labels.
    * Its output is deliberately NOT the fusion contract (no ``risk_score`` /
      ``risk_level``), is never passed to ``fuse_modalities`` and never influences
      the overall assessment.
    * Acted speech from actors does not represent natural speech from students.

Artifacts live in ``models/voice_emotion/`` (separate from ``models/voice/``, which
is reserved for a future, properly labeled stress-risk model).
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .config import AUDIO_CONFIG, AudioConfig
from .features import COMBINED_FEATURE_NAMES
from .model import BaseVoiceClassifier, ModelError, ModelNotFoundError, load_voice_model
from .predict import (
    PredictionModelError,
    TRAINING_REPORT_FILENAME,
    _validate_feature_dict,
    extract_voice_features_from_audio,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMOTION_MODELS_DIR = PROJECT_ROOT / "models" / "voice_emotion"

EMOTION_TASK = "emotion_recognition"
#: Acted emotion categories (CREMA-D). Independent of the stress-risk labels.
EMOTION_LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad")

#: CREMA-D utterances are short (about 1-5 s), so the minimum-duration gate used for
#: 30-60 s student recordings is relaxed. Training and inference share this config.
EMOTION_AUDIO_CONFIG: AudioConfig = dataclasses.replace(
    AUDIO_CONFIG, min_duration_seconds=1.0, min_duration_after_trim=0.5
)

NOTICE = (
    "Experimental, acted-speech emotion category. Not a measure of stress or mental health, "
    "not used in the overall assessment, and not validated on real student speech."
)


class EmotionModelNotTrainedError(ModelNotFoundError):
    """No emotion model artifact is available."""


class EmotionModelLoadError(PredictionModelError):
    """An emotion model artifact exists but cannot be used."""


def read_emotion_report() -> Optional[Dict[str, Any]]:
    path = EMOTION_MODELS_DIR / TRAINING_REPORT_FILENAME
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return report if isinstance(report, dict) else None


def load_emotion_model() -> tuple[BaseVoiceClassifier, str]:
    """Load the emotion model; never returns a model that was not trained for this task."""
    selector = EMOTION_MODELS_DIR / "selected_model.json"
    if not selector.is_file():
        raise EmotionModelNotTrainedError("No emotion model has been trained.")
    try:
        name = json.loads(selector.read_text(encoding="utf-8")).get("selected_model")
    except (OSError, ValueError, AttributeError) as error:
        raise EmotionModelLoadError("Emotion selector file could not be read.") from error
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise EmotionModelLoadError("Emotion selector does not name a valid model.")
    artifact = EMOTION_MODELS_DIR / f"{name}.joblib"
    if not artifact.is_file():
        raise EmotionModelNotTrainedError("Emotion model artifact was not found.")
    report = read_emotion_report()
    if not report or report.get("task") != EMOTION_TASK or list(report.get("classes", [])) != list(EMOTION_LABELS):
        raise EmotionModelLoadError("Emotion model report is missing or does not match this task.")
    try:
        model = load_voice_model(artifact)
    except ModelError as error:
        raise EmotionModelLoadError("Emotion model artifact could not be loaded.") from error
    except Exception as error:
        raise EmotionModelLoadError("Emotion model artifact is unreadable.") from error
    if list(model.feature_names_ or []) != list(COMBINED_FEATURE_NAMES) or set(model.classes_) != set(EMOTION_LABELS):
        raise EmotionModelLoadError("Emotion model does not match the current features or labels.")
    return model, name


def predict_emotion_from_features(features: Dict[str, Any], model: Optional[BaseVoiceClassifier] = None) -> Dict[str, Any]:
    """Predict the acted-emotion category from the 41 acoustic features (all required)."""
    cleaned = _validate_feature_dict(features)
    missing = [name for name in COMBINED_FEATURE_NAMES if name not in cleaned]
    if missing:
        raise ValueError(f"Voice features are incomplete: {len(missing)} of {len(COMBINED_FEATURE_NAMES)} missing.")
    if model is None:
        model, _ = load_emotion_model()
    row = pd.DataFrame([{name: cleaned[name] for name in COMBINED_FEATURE_NAMES}])[list(COMBINED_FEATURE_NAMES)]
    try:
        probabilities = model.predict_proba_dict(row)[0]
    except Exception as error:
        raise PredictionModelError("Emotion model inference failed.") from error
    if set(probabilities) != set(EMOTION_LABELS):
        raise PredictionModelError("Emotion model classes do not match the expected labels.")
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return {
        "task": EMOTION_TASK,
        "experimental": True,
        "predicted_emotion": ranked[0][0],
        "confidence": float(ranked[0][1]),
        "probabilities": {label: float(probabilities[label]) for label in EMOTION_LABELS},
        "notice": NOTICE,
    }


def predict_emotion_from_audio(audio_file_path: Union[str, Path], model: Optional[BaseVoiceClassifier] = None) -> Dict[str, Any]:
    """Audio file -> emotion category (separate from, and never part of, the stress-risk contract)."""
    if model is None:
        model, _ = load_emotion_model()
    features = extract_voice_features_from_audio(audio_file_path, audio_config=EMOTION_AUDIO_CONFIG)
    return predict_emotion_from_features(features, model=model)
