"""
Voice Modality — Prediction Interface

Provides the predict_voice() inference interface for the trained voice model,
matching the standardized multimodal fusion contract used by the questionnaire
and behavioral modalities.

STATUS: No trained voice model currently exists (models/voice/ contains only
a .gitkeep placeholder). Training requires a real labeled voice dataset,
which has not yet been finalized/obtained. This module intentionally does
NOT fabricate a trained model or synthetic predictions.

predict_voice() is fully wired to the standardized contract shape and will
work as soon as a trained model artifact is placed at
models/voice/<model_name>.joblib and voice_feature_names.json /
voice_explainability_metadata.json are populated (mirroring the layout
Member 1 used for models/questionnaire/). Until then, it raises
VoiceModelNotTrainedError, which the backend/fusion layer treats exactly
like any other missing modality: available=False, never a crash.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .model import ModelNotFoundError
from .features import COMBINED_FEATURE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models" / "voice"
RESULTS_DIR = PROJECT_ROOT / "results" / "voice"
RISK_WEIGHTS = {"low": 0.0, "moderate": 0.5, "high": 1.0}


class VoiceModelNotTrainedError(ModelNotFoundError):
    """Raised when predict_voice() is called before a voice model has been trained.

    Subclasses the project's own ModelNotFoundError (see model.py) so callers
    already handling voice model errors catch this too. The backend/API layer
    should catch this specifically and mark the voice modality unavailable
    rather than treating it as an unexpected failure.
    """


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def load_selected_voice_model():
    """Load the trained voice model pipeline, once one exists.

    Raises VoiceModelNotTrainedError until a real model is trained on a
    labeled voice dataset and saved to models/voice/.
    """
    selector = MODELS_DIR / "selected_model.json"
    if not selector.exists():
        raise VoiceModelNotTrainedError(
            "No trained voice model is available yet. Voice model training is "
            "pending a suitable real labeled voice dataset; this modality is "
            "currently optional/unavailable, matching the project's treatment "
            "of other not-yet-implemented optional modalities (e.g. face)."
        )
    selected = _load_json(selector)
    model_name = selected.get("selected_model")
    from .model import load_voice_model  # local import to avoid unused import when untrained
    artifact = MODELS_DIR / f"{model_name}.joblib"
    if not artifact.exists():
        raise VoiceModelNotTrainedError(f"Selected voice model artifact was not found: {artifact}")
    return load_voice_model(artifact), model_name


def predict_voice(features: Dict[str, Any]) -> Dict[str, Any]:
    """Predict voice-derived stress risk and return the standardized fusion contract.

    Parameters
    ----------
    features:
        A dict of the 41 combined acoustic features (COMBINED_FEATURE_NAMES),
        as produced by src.ai_model.voice.features.extract_combined_dict()
        from a preprocessed audio recording.

    Raises
    ------
    VoiceModelNotTrainedError
        Always, until a trained model artifact exists at models/voice/.
        Callers (the backend API, the fusion integration test harness)
        must catch this and treat voice as an unavailable modality --
        exactly like a missing optional input, never a crash.
    """
    model, model_name = load_selected_voice_model()  # raises VoiceModelNotTrainedError today

    # --- Everything below is the real inference path, ready to run the
    # --- moment a trained model exists. It intentionally mirrors
    # --- src/ai_model/questionnaire/predict.py's contract logic so the
    # --- three modalities stay consistent.
    ordered = {name: features.get(name, np.nan) for name in COMBINED_FEATURE_NAMES}
    probability_map = model.predict_proba_dict([ordered])[0]
    if set(probability_map) != set(RISK_WEIGHTS):
        raise ValueError("Voice model classes must be exactly low, moderate, and high.")
    risk_level = max(probability_map, key=probability_map.get)
    risk_score = float(sum(RISK_WEIGHTS[level] * probability_map[level] for level in RISK_WEIGHTS))
    ordered_probs = sorted(probability_map.values(), reverse=True)
    confidence = float(np.clip(0.5 * ordered_probs[0] + 0.5 * (ordered_probs[0] - ordered_probs[1]), 0.0, 1.0))

    total = len(COMBINED_FEATURE_NAMES)
    available = sum(1 for value in ordered.values() if value is not None and not (isinstance(value, float) and np.isnan(value)))

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


if __name__ == "__main__":
    try:
        predict_voice({})
    except VoiceModelNotTrainedError as error:
        print("VOICE MODEL STATUS: not yet trained (expected)")
        print(f"  {error}")
        print("\nThis is correct project state -- integrate voice as an optional")
        print("pending modality. Do not fabricate a model to make this pass.")
