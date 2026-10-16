"""Shared test-only helpers for voice-contract tests.

The model built here is trained on RANDOM synthetic features purely to exercise
the code path (shapes, contract fields, error handling). It carries no
predictive meaning and must never be presented as a real voice model.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from src.ai_model.voice.features import COMBINED_FEATURE_NAMES
from src.ai_model.voice.model import VoiceRandomForest

RISK_CLASSES = ["low", "moderate", "high"]


def make_test_voice_model(classes=None):
    """Fit a tiny RandomForest on random synthetic features (test-only)."""
    classes = classes or RISK_CLASSES
    rng = np.random.RandomState(0)
    n = 60
    X = pd.DataFrame(rng.randn(n, len(COMBINED_FEATURE_NAMES)), columns=list(COMBINED_FEATURE_NAMES))
    y = np.array(classes * (n // len(classes)))[:n]
    return VoiceRandomForest(random_state=0).fit(X, y)


def full_feature_dict(value: float = 0.1):
    return {name: value + 0.01 * i for i, name in enumerate(COMBINED_FEATURE_NAMES)}


def write_test_wav(path: Path, seconds: float = 6.0, sample_rate: int = 16000) -> Path:
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    samples = (0.6 * np.sin(2.0 * np.pi * 220.0 * t) * 32767).astype(np.int16)
    wavfile.write(str(path), sample_rate, samples)
    return path
