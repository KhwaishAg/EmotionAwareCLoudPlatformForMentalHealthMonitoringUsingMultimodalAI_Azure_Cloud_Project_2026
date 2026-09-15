"""
Voice Modality — Centralized Configuration

All configurable parameters for the voice processing pipeline are defined here.
This prevents magic numbers from being scattered across modules and ensures
reproducibility.

Sections:
    AudioConfig     - Audio input specifications (sample rate, channels, duration)
    FeatureConfig   - Acoustic feature extraction parameters (MFCC, pitch, spectral)
    ModelConfig     - ML model training settings (random seed, test size, model type)
    PathConfig      - Directory and file path conventions
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ===========================================================================
# Audio Input Configuration
# ===========================================================================

@dataclass(frozen=True)
class AudioConfig:
    """Target audio input specifications for the voice pipeline."""

    # Target sample rate after resampling (Hz)
    target_sample_rate: int = 16_000

    # Target number of audio channels
    target_channels: int = 1  # mono

    # Acceptable recording duration range (seconds)
    min_duration_seconds: float = 5.0
    max_duration_seconds: float = 120.0

    # Recommended recording duration (seconds)
    recommended_min_duration: float = 30.0
    recommended_max_duration: float = 60.0

    # Supported input file extensions
    supported_extensions: tuple = (".wav", ".WAV")

    # Amplitude normalization: peak normalization target
    normalization_peak: float = 0.95

    # Silence trimming: top_db threshold for librosa.effects.trim
    silence_top_db: int = 25

    # Minimum duration after silence trimming (seconds)
    # Protects against removing meaningful low-energy speech
    min_duration_after_trim: float = 3.0


# ===========================================================================
# Feature Extraction Configuration
# ===========================================================================

@dataclass(frozen=True)
class FeatureConfig:
    """Parameters controlling acoustic feature extraction."""

    # MFCC
    n_mfcc: int = 13
    mfcc_hop_length: int = 512
    mfcc_n_fft: int = 2048

    # Pitch (fundamental frequency / F0)
    pitch_fmin: float = 50.0   # Hz — lower bound for pitch detection
    pitch_fmax: float = 500.0  # Hz — upper bound for pitch detection

    # Spectral features
    spectral_hop_length: int = 512
    spectral_n_fft: int = 2048

    # Aggregation statistics applied to frame-level features
    # to produce a fixed-length vector from variable-length audio
    aggregation_stats: tuple = ("mean", "std")


# ===========================================================================
# Model Configuration
# ===========================================================================

@dataclass(frozen=True)
class ModelConfig:
    """ML model training and evaluation settings."""

    # Random seed for reproducibility
    random_seed: int = 42

    # Train / validation / test split ratios
    test_size: float = 0.2
    validation_size: float = 0.15  # of remaining after test split

    # Default model type for initial baseline
    default_model_type: str = "random_forest"

    # Label configuration — NOT hardcoded to specific project labels.
    # These are configurable placeholders that must be updated once
    # the project's final target label taxonomy is established.
    label_column: str = "target"
    participant_id_column: str = "participant_id"


# ===========================================================================
# Path Configuration
# ===========================================================================

@dataclass(frozen=True)
class PathConfig:
    """Directory and file path conventions for the voice pipeline."""

    # Relative to repository root
    raw_audio_dir: str = "data/raw/audio"
    processed_dir: str = "data/processed"
    models_dir: str = "models/voice"
    voice_module_dir: str = "src/ai_model/voice"

    # Expected metadata filename
    metadata_filename: str = "metadata.csv"

    # Feature output filename
    features_filename: str = "audio_features.csv"


# ===========================================================================
# Default Instances
# ===========================================================================

AUDIO_CONFIG = AudioConfig()
FEATURE_CONFIG = FeatureConfig()
MODEL_CONFIG = ModelConfig()
PATH_CONFIG = PathConfig()
