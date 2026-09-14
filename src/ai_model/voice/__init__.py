"""
Voice Modality Package — Emotion Aware Cloud Platform

This package implements the voice-derived stress/risk indicator pipeline
for the Multimodal Student Stress/Risk Assessment system.

Modules:
    config          - Centralized configuration (audio specs, feature params, model settings)
    recording       - Audio input interface, WAV loading, and recording abstraction
    preprocessing   - Audio loading, mono conversion, resampling, normalization, silence handling
    features        - Acoustic feature extraction (MFCC, pitch, energy, spectral)
    model           - Voice ML model training (Logistic Regression, Random Forest)
    evaluation      - Model evaluation metrics and confusion matrix
    predict         - Inference interface for the trained voice model

Note:
    This module produces voice-level assessments (risk scores, class probabilities)
    that are consumed by the future multimodal fusion component.
    It is NOT a clinical diagnostic tool.
"""

__version__ = "0.1.0"

from .config import (
    AUDIO_CONFIG,
    FEATURE_CONFIG,
    MODEL_CONFIG,
    PATH_CONFIG,
    AudioConfig,
    FeatureConfig,
    ModelConfig,
    PathConfig,
)
from .recording import (
    STANDARDIZED_RECORDING_PROMPT,
    AudioChannelError,
    AudioCorruptError,
    AudioDataError,
    AudioDurationError,
    AudioError,
    AudioFormatError,
    AudioInputSource,
    AudioNotFoundError,
    AudioRecording,
    AudioSampleRateError,
    AudioValidationError,
    BufferInputSource,
    LocalFileInputSource,
    ValidationResult,
    VoiceRecordingInterface,
    extract_participant_id,
    load_wav_bytes,
    load_wav_file,
    save_wav_file,
    validate_audio_file,
    validate_audio_recording,
)


