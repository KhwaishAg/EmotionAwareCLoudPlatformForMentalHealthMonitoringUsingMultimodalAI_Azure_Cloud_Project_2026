"""
Voice Modality Package — Emotion Aware Cloud Platform

This package implements the voice-derived stress/risk indicator pipeline
for the Multimodal Student Stress/Risk Assessment system.

Modules:
    config          - Centralized configuration (audio specs, feature params, model settings)
    recording       - Audio input interface, WAV loading, and recording abstraction
    preprocessing   - Audio loading, mono conversion, resampling, normalization, silence handling
    features        - Acoustic feature extraction (MFCC, pitch, energy, spectral)
    dataset         - Labeled audio dataset loader converting recordings into 41-feature datasets
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
from .preprocessing import (
    PreprocessedAudio,
    VoicePreprocessor,
    convert_to_mono,
    normalize_amplitude,
    preprocess_audio,
    resample_audio,
    trim_silence,
)
from .features import (
    AUDIO_FEATURE_NAMES,
    COMBINED_FEATURE_NAMES,
    ENERGY_FEATURE_NAMES,
    PITCH_FEATURE_NAMES,
    SPECTRAL_FEATURE_NAMES,
    AudioFeatureDict,
    CombinedFeatureDict,
    EnergyFeatureDict,
    FeatureExtractionError,
    SpectralFeatureDict,
    VoiceFeatureExtractor,
    compute_energy_frames,
    compute_mfcc_frames,
    compute_pitch_frames,
    compute_spectral_frames,
    extract_audio_feature_dict,
    extract_audio_features,
    extract_combined,
    extract_combined_dict,
    extract_combined_features,
    extract_energy,
    extract_energy_dict,
    extract_energy_features,
    extract_mfcc,
    extract_mfcc_dict,
    extract_mfcc_features,
    extract_pitch,
    extract_pitch_dict,
    extract_pitch_features,
    extract_spectral,
    extract_spectral_dict,
    extract_spectral_features,
    extract_voice_dict,
    extract_voice_features,
    get_audio_feature_names,
    get_combined_feature_names,
    get_energy_feature_names,
    get_mfcc_feature_names,
    get_pitch_feature_names,
    get_spectral_feature_names,
)
from .dataset import (
    AudioDatasetLoader,
    DatasetAudioNotFoundError,
    DatasetError,
    DatasetExtractionError,
    DatasetLabelError,
    DatasetMetadataError,
    DatasetNotFoundError,
    VoiceDatasetLoader,
    load_audio_dataset,
    load_voice_dataset,
    save_voice_dataset,
)
from .model import (
    BaseVoiceClassifier,
    ModelError,
    ModelInputError,
    ModelNotFittedError,
    VoiceBaselineClassifier,
    VoiceLogisticRegression,
    VoiceLogisticRegressionClassifier,
    VoiceRandomForest,
    VoiceRandomForestClassifier,
    separate_features_and_target,
)
from .evaluation import (
    ConfusionMatrixResult,
    EvaluationError,
    EvaluationInputError,
    EvaluationMetrics,
    VoiceConfusionMatrix,
    VoiceConfusionMatrixResult,
    VoiceEvaluationMetrics,
    VoiceModelMetrics,
    calculate_classification_metrics,
    calculate_voice_confusion_matrix,
    calculate_voice_metrics,
    compute_confusion_matrix,
    compute_voice_confusion_matrix,
    compute_voice_metrics,
    evaluate_confusion_matrix,
    evaluate_voice_confusion_matrix,
    evaluate_voice_model,
)




