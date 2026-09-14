"""
Voice Modality — Acoustic Feature Extraction: MFCC Extraction

Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from preprocessed
16 kHz mono audio signals for stress and risk classification models.

Features:
    - Configurable number of coefficients (default: 13, per FeatureConfig)
    - Frame-level STFT/Mel-spectrogram analysis via librosa
    - Fixed-length feature representation using summary statistics (mean, std)
    - Reusable controller (VoiceFeatureExtractor) and functional API
    - Robust input validation and edge-case handling

Specifications adhere to:
    - Centralized config: src/ai_model/voice/config.py (FeatureConfig, AudioConfig)
    - Input: 16 kHz, mono, 1D float32 waveform (or PreprocessedAudio/AudioRecording)
    - Output: Fixed-length 1D float32 numpy vector of size 2 * n_mfcc (26 features for 13 MFCCs)

Note:
    - Pitch, energy, spectral features, and ML modeling are deferred to subsequent commits.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import librosa
import numpy as np

from .config import (
    AUDIO_CONFIG,
    FEATURE_CONFIG,
    AudioConfig,
    FeatureConfig,
)
from .preprocessing import PreprocessedAudio
from .recording import (
    AudioDataError,
    AudioError,
    AudioRecording,
    AudioSampleRateError,
)


# ===========================================================================
# Exceptions
# ===========================================================================

class FeatureExtractionError(AudioError):
    """Base exception raised when acoustic feature extraction fails."""
    pass


# ===========================================================================
# Feature Names Helper
# ===========================================================================

def get_mfcc_feature_names(
    n_mfcc: int = FEATURE_CONFIG.n_mfcc,
    stats: Tuple[str, ...] = FEATURE_CONFIG.aggregation_stats,
) -> List[str]:
    """
    Generate standard feature column names for aggregated MFCC vectors.

    Parameters:
        n_mfcc: Number of MFCC coefficients (default: 13).
        stats: Aggregation statistics applied (default: ('mean', 'std')).

    Returns:
        List of feature name strings, e.g.:
        ['mfcc_1_mean', ..., 'mfcc_13_mean', 'mfcc_1_std', ..., 'mfcc_13_std'].
    """
    if n_mfcc <= 0:
        raise ValueError(f"n_mfcc must be a positive integer, got {n_mfcc}.")

    feature_names: List[str] = []
    for stat in stats:
        for i in range(1, n_mfcc + 1):
            feature_names.append(f"mfcc_{i}_{stat}")

    return feature_names


# ===========================================================================
# Frame-Level MFCC Extraction
# ===========================================================================

def compute_mfcc_frames(
    audio_data: np.ndarray,
    sample_rate: int = AUDIO_CONFIG.target_sample_rate,
    config: FeatureConfig = FEATURE_CONFIG,
    n_mfcc: Optional[int] = None,
) -> np.ndarray:
    """
    Compute frame-level MFCC matrix from a 1D audio waveform.

    Parameters:
        audio_data: 1D numpy array containing mono audio samples.
        sample_rate: Sampling frequency in Hz (default: 16,000 Hz).
        config: FeatureConfig containing default extraction parameters.
        n_mfcc: Optional override for number of coefficients (default: config.n_mfcc).

    Returns:
        2D numpy array of shape (n_mfcc, n_frames) with float32 dtype.

    Raises:
        AudioDataError: If audio_data is empty, not 1D, or contains non-finite values.
        AudioSampleRateError: If sample_rate <= 0.
        ValueError: If n_mfcc <= 0.
    """
    # 1. Validate audio array
    if audio_data is None or audio_data.size == 0:
        raise AudioDataError("Cannot extract MFCC features from empty audio array.")

    if not np.all(np.isfinite(audio_data)):
        raise AudioDataError("Audio waveform contains non-finite values (NaN or Inf).")

    if audio_data.ndim != 1:
        raise AudioDataError(
            f"Input audio must be a 1D mono waveform, got shape {audio_data.shape}."
        )

    # 2. Validate sample rate
    if sample_rate <= 0:
        raise AudioSampleRateError(f"Sample rate must be positive, got {sample_rate}.")

    # 3. Determine n_mfcc
    num_coeffs = n_mfcc if n_mfcc is not None else config.n_mfcc
    if num_coeffs <= 0:
        raise ValueError(f"n_mfcc must be a positive integer, got {num_coeffs}.")

    # Ensure float32 representation
    y = np.asarray(audio_data, dtype=np.float32)

    # Handle short audio length safely
    n_fft = config.mfcc_n_fft
    hop_length = config.mfcc_hop_length

    # If audio is shorter than n_fft, silence user warning from librosa
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_fft=.*is too large for input signal.*")
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sample_rate,
            n_mfcc=num_coeffs,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    return mfccs.astype(np.float32)


# ===========================================================================
# Fixed-Length Aggregated MFCC Feature Extraction
# ===========================================================================

def extract_mfcc_features(
    audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
    sample_rate: Optional[int] = None,
    config: FeatureConfig = FEATURE_CONFIG,
    n_mfcc: Optional[int] = None,
) -> np.ndarray:
    """
    Extract fixed-length MFCC feature vector using mean and standard deviation per coefficient.

    Parameters:
        audio: 1D numpy array, PreprocessedAudio container, or AudioRecording container.
        sample_rate: Optional sampling rate in Hz (defaults to audio container's rate or 16,000 Hz).
        config: FeatureConfig containing extraction and aggregation settings.
        n_mfcc: Optional override for number of coefficients (default: config.n_mfcc = 13).

    Returns:
        1D numpy array of shape (2 * n_mfcc,) containing [means..., stds...] in float32.

    Raises:
        AudioDataError: If audio is empty, non-finite, or not 1D mono.
        AudioSampleRateError: If sample rate is non-positive.
        ValueError: If n_mfcc <= 0.
        TypeError: If audio is not a supported input type.
    """
    # Unpack audio and sample rate from input containers
    if isinstance(audio, PreprocessedAudio):
        audio_data = audio.audio_data
        sr = sample_rate if sample_rate is not None else audio.sample_rate
    elif isinstance(audio, AudioRecording):
        audio_data = audio.audio_data
        sr = sample_rate if sample_rate is not None else audio.sample_rate
    elif isinstance(audio, np.ndarray):
        audio_data = audio
        sr = sample_rate if sample_rate is not None else AUDIO_CONFIG.target_sample_rate
    else:
        raise TypeError(
            f"Unsupported audio input type: {type(audio)}. "
            "Expected np.ndarray, PreprocessedAudio, or AudioRecording."
        )

    # Compute frame-level MFCCs
    mfcc_frames = compute_mfcc_frames(
        audio_data=audio_data,
        sample_rate=sr,
        config=config,
        n_mfcc=n_mfcc,
    )

    # Aggregate along time frames (axis=1) for each requested statistic
    stat_vectors: List[np.ndarray] = []
    for stat in config.aggregation_stats:
        if stat == "mean":
            stat_vectors.append(np.mean(mfcc_frames, axis=1))
        elif stat == "std":
            stat_vectors.append(np.std(mfcc_frames, axis=1))
        elif stat == "median":
            stat_vectors.append(np.median(mfcc_frames, axis=1))
        elif stat == "min":
            stat_vectors.append(np.min(mfcc_frames, axis=1))
        elif stat == "max":
            stat_vectors.append(np.max(mfcc_frames, axis=1))
        else:
            raise ValueError(f"Unsupported aggregation statistic: '{stat}'.")

    feature_vector = np.concatenate(stat_vectors).astype(np.float32)
    return feature_vector


# Alias for concise API usage
extract_mfcc = extract_mfcc_features


def extract_mfcc_dict(
    audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
    sample_rate: Optional[int] = None,
    config: FeatureConfig = FEATURE_CONFIG,
    n_mfcc: Optional[int] = None,
) -> Dict[str, float]:
    """
    Extract fixed-length MFCC features and return them as a dictionary of named features.

    Parameters:
        audio: 1D numpy array, PreprocessedAudio, or AudioRecording.
        sample_rate: Audio sampling frequency in Hz.
        config: FeatureConfig settings.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        Dictionary mapping feature names (e.g. 'mfcc_1_mean', 'mfcc_1_std') to float values.
    """
    num_coeffs = n_mfcc if n_mfcc is not None else config.n_mfcc
    feature_vector = extract_mfcc_features(
        audio=audio,
        sample_rate=sample_rate,
        config=config,
        n_mfcc=num_coeffs,
    )
    names = get_mfcc_feature_names(n_mfcc=num_coeffs, stats=config.aggregation_stats)

    return {name: float(val) for name, val in zip(names, feature_vector)}


# ===========================================================================
# Reusable Feature Extractor Controller
# ===========================================================================

class VoiceFeatureExtractor:
    """
    Controller for acoustic feature extraction in the Voice modality pipeline.

    Provides modular extraction methods and manages centralized feature configurations.
    """

    def __init__(
        self,
        config: FeatureConfig = FEATURE_CONFIG,
        audio_config: AudioConfig = AUDIO_CONFIG,
    ):
        self.config = config
        self.audio_config = audio_config

    def get_feature_names(self, n_mfcc: Optional[int] = None) -> List[str]:
        """Get list of feature names for the configured MFCC extraction."""
        num = n_mfcc if n_mfcc is not None else self.config.n_mfcc
        return get_mfcc_feature_names(n_mfcc=num, stats=self.config.aggregation_stats)

    def compute_frames(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None,
        n_mfcc: Optional[int] = None,
    ) -> np.ndarray:
        """Compute frame-level MFCC matrix."""
        sr = sample_rate if sample_rate is not None else self.audio_config.target_sample_rate
        return compute_mfcc_frames(
            audio_data=audio_data,
            sample_rate=sr,
            config=self.config,
            n_mfcc=n_mfcc,
        )

    def extract_features(
        self,
        audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
        sample_rate: Optional[int] = None,
        n_mfcc: Optional[int] = None,
    ) -> np.ndarray:
        """Extract fixed-length MFCC feature vector."""
        return extract_mfcc_features(
            audio=audio,
            sample_rate=sample_rate,
            config=self.config,
            n_mfcc=n_mfcc,
        )

    def extract_dict(
        self,
        audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
        sample_rate: Optional[int] = None,
        n_mfcc: Optional[int] = None,
    ) -> Dict[str, float]:
        """Extract fixed-length MFCC features as a named dictionary."""
        return extract_mfcc_dict(
            audio=audio,
            sample_rate=sample_rate,
            config=self.config,
            n_mfcc=n_mfcc,
        )
