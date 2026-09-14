"""
Voice Modality — Acoustic Feature Extraction: MFCC, Pitch, Energy & Spectral Extraction

Extracts acoustic feature representations from preprocessed 16 kHz mono
audio signals for stress and risk classification models.

Features Implemented:
    - Mel-Frequency Cepstral Coefficients (MFCCs): Configurable (default 13),
      aggregated using mean and standard deviation per coefficient (fixed-length 26D).
    - Fundamental Frequency / Pitch (F0): Computed via librosa.pyin with
      mean, standard deviation, minimum, and maximum extracted from valid voiced frames
      only (fixed-length 4D).
    - Root Mean Square (RMS) Energy: Frame-level energy contour aggregated
      using mean, standard deviation, and variation/range (fixed-length 3D).
    - Spectral Descriptors: Spectral centroid, spectral bandwidth, spectral rolloff,
      and zero-crossing rate aggregated using mean and standard deviation (fixed-length 8D).
    - Robust unvoiced/silent audio handling without NaN or Inf values.
    - Reusable controller (VoiceFeatureExtractor) and functional APIs.

Specifications adhere to:
    - Centralized config: src/ai_model/voice/config.py (FeatureConfig, AudioConfig)
    - Input: 16 kHz, mono, 1D float32 waveform (or PreprocessedAudio/AudioRecording)

Note:
    - ML modeling and multimodal fusion are deferred to subsequent commits.
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
# Feature Names Helpers & Constants
# ===========================================================================

PITCH_FEATURE_NAMES: Tuple[str, ...] = (
    "pitch_mean",
    "pitch_std",
    "pitch_min",
    "pitch_max",
)

ENERGY_FEATURE_NAMES: Tuple[str, ...] = (
    "energy_mean",
    "energy_std",
    "energy_range",
)

SPECTRAL_FEATURE_NAMES: Tuple[str, ...] = (
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
)


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


def get_pitch_feature_names() -> List[str]:
    """
    Generate standard feature column names for aggregated pitch (F0) vectors.

    Returns:
        List of pitch feature names: ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max'].
    """
    return list(PITCH_FEATURE_NAMES)


def get_energy_feature_names() -> List[str]:
    """
    Generate standard feature column names for aggregated RMS energy vectors.

    Returns:
        List of energy feature names: ['energy_mean', 'energy_std', 'energy_range'].
    """
    return list(ENERGY_FEATURE_NAMES)


class EnergyFeatureDict(dict):
    """
    Dictionary container for energy features.
    Provides transparent access for both 'energy_range' and 'energy_variation'
    as well as 'energy_rms_*' aliases.
    """

    def __getitem__(self, key: str) -> float:
        if key in self:
            return super().__getitem__(key)
        if key in ("energy_variation", "energy_rms_range", "energy_rms_variation") and "energy_range" in self:
            return super().__getitem__("energy_range")
        if key == "energy_rms_mean" and "energy_mean" in self:
            return super().__getitem__("energy_mean")
        if key == "energy_rms_std" and "energy_std" in self:
            return super().__getitem__("energy_std")
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def get_spectral_feature_names() -> List[str]:
    """
    Generate standard feature column names for aggregated spectral feature vectors.

    Returns:
        List of spectral feature names (length 8):
        ['spectral_centroid_mean', 'spectral_centroid_std',
         'spectral_bandwidth_mean', 'spectral_bandwidth_std',
         'spectral_rolloff_mean', 'spectral_rolloff_std',
         'zero_crossing_rate_mean', 'zero_crossing_rate_std'].
    """
    return list(SPECTRAL_FEATURE_NAMES)


class SpectralFeatureDict(dict):
    """
    Dictionary container for spectral features.
    Provides transparent access for standard names and convenient aliases
    (e.g., 'zcr_mean' for 'zero_crossing_rate_mean').
    """

    _ALIASES = {
        "zcr_mean": "zero_crossing_rate_mean",
        "zcr_std": "zero_crossing_rate_std",
        "centroid_mean": "spectral_centroid_mean",
        "centroid_std": "spectral_centroid_std",
        "bandwidth_mean": "spectral_bandwidth_mean",
        "bandwidth_std": "spectral_bandwidth_std",
        "rolloff_mean": "spectral_rolloff_mean",
        "rolloff_std": "spectral_rolloff_std",
    }

    def __getitem__(self, key: str) -> float:
        if key in self:
            return super().__getitem__(key)
        if key in self._ALIASES and self._ALIASES[key] in self:
            return super().__getitem__(self._ALIASES[key])
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


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
# Frame-Level Pitch (F0) Extraction
# ===========================================================================

def compute_pitch_frames(
    audio_data: np.ndarray,
    sample_rate: int = AUDIO_CONFIG.target_sample_rate,
    config: FeatureConfig = FEATURE_CONFIG,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute frame-level fundamental frequency (F0) track using librosa.pyin.

    Parameters:
        audio_data: 1D numpy array of mono audio waveform.
        sample_rate: Audio sampling frequency in Hz (default: 16,000 Hz).
        config: FeatureConfig containing pitch frequency bounds.
        fmin: Minimum fundamental frequency in Hz (default: config.pitch_fmin = 50.0).
        fmax: Maximum fundamental frequency in Hz (default: config.pitch_fmax = 500.0).

    Returns:
        Tuple of (f0, voiced_flag, voiced_probabilities):
            - f0: 1D numpy array of fundamental frequencies in Hz (unvoiced frames are NaN).
            - voiced_flag: 1D boolean array indicating voiced frames.
            - voiced_probabilities: 1D numpy array of voicing probability per frame.

    Raises:
        AudioDataError: If audio_data is empty, not 1D, or contains non-finite values.
        AudioSampleRateError: If sample_rate <= 0.
        ValueError: If fmin <= 0 or fmax <= fmin.
    """
    # 1. Validate audio array
    if audio_data is None or audio_data.size == 0:
        raise AudioDataError("Cannot extract pitch features from empty audio array.")

    if not np.all(np.isfinite(audio_data)):
        raise AudioDataError("Audio waveform contains non-finite values (NaN or Inf).")

    if audio_data.ndim != 1:
        raise AudioDataError(
            f"Input audio must be a 1D mono waveform, got shape {audio_data.shape}."
        )

    # 2. Validate sample rate
    if sample_rate <= 0:
        raise AudioSampleRateError(f"Sample rate must be positive, got {sample_rate}.")

    # 3. Validate pitch frequency bounds
    min_f = float(fmin if fmin is not None else config.pitch_fmin)
    max_f = float(fmax if fmax is not None else config.pitch_fmax)
    if min_f <= 0.0 or max_f <= min_f:
        raise ValueError(
            f"Invalid pitch bounds: fmin={min_f}, fmax={max_f}. Must satisfy 0 < fmin < fmax."
        )

    y = np.asarray(audio_data, dtype=np.float32)

    # Completely silent audio optimization: return empty/unvoiced frames directly
    if np.max(np.abs(y)) == 0.0:
        n_frames = max(1, int(np.ceil(len(y) / config.mfcc_hop_length)))
        f0 = np.full(n_frames, np.nan, dtype=np.float32)
        voiced_flag = np.zeros(n_frames, dtype=bool)
        voiced_probs = np.zeros(n_frames, dtype=np.float32)
        return f0, voiced_flag, voiced_probs

    # Probabilistic YIN fundamental frequency estimation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y=y,
            fmin=min_f,
            fmax=max_f,
            sr=sample_rate,
            hop_length=config.mfcc_hop_length,
        )

    return f0.astype(np.float32), voiced_flag.astype(bool), voiced_probs.astype(np.float32)


# ===========================================================================
# Fixed-Length Aggregated Pitch Feature Extraction
# ===========================================================================

def extract_pitch_features(
    audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
    sample_rate: Optional[int] = None,
    config: FeatureConfig = FEATURE_CONFIG,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    Extract fixed-length pitch features (mean, std, min, max) from valid voiced frames only.

    - Uses librosa.pyin for accurate fundamental frequency (F0) estimation.
    - Aggregates exclusively over voiced frames (where voicing is detected and F0 is finite).
    - Safely handles unvoiced or silent audio by returning 0.0 for all statistics,
      preventing NaN or Inf from propagating downstream.
    - Returns a 1D float32 array of shape (4,) corresponding to:
      [pitch_mean, pitch_std, pitch_min, pitch_max].

    Parameters:
        audio: 1D numpy array, PreprocessedAudio container, or AudioRecording container.
        sample_rate: Audio sampling frequency in Hz (defaults to container rate or 16,000 Hz).
        config: FeatureConfig containing pitch frequency thresholds.
        fmin: Minimum fundamental frequency in Hz (default: config.pitch_fmin = 50.0).
        fmax: Maximum fundamental frequency in Hz (default: config.pitch_fmax = 500.0).

    Returns:
        1D numpy array of shape (4,) with dtype float32: [mean, std, min, max].

    Raises:
        AudioDataError: If audio is empty, non-finite, or not 1D mono.
        AudioSampleRateError: If sample rate <= 0.
        ValueError: If fmin <= 0 or fmax <= fmin.
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

    f0, voiced_flag, _ = compute_pitch_frames(
        audio_data=audio_data,
        sample_rate=sr,
        config=config,
        fmin=fmin,
        fmax=fmax,
    )

    # Filter strictly for valid voiced frames
    valid_f0 = f0[voiced_flag & np.isfinite(f0)]

    # Handle unvoiced or silent audio safely without NaN/Inf
    if len(valid_f0) == 0:
        return np.zeros(4, dtype=np.float32)

    pitch_mean = float(np.mean(valid_f0))
    pitch_std = float(np.std(valid_f0))
    pitch_min = float(np.min(valid_f0))
    pitch_max = float(np.max(valid_f0))

    return np.array([pitch_mean, pitch_std, pitch_min, pitch_max], dtype=np.float32)


# Alias for concise API usage
extract_pitch = extract_pitch_features


def extract_pitch_dict(
    audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
    sample_rate: Optional[int] = None,
    config: FeatureConfig = FEATURE_CONFIG,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> Dict[str, float]:
    """
    Extract fixed-length pitch features and return them as a dictionary of named features.

    Parameters:
        audio: 1D numpy array, PreprocessedAudio, or AudioRecording.
        sample_rate: Audio sampling frequency in Hz.
        config: FeatureConfig settings.
        fmin: Minimum fundamental frequency in Hz.
        fmax: Maximum fundamental frequency in Hz.

    Returns:
        Dictionary mapping feature names ('pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max')
        to float values.
    """
    feature_vector = extract_pitch_features(
        audio=audio,
        sample_rate=sample_rate,
        config=config,
        fmin=fmin,
        fmax=fmax,
    )
    names = get_pitch_feature_names()

    return {name: float(val) for name, val in zip(names, feature_vector)}


# ===========================================================================
# Frame-Level RMS Energy Extraction
# ===========================================================================

def compute_energy_frames(
    audio_data: np.ndarray,
    sample_rate: int = AUDIO_CONFIG.target_sample_rate,
    config: FeatureConfig = FEATURE_CONFIG,
    frame_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> np.ndarray:
    """
    Compute frame-level Root Mean Square (RMS) energy contour using librosa.feature.rms.

    Parameters:
        audio_data: 1D numpy array of mono audio waveform.
        sample_rate: Audio sampling rate in Hz (default: 16,000 Hz).
        config: FeatureConfig containing default frame and hop sizes.
        frame_length: Analysis window length in samples (default: config.mfcc_n_fft = 2048).
        hop_length: Hop length between analysis frames in samples (default: config.mfcc_hop_length = 512).

    Returns:
        2D numpy array of shape (1, n_frames) with float32 dtype.

    Raises:
        AudioDataError: If audio_data is empty, not 1D, or contains non-finite values.
        AudioSampleRateError: If sample_rate <= 0.
        ValueError: If frame_length <= 0 or hop_length <= 0.
    """
    if audio_data is None or audio_data.size == 0:
        raise AudioDataError("Cannot extract energy features from empty audio array.")

    if not np.all(np.isfinite(audio_data)):
        raise AudioDataError("Audio waveform contains non-finite values (NaN or Inf).")

    if audio_data.ndim != 1:
        raise AudioDataError(
            f"Input audio must be a 1D mono waveform, got shape {audio_data.shape}."
        )

    if sample_rate <= 0:
        raise AudioSampleRateError(f"Sample rate must be positive, got {sample_rate}.")

    frame_len = frame_length if frame_length is not None else config.mfcc_n_fft
    hop_len = hop_length if hop_length is not None else config.mfcc_hop_length

    if frame_len <= 0:
        raise ValueError(f"frame_length must be positive, got {frame_len}.")
    if hop_len <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_len}.")

    y = np.asarray(audio_data, dtype=np.float32)

    # Completely silent audio: return zeros directly
    if np.max(np.abs(y)) == 0.0:
        n_frames = max(1, int(np.ceil(len(y) / hop_len)))
        return np.zeros((1, n_frames), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_len,
            hop_length=hop_len,
        )

    return rms.astype(np.float32)


# ===========================================================================
# Fixed-Length Aggregated RMS Energy Feature Extraction
# ===========================================================================

def extract_energy_features(
    audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
    sample_rate: Optional[int] = None,
    config: FeatureConfig = FEATURE_CONFIG,
    frame_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> np.ndarray:
    """
    Extract fixed-length RMS energy features (mean, std, range/variation) from audio.

    - Computes frame-level RMS energy contour reflecting vocal intensity.
    - Summarizes contour with mean, standard deviation, and variation (range = max - min).
    - Safely handles completely silent audio by returning 0.0 for all statistics without NaN/Inf.
    - Returns fixed-length 1D float32 array of shape (3,): [energy_mean, energy_std, energy_range].

    Parameters:
        audio: 1D numpy array, PreprocessedAudio container, or AudioRecording container.
        sample_rate: Audio sampling rate in Hz (defaults to container rate or 16,000 Hz).
        config: FeatureConfig extraction settings.
        frame_length: Analysis window size (default: config.mfcc_n_fft = 2048).
        hop_length: Hop length between frames (default: config.mfcc_hop_length = 512).

    Returns:
        1D numpy array of shape (3,) with float32 dtype: [mean, std, range].

    Raises:
        AudioDataError: If audio is empty, non-finite, or not 1D mono.
        AudioSampleRateError: If sample rate <= 0.
        ValueError: If frame_length or hop_length <= 0.
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

    rms_matrix = compute_energy_frames(
        audio_data=audio_data,
        sample_rate=sr,
        config=config,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    rms_frames = rms_matrix.flatten()

    if len(rms_frames) == 0 or np.max(np.abs(audio_data)) == 0.0:
        return np.zeros(3, dtype=np.float32)

    energy_mean = float(np.mean(rms_frames))
    energy_std = float(np.std(rms_frames))
    energy_range = float(np.max(rms_frames) - np.min(rms_frames))

    return np.array([energy_mean, energy_std, energy_range], dtype=np.float32)


# Alias for concise API usage
extract_energy = extract_energy_features


def extract_energy_dict(
    audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
    sample_rate: Optional[int] = None,
    config: FeatureConfig = FEATURE_CONFIG,
    frame_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> EnergyFeatureDict:
    """
    Extract fixed-length energy features and return them as an EnergyFeatureDict.

    Parameters:
        audio: 1D numpy array, PreprocessedAudio, or AudioRecording.
        sample_rate: Audio sampling frequency in Hz.
        config: FeatureConfig settings.
        frame_length: Analysis window size.
        hop_length: Hop length.

    Returns:
        EnergyFeatureDict mapping feature names ('energy_mean', 'energy_std', 'energy_range')
        to float values (supports 'energy_variation' access).
    """
    features = extract_energy_features(
        audio=audio,
        sample_rate=sample_rate,
        config=config,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    return EnergyFeatureDict({
        "energy_mean": float(features[0]),
        "energy_std": float(features[1]),
        "energy_range": float(features[2]),
    })


# ===========================================================================
# Frame-Level Spectral Feature Extraction
# ===========================================================================

def compute_spectral_frames(
    audio_data: np.ndarray,
    sample_rate: int = AUDIO_CONFIG.target_sample_rate,
    config: FeatureConfig = FEATURE_CONFIG,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    roll_percent: float = 0.85,
) -> Dict[str, np.ndarray]:
    """
    Compute frame-level spectral descriptors from an audio waveform.

    Calculates:
        - spectral_centroid: Center of mass of the spectrum per frame (Hz)
        - spectral_bandwidth: Spectral spread around the centroid per frame (Hz)
        - spectral_rolloff: Frequency below which roll_percent of energy lies (Hz)
        - zero_crossing_rate: Rate of sign changes in the signal per frame

    Parameters:
        audio_data: 1D numpy array of audio samples.
        sample_rate: Audio sampling frequency in Hz (default: 16,000 Hz).
        config: FeatureConfig settings.
        n_fft: Window / FFT length (default: config.spectral_n_fft = 2048).
        hop_length: Hop length between frames (default: config.spectral_hop_length = 512).
        roll_percent: Roll-off percentage (default: 0.85).

    Returns:
        Dictionary mapping descriptor names to 2D numpy arrays of shape (1, n_frames).

    Raises:
        AudioDataError: If audio_data is empty, non-finite, or not 1D mono.
        AudioSampleRateError: If sample_rate <= 0.
        ValueError: If n_fft, hop_length, or roll_percent are invalid.
    """
    if not isinstance(audio_data, np.ndarray):
        raise AudioDataError(f"audio_data must be a numpy.ndarray, got {type(audio_data)}.")

    if audio_data.size == 0:
        raise AudioDataError("audio_data is empty.")

    if audio_data.ndim != 1:
        raise AudioDataError(
            f"audio_data must be a 1D mono array, got shape {audio_data.shape} with {audio_data.ndim} dimensions."
        )

    if not np.all(np.isfinite(audio_data)):
        raise AudioDataError("audio_data contains NaN or Inf values.")

    if sample_rate <= 0:
        raise AudioSampleRateError(
            f"sample_rate must be a positive integer, got {sample_rate}."
        )

    fft_len = n_fft if n_fft is not None else config.spectral_n_fft
    hop = hop_length if hop_length is not None else config.spectral_hop_length

    if fft_len <= 0:
        raise ValueError(f"n_fft must be a positive integer, got {fft_len}.")
    if hop <= 0:
        raise ValueError(f"hop_length must be a positive integer, got {hop}.")
    if not (0.0 < roll_percent <= 1.0):
        raise ValueError(f"roll_percent must be in (0.0, 1.0], got {roll_percent}.")

    # Pad if shorter than n_fft
    y_float = audio_data.astype(np.float32)
    if y_float.shape[0] < fft_len:
        pad_len = fft_len - y_float.shape[0]
        y_proc = np.pad(y_float, (0, pad_len), mode="constant")
    else:
        y_proc = y_float

    centroid = librosa.feature.spectral_centroid(
        y=y_proc,
        sr=sample_rate,
        n_fft=fft_len,
        hop_length=hop,
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y_proc,
        sr=sample_rate,
        n_fft=fft_len,
        hop_length=hop,
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=y_proc,
        sr=sample_rate,
        n_fft=fft_len,
        hop_length=hop,
        roll_percent=roll_percent,
    )
    zcr = librosa.feature.zero_crossing_rate(
        y=y_proc,
        frame_length=fft_len,
        hop_length=hop,
    )

    return {
        "spectral_centroid": centroid.astype(np.float32),
        "spectral_bandwidth": bandwidth.astype(np.float32),
        "spectral_rolloff": rolloff.astype(np.float32),
        "zero_crossing_rate": zcr.astype(np.float32),
    }


# ===========================================================================
# Aggregated Spectral Feature Extraction
# ===========================================================================

def extract_spectral_features(
    audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
    sample_rate: Optional[int] = None,
    config: FeatureConfig = FEATURE_CONFIG,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    roll_percent: float = 0.85,
) -> np.ndarray:
    """
    Extract fixed-length spectral features from an audio recording or waveform.

    Descriptors Extracted:
        1. Spectral Centroid (mean, std)
        2. Spectral Bandwidth (mean, std)
        3. Spectral Rolloff (mean, std)
        4. Zero-Crossing Rate (mean, std)

    Returns a fixed-length 1D float32 array of shape (8,):
    [centroid_mean, centroid_std, bandwidth_mean, bandwidth_std,
     rolloff_mean, rolloff_std, zcr_mean, zcr_std].

    Handles silent audio safely without producing NaN or Inf values.

    Parameters:
        audio: 1D numpy array, PreprocessedAudio container, or AudioRecording container.
        sample_rate: Audio sampling rate in Hz (defaults to container rate or 16,000 Hz).
        config: FeatureConfig extraction settings.
        n_fft: FFT window size (default: config.spectral_n_fft = 2048).
        hop_length: Hop length between frames (default: config.spectral_hop_length = 512).
        roll_percent: Spectral roll-off percentage (default: 0.85).

    Returns:
        1D numpy array of shape (8,) with float32 dtype.

    Raises:
        AudioDataError: If audio is empty, non-finite, or not 1D mono.
        AudioSampleRateError: If sample rate <= 0.
        ValueError: If n_fft or hop_length <= 0.
        TypeError: If audio is not a supported input type.
    """
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

    frames = compute_spectral_frames(
        audio_data=audio_data,
        sample_rate=sr,
        config=config,
        n_fft=n_fft,
        hop_length=hop_length,
        roll_percent=roll_percent,
    )

    if len(audio_data) == 0 or np.max(np.abs(audio_data)) == 0.0:
        return np.zeros(len(SPECTRAL_FEATURE_NAMES), dtype=np.float32)

    centroid_frames = frames["spectral_centroid"].flatten()
    bandwidth_frames = frames["spectral_bandwidth"].flatten()
    rolloff_frames = frames["spectral_rolloff"].flatten()
    zcr_frames = frames["zero_crossing_rate"].flatten()

    def _safe_mean_std(arr: np.ndarray) -> Tuple[float, float]:
        if len(arr) == 0:
            return 0.0, 0.0
        finite_vals = arr[np.isfinite(arr)]
        if len(finite_vals) == 0:
            return 0.0, 0.0
        return float(np.mean(finite_vals)), float(np.std(finite_vals))

    c_mean, c_std = _safe_mean_std(centroid_frames)
    b_mean, b_std = _safe_mean_std(bandwidth_frames)
    r_mean, r_std = _safe_mean_std(rolloff_frames)
    z_mean, z_std = _safe_mean_std(zcr_frames)

    feature_vec = np.array([
        c_mean, c_std,
        b_mean, b_std,
        r_mean, r_std,
        z_mean, z_std,
    ], dtype=np.float32)

    return np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# Alias for concise API usage
extract_spectral = extract_spectral_features


def extract_spectral_dict(
    audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
    sample_rate: Optional[int] = None,
    config: FeatureConfig = FEATURE_CONFIG,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    roll_percent: float = 0.85,
) -> SpectralFeatureDict:
    """
    Extract fixed-length spectral features and return them as a SpectralFeatureDict.

    Parameters:
        audio: 1D numpy array, PreprocessedAudio, or AudioRecording.
        sample_rate: Audio sampling frequency in Hz.
        config: FeatureConfig settings.
        n_fft: FFT window size.
        hop_length: Hop length.
        roll_percent: Roll-off percentage.

    Returns:
        SpectralFeatureDict mapping feature names to float values.
    """
    features = extract_spectral_features(
        audio=audio,
        sample_rate=sample_rate,
        config=config,
        n_fft=n_fft,
        hop_length=hop_length,
        roll_percent=roll_percent,
    )

    return SpectralFeatureDict({
        "spectral_centroid_mean": float(features[0]),
        "spectral_centroid_std": float(features[1]),
        "spectral_bandwidth_mean": float(features[2]),
        "spectral_bandwidth_std": float(features[3]),
        "spectral_rolloff_mean": float(features[4]),
        "spectral_rolloff_std": float(features[5]),
        "zero_crossing_rate_mean": float(features[6]),
        "zero_crossing_rate_std": float(features[7]),
    })


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

    # --- MFCC Methods ---

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

    # --- Pitch Methods ---

    def get_pitch_feature_names(self) -> List[str]:
        """Get list of pitch feature names."""
        return get_pitch_feature_names()

    def compute_pitch_frames(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute frame-level pitch track and voicing indicators."""
        sr = sample_rate if sample_rate is not None else self.audio_config.target_sample_rate
        return compute_pitch_frames(
            audio_data=audio_data,
            sample_rate=sr,
            config=self.config,
            fmin=fmin,
            fmax=fmax,
        )

    def extract_pitch(
        self,
        audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
        sample_rate: Optional[int] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
    ) -> np.ndarray:
        """Extract fixed-length pitch feature vector."""
        return extract_pitch_features(
            audio=audio,
            sample_rate=sample_rate,
            config=self.config,
            fmin=fmin,
            fmax=fmax,
        )

    def extract_pitch_dict(
        self,
        audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
        sample_rate: Optional[int] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
    ) -> Dict[str, float]:
        """Extract fixed-length pitch features as a named dictionary."""
        return extract_pitch_dict(
            audio=audio,
            sample_rate=sample_rate,
            config=self.config,
            fmin=fmin,
            fmax=fmax,
        )

    # --- Energy Methods ---

    def get_energy_feature_names(self) -> List[str]:
        """Get list of energy feature names."""
        return get_energy_feature_names()

    def compute_energy_frames(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None,
        frame_length: Optional[int] = None,
        hop_length: Optional[int] = None,
    ) -> np.ndarray:
        """Compute frame-level RMS energy contour."""
        sr = sample_rate if sample_rate is not None else self.audio_config.target_sample_rate
        return compute_energy_frames(
            audio_data=audio_data,
            sample_rate=sr,
            config=self.config,
            frame_length=frame_length,
            hop_length=hop_length,
        )

    def extract_energy(
        self,
        audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
        sample_rate: Optional[int] = None,
        frame_length: Optional[int] = None,
        hop_length: Optional[int] = None,
    ) -> np.ndarray:
        """Extract fixed-length energy feature vector."""
        return extract_energy_features(
            audio=audio,
            sample_rate=sample_rate,
            config=self.config,
            frame_length=frame_length,
            hop_length=hop_length,
        )

    def extract_energy_dict(
        self,
        audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
        sample_rate: Optional[int] = None,
        frame_length: Optional[int] = None,
        hop_length: Optional[int] = None,
    ) -> EnergyFeatureDict:
        """Extract fixed-length energy features as an EnergyFeatureDict."""
        return extract_energy_dict(
            audio=audio,
            sample_rate=sample_rate,
            config=self.config,
            frame_length=frame_length,
            hop_length=hop_length,
        )

    # --- Spectral Methods ---

    def get_spectral_feature_names(self) -> List[str]:
        """Get list of spectral feature names."""
        return get_spectral_feature_names()

    def compute_spectral_frames(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        roll_percent: float = 0.85,
    ) -> Dict[str, np.ndarray]:
        """Compute frame-level spectral descriptor matrices."""
        sr = sample_rate if sample_rate is not None else self.audio_config.target_sample_rate
        return compute_spectral_frames(
            audio_data=audio_data,
            sample_rate=sr,
            config=self.config,
            n_fft=n_fft,
            hop_length=hop_length,
            roll_percent=roll_percent,
        )

    def extract_spectral(
        self,
        audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
        sample_rate: Optional[int] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        roll_percent: float = 0.85,
    ) -> np.ndarray:
        """Extract fixed-length spectral feature vector."""
        return extract_spectral_features(
            audio=audio,
            sample_rate=sample_rate,
            config=self.config,
            n_fft=n_fft,
            hop_length=hop_length,
            roll_percent=roll_percent,
        )

    def extract_spectral_dict(
        self,
        audio: Union[np.ndarray, PreprocessedAudio, AudioRecording],
        sample_rate: Optional[int] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        roll_percent: float = 0.85,
    ) -> SpectralFeatureDict:
        """Extract fixed-length spectral features as a SpectralFeatureDict."""
        return extract_spectral_dict(
            audio=audio,
            sample_rate=sample_rate,
            config=self.config,
            n_fft=n_fft,
            hop_length=hop_length,
            roll_percent=roll_percent,
        )
