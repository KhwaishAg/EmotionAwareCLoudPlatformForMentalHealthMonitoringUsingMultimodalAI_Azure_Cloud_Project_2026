"""
Voice Modality — Audio Preprocessing Pipeline

Implements the modular, deterministic preprocessing pipeline for voice recordings:
    1. Audio Ingestion: Load from local file, in-memory bytes, or AudioRecording.
    2. Mono Conversion: Downmix multi-channel / stereo audio to single-channel mono.
    3. 16 kHz Resampling: Polyphase resampling to 16,000 Hz if sample rate differs.
    4. Silence Handling: Detect and trim leading/trailing silence using RMS threshold.
    5. Amplitude Normalization: Scale waveform peak to target peak (default 0.95).
    6. Clean Waveform Output: Return PreprocessedAudio container ready for downstream stages.

Specifications adhere to:
    - Centralized config: src/ai_model/voice/config.py (AudioConfig)
    - Recording protocol: docs/audio_pipeline.md
    - Target: WAV, mono, 16 kHz, ~30-60 seconds.

Note:
    - Feature extraction and model inference are handled in subsequent modules.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal

from .config import AUDIO_CONFIG, AudioConfig
from .recording import (
    AudioDataError,
    AudioRecording,
    AudioSampleRateError,
    load_wav_bytes,
    load_wav_file,
)


# ===========================================================================
# Preprocessed Audio Data Structure
# ===========================================================================

@dataclass
class PreprocessedAudio:
    """
    Standardized clean waveform container produced by the preprocessing pipeline.

    Attributes:
        audio_data: 1D numpy array of clean float32 mono 16 kHz audio waveform.
        sample_rate: Audio sampling rate in Hz (standardized to 16,000 Hz).
        channels: Number of channels (always 1 for mono).
        duration_seconds: Duration of the waveform in seconds.
        participant_id: Optional anonymous participant identifier (e.g., 'P001').
        metadata: Tracking metadata for preprocessing operations applied.
    """

    audio_data: np.ndarray
    sample_rate: int = 16_000
    channels: int = 1
    duration_seconds: float = 0.0
    participant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure 1D shape and float32 dtype
        if self.audio_data.ndim != 1:
            raise AudioDataError(
                f"Preprocessed audio waveform must be 1-dimensional (got shape {self.audio_data.shape})."
            )
        if self.audio_data.dtype != np.float32:
            self.audio_data = self.audio_data.astype(np.float32)

        if self.duration_seconds <= 0.0 and self.sample_rate > 0:
            self.duration_seconds = len(self.audio_data) / float(self.sample_rate)

    @property
    def num_samples(self) -> int:
        """Total number of waveform sample frames."""
        return len(self.audio_data)

    @property
    def peak_amplitude(self) -> float:
        """Maximum absolute amplitude of the waveform."""
        if len(self.audio_data) == 0:
            return 0.0
        return float(np.max(np.abs(self.audio_data)))

    @property
    def is_silent(self) -> bool:
        """Check if the waveform consists entirely of zero amplitude."""
        return self.peak_amplitude == 0.0

    def to_recording(self, source_path: Optional[str] = None) -> AudioRecording:
        """
        Convert this PreprocessedAudio back into an AudioRecording container.
        """
        return AudioRecording(
            audio_data=self.audio_data,
            sample_rate=self.sample_rate,
            channels=self.channels,
            duration_seconds=self.duration_seconds,
            participant_id=self.participant_id,
            source_path=source_path,
            metadata=dict(self.metadata),
        )


# ===========================================================================
# Modular Preprocessing Operations
# ===========================================================================

def convert_to_mono(audio_data: np.ndarray) -> np.ndarray:
    """
    Convert an audio array to a single-channel (mono) float32 waveform.

    - If 1D: returned as a 1D float32 array.
    - If 2D with single column (N, 1): squeezed to 1D.
    - If 2D multi-channel (N, C): channels are averaged across axis 1 (arithmetic mean downmix).

    Parameters:
        audio_data: Numpy array of audio waveform.

    Returns:
        1D float32 numpy array.

    Raises:
        AudioDataError: If audio_data is empty, contains non-finite values, or has invalid dimensions.
    """
    if audio_data is None or audio_data.size == 0:
        raise AudioDataError("Cannot convert empty audio array to mono.")

    if not np.all(np.isfinite(audio_data)):
        raise AudioDataError("Audio waveform contains non-finite values (NaN or Inf).")

    if audio_data.ndim == 1:
        return audio_data.astype(np.float32, copy=True)
    elif audio_data.ndim == 2:
        num_channels = audio_data.shape[1]
        if num_channels == 1:
            return audio_data[:, 0].astype(np.float32, copy=True)
        else:
            # Multi-channel downmix: average all channels
            return np.mean(audio_data, axis=1, dtype=np.float32)
    else:
        raise AudioDataError(
            f"Unsupported audio array dimension {audio_data.ndim}. Expected 1D or 2D array."
        )


def resample_audio(
    audio_data: np.ndarray,
    orig_sample_rate: int,
    target_sample_rate: int = AUDIO_CONFIG.target_sample_rate,
) -> Tuple[np.ndarray, bool]:
    """
    Resample a 1D audio waveform to target_sample_rate (default 16 kHz).

    - Only resamples when orig_sample_rate != target_sample_rate.
    - If orig_sample_rate == target_sample_rate, returns original waveform without resampling.
    - Uses polyphase filtering (scipy.signal.resample_poly) with anti-aliasing filter
      to preserve audio quality.
    - Falls back to FFT-based sinc interpolation if polyphase factor is extreme.

    Parameters:
        audio_data: 1D numpy array of audio samples.
        orig_sample_rate: Original sampling frequency in Hz.
        target_sample_rate: Target sampling frequency in Hz (default: 16,000 Hz).

    Returns:
        Tuple of (resampled_1d_float32_array, was_resampled_boolean).

    Raises:
        AudioSampleRateError: If orig_sample_rate or target_sample_rate is non-positive.
        AudioDataError: If audio_data is empty or contains non-finite values.
    """
    if orig_sample_rate <= 0:
        raise AudioSampleRateError(
            f"Original sample rate must be a positive integer > 0 (got {orig_sample_rate})."
        )
    if target_sample_rate <= 0:
        raise AudioSampleRateError(
            f"Target sample rate must be a positive integer > 0 (got {target_sample_rate})."
        )

    if audio_data is None or audio_data.size == 0:
        raise AudioDataError("Cannot resample empty audio array.")

    if not np.all(np.isfinite(audio_data)):
        raise AudioDataError("Audio waveform contains non-finite values (NaN or Inf).")

    # Fast passthrough: no resampling needed if rates match
    if orig_sample_rate == target_sample_rate:
        return audio_data.astype(np.float32, copy=True), False

    # Ensure 1D
    if audio_data.ndim != 1:
        raise AudioDataError(f"resample_audio expects 1D array, got shape {audio_data.shape}.")

    # Rational polyphase resampling
    gcd = math.gcd(target_sample_rate, orig_sample_rate)
    up = target_sample_rate // gcd
    down = orig_sample_rate // gcd

    try:
        resampled = signal.resample_poly(audio_data, up, down).astype(np.float32)
    except Exception:
        # Fallback to FFT-based sinc interpolation
        target_length = int(round(len(audio_data) * target_sample_rate / orig_sample_rate))
        resampled = signal.resample(audio_data, target_length).astype(np.float32)

    return resampled, True


def trim_silence(
    audio_data: np.ndarray,
    sample_rate: int = AUDIO_CONFIG.target_sample_rate,
    top_db: int = AUDIO_CONFIG.silence_top_db,
    frame_length: int = 512,
    min_duration_after_trim: float = AUDIO_CONFIG.min_duration_after_trim,
) -> Tuple[np.ndarray, bool, int, int]:
    """
    Trim leading and trailing silence from a 1D audio waveform using an RMS energy threshold.

    - Detects speech onset and offset using frame-level RMS energy relative to peak energy.
    - Preserves internal pauses (hesitations, phrase gaps) essential for stress assessment.
    - Safely handles completely silent audio by returning the original array untrimmed.
    - Enforces min_duration_after_trim to prevent over-trimming meaningful low-energy speech.

    Parameters:
        audio_data: 1D numpy array of float32 audio waveform.
        sample_rate: Audio sampling frequency in Hz (default: 16,000 Hz).
        top_db: Decibel threshold below peak RMS to classify as silence (default: 25 dB).
        frame_length: Analysis window size in samples (default: 512).
        min_duration_after_trim: Minimum duration in seconds required after trimming (default: 3.0s).

    Returns:
        Tuple of (trimmed_waveform, was_trimmed_bool, leading_samples_count, trailing_samples_count).

    Raises:
        AudioDataError: If audio_data is empty or contains non-finite values.
    """
    if audio_data is None or audio_data.size == 0:
        raise AudioDataError("Cannot trim silence from empty audio array.")

    if not np.all(np.isfinite(audio_data)):
        raise AudioDataError("Audio waveform contains non-finite values (NaN or Inf).")

    if audio_data.ndim != 1:
        raise AudioDataError(f"trim_silence expects 1D array, got shape {audio_data.shape}.")

    peak = float(np.max(np.abs(audio_data)))
    if peak == 0.0:
        # Completely silent audio: return untrimmed safely
        return audio_data.astype(np.float32, copy=True), False, 0, 0

    # Moving average frame-level RMS calculation
    win_len = min(frame_length, len(audio_data))
    window = np.ones(win_len, dtype=np.float32) / win_len
    squared = audio_data ** 2
    rms = np.sqrt(np.convolve(squared, window, mode="same"))

    max_rms = float(np.max(rms))
    if max_rms == 0.0:
        return audio_data.astype(np.float32, copy=True), False, 0, 0

    # Decibel threshold relative to max RMS
    threshold = max_rms * (10.0 ** (-float(top_db) / 20.0))
    non_silent_indices = np.where(rms >= threshold)[0]

    if len(non_silent_indices) == 0:
        return audio_data.astype(np.float32, copy=True), False, 0, 0

    # Expand slightly by half-window to avoid clipping speech onsets/offsets
    start_idx = max(0, int(non_silent_indices[0]) - win_len // 2)
    end_idx = min(len(audio_data), int(non_silent_indices[-1]) + win_len // 2)

    trimmed_length = end_idx - start_idx
    min_samples = int(min_duration_after_trim * sample_rate)

    # Protect against removing meaningful speech below min_duration_after_trim
    if trimmed_length < min_samples:
        if len(audio_data) >= min_samples:
            # Expand around speech center to preserve at least min_samples
            center = (start_idx + end_idx) // 2
            half_min = min_samples // 2
            start_idx = max(0, center - half_min)
            end_idx = min(len(audio_data), start_idx + min_samples)
            if end_idx - start_idx < min_samples and start_idx > 0:
                start_idx = max(0, end_idx - min_samples)
        else:
            return audio_data.astype(np.float32, copy=True), False, 0, 0

    leading_samples = start_idx
    trailing_samples = len(audio_data) - end_idx

    was_trimmed = (leading_samples > 0 or trailing_samples > 0)
    trimmed_waveform = audio_data[start_idx:end_idx].astype(np.float32, copy=True)

    return trimmed_waveform, was_trimmed, leading_samples, trailing_samples


def normalize_amplitude(
    audio_data: np.ndarray,
    target_peak: float = 0.95,
) -> Tuple[np.ndarray, float]:
    """
    Normalize waveform amplitude so the peak absolute amplitude equals target_peak.

    - Deterministic and linear: preserves waveform shape and dynamic variation.
    - Safely handles silent audio (all zeros) without division by zero.
    - Ensures the resulting peak does not exceed 1.0.

    Parameters:
        audio_data: 1D or 2D float32 audio array.
        target_peak: Target peak amplitude in range (0.0, 1.0]. Default is 0.95.

    Returns:
        Tuple of (normalized_audio_array, original_peak_amplitude).

    Raises:
        AudioDataError: If audio_data is empty, contains non-finite values, or target_peak is invalid.
    """
    if audio_data is None or audio_data.size == 0:
        raise AudioDataError("Cannot normalize empty audio array.")

    if not np.all(np.isfinite(audio_data)):
        raise AudioDataError("Audio waveform contains non-finite values (NaN or Inf).")

    if not (0.0 < target_peak <= 1.0):
        raise AudioDataError(f"Target peak must be in (0.0, 1.0], got {target_peak}.")

    original_peak = float(np.max(np.abs(audio_data)))

    # Silent / flatline audio protection
    if original_peak == 0.0:
        return audio_data.astype(np.float32, copy=True), 0.0

    scale_factor = target_peak / original_peak
    normalized = (audio_data * scale_factor).astype(np.float32)

    return normalized, original_peak


# ===========================================================================
# Full Pipeline Execution
# ===========================================================================

def preprocess_audio(
    audio: Union[AudioRecording, str, Path, bytes],
    config: AudioConfig = AUDIO_CONFIG,
    participant_id: Optional[str] = None,
) -> PreprocessedAudio:
    """
    Execute the Voice preprocessing pipeline:
        1. Ingest audio from AudioRecording, file path, or byte buffer.
        2. Validate audio structure and contents.
        3. Convert audio to mono (averaging channels if stereo/multichannel).
        4. Resample audio to 16 kHz (config.target_sample_rate) if different.
        5. Silence handling: Trim leading and trailing silence (RMS threshold).
        6. Normalize amplitude to target peak (config.normalization_peak = 0.95).
        7. Return clean PreprocessedAudio container ready for downstream stages.

    Parameters:
        audio: AudioRecording instance, file path (str/Path), or raw WAV bytes.
        config: Centralized AudioConfig settings.
        participant_id: Optional anonymous participant identifier to assign.

    Returns:
        PreprocessedAudio container with clean mono 16 kHz waveform.

    Raises:
        AudioDataError: If audio is corrupt, empty, or contains non-finite values.
        AudioNotFoundError: If audio file path does not exist.
        AudioFormatError: If file format is unsupported.
    """
    # 1. Ingest audio into AudioRecording
    if isinstance(audio, AudioRecording):
        recording = audio
        pid = participant_id or recording.participant_id
    elif isinstance(audio, (str, Path)):
        recording = load_wav_file(audio, config)
        pid = participant_id or recording.participant_id
    elif isinstance(audio, (bytes, bytearray)):
        recording = load_wav_bytes(bytes(audio), participant_id=participant_id, config=config)
        pid = participant_id or recording.participant_id
    else:
        raise TypeError(
            f"Unsupported input type for preprocess_audio: {type(audio)}. "
            "Expected AudioRecording, str, Path, or bytes."
        )

    # 2. Validate basic audio integrity
    if recording.num_samples == 0 or recording.audio_data.size == 0:
        raise AudioDataError("Audio contains zero samples and cannot be preprocessed.")

    if not np.all(np.isfinite(recording.audio_data)):
        raise AudioDataError("Audio contains non-finite values (NaN or Inf) and cannot be preprocessed.")

    original_channels = recording.channels
    orig_sample_rate = recording.sample_rate

    # 3. Convert to mono
    mono_waveform = convert_to_mono(recording.audio_data)

    # 4. Resample to 16 kHz if sample rate differs
    resampled_waveform, was_resampled = resample_audio(
        mono_waveform,
        orig_sample_rate=orig_sample_rate,
        target_sample_rate=config.target_sample_rate,
    )

    # 5. Silence handling: trim leading and trailing silence (after resampling, before normalization)
    trimmed_waveform, silence_trimmed, leading_samples, trailing_samples = trim_silence(
        resampled_waveform,
        sample_rate=config.target_sample_rate,
        top_db=config.silence_top_db,
        min_duration_after_trim=config.min_duration_after_trim,
    )

    # 6. Amplitude normalization
    normalized_waveform, orig_peak = normalize_amplitude(
        trimmed_waveform,
        target_peak=config.normalization_peak,
    )

    # 7. Metadata tracking
    duration_seconds = len(normalized_waveform) / float(config.target_sample_rate)
    metadata = {
        "original_channels": original_channels,
        "converted_to_mono": (original_channels != 1),
        "original_sample_rate": orig_sample_rate,
        "target_sample_rate": config.target_sample_rate,
        "was_resampled": was_resampled,
        "silence_trimmed": silence_trimmed,
        "leading_silence_samples": leading_samples,
        "trailing_silence_samples": trailing_samples,
        "original_peak": orig_peak,
        "normalized_peak": float(np.max(np.abs(normalized_waveform))) if len(normalized_waveform) > 0 else 0.0,
        "normalization_target_peak": config.normalization_peak,
    }

    return PreprocessedAudio(
        audio_data=normalized_waveform,
        sample_rate=config.target_sample_rate,
        channels=1,
        duration_seconds=duration_seconds,
        participant_id=pid,
        metadata=metadata,
    )


# ===========================================================================
# Preprocessor Class Abstraction
# ===========================================================================

class VoicePreprocessor:
    """
    Reusable controller for the Voice audio preprocessing pipeline.

    Encapsulates preprocessing configurations and exposes modular methods
    as well as full pipeline execution.
    """

    def __init__(self, config: AudioConfig = AUDIO_CONFIG):
        self.config = config

    def to_mono(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert audio waveform to mono."""
        return convert_to_mono(audio_data)

    def resample(
        self,
        audio_data: np.ndarray,
        orig_sample_rate: int,
        target_sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, bool]:
        """Resample audio waveform to target sample rate (default 16 kHz)."""
        target = target_sample_rate if target_sample_rate is not None else self.config.target_sample_rate
        return resample_audio(audio_data, orig_sample_rate, target_sample_rate=target)

    def trim_silence(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None,
        top_db: Optional[int] = None,
        min_duration_after_trim: Optional[float] = None,
    ) -> Tuple[np.ndarray, bool, int, int]:
        """Trim leading and trailing silence from audio waveform."""
        sr = sample_rate if sample_rate is not None else self.config.target_sample_rate
        db = top_db if top_db is not None else self.config.silence_top_db
        min_dur = (
            min_duration_after_trim
            if min_duration_after_trim is not None
            else self.config.min_duration_after_trim
        )
        return trim_silence(audio_data, sample_rate=sr, top_db=db, min_duration_after_trim=min_dur)

    def normalize(
        self,
        audio_data: np.ndarray,
        target_peak: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Normalize audio waveform peak amplitude."""
        peak = target_peak if target_peak is not None else self.config.normalization_peak
        return normalize_amplitude(audio_data, target_peak=peak)

    def process(
        self,
        audio: Union[AudioRecording, str, Path, bytes],
        participant_id: Optional[str] = None,
    ) -> PreprocessedAudio:
        """
        Run the complete preprocessing pipeline on an audio input.
        """
        return preprocess_audio(audio, config=self.config, participant_id=participant_id)
