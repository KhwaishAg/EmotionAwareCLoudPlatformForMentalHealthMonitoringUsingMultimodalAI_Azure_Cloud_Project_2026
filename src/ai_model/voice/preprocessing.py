"""
Voice Modality — Audio Preprocessing Pipeline

Implements the modular, deterministic preprocessing pipeline for voice recordings:
    1. Audio Ingestion: Load from local file, in-memory bytes, or AudioRecording.
    2. Mono Conversion: Downmix multi-channel / stereo audio to single-channel mono.
    3. Amplitude Normalization: Scale waveform peak to target peak (default 0.95).
    4. Clean Waveform Output: Return PreprocessedAudio container ready for downstream stages.

Specifications adhere to:
    - Centralized config: src/ai_model/voice/config.py (AudioConfig)
    - Recording protocol: docs/audio_pipeline.md

Note:
    - Sample rate is preserved as-is in this stage (resampling to 16 kHz is Commit 6).
    - Silence trimming is deferred to Commit 7.
    - Feature extraction and model inference are handled in subsequent modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .config import AUDIO_CONFIG, AudioConfig
from .recording import (
    AudioDataError,
    AudioRecording,
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
        audio_data: 1D numpy array of clean float32 mono audio waveform.
        sample_rate: Audio sampling rate in Hz (preserved from input).
        channels: Number of channels (always 1 for mono).
        duration_seconds: Duration of the waveform in seconds.
        participant_id: Optional anonymous participant identifier (e.g., 'P001').
        metadata: Tracking metadata for preprocessing operations applied.
    """

    audio_data: np.ndarray
    sample_rate: int
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
        4. Normalize amplitude to target peak (config.normalization_peak = 0.95).
        5. Return clean PreprocessedAudio container.

    Note:
        Sample rate is preserved as-is. Resampling is deferred to Commit 6.
        Silence trimming is deferred to Commit 7.

    Parameters:
        audio: AudioRecording instance, file path (str/Path), or raw WAV bytes.
        config: Centralized AudioConfig settings.
        participant_id: Optional anonymous participant identifier to assign.

    Returns:
        PreprocessedAudio container with clean mono waveform.

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
    sample_rate = recording.sample_rate

    # 3. Convert to mono
    mono_waveform = convert_to_mono(recording.audio_data)

    # 4. Amplitude normalization
    normalized_waveform, orig_peak = normalize_amplitude(
        mono_waveform,
        target_peak=config.normalization_peak,
    )

    # 5. Metadata tracking
    duration_seconds = len(normalized_waveform) / float(sample_rate)
    metadata = {
        "original_channels": original_channels,
        "converted_to_mono": (original_channels != 1),
        "original_peak": orig_peak,
        "normalized_peak": float(np.max(np.abs(normalized_waveform))) if len(normalized_waveform) > 0 else 0.0,
        "sample_rate_preserved": sample_rate,
        "normalization_target_peak": config.normalization_peak,
    }

    return PreprocessedAudio(
        audio_data=normalized_waveform,
        sample_rate=sample_rate,
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
