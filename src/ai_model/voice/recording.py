"""
Voice Modality — Audio Recording & Input Interface

Provides the functional Python audio input interface for the voice modality.
Handles local WAV file ingestion, in-memory audio byte loading (for frontend/API
integration), target specification verification, and standardized recording prompt
delivery.

Design:
    - AudioRecording        : Structured representation of loaded/recorded audio.
    - AudioInputSource      : Abstract base class for extensible input sources.
    - LocalFileInputSource  : Audio source from local filesystem.
    - BufferInputSource     : Audio source from in-memory byte buffer (frontend/REST API).
    - VoiceRecordingInterface: Unified controller for audio ingestion, verification,
                              and storage.

Specifications adhere to:
    - Centralized config: src/ai_model/voice/config.py (AudioConfig)
    - Recording protocol: docs/audio_pipeline.md
    - Target: WAV, mono, 16 kHz, ~30-60 seconds.

Note:
    Preprocessing (resampling, silence trimming, normalization) and acoustic
    feature extraction are handled in separate downstream pipeline stages.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import io
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import scipy.io.wavfile as wavfile

from .config import AUDIO_CONFIG, AudioConfig


# ===========================================================================
# Standardized Prompt Constant (from docs/audio_pipeline.md)
# ===========================================================================

STANDARDIZED_RECORDING_PROMPT = (
    "Please describe your recent academic workload, your typical college day, "
    "and how you have been feeling about your studies."
)


# ===========================================================================
# Exceptions
# ===========================================================================

class AudioError(Exception):
    """Base exception for all audio input, recording, and validation errors."""
    pass


class AudioNotFoundError(AudioError, FileNotFoundError):
    """Raised when an audio file cannot be found at the specified path."""
    pass


class AudioFormatError(AudioError, ValueError):
    """Raised when an audio file extension or format is unsupported."""
    pass


class AudioCorruptError(AudioError, ValueError):
    """Raised when an audio file or stream is empty, truncated, or unreadable."""
    pass


class AudioValidationError(AudioError, ValueError):
    """Raised when an audio recording fails pipeline validation constraints."""
    pass


class AudioDurationError(AudioValidationError):
    """Raised when audio duration falls outside acceptable pipeline limits (5-120s)."""
    pass


class AudioChannelError(AudioValidationError):
    """Raised when audio channel count is invalid or unsupported."""
    pass


class AudioSampleRateError(AudioValidationError):
    """Raised when audio sample rate is invalid or unsupported."""
    pass


class AudioDataError(AudioValidationError):
    """Raised when audio waveform data is invalid (e.g. non-finite, NaN, Inf, empty)."""
    pass


# ===========================================================================
# Participant ID Helper
# ===========================================================================

# Anonymous participant pattern: e.g., P001, P042, P1000
PARTICIPANT_ID_PATTERN = re.compile(r"^(P\d{3,})", re.IGNORECASE)


def extract_participant_id(identifier_or_path: Union[str, Path]) -> Optional[str]:
    """
    Extract an anonymous participant identifier (e.g. 'P001') from a filename
    or identifier string.

    Parameters:
        identifier_or_path: Filename, path, or string to parse.

    Returns:
        The uppercase anonymous ID (e.g., 'P001') if matched, else None.
    """
    stem = Path(identifier_or_path).stem
    match = PARTICIPANT_ID_PATTERN.match(stem)
    if match:
        return match.group(1).upper()
    return None


# ===========================================================================
# Audio Validation Result
# ===========================================================================

class ValidationResult(dict):
    """
    Structured outcome of validating an audio recording or file against pipeline constraints.

    Subclasses dict to preserve backward compatibility with dictionary-style
    key access (e.g., result['is_valid']), while offering attribute access
    (e.g., result.is_valid) and error-raising helpers.
    """

    def __init__(
        self,
        is_valid: bool,
        is_target_spec: bool,
        meets_recommended_duration: bool,
        is_target_sample_rate: bool,
        is_mono: bool,
        has_valid_data: bool,
        is_silent: bool,
        sample_rate: int,
        channels: int,
        duration_seconds: float,
        num_samples: int,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        errors = errors or []
        warnings = warnings or []
        data = {
            "is_valid": is_valid,
            "is_target_spec": is_target_spec,
            "meets_recommended_duration": meets_recommended_duration,
            "is_target_sample_rate": is_target_sample_rate,
            "is_mono": is_mono,
            "has_valid_data": has_valid_data,
            "is_silent": is_silent,
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_seconds": round(duration_seconds, 3),
            "num_samples": num_samples,
            "errors": errors,
            "warnings": warnings,
            **kwargs,
        }
        super().__init__(data)
        self.__dict__.update(data)

    def raise_if_invalid(self) -> None:
        """
        Raise a typed AudioValidationError subclass if validation failed.
        """
        if not self["is_valid"]:
            errors = self.get("errors", [])
            msg = "; ".join(errors) if errors else "Audio validation failed."
            for err in errors:
                err_lower = err.lower()
                if "duration" in err_lower:
                    raise AudioDurationError(err)
                if "channel" in err_lower:
                    raise AudioChannelError(err)
                if "sample rate" in err_lower:
                    raise AudioSampleRateError(err)
                if any(kw in err_lower for kw in ("nan", "inf", "sample", "data", "dimension", "shape", "empty")):
                    raise AudioDataError(err)
            raise AudioValidationError(msg)


# ===========================================================================
# Audio Recording Data Structure
# ===========================================================================

@dataclass
class AudioRecording:
    """
    Structured representation of a loaded or recorded audio sample.

    Attributes:
        audio_data: Numpy array of audio waveform (normalized float32 in [-1.0, 1.0]).
                    Shape is (num_samples,) for mono, (num_samples, channels) for stereo.
        sample_rate: Audio sampling frequency in Hz.
        channels: Number of audio channels (1 = mono, 2 = stereo).
        duration_seconds: Duration of the audio in seconds.
        participant_id: Optional anonymous participant identifier (e.g., 'P001').
        source_path: Path of the source file if loaded from disk, else None.
        metadata: Additional metadata dictionary (bit depth, frame count, etc.).
    """

    audio_data: np.ndarray
    sample_rate: int
    channels: int
    duration_seconds: float
    participant_id: Optional[str] = None
    source_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        """Total number of audio sample frames."""
        return len(self.audio_data)

    @property
    def is_mono(self) -> bool:
        """Check if audio is single-channel (mono)."""
        return self.channels == 1

    def is_target_spec(self, config: AudioConfig = AUDIO_CONFIG) -> bool:
        """
        Check if recording satisfies the ideal target pipeline specifications:
        mono, 16 kHz sample rate, within recommended 30-60 second duration,
        and valid finite waveform data.
        """
        return (
            self.channels == config.target_channels
            and self.sample_rate == config.target_sample_rate
            and config.recommended_min_duration <= self.duration_seconds <= config.recommended_max_duration
            and self.num_samples > 0
            and bool(np.all(np.isfinite(self.audio_data)))
        )

    def validate(
        self,
        config: AudioConfig = AUDIO_CONFIG,
        strict: bool = False,
    ) -> ValidationResult:
        """
        Validate the recording against configured pipeline constraints.

        Checks:
            - Waveform data integrity (non-empty, finite values, no NaN/Inf)
            - Channel count (valid positive count, mono target)
            - Sample rate (valid positive rate, 16 kHz target)
            - Duration limits (5-120s acceptable bounds, 30-60s recommended)
            - Silence detection and amplitude range

        Parameters:
            config: Centralized AudioConfig settings.
            strict: If True, raises a specific AudioValidationError if invalid.

        Returns:
            ValidationResult with detailed flags, errors, and warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Waveform Data Integrity
        has_valid_data = True
        if self.num_samples == 0 or self.audio_data.size == 0:
            errors.append("Audio contains zero samples.")
            has_valid_data = False
        elif not np.all(np.isfinite(self.audio_data)):
            errors.append("Audio data contains non-finite values (NaN or Inf).")
            has_valid_data = False

        # 2. Channel Count
        if self.channels <= 0:
            errors.append(f"Channel count must be a positive integer > 0 (got {self.channels}).")
        elif self.channels > 8:
            errors.append(f"Channel count ({self.channels}) exceeds supported multichannel limit (8).")

        # Dimensional consistency between array and declared channels
        if has_valid_data and self.channels > 0:
            if self.channels == 1:
                if self.audio_data.ndim != 1 and not (self.audio_data.ndim == 2 and self.audio_data.shape[1] == 1):
                    errors.append(f"Audio array shape {self.audio_data.shape} does not match mono channel count.")
                    has_valid_data = False
            else:
                if self.audio_data.ndim != 2 or self.audio_data.shape[1] != self.channels:
                    errors.append(
                        f"Audio array shape {self.audio_data.shape} does not match expected {self.channels} channels."
                    )
                    has_valid_data = False

        if self.channels > 0 and self.channels != config.target_channels:
            warnings.append(
                f"Audio has {self.channels} channels; target is "
                f"{config.target_channels} channel (mono conversion will be required in preprocessing)."
            )

        # 3. Sample Rate
        if self.sample_rate <= 0:
            errors.append(f"Sample rate must be a positive integer > 0 (got {self.sample_rate} Hz).")
        elif self.sample_rate < 8000:
            warnings.append(
                f"Sample rate ({self.sample_rate} Hz) is very low for speech analysis; "
                f"target is {config.target_sample_rate} Hz."
            )

        if self.sample_rate > 0 and self.sample_rate != config.target_sample_rate:
            warnings.append(
                f"Sample rate is {self.sample_rate} Hz; target is "
                f"{config.target_sample_rate} Hz (resampling will be required in preprocessing)."
            )

        # 4. Duration Limits (5-120s acceptable, 30-60s recommended)
        if self.duration_seconds < config.min_duration_seconds:
            errors.append(
                f"Duration ({self.duration_seconds:.2f}s) is below the minimum acceptable "
                f"threshold ({config.min_duration_seconds:.1f}s)."
            )
        elif self.duration_seconds > config.max_duration_seconds:
            errors.append(
                f"Duration ({self.duration_seconds:.2f}s) exceeds the maximum allowed "
                f"duration ({config.max_duration_seconds:.1f}s)."
            )
        elif self.duration_seconds < config.recommended_min_duration:
            warnings.append(
                f"Duration ({self.duration_seconds:.2f}s) is acceptable but shorter than recommended "
                f"({config.recommended_min_duration:.1f}-{config.recommended_max_duration:.1f}s)."
            )
        elif self.duration_seconds > config.recommended_max_duration:
            warnings.append(
                f"Duration ({self.duration_seconds:.2f}s) is acceptable but longer than recommended "
                f"({config.recommended_min_duration:.1f}-{config.recommended_max_duration:.1f}s)."
            )

        # 5. Silence & Amplitude Checks
        is_silent = False
        if has_valid_data and self.num_samples > 0:
            peak_amplitude = float(np.max(np.abs(self.audio_data)))
            if peak_amplitude == 0.0:
                is_silent = True
                warnings.append("Audio waveform contains only silence (all zero values).")
            elif peak_amplitude > 1.001:
                warnings.append(
                    f"Audio waveform exhibits amplitude clipping/over-range (peak: {peak_amplitude:.3f} > 1.0)."
                )

        # Overall validity
        is_valid = len(errors) == 0
        meets_recommended = (
            is_valid
            and config.recommended_min_duration <= self.duration_seconds <= config.recommended_max_duration
        )
        is_target_spec = (
            is_valid
            and meets_recommended
            and self.channels == config.target_channels
            and self.sample_rate == config.target_sample_rate
        )

        result = ValidationResult(
            is_valid=is_valid,
            is_target_spec=is_target_spec,
            meets_recommended_duration=meets_recommended,
            is_target_sample_rate=(self.sample_rate == config.target_sample_rate),
            is_mono=self.is_mono,
            has_valid_data=has_valid_data,
            is_silent=is_silent,
            sample_rate=self.sample_rate,
            channels=self.channels,
            duration_seconds=self.duration_seconds,
            num_samples=self.num_samples,
            errors=errors,
            warnings=warnings,
        )

        if strict:
            result.raise_if_invalid()

        return result



# ===========================================================================
# Audio Normalization & Array Helper
# ===========================================================================

def _to_float32_waveform(raw_array: np.ndarray) -> np.ndarray:
    """
    Convert raw integer or float audio array from wav reader to float32 in [-1.0, 1.0].
    Preserves channel structure without modifying sample rate or channel count.
    """
    dtype = raw_array.dtype

    if np.issubdtype(dtype, np.integer):
        if dtype == np.int16:
            return raw_array.astype(np.float32) / 32768.0
        elif dtype == np.int32:
            return raw_array.astype(np.float32) / 2147483648.0
        elif dtype == np.uint8:
            # 8-bit unsigned PCM (128 is center)
            return (raw_array.astype(np.float32) - 128.0) / 128.0
        else:
            max_val = float(np.iinfo(dtype).max)
            return raw_array.astype(np.float32) / max_val
    elif np.issubdtype(dtype, np.floating):
        return raw_array.astype(np.float32)

    return raw_array.astype(np.float32)


# ===========================================================================
# Core Audio Loading Functions
# ===========================================================================

def load_wav_file(
    file_path: Union[str, Path],
    config: AudioConfig = AUDIO_CONFIG,
) -> AudioRecording:
    """
    Load an existing local WAV audio file into an AudioRecording container.

    Parameters:
        file_path: Absolute or relative path to the local WAV file.
        config: Centralized AudioConfig settings.

    Returns:
        AudioRecording instance with loaded audio data, sample rate, channels,
        duration, participant ID, and metadata.

    Raises:
        AudioNotFoundError: If the file does not exist or path is not a file.
        AudioFormatError: If the file extension is not supported by config.
        AudioCorruptError: If the file is empty or cannot be decoded as a valid WAV.
    """
    path = Path(file_path).resolve()

    # 1. Existence check
    if not path.exists():
        raise AudioNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise AudioNotFoundError(f"Specified path is not a file: {path}")

    # 2. Format / extension validation
    ext = path.suffix.lower()
    valid_exts = tuple(e.lower() for e in config.supported_extensions)
    if ext not in valid_exts:
        raise AudioFormatError(
            f"Unsupported audio format '{path.suffix}'. "
            f"Supported extensions: {config.supported_extensions}"
        )

    # 3. Non-empty check
    file_size = path.stat().st_size
    if file_size == 0:
        raise AudioCorruptError(f"Audio file is empty (0 bytes): {path}")

    # 4. Decode WAV
    try:
        sample_rate, raw_data = wavfile.read(str(path))
    except Exception as exc:
        raise AudioCorruptError(f"Failed to read WAV file '{path}': {exc}") from exc

    if raw_data.size == 0 or sample_rate <= 0:
        raise AudioCorruptError(f"Audio file contains zero samples or invalid sample rate: {path}")

    # 5. Extract dimensions
    channels = 1 if raw_data.ndim == 1 else raw_data.shape[1]
    num_samples = raw_data.shape[0]
    duration_seconds = float(num_samples) / float(sample_rate)

    # 6. Normalize to float32 waveform
    audio_data = _to_float32_waveform(raw_data)

    # 7. Extract participant ID if available in filename
    participant_id = extract_participant_id(path)

    metadata = {
        "file_size_bytes": file_size,
        "raw_dtype": str(raw_data.dtype),
        "num_samples": num_samples,
    }

    return AudioRecording(
        audio_data=audio_data,
        sample_rate=int(sample_rate),
        channels=int(channels),
        duration_seconds=duration_seconds,
        participant_id=participant_id,
        source_path=str(path),
        metadata=metadata,
    )


def load_wav_bytes(
    data: bytes,
    participant_id: Optional[str] = None,
    config: AudioConfig = AUDIO_CONFIG,
) -> AudioRecording:
    """
    Load a WAV audio recording from an in-memory byte buffer.
    Designed for browser frontend uploads and API endpoints.

    Parameters:
        data: Raw bytes representing a complete WAV file.
        participant_id: Optional anonymous participant ID to attach.
        config: Centralized AudioConfig settings.

    Returns:
        AudioRecording instance.

    Raises:
        AudioCorruptError: If the byte buffer is empty or cannot be parsed as WAV.
    """
    if not data or len(data) == 0:
        raise AudioCorruptError("Audio byte buffer is empty (0 bytes).")

    buffer = io.BytesIO(data)
    try:
        sample_rate, raw_data = wavfile.read(buffer)
    except Exception as exc:
        raise AudioCorruptError(f"Failed to parse audio bytes as valid WAV: {exc}") from exc

    if raw_data.size == 0 or sample_rate <= 0:
        raise AudioCorruptError("Audio byte stream contains zero samples or invalid sample rate.")

    channels = 1 if raw_data.ndim == 1 else raw_data.shape[1]
    num_samples = raw_data.shape[0]
    duration_seconds = float(num_samples) / float(sample_rate)
    audio_data = _to_float32_waveform(raw_data)

    metadata = {
        "file_size_bytes": len(data),
        "raw_dtype": str(raw_data.dtype),
        "num_samples": num_samples,
    }

    return AudioRecording(
        audio_data=audio_data,
        sample_rate=int(sample_rate),
        channels=int(channels),
        duration_seconds=duration_seconds,
        participant_id=participant_id,
        source_path=None,
        metadata=metadata,
    )


def save_wav_file(
    recording: AudioRecording,
    output_path: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """
    Save an AudioRecording to disk as a standard 16-bit PCM WAV file.

    Parameters:
        recording: AudioRecording to save.
        output_path: Destination file path.
        overwrite: If False and destination file exists, raises FileExistsError.

    Returns:
        Resolved Path of saved WAV file.
    """
    path = Path(output_path).resolve()

    if path.exists() and not overwrite:
        raise FileExistsError(f"Target file already exists and overwrite=False: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert float32 back to 16-bit signed PCM
    pcm_data = np.clip(recording.audio_data * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(str(path), recording.sample_rate, pcm_data)

    return path


# ===========================================================================
# Validation Functions
# ===========================================================================

def validate_audio_recording(
    recording: AudioRecording,
    config: AudioConfig = AUDIO_CONFIG,
    strict: bool = False,
) -> ValidationResult:
    """
    Validate an in-memory AudioRecording against pipeline constraints.

    Parameters:
        recording: AudioRecording to validate.
        config: Centralized AudioConfig settings.
        strict: If True, raises a typed AudioValidationError on failure.

    Returns:
        ValidationResult with validation flags and diagnostics.
    """
    return recording.validate(config=config, strict=strict)


def validate_audio_file(
    file_path: Union[str, Path],
    config: AudioConfig = AUDIO_CONFIG,
    strict: bool = False,
) -> ValidationResult:
    """
    Validate a local audio file against format, corruptibility, and pipeline constraints.

    Parameters:
        file_path: Absolute or relative path to the local audio file.
        config: Centralized AudioConfig settings.
        strict: If True, raises typed exceptions on failure.

    Returns:
        ValidationResult with validation flags, errors, and warnings.
    """
    path = Path(file_path).resolve()

    # 1. Existence check
    if not path.exists() or not path.is_file():
        err = f"Audio file not found: {path}"
        if strict:
            raise AudioNotFoundError(err)
        return ValidationResult(
            is_valid=False,
            is_target_spec=False,
            meets_recommended_duration=False,
            is_target_sample_rate=False,
            is_mono=False,
            has_valid_data=False,
            is_silent=False,
            sample_rate=0,
            channels=0,
            duration_seconds=0.0,
            num_samples=0,
            errors=[err],
            warnings=[],
        )

    # 2. Format / extension validation
    ext = path.suffix.lower()
    valid_exts = tuple(e.lower() for e in config.supported_extensions)
    if ext not in valid_exts:
        err = (
            f"Unsupported audio format '{path.suffix}'. "
            f"Supported extensions: {config.supported_extensions}"
        )
        if strict:
            raise AudioFormatError(err)
        return ValidationResult(
            is_valid=False,
            is_target_spec=False,
            meets_recommended_duration=False,
            is_target_sample_rate=False,
            is_mono=False,
            has_valid_data=False,
            is_silent=False,
            sample_rate=0,
            channels=0,
            duration_seconds=0.0,
            num_samples=0,
            errors=[err],
            warnings=[],
        )

    # 3. Read & decode WAV
    try:
        recording = load_wav_file(path, config)
    except (AudioNotFoundError, AudioFormatError, AudioCorruptError) as exc:
        if strict:
            raise
        return ValidationResult(
            is_valid=False,
            is_target_spec=False,
            meets_recommended_duration=False,
            is_target_sample_rate=False,
            is_mono=False,
            has_valid_data=False,
            is_silent=False,
            sample_rate=0,
            channels=0,
            duration_seconds=0.0,
            num_samples=0,
            errors=[str(exc)],
            warnings=[],
        )

    return recording.validate(config=config, strict=strict)


# ===========================================================================
# Extensible Audio Input Sources (Clean Abstraction)
# ===========================================================================

class AudioInputSource(ABC):
    """
    Abstract Base Class for an audio input source.
    Enables pluggable audio ingestion mechanisms (filesystem, in-memory buffer,
    frontend streaming, simulated microphone).
    """

    @abstractmethod
    def read(self) -> AudioRecording:
        """Read and return an AudioRecording from this source."""
        pass


class LocalFileInputSource(AudioInputSource):
    """Audio input source for local filesystem WAV files."""

    def __init__(
        self,
        file_path: Union[str, Path],
        config: AudioConfig = AUDIO_CONFIG,
    ):
        self.file_path = Path(file_path)
        self.config = config

    def read(self) -> AudioRecording:
        return load_wav_file(self.file_path, self.config)


class BufferInputSource(AudioInputSource):
    """Audio input source for in-memory byte buffers (e.g. from frontend HTTP POST)."""

    def __init__(
        self,
        data: bytes,
        participant_id: Optional[str] = None,
        config: AudioConfig = AUDIO_CONFIG,
    ):
        self.data = data
        self.participant_id = participant_id
        self.config = config

    def read(self) -> AudioRecording:
        return load_wav_bytes(self.data, self.participant_id, self.config)


# ===========================================================================
# Voice Recording Interface (High-level controller)
# ===========================================================================

class VoiceRecordingInterface:
    """
    Functional Python audio input interface for the Voice modality.

    Provides a clean recording and input abstraction that:
        - Loads local WAV audio files.
        - Ingests in-memory audio buffers from frontend/API submissions.
        - Validates audio against target pipeline specifications (WAV, mono, 16 kHz, ~30-60s).
        - Exposes the standardized recording prompt for participant elicitation.
        - Provides save/export functionality for temporary staging or raw dataset storage.
    """

    def __init__(self, config: AudioConfig = AUDIO_CONFIG):
        self.config = config
        self.prompt = STANDARDIZED_RECORDING_PROMPT

    def get_target_specs(self) -> Dict[str, Any]:
        """
        Return target audio specifications for frontend or client reference.
        """
        return {
            "format": "WAV",
            "channels": self.config.target_channels,
            "sample_rate_hz": self.config.target_sample_rate,
            "min_duration_seconds": self.config.min_duration_seconds,
            "max_duration_seconds": self.config.max_duration_seconds,
            "recommended_min_duration_seconds": self.config.recommended_min_duration,
            "recommended_max_duration_seconds": self.config.recommended_max_duration,
            "supported_extensions": self.config.supported_extensions,
        }

    def load_file(self, file_path: Union[str, Path]) -> AudioRecording:
        """
        Load an existing local WAV file.
        """
        return load_wav_file(file_path, self.config)

    def load_bytes(
        self,
        data: bytes,
        participant_id: Optional[str] = None,
    ) -> AudioRecording:
        """
        Ingest audio bytes (e.g., from frontend file upload or audio recording API).
        """
        return load_wav_bytes(data, participant_id, self.config)

    def ingest(self, source: AudioInputSource) -> AudioRecording:
        """
        Ingest an audio recording through any AudioInputSource implementation.
        """
        return source.read()

    def verify(
        self,
        recording: AudioRecording,
        strict: bool = False,
    ) -> ValidationResult:
        """
        Verify whether an AudioRecording meets pipeline requirements.
        """
        return recording.validate(self.config, strict=strict)

    def validate_file(
        self,
        file_path: Union[str, Path],
        strict: bool = False,
    ) -> ValidationResult:
        """
        Validate a local audio file against format, corruptibility, and pipeline constraints.
        """
        return validate_audio_file(file_path, self.config, strict=strict)

    def save(
        self,
        recording: AudioRecording,
        output_path: Union[str, Path],
        overwrite: bool = False,
    ) -> Path:
        """
        Save an AudioRecording to disk as a standard PCM WAV file.
        """
        return save_wav_file(recording, output_path, overwrite=overwrite)

