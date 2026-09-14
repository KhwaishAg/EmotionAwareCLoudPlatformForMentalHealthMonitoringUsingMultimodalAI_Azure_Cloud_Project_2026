"""
Tests for the Voice Modality Pipeline — Recording & Audio Input Interface.

Covers:
    - Audio loading from local WAV files
    - Reading sample rate, duration, and channel dimensions
    - Audio loading from in-memory byte buffers (frontend integration)
    - Error handling for missing, corrupt, and unsupported audio files
    - Anonymous participant ID extraction
    - Recording interface abstraction and target specification validation
    - Robust audio validation for format, corruption, duration limits,
      sample rate, channel count, and waveform data integrity

Note:
    All tests use synthetic audio generated on-the-fly via numpy/scipy.
    No private participant recordings are stored or committed.
"""

import io
from pathlib import Path
import numpy as np
import pytest
import scipy.io.wavfile as wavfile

from src.ai_model.voice.config import AUDIO_CONFIG, AudioConfig
from src.ai_model.voice.recording import (
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
from src.ai_model.voice.preprocessing import (
    PreprocessedAudio,
    VoicePreprocessor,
    convert_to_mono,
    normalize_amplitude,
    preprocess_audio,
    resample_audio,
)




# ===========================================================================
# Fixtures — Synthetic Audio Generators (No real audio files used)
# ===========================================================================

def _create_synthetic_wav_bytes(
    duration_seconds: float = 30.0,
    sample_rate: int = 16_000,
    channels: int = 1,
    frequency: float = 440.0,
) -> bytes:
    """Generate in-memory WAV bytes containing a synthetic sine wave."""
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
    wave = np.sin(2 * np.pi * frequency * t)

    # 16-bit PCM conversion
    pcm_data = (wave * 32767 * 0.8).astype(np.int16)

    if channels == 2:
        pcm_data = np.column_stack([pcm_data, pcm_data])

    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, pcm_data)
    return buffer.getvalue()


@pytest.fixture
def synthetic_wav_file(tmp_path: Path) -> Path:
    """Fixture: Create a temporary 30s 16kHz mono WAV file for testing."""
    wav_bytes = _create_synthetic_wav_bytes(duration_seconds=30.0, sample_rate=16_000, channels=1)
    file_path = tmp_path / "P001.wav"
    file_path.write_bytes(wav_bytes)
    return file_path


# ===========================================================================
# Tests — Audio Loading and Dimension Verification
# ===========================================================================

def test_load_valid_wav_file(synthetic_wav_file: Path):
    """Verify a valid WAV file can be loaded and sample rate/duration are read correctly."""
    recording = load_wav_file(synthetic_wav_file)

    assert isinstance(recording, AudioRecording)
    assert recording.sample_rate == 16_000
    assert recording.channels == 1
    assert recording.is_mono is True
    assert abs(recording.duration_seconds - 30.0) < 0.05
    assert recording.num_samples == 16_000 * 30
    assert recording.participant_id == "P001"
    assert recording.source_path == str(synthetic_wav_file.resolve())

    # Normalized float32 range check [-1.0, 1.0]
    assert recording.audio_data.dtype == np.float32
    assert recording.audio_data.max() <= 1.0
    assert recording.audio_data.min() >= -1.0


def test_load_wav_bytes():
    """Verify audio can be loaded from in-memory bytes (for frontend/API upload)."""
    wav_bytes = _create_synthetic_wav_bytes(duration_seconds=5.0, sample_rate=16_000, channels=1)
    recording = load_wav_bytes(wav_bytes, participant_id="P042")

    assert recording.sample_rate == 16_000
    assert recording.channels == 1
    assert abs(recording.duration_seconds - 5.0) < 0.05
    assert recording.participant_id == "P042"
    assert recording.source_path is None


# ===========================================================================
# Tests — Error Handling for Missing / Corrupt / Unsupported Files
# ===========================================================================

def test_load_missing_file_raises_not_found(tmp_path: Path):
    """Verify loading a non-existent file raises AudioNotFoundError."""
    non_existent = tmp_path / "non_existent_audio.wav"
    with pytest.raises(AudioNotFoundError, match="Audio file not found"):
        load_wav_file(non_existent)


def test_load_unsupported_format_raises_format_error(tmp_path: Path):
    """Verify loading an unsupported file extension raises AudioFormatError."""
    fake_mp3 = tmp_path / "recording.mp3"
    fake_mp3.write_text("not an audio file")
    with pytest.raises(AudioFormatError, match="Unsupported audio format"):
        load_wav_file(fake_mp3)


def test_load_empty_file_raises_corrupt_error(tmp_path: Path):
    """Verify loading a 0-byte file raises AudioCorruptError."""
    empty_wav = tmp_path / "empty.wav"
    empty_wav.write_bytes(b"")
    with pytest.raises(AudioCorruptError, match="empty"):
        load_wav_file(empty_wav)


def test_load_corrupt_bytes_raises_corrupt_error(tmp_path: Path):
    """Verify loading corrupted WAV data raises AudioCorruptError."""
    corrupt_wav = tmp_path / "corrupt.wav"
    corrupt_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt invalid corrupt payload")
    with pytest.raises(AudioCorruptError, match="Failed to read WAV file"):
        load_wav_file(corrupt_wav)


# ===========================================================================
# Tests — Specification Validation & Abstraction
# ===========================================================================

def test_target_spec_validation(synthetic_wav_file: Path):
    """Verify validation flags accurately assess target pipeline constraints."""
    recording = load_wav_file(synthetic_wav_file)
    validation = recording.validate(AUDIO_CONFIG)

    assert validation["is_valid"] is True
    assert validation["is_target_spec"] is True
    assert validation["meets_recommended_duration"] is True
    assert validation["is_target_sample_rate"] is True
    assert validation["is_mono"] is True
    assert len(validation["warnings"]) == 0


def test_recording_interface_and_sources(synthetic_wav_file: Path):
    """Verify VoiceRecordingInterface methods and pluggable input sources."""
    interface = VoiceRecordingInterface()

    # Prompt check
    assert interface.prompt == STANDARDIZED_RECORDING_PROMPT
    assert "academic workload" in interface.prompt

    # Target specs check
    specs = interface.get_target_specs()
    assert specs["sample_rate_hz"] == 16_000
    assert specs["channels"] == 1
    assert specs["recommended_min_duration_seconds"] == 30.0
    assert specs["recommended_max_duration_seconds"] == 60.0

    # Test LocalFileInputSource
    source = LocalFileInputSource(synthetic_wav_file)
    rec1 = interface.ingest(source)
    assert rec1.sample_rate == 16_000

    # Test BufferInputSource
    wav_bytes = synthetic_wav_file.read_bytes()
    buf_source = BufferInputSource(wav_bytes, participant_id="P001")
    rec2 = interface.ingest(buf_source)
    assert rec2.sample_rate == 16_000
    assert rec2.participant_id == "P001"


def test_save_and_reload_wav_file(synthetic_wav_file: Path, tmp_path: Path):
    """Verify AudioRecording can be saved as a standard WAV and loaded back."""
    recording = load_wav_file(synthetic_wav_file)
    saved_path = tmp_path / "saved_P001.wav"

    save_wav_file(recording, saved_path)
    assert saved_path.exists()

    reloaded = load_wav_file(saved_path)
    assert reloaded.sample_rate == recording.sample_rate
    assert reloaded.channels == recording.channels
    assert abs(reloaded.duration_seconds - recording.duration_seconds) < 0.05


def test_extract_participant_id():
    """Verify anonymous participant ID extraction avoids PII and parses correctly."""
    assert extract_participant_id("P001.wav") == "P001"
    assert extract_participant_id("p045_recording.wav") == "P045"
    assert extract_participant_id("data/raw/audio/P120.WAV") == "P120"
    assert extract_participant_id("student_john_smith.wav") is None


# ===========================================================================
# Tests — Robust Validation (Commit 4)
# ===========================================================================

def test_validation_boundary_durations():
    """
    Test duration validation across boundary conditions:
        - Below acceptable minimum (< 5.0s) -> invalid, raises AudioDurationError
        - Exact minimum boundary (5.0s) -> valid, but below recommended
        - Recommended minimum boundary (30.0s) -> valid, recommended
        - Recommended maximum boundary (60.0s) -> valid, recommended
        - Acceptable upper duration (75.0s) -> valid, but above recommended
        - Exact maximum boundary (120.0s) -> valid, but above recommended
        - Exceeding acceptable maximum (> 120.0s) -> invalid, raises AudioDurationError
    """
    sr = 16_000

    # 1. Below acceptable minimum: 4.9s (< 5.0s)
    rec_short = AudioRecording(
        audio_data=np.ones(int(4.9 * sr), dtype=np.float32) * 0.1,
        sample_rate=sr,
        channels=1,
        duration_seconds=4.9,
    )
    val_short = rec_short.validate()
    assert val_short.is_valid is False
    assert val_short.meets_recommended_duration is False
    assert any("below the minimum acceptable" in err for err in val_short.errors)
    with pytest.raises(AudioDurationError, match="below the minimum"):
        rec_short.validate(strict=True)

    # 2. Exact minimum acceptable boundary: 5.0s
    rec_min = AudioRecording(
        audio_data=np.ones(int(5.0 * sr), dtype=np.float32) * 0.1,
        sample_rate=sr,
        channels=1,
        duration_seconds=5.0,
    )
    val_min = rec_min.validate()
    assert val_min.is_valid is True
    assert val_min.meets_recommended_duration is False
    assert any("shorter than recommended" in w for w in val_min.warnings)

    # 3. Recommended minimum boundary: 30.0s
    rec_rec_min = AudioRecording(
        audio_data=np.ones(int(30.0 * sr), dtype=np.float32) * 0.1,
        sample_rate=sr,
        channels=1,
        duration_seconds=30.0,
    )
    val_rec_min = rec_rec_min.validate()
    assert val_rec_min.is_valid is True
    assert val_rec_min.meets_recommended_duration is True
    assert val_rec_min.is_target_spec is True
    assert len(val_rec_min.warnings) == 0

    # 4. Recommended maximum boundary: 60.0s
    rec_rec_max = AudioRecording(
        audio_data=np.ones(int(60.0 * sr), dtype=np.float32) * 0.1,
        sample_rate=sr,
        channels=1,
        duration_seconds=60.0,
    )
    val_rec_max = rec_rec_max.validate()
    assert val_rec_max.is_valid is True
    assert val_rec_max.meets_recommended_duration is True
    assert val_rec_max.is_target_spec is True
    assert len(val_rec_max.warnings) == 0

    # 5. Above recommended but within acceptable: 75.0s
    rec_above_rec = AudioRecording(
        audio_data=np.ones(int(75.0 * sr), dtype=np.float32) * 0.1,
        sample_rate=sr,
        channels=1,
        duration_seconds=75.0,
    )
    val_above_rec = rec_above_rec.validate()
    assert val_above_rec.is_valid is True
    assert val_above_rec.meets_recommended_duration is False
    assert any("longer than recommended" in w for w in val_above_rec.warnings)

    # 6. Exact maximum acceptable boundary: 120.0s
    rec_max = AudioRecording(
        audio_data=np.ones(int(120.0 * sr), dtype=np.float32) * 0.1,
        sample_rate=sr,
        channels=1,
        duration_seconds=120.0,
    )
    val_max = rec_max.validate()
    assert val_max.is_valid is True
    assert val_max.meets_recommended_duration is False

    # 7. Exceeding acceptable maximum: 120.5s (> 120.0s)
    rec_long = AudioRecording(
        audio_data=np.ones(int(120.5 * sr), dtype=np.float32) * 0.1,
        sample_rate=sr,
        channels=1,
        duration_seconds=120.5,
    )
    val_long = rec_long.validate()
    assert val_long.is_valid is False
    assert any("exceeds the maximum allowed" in err for err in val_long.errors)
    with pytest.raises(AudioDurationError, match="exceeds the maximum"):
        rec_long.validate(strict=True)


def test_validation_sample_rate_variations():
    """Verify validation behavior for target, non-target, and invalid sample rates."""
    # Target 16 kHz sample rate
    rec_16k = AudioRecording(
        audio_data=np.ones(16_000 * 30, dtype=np.float32) * 0.1,
        sample_rate=16_000,
        channels=1,
        duration_seconds=30.0,
    )
    val_16k = rec_16k.validate()
    assert val_16k.is_target_sample_rate is True
    assert val_16k.is_valid is True

    # Non-target 44.1 kHz (valid speech audio, but requires resampling in preprocessing)
    rec_44k = AudioRecording(
        audio_data=np.ones(44_100 * 30, dtype=np.float32) * 0.1,
        sample_rate=44_100,
        channels=1,
        duration_seconds=30.0,
    )
    val_44k = rec_44k.validate()
    assert val_44k.is_valid is True
    assert val_44k.is_target_sample_rate is False
    assert val_44k.is_target_spec is False
    assert any("resampling will be required" in w for w in val_44k.warnings)

    # Invalid non-positive sample rate (0 Hz)
    rec_0hz = AudioRecording(
        audio_data=np.ones(1000, dtype=np.float32) * 0.1,
        sample_rate=0,
        channels=1,
        duration_seconds=30.0,
    )
    val_0hz = rec_0hz.validate()
    assert val_0hz.is_valid is False
    assert any("sample rate" in err.lower() for err in val_0hz.errors)
    with pytest.raises(AudioSampleRateError, match="Sample rate"):
        rec_0hz.validate(strict=True)


def test_validation_channel_count():
    """Verify validation behavior for mono, stereo, and invalid channel configurations."""
    sr = 16_000

    # Mono (1 channel)
    rec_mono = AudioRecording(
        audio_data=np.ones(sr * 30, dtype=np.float32) * 0.1,
        sample_rate=sr,
        channels=1,
        duration_seconds=30.0,
    )
    val_mono = rec_mono.validate()
    assert val_mono.is_mono is True
    assert val_mono.is_valid is True

    # Stereo (2 channels) — acceptable input, flagged for mono conversion in preprocessing
    stereo_data = np.ones((sr * 30, 2), dtype=np.float32) * 0.1
    rec_stereo = AudioRecording(
        audio_data=stereo_data,
        sample_rate=sr,
        channels=2,
        duration_seconds=30.0,
    )
    val_stereo = rec_stereo.validate()
    assert val_stereo.is_valid is True
    assert val_stereo.is_mono is False
    assert val_stereo.is_target_spec is False
    assert any("mono conversion will be required" in w for w in val_stereo.warnings)

    # Invalid channel count (0 channels)
    rec_zero_chan = AudioRecording(
        audio_data=np.ones(sr * 30, dtype=np.float32) * 0.1,
        sample_rate=sr,
        channels=0,
        duration_seconds=30.0,
    )
    val_zero_chan = rec_zero_chan.validate()
    assert val_zero_chan.is_valid is False
    assert any("channel count" in err.lower() for err in val_zero_chan.errors)
    with pytest.raises(AudioChannelError, match="Channel count"):
        rec_zero_chan.validate(strict=True)


def test_validation_audio_data_integrity():
    """Verify detection of NaN, Inf, zero samples, and silent audio waveforms."""
    sr = 16_000

    # NaN in audio data
    nan_data = np.ones(sr * 30, dtype=np.float32) * 0.1
    nan_data[100] = np.nan
    rec_nan = AudioRecording(
        audio_data=nan_data,
        sample_rate=sr,
        channels=1,
        duration_seconds=30.0,
    )
    val_nan = rec_nan.validate()
    assert val_nan.is_valid is False
    assert val_nan.has_valid_data is False
    assert any("nan or inf" in err.lower() for err in val_nan.errors)
    with pytest.raises(AudioDataError, match="non-finite"):
        rec_nan.validate(strict=True)

    # Inf in audio data
    inf_data = np.ones(sr * 30, dtype=np.float32) * 0.1
    inf_data[250] = np.inf
    rec_inf = AudioRecording(
        audio_data=inf_data,
        sample_rate=sr,
        channels=1,
        duration_seconds=30.0,
    )
    val_inf = rec_inf.validate()
    assert val_inf.is_valid is False
    assert val_inf.has_valid_data is False
    with pytest.raises(AudioDataError, match="non-finite"):
        rec_inf.validate(strict=True)

    # Zero samples (empty array)
    rec_empty = AudioRecording(
        audio_data=np.array([], dtype=np.float32),
        sample_rate=sr,
        channels=1,
        duration_seconds=0.0,
    )
    val_empty = rec_empty.validate()
    assert val_empty.is_valid is False
    assert val_empty.has_valid_data is False
    with pytest.raises(AudioDataError, match="zero samples"):
        rec_empty.validate(strict=True)

    # Silent audio (all zeros) — valid duration, but flagged with warning
    rec_silent = AudioRecording(
        audio_data=np.zeros(sr * 30, dtype=np.float32),
        sample_rate=sr,
        channels=1,
        duration_seconds=30.0,
    )
    val_silent = rec_silent.validate()
    assert val_silent.is_valid is True
    assert val_silent.is_silent is True
    assert any("silence" in w.lower() for w in val_silent.warnings)


def test_validate_audio_file_helper(synthetic_wav_file: Path, tmp_path: Path):
    """Verify validate_audio_file against valid, missing, corrupt, and unsupported files."""
    # 1. Valid file
    res_valid = validate_audio_file(synthetic_wav_file)
    assert isinstance(res_valid, ValidationResult)
    assert res_valid.is_valid is True
    assert res_valid.is_target_spec is True

    # 2. Missing file
    missing = tmp_path / "does_not_exist.wav"
    res_missing = validate_audio_file(missing, strict=False)
    assert res_missing.is_valid is False
    with pytest.raises(AudioNotFoundError):
        validate_audio_file(missing, strict=True)

    # 3. Unsupported extension
    fake_txt = tmp_path / "audio.txt"
    fake_txt.write_text("not a wav file")
    res_format = validate_audio_file(fake_txt, strict=False)
    assert res_format.is_valid is False
    with pytest.raises(AudioFormatError):
        validate_audio_file(fake_txt, strict=True)

    # 4. Corrupt / empty file
    empty_wav = tmp_path / "zero_bytes.wav"
    empty_wav.write_bytes(b"")
    res_empty = validate_audio_file(empty_wav, strict=False)
    assert res_empty.is_valid is False
    with pytest.raises(AudioCorruptError):
        validate_audio_file(empty_wav, strict=True)


def test_interface_validation_methods(synthetic_wav_file: Path):
    """Verify VoiceRecordingInterface verify and validate_file methods."""
    interface = VoiceRecordingInterface()
    rec = interface.load_file(synthetic_wav_file)

    # verify method
    val = interface.verify(rec)
    assert val.is_valid is True
    assert val.is_target_spec is True

    # validate_file method
    val_file = interface.validate_file(synthetic_wav_file)
    assert val_file.is_valid is True

    # validate_audio_recording standalone function
    val_rec = validate_audio_recording(rec)
    assert val_rec.is_valid is True


# ===========================================================================
# Tests — Audio Preprocessing Pipeline (Commit 5)
# ===========================================================================

def test_convert_to_mono_from_1d():
    """Verify convert_to_mono preserves already mono 1D audio."""
    mono_in = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    mono_out = convert_to_mono(mono_in)
    assert mono_out.ndim == 1
    assert mono_out.dtype == np.float32
    assert np.allclose(mono_out, mono_in)


def test_convert_to_mono_from_2d_single_channel():
    """Verify convert_to_mono handles (N, 1) column vectors."""
    col_in = np.array([[0.1], [-0.2], [0.3]], dtype=np.float32)
    mono_out = convert_to_mono(col_in)
    assert mono_out.ndim == 1
    assert mono_out.shape == (3,)
    assert np.allclose(mono_out, [0.1, -0.2, 0.3])


def test_convert_to_mono_from_stereo():
    """Verify convert_to_mono correctly averages left and right channels."""
    stereo_in = np.array([[0.2, 0.4], [0.6, 0.8], [-0.4, -0.2]], dtype=np.float32)
    mono_out = convert_to_mono(stereo_in)
    assert mono_out.ndim == 1
    assert mono_out.shape == (3,)
    expected = np.array([0.3, 0.7, -0.3], dtype=np.float32)
    assert np.allclose(mono_out, expected)


def test_convert_to_mono_multi_channel():
    """Verify convert_to_mono handles 4-channel audio by averaging across channels."""
    quad_in = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    mono_out = convert_to_mono(quad_in)
    assert mono_out.shape == (1,)
    assert np.isclose(mono_out[0], 0.25)


def test_convert_to_mono_invalid_inputs():
    """Verify convert_to_mono safely rejects empty, non-finite, or 3D arrays."""
    with pytest.raises(AudioDataError, match="empty"):
        convert_to_mono(np.array([], dtype=np.float32))

    with pytest.raises(AudioDataError, match="non-finite"):
        convert_to_mono(np.array([0.1, np.nan, 0.3], dtype=np.float32))

    with pytest.raises(AudioDataError, match="Unsupported audio array dimension"):
        convert_to_mono(np.zeros((2, 2, 2), dtype=np.float32))


def test_normalize_amplitude_deterministic():
    """Verify normalize_amplitude scales waveform peak to target peak (0.95)."""
    data = np.array([-0.5, 0.2, 0.4, -0.1], dtype=np.float32)
    norm, orig_peak = normalize_amplitude(data, target_peak=0.95)

    assert np.isclose(orig_peak, 0.5)
    assert np.isclose(float(np.max(np.abs(norm))), 0.95)
    assert norm.dtype == np.float32
    assert np.isclose(norm[0], -0.95)
    assert np.isclose(norm[1], 0.2 * (0.95 / 0.5))


def test_normalize_amplitude_custom_target_peak():
    """Verify normalize_amplitude respects custom target peak (e.g. 0.8)."""
    data = np.array([0.1, -0.4, 0.2], dtype=np.float32)
    norm, orig_peak = normalize_amplitude(data, target_peak=0.8)
    assert np.isclose(orig_peak, 0.4)
    assert np.isclose(float(np.max(np.abs(norm))), 0.8)


def test_normalize_amplitude_silent_audio():
    """Verify normalize_amplitude handles all-zero silent audio without division by zero."""
    silent = np.zeros(100, dtype=np.float32)
    norm, orig_peak = normalize_amplitude(silent, target_peak=0.95)

    assert orig_peak == 0.0
    assert np.all(norm == 0.0)
    assert norm.shape == (100,)


def test_normalize_amplitude_invalid_inputs():
    """Verify normalize_amplitude safely rejects empty, non-finite, or out-of-range target peaks."""
    with pytest.raises(AudioDataError, match="empty"):
        normalize_amplitude(np.array([]))

    with pytest.raises(AudioDataError, match="non-finite"):
        normalize_amplitude(np.array([np.nan]))

    with pytest.raises(AudioDataError, match="Target peak must be in"):
        normalize_amplitude(np.array([0.5]), target_peak=1.5)

    with pytest.raises(AudioDataError, match="Target peak must be in"):
        normalize_amplitude(np.array([0.5]), target_peak=0.0)


def test_preprocess_audio_pipeline_from_file(synthetic_wav_file: Path):
    """Verify full preprocessing pipeline on a valid WAV file."""
    preprocessed = preprocess_audio(synthetic_wav_file)

    assert isinstance(preprocessed, PreprocessedAudio)
    assert preprocessed.channels == 1
    assert preprocessed.sample_rate == 16_000
    assert preprocessed.audio_data.ndim == 1
    assert preprocessed.participant_id == "P001"
    assert np.isclose(preprocessed.peak_amplitude, 0.95, atol=1e-4)
    assert preprocessed.metadata["converted_to_mono"] is False
    assert preprocessed.metadata["normalization_target_peak"] == 0.95


def test_preprocess_audio_downmixes_stereo_and_resamples_44k_to_16k():
    """
    Verify pipeline converts stereo to mono and resamples 44.1 kHz input
    to the target 16 kHz standard.
    """
    sr = 44_100
    duration = 5.0
    stereo_bytes = _create_synthetic_wav_bytes(
        duration_seconds=duration,
        sample_rate=sr,
        channels=2,
    )

    preprocessed = preprocess_audio(stereo_bytes, participant_id="P099")

    # Sample rate must be resampled to 16 kHz target
    assert preprocessed.sample_rate == 16_000
    # Must be mono
    assert preprocessed.channels == 1
    assert preprocessed.audio_data.ndim == 1
    # Sample count must match 16 kHz * 5s
    assert abs(len(preprocessed.audio_data) - int(16_000 * duration)) <= 1
    # Normalized to 0.95 peak
    assert np.isclose(preprocessed.peak_amplitude, 0.95, atol=1e-4)
    assert preprocessed.metadata["converted_to_mono"] is True
    assert preprocessed.metadata["original_channels"] == 2
    assert preprocessed.metadata["was_resampled"] is True
    assert preprocessed.metadata["original_sample_rate"] == 44_100
    assert preprocessed.metadata["target_sample_rate"] == 16_000
    assert preprocessed.participant_id == "P099"



def test_preprocess_audio_to_recording():
    """Verify PreprocessedAudio can convert back to AudioRecording."""
    sr = 16_000
    pre = PreprocessedAudio(
        audio_data=np.ones(sr * 5, dtype=np.float32) * 0.95,
        sample_rate=sr,
        channels=1,
        participant_id="P002",
        metadata={"step": "test"},
    )
    rec = pre.to_recording()
    assert isinstance(rec, AudioRecording)
    assert rec.sample_rate == sr
    assert rec.channels == 1
    assert rec.participant_id == "P002"
    assert rec.is_mono is True


def test_voice_preprocessor_controller(synthetic_wav_file: Path):
    """Verify VoicePreprocessor class methods and operations."""
    preprocessor = VoicePreprocessor()
    result = preprocessor.process(synthetic_wav_file)

    assert isinstance(result, PreprocessedAudio)
    assert result.channels == 1
    assert np.isclose(result.peak_amplitude, 0.95, atol=1e-4)

    # Test class to_mono and normalize helpers
    mono = preprocessor.to_mono(np.array([[0.2, 0.4]], dtype=np.float32))
    assert np.isclose(mono[0], 0.3)

    norm, orig_pk = preprocessor.normalize(np.array([0.5], dtype=np.float32))
    assert np.isclose(norm[0], 0.95)
    assert np.isclose(orig_pk, 0.5)


def test_preprocess_audio_invalid_inputs():
    """Verify preprocess_audio safely handles invalid types and empty data."""
    with pytest.raises(TypeError, match="Unsupported input type"):
        preprocess_audio(12345)  # type: ignore

    empty_rec = AudioRecording(
        audio_data=np.array([], dtype=np.float32),
        sample_rate=16000,
        channels=1,
        duration_seconds=0.0,
    )
    with pytest.raises(AudioDataError, match="zero samples"):
        preprocess_audio(empty_rec)


# ===========================================================================
# Tests — 16 kHz Resampling (Commit 6)
# ===========================================================================

def test_resample_audio_44k_to_16k():
    """Verify resample_audio converts 44.1 kHz audio to exactly 16 kHz."""
    orig_sr = 44_100
    target_sr = 16_000
    duration = 1.0

    t = np.linspace(0, duration, int(orig_sr * duration), endpoint=False)
    orig_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    resampled, was_resampled = resample_audio(
        orig_data,
        orig_sample_rate=orig_sr,
        target_sample_rate=target_sr,
    )

    assert was_resampled is True
    assert resampled.ndim == 1
    assert resampled.dtype == np.float32
    assert len(resampled) == int(target_sr * duration)
    assert abs(len(resampled) - 16_000) <= 1


def test_resample_audio_16k_passthrough():
    """Verify resample_audio returns identical data without resampling when already 16 kHz."""
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    resampled, was_resampled = resample_audio(
        data,
        orig_sample_rate=sr,
        target_sample_rate=sr,
    )

    assert was_resampled is False
    assert resampled.ndim == 1
    assert len(resampled) == sr
    assert np.allclose(resampled, data)


def test_resample_audio_48k_to_16k():
    """Verify resample_audio handles 48 kHz studio audio down to 16 kHz."""
    orig_sr = 48_000
    target_sr = 16_000
    data = np.ones(orig_sr, dtype=np.float32) * 0.5

    resampled, was_resampled = resample_audio(
        data,
        orig_sample_rate=orig_sr,
        target_sample_rate=target_sr,
    )

    assert was_resampled is True
    assert len(resampled) == target_sr


def test_resample_audio_invalid_inputs():
    """Verify resample_audio safely rejects invalid sample rates and corrupted audio data."""
    valid_data = np.ones(1000, dtype=np.float32)

    with pytest.raises(AudioSampleRateError, match="Original sample rate"):
        resample_audio(valid_data, orig_sample_rate=0, target_sample_rate=16_000)

    with pytest.raises(AudioSampleRateError, match="Target sample rate"):
        resample_audio(valid_data, orig_sample_rate=44_100, target_sample_rate=-16_000)

    with pytest.raises(AudioDataError, match="empty"):
        resample_audio(np.array([], dtype=np.float32), orig_sample_rate=44_100, target_sample_rate=16_000)

    with pytest.raises(AudioDataError, match="non-finite"):
        resample_audio(np.array([0.1, np.nan]), orig_sample_rate=44_100, target_sample_rate=16_000)


def test_preprocess_pipeline_16k_passthrough(synthetic_wav_file: Path):
    """Verify preprocessing pipeline preserves already 16 kHz sample rate without resampling."""
    preprocessed = preprocess_audio(synthetic_wav_file)

    assert preprocessed.sample_rate == 16_000
    assert preprocessed.metadata["was_resampled"] is False
    assert preprocessed.metadata["original_sample_rate"] == 16_000
    assert preprocessed.metadata["target_sample_rate"] == 16_000


def test_preprocess_pipeline_44k_resampling():
    """Verify full preprocessing pipeline ingests 44.1 kHz audio and produces clean 16 kHz mono waveform."""
    duration = 5.0
    wav_bytes = _create_synthetic_wav_bytes(
        duration_seconds=duration,
        sample_rate=44_100,
        channels=1,
    )

    preprocessed = preprocess_audio(wav_bytes)

    assert preprocessed.sample_rate == 16_000
    assert preprocessed.channels == 1
    assert preprocessed.audio_data.ndim == 1
    assert preprocessed.metadata["was_resampled"] is True
    assert preprocessed.metadata["original_sample_rate"] == 44_100
    assert preprocessed.metadata["target_sample_rate"] == 16_000
    assert abs(len(preprocessed.audio_data) - int(16_000 * duration)) <= 1
    assert np.isclose(preprocessed.peak_amplitude, 0.95, atol=1e-4)


def test_voice_preprocessor_resample_helper():
    """Verify VoicePreprocessor.resample helper method."""
    preprocessor = VoicePreprocessor()
    data = np.ones(44_100, dtype=np.float32) * 0.5
    resampled, was_resampled = preprocessor.resample(data, orig_sample_rate=44_100)

    assert was_resampled is True
    assert len(resampled) == 16_000


