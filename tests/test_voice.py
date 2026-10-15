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
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import pytest
import scipy.io.wavfile as wavfile

from src.ai_model.voice.config import AUDIO_CONFIG, FEATURE_CONFIG, AudioConfig, FeatureConfig
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
    trim_silence,
)
from src.ai_model.voice.features import (
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
from src.ai_model.voice.dataset import (
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
from src.ai_model.voice.model import (
    BaseVoiceClassifier,
    ModelCorruptError,
    ModelError,
    ModelInputError,
    ModelNotFoundError,
    ModelNotFittedError,
    ModelSerializationError,
    VoiceBaselineClassifier,
    VoiceLogisticRegression,
    VoiceLogisticRegressionClassifier,
    VoiceRandomForest,
    VoiceRandomForestClassifier,
    load_voice_model,
    save_voice_model,
    separate_features_and_target,
)
from src.ai_model.voice.evaluation import (
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


# ===========================================================================
# Tests — Silence Handling & Trimming (Commit 7)
# ===========================================================================

def test_trim_silence_leading_and_trailing():
    """Verify trim_silence detects and trims leading and trailing silence."""
    sr = 16_000
    leading_silence = np.zeros(sr * 1, dtype=np.float32)       # 1 sec silence
    t = np.linspace(0, 4.0, sr * 4, endpoint=False)
    speech = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)  # 4 sec tone
    trailing_silence = np.zeros(sr * 1, dtype=np.float32)      # 1 sec silence
    audio = np.concatenate([leading_silence, speech, trailing_silence])

    trimmed, was_trimmed, leading, trailing = trim_silence(audio, sample_rate=sr)

    assert was_trimmed is True
    assert abs(leading - 16_000) <= 512
    assert abs(trailing - 16_000) <= 512
    assert abs(len(trimmed) - 64_000) <= 1024
    assert len(trimmed) < len(audio)
    assert np.isclose(np.max(np.abs(trimmed)), 0.5, atol=1e-3)


def test_trim_silence_preserves_internal_pauses():
    """Verify trim_silence preserves internal conversational pauses while removing outer silence."""
    sr = 16_000
    lead_silence = np.zeros(sr * 1, dtype=np.float32)          # 1 sec leading
    t1 = np.linspace(0, 3.0, sr * 3, endpoint=False)
    speech1 = (np.sin(2 * np.pi * 440 * t1) * 0.5).astype(np.float32)  # 3 sec speech
    internal_pause = np.zeros(sr * 2, dtype=np.float32)        # 2 sec internal pause
    t2 = np.linspace(0, 3.0, sr * 3, endpoint=False)
    speech2 = (np.sin(2 * np.pi * 880 * t2) * 0.5).astype(np.float32)  # 3 sec speech
    trail_silence = np.zeros(sr * 1, dtype=np.float32)         # 1 sec trailing
    audio = np.concatenate([lead_silence, speech1, internal_pause, speech2, trail_silence])

    trimmed, was_trimmed, leading, trailing = trim_silence(audio, sample_rate=sr)

    assert was_trimmed is True
    assert abs(leading - 16_000) <= 512
    assert abs(trailing - 16_000) <= 512
    assert abs(len(trimmed) - (8 * sr)) <= 1024

    # Verify the internal pause is intact within the trimmed audio
    pause_segment = trimmed[int(3.5 * sr) : int(4.5 * sr)]
    assert np.allclose(pause_segment, 0.0)


def test_trim_silence_completely_silent_audio():
    """Verify completely silent audio is returned intact without error or truncation."""
    sr = 16_000
    silent_audio = np.zeros(sr * 5, dtype=np.float32)

    trimmed, was_trimmed, leading, trailing = trim_silence(silent_audio, sample_rate=sr)

    assert was_trimmed is False
    assert leading == 0
    assert trailing == 0
    assert len(trimmed) == len(silent_audio)
    assert np.array_equal(trimmed, silent_audio)


def test_trim_silence_no_silence_to_trim():
    """Verify audio with speech throughout is not truncated."""
    sr = 16_000
    t = np.linspace(0, 5.0, sr * 5, endpoint=False)
    continuous_speech = (np.sin(2 * np.pi * 440 * t) * 0.6).astype(np.float32)

    trimmed, was_trimmed, leading, trailing = trim_silence(continuous_speech, sample_rate=sr)

    assert was_trimmed is False
    assert leading == 0
    assert trailing == 0
    assert len(trimmed) == len(continuous_speech)


def test_trim_silence_min_duration_protection():
    """Verify min_duration_after_trim prevents over-trimming short utterances."""
    sr = 16_000
    lead = np.zeros(sr * 2, dtype=np.float32)
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    speech = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    trail = np.zeros(int(sr * 2.5), dtype=np.float32)
    audio = np.concatenate([lead, speech, trail])

    min_dur = 3.0
    trimmed, was_trimmed, leading, trailing = trim_silence(
        audio, sample_rate=sr, min_duration_after_trim=min_dur
    )

    assert len(trimmed) >= int(min_dur * sr)


def test_trim_silence_invalid_inputs():
    """Verify trim_silence rejects empty, non-finite, and multi-dimensional inputs."""
    with pytest.raises(AudioDataError, match="empty"):
        trim_silence(np.array([], dtype=np.float32))

    with pytest.raises(AudioDataError, match="non-finite"):
        trim_silence(np.array([0.1, np.nan, 0.2], dtype=np.float32))

    with pytest.raises(AudioDataError, match="1D"):
        trim_silence(np.zeros((100, 2), dtype=np.float32))


def test_preprocess_pipeline_with_silence_trimming():
    """Verify the full preprocess_audio pipeline trims silence and records metadata."""
    sr = 16_000
    lead_silence = np.zeros(sr * 1, dtype=np.int16)
    t = np.linspace(0, 5.0, sr * 5, endpoint=False)
    speech = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    trail_silence = np.zeros(sr * 1, dtype=np.int16)
    pcm_data = np.concatenate([lead_silence, speech, trail_silence])

    buffer = io.BytesIO()
    wavfile.write(buffer, sr, pcm_data)
    wav_bytes = buffer.getvalue()

    preprocessed = preprocess_audio(wav_bytes)

    assert preprocessed.sample_rate == 16_000
    assert preprocessed.channels == 1
    assert preprocessed.audio_data.ndim == 1
    assert preprocessed.metadata["silence_trimmed"] is True
    assert preprocessed.metadata["leading_silence_samples"] > 0
    assert preprocessed.metadata["trailing_silence_samples"] > 0
    assert abs(preprocessed.duration_seconds - 5.0) <= 0.2
    assert np.isclose(preprocessed.peak_amplitude, 0.95, atol=1e-4)


def test_preprocess_pipeline_completely_silent_audio():
    """Verify preprocess_audio safely processes a completely silent audio file."""
    sr = 16_000
    pcm_data = np.zeros(sr * 5, dtype=np.int16)

    buffer = io.BytesIO()
    wavfile.write(buffer, sr, pcm_data)
    wav_bytes = buffer.getvalue()

    preprocessed = preprocess_audio(wav_bytes)

    assert preprocessed.sample_rate == 16_000
    assert preprocessed.channels == 1
    assert preprocessed.metadata["silence_trimmed"] is False
    assert preprocessed.metadata["leading_silence_samples"] == 0
    assert preprocessed.metadata["trailing_silence_samples"] == 0
    assert preprocessed.peak_amplitude == 0.0
    assert np.allclose(preprocessed.audio_data, 0.0)


def test_voice_preprocessor_trim_silence_method():
    """Verify VoicePreprocessor controller exposes trim_silence correctly."""
    preprocessor = VoicePreprocessor()
    sr = 16_000
    audio = np.concatenate([
        np.zeros(sr, dtype=np.float32),
        np.ones(sr * 4, dtype=np.float32) * 0.5,
        np.zeros(sr, dtype=np.float32),
    ])

    trimmed, was_trimmed, leading, trailing = preprocessor.trim_silence(audio)

    assert was_trimmed is True
    assert leading > 0
    assert trailing > 0
    assert len(trimmed) < len(audio)


# ===========================================================================
# Tests — MFCC Feature Extraction (Commit 8)
# ===========================================================================

def test_extract_mfcc_shape_default_13():
    """Verify default MFCC extraction produces fixed-length 26-element vector (13 mean + 13 std)."""
    sr = 16_000
    t = np.linspace(0, 5.0, sr * 5, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)

    features = extract_mfcc_features(audio, sample_rate=sr)

    assert isinstance(features, np.ndarray)
    assert features.shape == (26,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_extract_mfcc_shape_custom_n_mfcc():
    """Verify configurable n_mfcc parameter controls output feature dimensionality."""
    sr = 16_000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)

    # Custom n_mfcc = 20 -> 20 means + 20 stds = 40 features
    features_20 = extract_mfcc_features(audio, sample_rate=sr, n_mfcc=20)
    assert features_20.shape == (40,)
    assert features_20.dtype == np.float32

    # Custom FeatureConfig with n_mfcc = 10 -> 20 features
    custom_cfg = FeatureConfig(n_mfcc=10)
    features_10 = extract_mfcc_features(audio, sample_rate=sr, config=custom_cfg)
    assert features_10.shape == (20,)


def test_extract_mfcc_determinism():
    """Verify identical audio input produces identical MFCC feature vectors."""
    sr = 16_000
    t = np.linspace(0, 4.0, sr * 4, endpoint=False)
    audio = (np.sin(2 * np.pi * 300 * t) * 0.5 + np.sin(2 * np.pi * 600 * t) * 0.3).astype(np.float32)

    f1 = extract_mfcc_features(audio, sample_rate=sr)
    f2 = extract_mfcc_features(audio, sample_rate=sr)

    assert np.array_equal(f1, f2)


def test_extract_mfcc_from_preprocessed_audio(synthetic_wav_file: Path):
    """Verify MFCC extraction directly accepts PreprocessedAudio container."""
    preprocessed = preprocess_audio(synthetic_wav_file)

    features = extract_mfcc_features(preprocessed)

    assert isinstance(features, np.ndarray)
    assert features.shape == (26,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_extract_mfcc_from_audio_recording(synthetic_wav_file: Path):
    """Verify MFCC extraction directly accepts AudioRecording container."""
    recording = load_wav_file(synthetic_wav_file)

    features = extract_mfcc_features(recording)

    assert isinstance(features, np.ndarray)
    assert features.shape == (26,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_extract_mfcc_completely_silent_audio():
    """Verify completely silent audio produces finite MFCC features without error."""
    sr = 16_000
    silent_audio = np.zeros(sr * 4, dtype=np.float32)

    features = extract_mfcc_features(silent_audio, sample_rate=sr)

    assert isinstance(features, np.ndarray)
    assert features.shape == (26,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_extract_mfcc_invalid_inputs():
    """Verify robust validation against empty, non-finite, multi-dimensional, and invalid inputs."""
    # Empty audio
    with pytest.raises(AudioDataError, match="empty"):
        extract_mfcc_features(np.array([], dtype=np.float32))

    # Non-finite values (NaN / Inf)
    with pytest.raises(AudioDataError, match="non-finite"):
        extract_mfcc_features(np.array([0.1, np.nan, 0.3], dtype=np.float32))

    with pytest.raises(AudioDataError, match="non-finite"):
        extract_mfcc_features(np.array([0.1, np.inf, 0.3], dtype=np.float32))

    # 2D stereo input (should be 1D mono)
    with pytest.raises(AudioDataError, match="1D"):
        extract_mfcc_features(np.zeros((16000, 2), dtype=np.float32))

    # Invalid sample rate
    with pytest.raises(AudioSampleRateError, match="positive"):
        extract_mfcc_features(np.ones(1000, dtype=np.float32), sample_rate=0)

    with pytest.raises(AudioSampleRateError, match="positive"):
        extract_mfcc_features(np.ones(1000, dtype=np.float32), sample_rate=-16000)

    # Invalid n_mfcc
    with pytest.raises(ValueError, match="positive"):
        extract_mfcc_features(np.ones(1000, dtype=np.float32), n_mfcc=0)

    with pytest.raises(ValueError, match="positive"):
        extract_mfcc_features(np.ones(1000, dtype=np.float32), n_mfcc=-5)

    # Unsupported input type
    with pytest.raises(TypeError, match="Unsupported audio input type"):
        extract_mfcc_features("not_audio_data")  # type: ignore


def test_get_mfcc_feature_names():
    """Verify feature names generation matches expected format and count."""
    names_13 = get_mfcc_feature_names(n_mfcc=13)
    assert len(names_13) == 26
    assert names_13[0] == "mfcc_1_mean"
    assert names_13[12] == "mfcc_13_mean"
    assert names_13[13] == "mfcc_1_std"
    assert names_13[25] == "mfcc_13_std"

    names_20 = get_mfcc_feature_names(n_mfcc=20)
    assert len(names_20) == 40
    assert names_20[0] == "mfcc_1_mean"
    assert names_20[19] == "mfcc_20_mean"
    assert names_20[20] == "mfcc_1_std"
    assert names_20[39] == "mfcc_20_std"

    with pytest.raises(ValueError, match="positive"):
        get_mfcc_feature_names(n_mfcc=0)


def test_extract_mfcc_dict():
    """Verify extract_mfcc_dict returns mapping of feature names to float values."""
    sr = 16_000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.7).astype(np.float32)

    feat_dict = extract_mfcc_dict(audio, sample_rate=sr)

    assert isinstance(feat_dict, dict)
    assert len(feat_dict) == 26
    assert "mfcc_1_mean" in feat_dict
    assert "mfcc_13_mean" in feat_dict
    assert "mfcc_1_std" in feat_dict
    assert "mfcc_13_std" in feat_dict
    for k, v in feat_dict.items():
        assert isinstance(k, str)
        assert isinstance(v, float)
        assert np.isfinite(v)


def test_compute_mfcc_frames():
    """Verify compute_mfcc_frames produces 2D time-frequency matrix."""
    sr = 16_000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.6).astype(np.float32)

    frames = compute_mfcc_frames(audio, sample_rate=sr, n_mfcc=13)

    assert isinstance(frames, np.ndarray)
    assert frames.ndim == 2
    assert frames.shape[0] == 13
    assert frames.shape[1] > 0
    assert frames.dtype == np.float32


def test_voice_feature_extractor_controller():
    """Verify VoiceFeatureExtractor controller exposes modular methods and configurations."""
    extractor = VoiceFeatureExtractor()
    sr = 16_000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    audio = (np.sin(2 * np.pi * 500 * t) * 0.5).astype(np.float32)

    # Controller feature extraction
    features = extractor.extract_features(audio, sample_rate=sr)
    assert features.shape == (26,)

    # Controller dict extraction
    feat_dict = extractor.extract_dict(audio, sample_rate=sr)
    assert len(feat_dict) == 26

    # Controller frame computation
    frames = extractor.compute_frames(audio, sample_rate=sr)
    assert frames.shape[0] == 13

    # Controller feature names
    names = extractor.get_feature_names()
    assert len(names) == 26


# ===========================================================================
# Tests — Pitch (F0) Feature Extraction (Commit 9)
# ===========================================================================

def test_extract_pitch_voiced_audio():
    """Verify pitch extraction computes accurate mean, std, min, and max for voiced tone."""
    sr = 16_000
    target_freq = 220.0  # A3 note
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    audio = (np.sin(2 * np.pi * target_freq * t) * 0.8).astype(np.float32)

    pitch_features = extract_pitch_features(audio, sample_rate=sr)

    assert isinstance(pitch_features, np.ndarray)
    assert pitch_features.shape == (4,)
    assert pitch_features.dtype == np.float32
    assert np.all(np.isfinite(pitch_features))

    pitch_mean, pitch_std, pitch_min, pitch_max = pitch_features
    # Mean frequency should be close to 220 Hz
    assert abs(pitch_mean - target_freq) < 15.0
    # Min should be <= Mean <= Max
    assert pitch_min <= pitch_mean <= pitch_max
    assert pitch_std >= 0.0


def test_extract_pitch_silent_audio():
    """Verify completely silent audio safely returns zero vector without NaN/Inf."""
    sr = 16_000
    silent_audio = np.zeros(sr * 2, dtype=np.float32)

    pitch_features = extract_pitch_features(silent_audio, sample_rate=sr)

    assert isinstance(pitch_features, np.ndarray)
    assert pitch_features.shape == (4,)
    assert pitch_features.dtype == np.float32
    assert not np.any(np.isnan(pitch_features))
    assert not np.any(np.isinf(pitch_features))
    assert np.all(pitch_features == 0.0)


def test_extract_pitch_unvoiced_audio():
    """Verify unvoiced noise audio safely returns finite values without NaN/Inf."""
    sr = 16_000
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.01, sr * 1).astype(np.float32)

    pitch_features = extract_pitch_features(noise, sample_rate=sr)

    assert isinstance(pitch_features, np.ndarray)
    assert pitch_features.shape == (4,)
    assert pitch_features.dtype == np.float32
    assert np.all(np.isfinite(pitch_features))


def test_extract_pitch_from_preprocessed_audio(synthetic_wav_file: Path):
    """Verify pitch extraction accepts PreprocessedAudio container."""
    preprocessed = preprocess_audio(synthetic_wav_file)

    pitch_features = extract_pitch_features(preprocessed)

    assert isinstance(pitch_features, np.ndarray)
    assert pitch_features.shape == (4,)
    assert pitch_features.dtype == np.float32
    assert np.all(np.isfinite(pitch_features))


def test_extract_pitch_from_audio_recording(synthetic_wav_file: Path):
    """Verify pitch extraction accepts AudioRecording container."""
    recording = load_wav_file(synthetic_wav_file)

    pitch_features = extract_pitch_features(recording)

    assert isinstance(pitch_features, np.ndarray)
    assert pitch_features.shape == (4,)
    assert pitch_features.dtype == np.float32
    assert np.all(np.isfinite(pitch_features))


def test_extract_pitch_invalid_inputs():
    """Verify safe validation against empty, non-finite, multi-dimensional, and invalid bounds."""
    # Empty audio
    with pytest.raises(AudioDataError, match="empty"):
        extract_pitch_features(np.array([], dtype=np.float32))

    # Non-finite values
    with pytest.raises(AudioDataError, match="non-finite"):
        extract_pitch_features(np.array([0.1, np.nan], dtype=np.float32))

    # 2D stereo array
    with pytest.raises(AudioDataError, match="1D"):
        extract_pitch_features(np.zeros((16000, 2), dtype=np.float32))

    # Invalid sample rate
    with pytest.raises(AudioSampleRateError, match="positive"):
        extract_pitch_features(np.ones(1000, dtype=np.float32), sample_rate=0)

    # Invalid pitch frequency bounds
    with pytest.raises(ValueError, match="Invalid pitch bounds"):
        extract_pitch_features(np.ones(1000, dtype=np.float32), fmin=-10.0, fmax=500.0)

    with pytest.raises(ValueError, match="Invalid pitch bounds"):
        extract_pitch_features(np.ones(1000, dtype=np.float32), fmin=500.0, fmax=100.0)

    # Unsupported type
    with pytest.raises(TypeError, match="Unsupported audio input type"):
        extract_pitch_features(12345)  # type: ignore


def test_get_pitch_feature_names():
    """Verify pitch feature names list."""
    names = get_pitch_feature_names()
    assert names == ["pitch_mean", "pitch_std", "pitch_min", "pitch_max"]
    assert len(names) == 4


def test_extract_pitch_dict():
    """Verify extract_pitch_dict produces clear dictionary mapping."""
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (np.sin(2 * np.pi * 300 * t) * 0.7).astype(np.float32)

    feat_dict = extract_pitch_dict(audio, sample_rate=sr)

    assert isinstance(feat_dict, dict)
    assert set(feat_dict.keys()) == {"pitch_mean", "pitch_std", "pitch_min", "pitch_max"}
    for k, v in feat_dict.items():
        assert isinstance(k, str)
        assert isinstance(v, float)
        assert np.isfinite(v)


def test_compute_pitch_frames():
    """Verify compute_pitch_frames returns f0 track, voiced flags, and voicing probabilities."""
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (np.sin(2 * np.pi * 250 * t) * 0.8).astype(np.float32)

    f0, voiced_flag, voiced_probs = compute_pitch_frames(audio, sample_rate=sr)

    assert isinstance(f0, np.ndarray)
    assert isinstance(voiced_flag, np.ndarray)
    assert isinstance(voiced_probs, np.ndarray)
    assert f0.ndim == 1
    assert voiced_flag.dtype == bool
    assert len(f0) == len(voiced_flag) == len(voiced_probs)


def test_voice_feature_extractor_pitch_methods():
    """Verify VoiceFeatureExtractor exposes pitch extraction methods."""
    extractor = VoiceFeatureExtractor()
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (np.sin(2 * np.pi * 350 * t) * 0.6).astype(np.float32)

    # Pitch vector
    pitch_vec = extractor.extract_pitch(audio, sample_rate=sr)
    assert pitch_vec.shape == (4,)
    assert pitch_vec.dtype == np.float32

    # Pitch dict
    pitch_dict = extractor.extract_pitch_dict(audio, sample_rate=sr)
    assert len(pitch_dict) == 4

    # Pitch frames
    f0, vf, vp = extractor.compute_pitch_frames(audio, sample_rate=sr)
    assert len(f0) > 0

    # Pitch names
    names = extractor.get_pitch_feature_names()
    assert names == ["pitch_mean", "pitch_std", "pitch_min", "pitch_max"]


def test_mfcc_behavior_unmodified():
    """Verify MFCC feature extraction behavior and output remain completely unchanged."""
    sr = 16_000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)

    mfcc_feats = extract_mfcc_features(audio, sample_rate=sr)
    assert mfcc_feats.shape == (26,)
    assert mfcc_feats.dtype == np.float32
    assert np.all(np.isfinite(mfcc_feats))


# ===========================================================================
# Tests — RMS Energy Feature Extraction (Commit 10)
# ===========================================================================

def test_extract_energy_normal_audio():
    """Verify energy extraction produces fixed-length 3D vector [mean, std, range]."""
    sr = 16_000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    # Amplitude modulated wave to give variation/range
    carrier = np.sin(2 * np.pi * 440 * t)
    modulator = 0.5 * (1.0 + np.sin(2 * np.pi * 2 * t))
    audio = (carrier * modulator * 0.8).astype(np.float32)

    energy_features = extract_energy_features(audio, sample_rate=sr)

    assert isinstance(energy_features, np.ndarray)
    assert energy_features.shape == (3,)
    assert energy_features.dtype == np.float32
    assert np.all(np.isfinite(energy_features))

    mean_val, std_val, range_val = energy_features
    assert mean_val > 0.0
    assert std_val > 0.0
    assert range_val > 0.0
    assert range_val >= std_val


def test_extract_energy_silent_audio():
    """Verify completely silent audio safely returns zero vector without NaN/Inf."""
    sr = 16_000
    silent_audio = np.zeros(sr * 3, dtype=np.float32)

    energy_features = extract_energy_features(silent_audio, sample_rate=sr)

    assert isinstance(energy_features, np.ndarray)
    assert energy_features.shape == (3,)
    assert energy_features.dtype == np.float32
    assert not np.any(np.isnan(energy_features))
    assert not np.any(np.isinf(energy_features))
    assert np.all(energy_features == 0.0)


def test_extract_energy_from_preprocessed_audio(synthetic_wav_file: Path):
    """Verify energy extraction accepts PreprocessedAudio container."""
    preprocessed = preprocess_audio(synthetic_wav_file)

    energy_features = extract_energy_features(preprocessed)

    assert isinstance(energy_features, np.ndarray)
    assert energy_features.shape == (3,)
    assert energy_features.dtype == np.float32
    assert np.all(np.isfinite(energy_features))


def test_extract_energy_from_audio_recording(synthetic_wav_file: Path):
    """Verify energy extraction accepts AudioRecording container."""
    recording = load_wav_file(synthetic_wav_file)

    energy_features = extract_energy_features(recording)

    assert isinstance(energy_features, np.ndarray)
    assert energy_features.shape == (3,)
    assert energy_features.dtype == np.float32
    assert np.all(np.isfinite(energy_features))


def test_extract_energy_invalid_inputs():
    """Verify safe validation against empty, non-finite, multi-dimensional, and invalid sizes."""
    # Empty audio
    with pytest.raises(AudioDataError, match="empty"):
        extract_energy_features(np.array([], dtype=np.float32))

    # Non-finite values
    with pytest.raises(AudioDataError, match="non-finite"):
        extract_energy_features(np.array([0.1, np.nan], dtype=np.float32))

    # 2D stereo array
    with pytest.raises(AudioDataError, match="1D"):
        extract_energy_features(np.zeros((16000, 2), dtype=np.float32))

    # Invalid sample rate
    with pytest.raises(AudioSampleRateError, match="positive"):
        extract_energy_features(np.ones(1000, dtype=np.float32), sample_rate=0)

    # Invalid frame or hop length
    with pytest.raises(ValueError, match="positive"):
        extract_energy_features(np.ones(1000, dtype=np.float32), frame_length=-2048)

    with pytest.raises(ValueError, match="positive"):
        extract_energy_features(np.ones(1000, dtype=np.float32), hop_length=0)

    # Unsupported type
    with pytest.raises(TypeError, match="Unsupported audio input type"):
        extract_energy_features("not_audio")  # type: ignore


def test_get_energy_feature_names():
    """Verify energy feature names list."""
    names = get_energy_feature_names()
    assert names == ["energy_mean", "energy_std", "energy_range"]
    assert len(names) == 3


def test_extract_energy_dict():
    """Verify extract_energy_dict produces clear dictionary mapping and alias support."""
    sr = 16_000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.7).astype(np.float32)

    feat_dict = extract_energy_dict(audio, sample_rate=sr)

    assert isinstance(feat_dict, dict)
    assert len(feat_dict) == 3
    assert "energy_mean" in feat_dict
    assert "energy_std" in feat_dict
    assert "energy_range" in feat_dict
    # Verify transparent alias access for variation
    assert feat_dict["energy_variation"] == feat_dict["energy_range"]
    assert feat_dict.get("energy_variation") == feat_dict["energy_range"]
    for k, v in feat_dict.items():
        assert isinstance(k, str)
        assert isinstance(v, float)
        assert np.isfinite(v)


def test_compute_energy_frames():
    """Verify compute_energy_frames returns 2D RMS energy contour."""
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.6).astype(np.float32)

    frames = compute_energy_frames(audio, sample_rate=sr)

    assert isinstance(frames, np.ndarray)
    assert frames.ndim == 2
    assert frames.shape[0] == 1
    assert frames.shape[1] > 0
    assert frames.dtype == np.float32
    assert np.all(np.isfinite(frames))


def test_voice_feature_extractor_energy_methods():
    """Verify VoiceFeatureExtractor exposes energy extraction methods."""
    extractor = VoiceFeatureExtractor()
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (np.sin(2 * np.pi * 350 * t) * 0.5).astype(np.float32)

    # Energy vector
    energy_vec = extractor.extract_energy(audio, sample_rate=sr)
    assert energy_vec.shape == (3,)
    assert energy_vec.dtype == np.float32

    # Energy dict
    energy_dict = extractor.extract_energy_dict(audio, sample_rate=sr)
    assert len(energy_dict) == 3

    # Energy frames
    frames = extractor.compute_energy_frames(audio, sample_rate=sr)
    assert frames.shape[0] == 1
    assert frames.shape[1] > 0

    # Energy names
    names = extractor.get_energy_feature_names()
    assert names == ["energy_mean", "energy_std", "energy_range"]


def test_mfcc_and_pitch_behavior_unmodified():
    """Verify MFCC and pitch extraction continue to operate unmodified."""
    sr = 16_000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)

    mfcc_feats = extract_mfcc_features(audio, sample_rate=sr)
    assert mfcc_feats.shape == (26,)

    pitch_feats = extract_pitch_features(audio, sample_rate=sr)
    assert pitch_feats.shape == (4,)


# ===========================================================================
# Spectral Feature Extraction Tests (Commit 11)
# ===========================================================================

def test_extract_spectral_normal_audio():
    """Verify extract_spectral_features produces 8-dim float32 vector with positive values."""
    sr = 16_000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    # Mix of two frequencies
    audio = (0.6 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

    features = extract_spectral_features(audio, sample_rate=sr)

    assert isinstance(features, np.ndarray)
    assert features.shape == (8,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))

    # Check that spectral centroid, bandwidth, rolloff, and zcr means are positive
    centroid_mean = features[0]
    bandwidth_mean = features[2]
    rolloff_mean = features[4]
    zcr_mean = features[6]

    assert centroid_mean > 0.0
    assert bandwidth_mean > 0.0
    assert rolloff_mean > 0.0
    assert zcr_mean > 0.0


def test_extract_spectral_silent_audio():
    """Verify extract_spectral_features handles complete silence safely without NaN/Inf."""
    sr = 16_000
    silent_audio = np.zeros(sr * 2, dtype=np.float32)

    features = extract_spectral_features(silent_audio, sample_rate=sr)

    assert isinstance(features, np.ndarray)
    assert features.shape == (8,)
    assert features.dtype == np.float32
    assert np.all(features == 0.0)
    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))


def test_extract_spectral_from_preprocessed_audio(synthetic_wav_file: Path):
    """Verify extract_spectral_features accepts PreprocessedAudio container."""
    preprocessed = preprocess_audio(synthetic_wav_file)

    features = extract_spectral_features(preprocessed)
    assert features.shape == (8,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_extract_spectral_from_audio_recording():
    """Verify extract_spectral_features accepts AudioRecording container."""
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (np.sin(2 * np.pi * 500 * t) * 0.8).astype(np.float32)
    rec = AudioRecording(audio_data=audio, sample_rate=sr, channels=1, duration_seconds=1.0)

    features = extract_spectral_features(rec)
    assert features.shape == (8,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_extract_spectral_invalid_inputs():
    """Verify input validation handles empty, non-finite, multidimensional, and bad params."""
    # Empty audio
    with pytest.raises(AudioDataError, match="empty"):
        extract_spectral_features(np.array([], dtype=np.float32))

    # NaN / Inf audio
    with pytest.raises(AudioDataError, match="NaN or Inf"):
        extract_spectral_features(np.array([0.1, np.nan, 0.3], dtype=np.float32))

    with pytest.raises(AudioDataError, match="NaN or Inf"):
        extract_spectral_features(np.array([0.1, np.inf, 0.3], dtype=np.float32))

    # Multidimensional audio
    with pytest.raises(AudioDataError, match="1D mono"):
        extract_spectral_features(np.zeros((16000, 2), dtype=np.float32))

    # Invalid sample rate
    with pytest.raises(AudioSampleRateError, match="positive"):
        extract_spectral_features(np.ones(1000, dtype=np.float32), sample_rate=0)

    # Invalid n_fft or hop_length
    with pytest.raises(ValueError, match="positive"):
        extract_spectral_features(np.ones(1000, dtype=np.float32), n_fft=-2048)

    with pytest.raises(ValueError, match="positive"):
        extract_spectral_features(np.ones(1000, dtype=np.float32), hop_length=0)

    with pytest.raises(ValueError, match="roll_percent"):
        extract_spectral_features(np.ones(1000, dtype=np.float32), roll_percent=0.0)

    # Unsupported type
    with pytest.raises(TypeError, match="Unsupported audio input type"):
        extract_spectral_features("not_audio")  # type: ignore


def test_get_spectral_feature_names():
    """Verify spectral feature names list and order."""
    names = get_spectral_feature_names()
    assert len(names) == 8
    expected = [
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
    ]
    assert names == expected
    assert tuple(names) == SPECTRAL_FEATURE_NAMES


def test_extract_spectral_dict():
    """Verify extract_spectral_dict produces dictionary output with alias support."""
    sr = 16_000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.7).astype(np.float32)

    feat_dict = extract_spectral_dict(audio, sample_rate=sr)

    assert isinstance(feat_dict, dict)
    assert len(feat_dict) == 8

    for name in SPECTRAL_FEATURE_NAMES:
        assert name in feat_dict
        assert isinstance(feat_dict[name], float)
        assert np.isfinite(feat_dict[name])

    # Verify alias lookups
    assert feat_dict["zcr_mean"] == feat_dict["zero_crossing_rate_mean"]
    assert feat_dict["zcr_std"] == feat_dict["zero_crossing_rate_std"]
    assert feat_dict["centroid_mean"] == feat_dict["spectral_centroid_mean"]
    assert feat_dict["centroid_std"] == feat_dict["spectral_centroid_std"]
    assert feat_dict["bandwidth_mean"] == feat_dict["spectral_bandwidth_mean"]
    assert feat_dict["bandwidth_std"] == feat_dict["spectral_bandwidth_std"]
    assert feat_dict["rolloff_mean"] == feat_dict["spectral_rolloff_mean"]
    assert feat_dict["rolloff_std"] == feat_dict["spectral_rolloff_std"]

    assert feat_dict.get("zcr_mean") == feat_dict["zero_crossing_rate_mean"]
    assert feat_dict.get("nonexistent", 99.0) == 99.0


def test_compute_spectral_frames():
    """Verify compute_spectral_frames returns frame matrices for all 4 descriptors."""
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.6).astype(np.float32)

    frames = compute_spectral_frames(audio, sample_rate=sr)

    assert isinstance(frames, dict)
    assert len(frames) == 4
    for key in ("spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zero_crossing_rate"):
        assert key in frames
        mat = frames[key]
        assert isinstance(mat, np.ndarray)
        assert mat.ndim == 2
        assert mat.shape[0] == 1
        assert mat.shape[1] > 0
        assert mat.dtype == np.float32
        assert np.all(np.isfinite(mat))


def test_voice_feature_extractor_spectral_methods():
    """Verify VoiceFeatureExtractor exposes spectral extraction methods."""
    extractor = VoiceFeatureExtractor()
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (np.sin(2 * np.pi * 400 * t) * 0.5).astype(np.float32)

    # Vector extraction
    vec = extractor.extract_spectral(audio, sample_rate=sr)
    assert vec.shape == (8,)
    assert vec.dtype == np.float32

    # Dict extraction
    fdict = extractor.extract_spectral_dict(audio, sample_rate=sr)
    assert len(fdict) == 8
    assert fdict["zcr_mean"] == fdict["zero_crossing_rate_mean"]

    # Frame extraction
    frames = extractor.compute_spectral_frames(audio, sample_rate=sr)
    assert len(frames) == 4

    # Names
    names = extractor.get_spectral_feature_names()
    assert len(names) == 8
    assert names == list(SPECTRAL_FEATURE_NAMES)


def test_mfcc_pitch_energy_behavior_unmodified():
    """Verify MFCC, pitch, and energy extraction continue to operate unmodified."""
    sr = 16_000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)

    mfcc_feats = extract_mfcc_features(audio, sample_rate=sr)
    assert mfcc_feats.shape == (26,)

    pitch_feats = extract_pitch_features(audio, sample_rate=sr)
    assert pitch_feats.shape == (4,)

    energy_feats = extract_energy_features(audio, sample_rate=sr)
    assert energy_feats.shape == (3,)


# ===========================================================================
# Combined Audio Feature Extraction Tests (Commit 12)
# ===========================================================================

def test_extract_combined_features_shape_and_dtype():
    """Verify extract_combined_features returns a 41-dim float32 vector."""
    sr = 16_000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 330 * t) + 0.3 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)

    features = extract_combined_features(audio, sample_rate=sr)

    assert isinstance(features, np.ndarray)
    assert features.shape == (41,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_combined_feature_names_and_ordering():
    """Verify combined feature names count (41) and strict deterministic ordering."""
    names = get_combined_feature_names()
    assert len(names) == 41
    assert len(COMBINED_FEATURE_NAMES) == 41
    assert len(AUDIO_FEATURE_NAMES) == 41
    assert names == list(COMBINED_FEATURE_NAMES)
    assert get_audio_feature_names() == names

    # Sub-component block checks
    mfcc_names = get_mfcc_feature_names()
    pitch_names = get_pitch_feature_names()
    energy_names = get_energy_feature_names()
    spectral_names = get_spectral_feature_names()

    assert names[:26] == mfcc_names
    assert names[26:30] == pitch_names
    assert names[30:33] == energy_names
    assert names[33:41] == spectral_names


def test_combined_component_consistency():
    """Verify combined feature extractor exactly matches individual extractors."""
    sr = 16_000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    audio = (0.6 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

    combined = extract_combined_features(audio, sample_rate=sr)
    mfcc = extract_mfcc_features(audio, sample_rate=sr)
    pitch = extract_pitch_features(audio, sample_rate=sr)
    energy = extract_energy_features(audio, sample_rate=sr)
    spectral = extract_spectral_features(audio, sample_rate=sr)

    assert np.allclose(combined[:26], mfcc, rtol=1e-5, atol=1e-5)
    assert np.allclose(combined[26:30], pitch, rtol=1e-5, atol=1e-5)
    assert np.allclose(combined[30:33], energy, rtol=1e-5, atol=1e-5)
    assert np.allclose(combined[33:41], spectral, rtol=1e-5, atol=1e-5)


def test_combined_features_determinism():
    """Verify combined feature extraction is deterministic across multiple calls."""
    sr = 16_000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    audio = (np.sin(2 * np.pi * 500 * t) * 0.7).astype(np.float32)

    vec1 = extract_combined_features(audio, sample_rate=sr)
    vec2 = extract_combined_features(audio, sample_rate=sr)

    assert np.array_equal(vec1, vec2)


def test_extract_combined_features_silent_audio():
    """Verify extract_combined_features safely processes silent audio without NaN/Inf."""
    sr = 16_000
    silent_audio = np.zeros(sr * 2, dtype=np.float32)

    features = extract_combined_features(silent_audio, sample_rate=sr)

    assert isinstance(features, np.ndarray)
    assert features.shape == (41,)
    assert features.dtype == np.float32
    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))

    # Pitch, energy, and spectral components should be 0.0 for silent audio
    assert np.all(features[26:30] == 0.0)  # pitch
    assert np.all(features[30:33] == 0.0)  # energy
    assert np.all(features[33:41] == 0.0)  # spectral
    # MFCC should be finite
    assert np.all(np.isfinite(features[:26]))


def test_extract_combined_features_invalid_inputs():
    """Verify input validation handles empty, non-finite, multidimensional, and bad params."""
    # Empty audio
    with pytest.raises(AudioDataError, match="empty"):
        extract_combined_features(np.array([], dtype=np.float32))

    # NaN / Inf audio
    with pytest.raises(AudioDataError, match="NaN or Inf"):
        extract_combined_features(np.array([0.1, np.nan, 0.3], dtype=np.float32))

    with pytest.raises(AudioDataError, match="NaN or Inf"):
        extract_combined_features(np.array([0.1, np.inf, 0.3], dtype=np.float32))

    # Multidimensional audio
    with pytest.raises(AudioDataError, match="1D mono"):
        extract_combined_features(np.zeros((16000, 2), dtype=np.float32))

    # Invalid sample rate
    with pytest.raises(AudioSampleRateError, match="positive"):
        extract_combined_features(np.ones(1000, dtype=np.float32), sample_rate=0)

    # Unsupported type
    with pytest.raises(TypeError, match="Unsupported audio input type"):
        extract_combined_features("not_audio")  # type: ignore


def test_extract_combined_features_containers(synthetic_wav_file: Path):
    """Verify combined extraction accepts PreprocessedAudio and AudioRecording containers."""
    # PreprocessedAudio
    preprocessed = preprocess_audio(synthetic_wav_file)
    feats_prep = extract_combined_features(preprocessed)
    assert feats_prep.shape == (41,)
    assert feats_prep.dtype == np.float32
    assert np.all(np.isfinite(feats_prep))

    # AudioRecording
    rec = AudioRecording(
        audio_data=preprocessed.audio_data,
        sample_rate=preprocessed.sample_rate,
        channels=1,
        duration_seconds=preprocessed.duration_seconds,
    )
    feats_rec = extract_combined_features(rec)
    assert feats_rec.shape == (41,)
    assert feats_rec.dtype == np.float32
    assert np.all(np.isfinite(feats_rec))


def test_extract_combined_dict_and_container():
    """Verify extract_combined_dict produces 41-element CombinedFeatureDict with properties."""
    sr = 16_000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.7).astype(np.float32)

    feat_dict = extract_combined_dict(audio, sample_rate=sr)

    assert isinstance(feat_dict, CombinedFeatureDict)
    assert len(feat_dict) == 41

    # Check all names are present and finite
    for name in COMBINED_FEATURE_NAMES:
        assert name in feat_dict
        assert isinstance(feat_dict[name], float)
        assert np.isfinite(feat_dict[name])

    # Check sub-dictionary accessors
    assert len(feat_dict.mfcc) == 26
    assert len(feat_dict.pitch) == 4
    assert len(feat_dict.energy) == 3
    assert len(feat_dict.spectral) == 8

    # Check to_vector
    vec = feat_dict.to_vector()
    assert vec.shape == (41,)
    assert vec.dtype == np.float32
    assert np.allclose(vec, extract_combined_features(audio, sample_rate=sr))

    # Check to_numpy alias
    assert np.array_equal(vec, feat_dict.to_numpy())


def test_voice_feature_extractor_combined_methods():
    """Verify VoiceFeatureExtractor exposes combined extraction methods and aliases."""
    extractor = VoiceFeatureExtractor()
    sr = 16_000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    audio = (np.sin(2 * np.pi * 400 * t) * 0.6).astype(np.float32)

    # Names
    names = extractor.get_combined_feature_names()
    assert len(names) == 41
    assert names == list(COMBINED_FEATURE_NAMES)

    # Vector extraction
    vec = extractor.extract_combined(audio, sample_rate=sr)
    assert vec.shape == (41,)
    assert vec.dtype == np.float32

    # Alias extraction
    vec_all = extractor.extract_all(audio, sample_rate=sr)
    assert np.array_equal(vec, vec_all)

    # Dict extraction
    fdict = extractor.extract_combined_dict(audio, sample_rate=sr)
    assert len(fdict) == 41
    assert isinstance(fdict, CombinedFeatureDict)

    # Alias dict extraction
    fdict_all = extractor.extract_all_dict(audio, sample_rate=sr)
    assert fdict == fdict_all


# ===========================================================================
# Audio Dataset Loader Tests (Commit 13)
# ===========================================================================

def _generate_test_audio_file(
    path: Path,
    frequency: float = 440.0,
    duration_seconds: float = 6.0,
    sample_rate: int = 16_000,
) -> Path:
    """Helper to generate a clean synthetic WAV file for dataset testing."""
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
    waveform = (0.7 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    pcm = (waveform * 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, pcm)
    return path


def test_dataset_loader_valid_synthetic_dataset(tmp_path: Path):
    """Verify loading labeled audio files into a 41-feature dataset DataFrame."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    f1 = _generate_test_audio_file(audio_dir / "P001_session1.wav", frequency=300.0)
    f2 = _generate_test_audio_file(audio_dir / "P002_session1.wav", frequency=450.0)
    f3 = _generate_test_audio_file(audio_dir / "P003_session1.wav", frequency=600.0)

    metadata_path = tmp_path / "metadata.csv"
    meta_df = pd.DataFrame([
        {"filename": f1.name, "participant_id": "P001", "target": "mild_stress"},
        {"filename": f2.name, "participant_id": "P002", "target": "moderate_stress"},
        {"filename": f3.name, "participant_id": "P003", "target": "severe_stress"},
    ])
    meta_df.to_csv(metadata_path, index=False)

    df = load_voice_dataset(metadata=metadata_path, audio_dir=audio_dir)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    # Expected columns: participant_id (1) + 41 features + target (1) = 43 columns
    expected_columns = ["participant_id"] + list(COMBINED_FEATURE_NAMES) + ["target"]
    assert list(df.columns) == expected_columns
    assert len(df.columns) == 43

    # Participant IDs must be preserved anonymously
    assert list(df["participant_id"]) == ["P001", "P002", "P003"]

    # Target labels must be preserved without hardcoding or mapping
    assert list(df["target"]) == ["mild_stress", "moderate_stress", "severe_stress"]

    # Feature columns must be float32 and finite
    for col in COMBINED_FEATURE_NAMES:
        assert df[col].dtype == np.float32
        assert not df[col].isna().any()
        assert np.all(np.isfinite(df[col]))


def test_dataset_loader_component_consistency(tmp_path: Path):
    """Verify DataFrame row values match direct pipeline preprocessing and extraction."""
    f1 = _generate_test_audio_file(tmp_path / "P001_check.wav", frequency=440.0)
    meta_df = pd.DataFrame([
        {"filename": f1.name, "participant_id": "P001", "target": 1},
    ])

    df = load_voice_dataset(metadata=meta_df, audio_dir=tmp_path)

    # Compute directly via preprocessing + combined feature pipeline
    prep = preprocess_audio(f1)
    direct_feats = extract_combined_features(prep)

    extracted_row = df.loc[0, list(COMBINED_FEATURE_NAMES)].to_numpy(dtype=np.float32)
    assert np.allclose(extracted_row, direct_feats, atol=1e-5)


def test_dataset_loader_missing_metadata_file(tmp_path: Path):
    """Verify DatasetNotFoundError is raised when metadata file does not exist."""
    missing_meta = tmp_path / "nonexistent_metadata.csv"
    with pytest.raises(DatasetNotFoundError, match="Metadata file does not exist"):
        load_voice_dataset(metadata=missing_meta, audio_dir=tmp_path)


def test_dataset_loader_empty_metadata(tmp_path: Path):
    """Verify DatasetMetadataError is raised when metadata is empty."""
    empty_csv = tmp_path / "empty_metadata.csv"
    pd.DataFrame().to_csv(empty_csv, index=False)

    with pytest.raises(DatasetMetadataError, match="empty"):
        load_voice_dataset(metadata=empty_csv, audio_dir=tmp_path)

    with pytest.raises(DatasetMetadataError, match="empty"):
        load_voice_dataset(metadata=pd.DataFrame(), audio_dir=tmp_path)


def test_dataset_loader_missing_columns(tmp_path: Path):
    """Verify DatasetMetadataError is raised when required columns cannot be found."""
    f1 = _generate_test_audio_file(tmp_path / "P001_test.wav")

    # Missing target label column
    bad_meta1 = pd.DataFrame([
        {"filename": f1.name, "participant_id": "P001"},
    ])
    with pytest.raises(DatasetMetadataError, match="target label"):
        load_voice_dataset(metadata=bad_meta1, audio_dir=tmp_path)

    # Missing filename column
    bad_meta2 = pd.DataFrame([
        {"participant_id": "P001", "target": "stress"},
    ])
    with pytest.raises(DatasetMetadataError, match="audio filename"):
        load_voice_dataset(metadata=bad_meta2, audio_dir=tmp_path)


def test_dataset_loader_missing_audio_file(tmp_path: Path):
    """Verify DatasetAudioNotFoundError is raised when an audio file does not exist."""
    meta_df = pd.DataFrame([
        {"filename": "does_not_exist.wav", "participant_id": "P001", "target": 0},
    ])

    with pytest.raises(DatasetAudioNotFoundError, match="does not exist"):
        load_voice_dataset(metadata=meta_df, audio_dir=tmp_path, on_error="raise")

    # Test on_error='skip' skips row
    with pytest.warns(UserWarning, match="does not exist"):
        # If all rows skipped, raises DatasetMetadataError
        with pytest.raises(DatasetMetadataError, match="No valid audio samples"):
            load_voice_dataset(metadata=meta_df, audio_dir=tmp_path, on_error="skip")


def test_dataset_loader_invalid_target_label(tmp_path: Path):
    """Verify DatasetLabelError is raised when target label is null/empty."""
    f1 = _generate_test_audio_file(tmp_path / "P001_lbl.wav")

    # None / NaN label
    meta_nan = pd.DataFrame([
        {"filename": f1.name, "participant_id": "P001", "target": None},
    ])
    with pytest.raises(DatasetLabelError, match="Invalid target label"):
        load_voice_dataset(metadata=meta_nan, audio_dir=tmp_path, on_error="raise")

    # Empty string label
    meta_empty_str = pd.DataFrame([
        {"filename": f1.name, "participant_id": "P001", "target": "   "},
    ])
    with pytest.raises(DatasetLabelError, match="Invalid target label"):
        load_voice_dataset(metadata=meta_empty_str, audio_dir=tmp_path, on_error="raise")


def test_dataset_loader_corrupt_audio_file(tmp_path: Path):
    """Verify DatasetExtractionError is raised when audio file cannot be loaded."""
    corrupt_file = tmp_path / "corrupt.wav"
    corrupt_file.write_bytes(b"THIS IS NOT A VALID WAV FILE")

    meta_df = pd.DataFrame([
        {"filename": corrupt_file.name, "participant_id": "P001", "target": 1},
    ])

    with pytest.raises(DatasetExtractionError, match="Feature extraction failed"):
        load_voice_dataset(metadata=meta_df, audio_dir=tmp_path, on_error="raise")


def test_dataset_loader_on_error_skip(tmp_path: Path):
    """Verify on_error='skip' processes valid audio while bypassing invalid files."""
    valid_file = _generate_test_audio_file(tmp_path / "valid.wav", frequency=400.0)
    corrupt_file = tmp_path / "corrupt.wav"
    corrupt_file.write_bytes(b"INVALID_WAV")

    meta_df = pd.DataFrame([
        {"filename": valid_file.name, "participant_id": "P001", "target": "ok"},
        {"filename": corrupt_file.name, "participant_id": "P002", "target": "corrupt"},
    ])

    with pytest.warns(UserWarning):
        df = load_voice_dataset(metadata=meta_df, audio_dir=tmp_path, on_error="skip")

    assert len(df) == 1
    assert df.loc[0, "participant_id"] == "P001"
    assert df.loc[0, "target"] == "ok"


def test_dataset_loader_custom_column_names(tmp_path: Path):
    """Verify explicit custom column names are supported properly."""
    f1 = _generate_test_audio_file(tmp_path / "sample_a.wav", frequency=350.0)

    custom_meta = pd.DataFrame([
        {"audio_clip": f1.name, "subject_code": "ANON_42", "risk_rating": 3.5},
    ])

    loader = AudioDatasetLoader(
        filename_column="audio_clip",
        participant_id_column="subject_code",
        label_column="risk_rating",
    )
    df = loader.load(metadata=custom_meta, audio_dir=tmp_path)

    assert len(df) == 1
    assert df.columns[0] == "subject_code"
    assert df.columns[-1] == "risk_rating"
    assert df.loc[0, "subject_code"] == "ANON_42"
    assert df.loc[0, "risk_rating"] == 3.5


def test_dataset_loader_anonymous_participant_id_fallback(tmp_path: Path):
    """Verify participant ID is extracted from filename if column is missing/empty."""
    f1 = _generate_test_audio_file(tmp_path / "P009_task1.wav", frequency=420.0)

    meta_df = pd.DataFrame([
        {"filename": f1.name, "participant_id": "", "target": "high"},
    ])

    df = load_voice_dataset(metadata=meta_df, audio_dir=tmp_path)
    assert len(df) == 1
    assert df.loc[0, "participant_id"] == "P009"


def test_dataset_loader_save_and_reload(tmp_path: Path):
    """Verify saving the extracted dataset DataFrame to CSV and reloading."""
    f1 = _generate_test_audio_file(tmp_path / "P001_rec.wav", frequency=500.0)
    meta_df = pd.DataFrame([
        {"filename": f1.name, "participant_id": "P001", "target": 0},
    ])

    loader = AudioDatasetLoader(audio_dir=tmp_path)
    df = loader.load(metadata=meta_df)

    out_csv = tmp_path / "output_features.csv"
    saved_path = loader.save(df, out_csv)
    assert saved_path.exists()

    reloaded = pd.read_csv(saved_path)
    assert len(reloaded) == 1
    assert len(reloaded.columns) == 43
    assert list(reloaded.columns) == list(df.columns)


def test_voice_data_module_reexports():
    """Verify src.ai_model.voice.data re-exports all dataset symbols properly."""
    from src.ai_model.voice.data import (
        AudioDatasetLoader as A1,
        VoiceDatasetLoader as V1,
        DatasetError as E1,
        load_voice_dataset as L1,
    )
    assert A1 is AudioDatasetLoader
    assert V1 is VoiceDatasetLoader
    assert E1 is DatasetError
    assert L1 is load_voice_dataset


# ===========================================================================
# Logistic Regression Baseline Model Tests (Commit 14)
# ===========================================================================

def _create_synthetic_feature_df(
    n_samples: int = 20,
    target_classes: Tuple[Any, ...] = ("low_risk", "high_risk"),
    random_seed: int = 42,
) -> pd.DataFrame:
    """Helper to generate a synthetic feature DataFrame matching dataset loader output."""
    rng = np.random.RandomState(random_seed)
    data: Dict[str, Any] = {
        "participant_id": [f"P{i:03d}" for i in range(1, n_samples + 1)]
    }

    # Generate 41 feature columns
    for name in COMBINED_FEATURE_NAMES:
        data[name] = rng.randn(n_samples).astype(np.float32)

    # Assign balanced target labels
    labels = [target_classes[i % len(target_classes)] for i in range(n_samples)]
    data["target"] = labels

    return pd.DataFrame(data)


def test_separate_features_and_target_valid():
    """Verify separate_features_and_target decouples feature matrix from metadata and label."""
    df = _create_synthetic_feature_df(n_samples=10, target_classes=("class_0", "class_1"))

    X, y = separate_features_and_target(df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (10, 41)
    assert len(y) == 10
    assert list(X.columns) == list(COMBINED_FEATURE_NAMES)
    assert "participant_id" not in X.columns
    assert "target" not in X.columns
    assert list(y) == list(df["target"])


def test_separate_features_and_target_custom_columns():
    """Verify separate_features_and_target supports custom target and id column names."""
    df = _create_synthetic_feature_df(n_samples=8)
    df = df.rename(columns={"target": "stress_outcome", "participant_id": "student_anon_id"})

    X, y = separate_features_and_target(
        df,
        target_column="stress_outcome",
        id_column="student_anon_id",
    )

    assert X.shape == (8, 41)
    assert len(y) == 8
    assert "student_anon_id" not in X.columns
    assert "stress_outcome" not in X.columns
    assert y.name == "stress_outcome"


def test_separate_features_and_target_explicit_feature_columns():
    """Verify separate_features_and_target respects explicit feature_columns subset."""
    df = _create_synthetic_feature_df(n_samples=5)
    selected = ["mfcc_1_mean", "energy_mean", "spectral_centroid_mean"]

    X, y = separate_features_and_target(df, feature_columns=selected)

    assert list(X.columns) == selected
    assert X.shape == (5, 3)


def test_separate_features_and_target_invalid_inputs():
    """Verify validation errors for invalid inputs to separate_features_and_target."""
    with pytest.raises(TypeError, match="pandas DataFrame"):
        separate_features_and_target("not_a_df")  # type: ignore

    with pytest.raises(ModelInputError, match="empty"):
        separate_features_and_target(pd.DataFrame())

    with pytest.raises(ModelInputError, match="Target column 'target' not found"):
        separate_features_and_target(pd.DataFrame({"col_a": [1, 2]}))

    with pytest.raises(ModelInputError, match="Specified feature columns not found"):
        df = _create_synthetic_feature_df(n_samples=4)
        separate_features_and_target(df, feature_columns=["nonexistent_feat"])


def test_logistic_regression_fit_and_predict_binary():
    """Verify VoiceLogisticRegression fit, predict, and predict_proba on binary target."""
    df = _create_synthetic_feature_df(n_samples=24, target_classes=("low", "high"), random_seed=123)

    model = VoiceLogisticRegression(random_state=42, max_iter=500)
    assert not model.is_fitted

    # Fit directly on combined dataset DataFrame
    fitted_model = model.fit(df)
    assert fitted_model is model
    assert model.is_fitted
    assert set(model.classes_) == {"high", "low"}
    assert model.n_features_in_ == 41

    # Predict
    preds = model.predict(df)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (24,)
    for p in preds:
        assert p in ("low", "high")

    # Predict Proba
    proba = model.predict_proba(df)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (24, 2)
    # Check row-wise sum to 1.0
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    # Predict Proba Dict
    proba_dicts = model.predict_proba_dict(df)
    assert len(proba_dicts) == 24
    for d in proba_dicts:
        assert isinstance(d, dict)
        assert set(d.keys()) == set(model.classes_)
        assert abs(sum(d.values()) - 1.0) < 1e-5


def test_logistic_regression_multiclass_arbitrary_labels():
    """Verify model handles multiclass without hardcoding labels or assuming binary/integers."""
    labels = ("minimal", "mild", "moderate", "severe")
    df = _create_synthetic_feature_df(n_samples=32, target_classes=labels, random_seed=99)

    model = VoiceLogisticRegression(random_state=42)
    X, y = separate_features_and_target(df)
    model.fit(X, y)

    assert model.is_fitted
    assert len(model.classes_) == 4
    assert set(model.classes_) == set(labels)

    preds = model.predict(X)
    assert len(preds) == 32
    for p in preds:
        assert p in labels

    proba = model.predict_proba(X)
    assert proba.shape == (32, 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_logistic_regression_configurable_random_state_and_max_iter():
    """Verify configurable random_state and max_iter parameters operate as expected."""
    df = _create_synthetic_feature_df(n_samples=20, random_seed=7)
    X, y = separate_features_and_target(df)

    # Custom configuration
    model1 = VoiceLogisticRegression(random_state=77, max_iter=250, C=0.5)
    assert model1.random_state == 77
    assert model1.max_iter == 250
    assert model1.C == 0.5

    model1.fit(X, y)

    # Identical random_state produces identical coefficients and predictions
    model2 = VoiceLogisticRegression(random_state=77, max_iter=250, C=0.5)
    model2.fit(X, y)

    assert np.allclose(model1.coef_, model2.coef_)
    assert np.allclose(model1.intercept_, model2.intercept_)
    assert np.array_equal(model1.predict(X), model2.predict(X))


def test_logistic_regression_unfitted_raises_error():
    """Verify calling predict, predict_proba, or accessing parameters before fit raises error."""
    model = VoiceLogisticRegression()
    dummy_X = np.ones((5, 41), dtype=np.float32)

    with pytest.raises(ModelNotFittedError, match="not fitted yet"):
        model.predict(dummy_X)

    with pytest.raises(ModelNotFittedError, match="not fitted yet"):
        model.predict_proba(dummy_X)

    with pytest.raises(ModelNotFittedError, match="not fitted yet"):
        _ = model.coef_

    with pytest.raises(ModelNotFittedError, match="not fitted yet"):
        _ = model.intercept_


def test_logistic_regression_input_validation():
    """Verify rigorous input validation across features, columns, and target."""
    model = VoiceLogisticRegression()

    # Empty features
    with pytest.raises(ModelInputError, match="empty"):
        model.fit(pd.DataFrame(), y=np.array([1, 2]))

    # Target is None and X cannot be decoupled
    with pytest.raises(ModelInputError, match="Target column 'target' not found"):
        model.fit(pd.DataFrame({"feat_1": [1.0, 2.0]}))

    # Features contains NaN
    df_nan = _create_synthetic_feature_df(n_samples=10)
    df_nan.loc[2, "mfcc_1_mean"] = np.nan
    with pytest.raises(ModelInputError, match="NaN"):
        model.fit(df_nan)

    # Features contains non-numeric column
    df_str = _create_synthetic_feature_df(n_samples=10)
    df_str["bad_feature"] = ["str"] * 10
    with pytest.raises(ModelInputError, match="non-numeric"):
        model.fit(df_str.drop(columns=["target"]), y=df_str["target"])

    # Target contains null
    df_valid = _create_synthetic_feature_df(n_samples=10)
    X, y = separate_features_and_target(df_valid)
    y_null = y.copy()
    y_null.iloc[1] = None
    with pytest.raises(ModelInputError, match="null/NaN"):
        model.fit(X, y_null)

    # Target with only 1 unique class
    y_single = ["class_a"] * 10
    with pytest.raises(ModelInputError, match="at least 2 distinct classes"):
        model.fit(X, y_single)

    # Mismatched sample counts
    with pytest.raises(ModelInputError, match="Sample count mismatch"):
        model.fit(X, y.iloc[:5])

    # Missing expected feature columns during predict
    model.fit(X, y)
    X_missing = X.drop(columns=["mfcc_1_mean"])
    with pytest.raises(ModelInputError, match="missing fitted feature columns"):
        model.predict(X_missing)


def test_logistic_regression_single_sample_prediction():
    """Verify predicting on a single sample dictionary or 1D vector."""
    df = _create_synthetic_feature_df(n_samples=16, random_seed=42)
    model = VoiceLogisticRegression(random_state=42)
    model.fit(df)

    # Single-sample dict
    single_dict = {col: float(df.loc[0, col]) for col in COMBINED_FEATURE_NAMES}
    pred_dict = model.predict(single_dict)
    assert pred_dict.shape == (1,)
    proba_dict = model.predict_proba(single_dict)
    assert proba_dict.shape == (1, 2)
    assert np.allclose(proba_dict.sum(), 1.0)

    # Single-sample 1D numpy vector
    single_vec = np.array([single_dict[c] for c in COMBINED_FEATURE_NAMES], dtype=np.float32)
    pred_vec = model.predict(single_vec)
    assert pred_vec.shape == (1,)
    assert pred_vec[0] == pred_dict[0]


def test_logistic_regression_feature_scaling_toggle():
    """Verify scale_features=True uses StandardScaler, scale_features=False operates unscaled."""
    df = _create_synthetic_feature_df(n_samples=16)
    X, y = separate_features_and_target(df)

    # Scaled (default)
    model_scaled = VoiceLogisticRegression(scale_features=True)
    model_scaled.fit(X, y)
    assert model_scaled.scaler_ is not None

    # Unscaled
    model_unscaled = VoiceLogisticRegression(scale_features=False)
    model_unscaled.fit(X, y)
    assert model_unscaled.scaler_ is None

    assert model_unscaled.is_fitted
    preds = model_unscaled.predict(X)
    assert len(preds) == 16


def test_logistic_regression_dataset_loader_integration(tmp_path: Path):
    """End-to-end integration: load dataset via AudioDatasetLoader and fit VoiceLogisticRegression."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Create 4 synthetic audio files
    f1 = _generate_test_audio_file(audio_dir / "P001.wav", frequency=300.0)
    f2 = _generate_test_audio_file(audio_dir / "P002.wav", frequency=350.0)
    f3 = _generate_test_audio_file(audio_dir / "P003.wav", frequency=500.0)
    f4 = _generate_test_audio_file(audio_dir / "P004.wav", frequency=550.0)

    meta_df = pd.DataFrame([
        {"filename": f1.name, "participant_id": "P001", "target": "calm"},
        {"filename": f2.name, "participant_id": "P002", "target": "calm"},
        {"filename": f3.name, "participant_id": "P003", "target": "stressed"},
        {"filename": f4.name, "participant_id": "P004", "target": "stressed"},
    ])
    meta_path = tmp_path / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    # 1. Load dataset with AudioDatasetLoader
    dataset_df = load_voice_dataset(metadata=meta_path, audio_dir=audio_dir)
    assert len(dataset_df) == 4
    assert len(dataset_df.columns) == 43

    # 2. Fit VoiceLogisticRegression directly on dataset_df
    clf = VoiceLogisticRegression(random_state=42, max_iter=200)
    clf.fit(dataset_df)

    assert clf.is_fitted
    assert set(clf.classes_) == {"calm", "stressed"}

    # 3. Predict on dataset_df
    predictions = clf.predict(dataset_df)
    assert len(predictions) == 4

    probabilities = clf.predict_proba(dataset_df)
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_voice_model_aliases():
    """Verify class aliases match VoiceLogisticRegression."""
    assert VoiceBaselineClassifier is VoiceLogisticRegression
    assert VoiceLogisticRegressionClassifier is VoiceLogisticRegression


# ===========================================================================
# Random Forest Model Tests (Commit 15)
# ===========================================================================

def test_random_forest_fit_and_predict_binary():
    """Verify VoiceRandomForest fits on binary labels and produces valid predictions and probabilities."""
    df = _create_synthetic_feature_df(n_samples=24, target_classes=("low_risk", "high_risk"), random_seed=42)
    clf = VoiceRandomForest(n_estimators=25, random_state=42)

    assert not clf.is_fitted
    clf.fit(df)
    assert clf.is_fitted
    assert set(clf.classes_) == {"high_risk", "low_risk"}
    assert clf.n_features_in_ == len(COMBINED_FEATURE_NAMES)

    # Predict class labels
    preds = clf.predict(df)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (24,)
    for p in preds:
        assert p in {"high_risk", "low_risk"}

    # Predict class probabilities
    proba = clf.predict_proba(df)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (24, 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)

    # Predict proba dict
    proba_dicts = clf.predict_proba_dict(df)
    assert len(proba_dicts) == 24
    for item in proba_dicts:
        assert set(item.keys()) == {"high_risk", "low_risk"}
        assert np.isclose(sum(item.values()), 1.0)


def test_random_forest_multiclass_arbitrary_labels():
    """Verify VoiceRandomForest supports multiclass classification with arbitrary string or numeric labels."""
    # 3 string classes
    df_multi = _create_synthetic_feature_df(
        n_samples=30,
        target_classes=("mild", "moderate", "severe"),
        random_seed=42,
    )
    clf = VoiceRandomForest(n_estimators=20, random_state=42)
    clf.fit(df_multi)

    assert clf.is_fitted
    assert len(clf.classes_) == 3
    assert set(clf.classes_) == {"mild", "moderate", "severe"}

    preds = clf.predict(df_multi)
    assert len(preds) == 30
    assert set(np.unique(preds)).issubset({"mild", "moderate", "severe"})

    proba = clf.predict_proba(df_multi)
    assert proba.shape == (30, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)

    # Integer labels (-1, 0, 1)
    df_int = _create_synthetic_feature_df(
        n_samples=21,
        target_classes=(-1, 0, 1),
        random_seed=101,
    )
    clf_int = VoiceRandomForest(n_estimators=15, random_state=42)
    clf_int.fit(df_int)
    assert set(clf_int.classes_) == {-1, 0, 1}
    preds_int = clf_int.predict(df_int)
    assert set(np.unique(preds_int)).issubset({-1, 0, 1})


def test_random_forest_configurable_hyperparameters():
    """Verify VoiceRandomForest honors configurable hyperparameters and produces deterministic output."""
    clf1 = VoiceRandomForest(
        n_estimators=15,
        max_depth=3,
        min_samples_split=4,
        class_weight="balanced",
        random_state=77,
    )
    assert clf1.n_estimators == 15
    assert clf1.max_depth == 3
    assert clf1.min_samples_split == 4
    assert clf1.class_weight == "balanced"
    assert clf1.random_state == 77
    assert clf1.classifier_.n_estimators == 15
    assert clf1.classifier_.max_depth == 3
    assert clf1.classifier_.min_samples_split == 4
    assert clf1.classifier_.class_weight == "balanced"

    df = _create_synthetic_feature_df(n_samples=20, random_seed=42)
    X, y = separate_features_and_target(df)

    clf1.fit(X, y)

    # Identical random_state produces identical predictions and feature importances
    clf2 = VoiceRandomForest(
        n_estimators=15,
        max_depth=3,
        min_samples_split=4,
        class_weight="balanced",
        random_state=77,
    )
    clf2.fit(X, y)

    assert np.array_equal(clf1.predict(X), clf2.predict(X))
    assert np.allclose(clf1.predict_proba(X), clf2.predict_proba(X))
    assert np.allclose(clf1.feature_importances_, clf2.feature_importances_)


def test_random_forest_unfitted_raises_error():
    """Verify calling predict, predict_proba, or accessing feature importances before fit raises error."""
    clf = VoiceRandomForest()
    dummy_X = np.ones((5, 41), dtype=np.float32)

    with pytest.raises(ModelNotFittedError, match="not fitted yet"):
        clf.predict(dummy_X)

    with pytest.raises(ModelNotFittedError, match="not fitted yet"):
        clf.predict_proba(dummy_X)

    with pytest.raises(ModelNotFittedError, match="not fitted yet"):
        clf.predict_proba_dict(dummy_X)

    with pytest.raises(ModelNotFittedError, match="not fitted yet"):
        _ = clf.feature_importances_

    with pytest.raises(ModelNotFittedError, match="not fitted yet"):
        _ = clf.feature_importances_dict


def test_random_forest_input_validation():
    """Verify rigorous input validation across features, columns, and target."""
    clf = VoiceRandomForest()

    # Empty features
    with pytest.raises(ModelInputError, match="empty"):
        clf.fit(pd.DataFrame(), y=np.array([1, 2]))

    # Target is None and X cannot be decoupled
    with pytest.raises(ModelInputError, match="Target column 'target' not found"):
        clf.fit(pd.DataFrame({"feat_1": [1.0, 2.0]}))

    # Features contains NaN
    df_nan = _create_synthetic_feature_df(n_samples=10)
    df_nan.loc[2, "mfcc_1_mean"] = np.nan
    with pytest.raises(ModelInputError, match="NaN"):
        clf.fit(df_nan)

    # Features contains non-numeric column
    df_str = _create_synthetic_feature_df(n_samples=10)
    df_str["bad_feature"] = ["str"] * 10
    with pytest.raises(ModelInputError, match="non-numeric"):
        clf.fit(df_str.drop(columns=["target"]), y=df_str["target"])

    # Target contains null
    df_valid = _create_synthetic_feature_df(n_samples=10)
    X, y = separate_features_and_target(df_valid)
    y_null = y.copy()
    y_null.iloc[1] = None
    with pytest.raises(ModelInputError, match="null/NaN"):
        clf.fit(X, y_null)

    # Target with only 1 unique class
    y_single = ["class_a"] * 10
    with pytest.raises(ModelInputError, match="at least 2 distinct classes"):
        clf.fit(X, y_single)

    # Mismatched sample counts
    with pytest.raises(ModelInputError, match="Sample count mismatch"):
        clf.fit(X, y.iloc[:5])

    # Missing expected feature columns during predict
    clf.fit(X, y)
    X_missing = X.drop(columns=["mfcc_1_mean"])
    with pytest.raises(ModelInputError, match="missing fitted feature columns"):
        clf.predict(X_missing)


def test_random_forest_feature_importances():
    """Verify feature_importances_ and feature_importances_dict properties."""
    df = _create_synthetic_feature_df(n_samples=30, random_seed=42)
    clf = VoiceRandomForest(n_estimators=30, random_state=42)
    clf.fit(df)

    importances = clf.feature_importances_
    assert isinstance(importances, np.ndarray)
    assert len(importances) == len(COMBINED_FEATURE_NAMES)
    assert np.all(importances >= 0.0)
    assert np.isclose(importances.sum(), 1.0)

    imp_dict = clf.feature_importances_dict
    assert isinstance(imp_dict, dict)
    assert len(imp_dict) == len(COMBINED_FEATURE_NAMES)
    for name in COMBINED_FEATURE_NAMES:
        assert name in imp_dict
        assert imp_dict[name] >= 0.0
    assert np.isclose(sum(imp_dict.values()), 1.0)


def test_random_forest_single_sample_prediction():
    """Verify predicting on a single sample dictionary or 1D vector."""
    df = _create_synthetic_feature_df(n_samples=16, random_seed=42)
    clf = VoiceRandomForest(n_estimators=20, random_state=42)
    clf.fit(df)

    # Single-sample dict
    single_dict = {col: float(df.loc[0, col]) for col in COMBINED_FEATURE_NAMES}
    pred_dict = clf.predict(single_dict)
    assert pred_dict.shape == (1,)
    proba_dict = clf.predict_proba(single_dict)
    assert proba_dict.shape == (1, 2)
    assert np.allclose(proba_dict.sum(), 1.0)

    # Single-sample 1D numpy vector
    single_vec = np.array([single_dict[c] for c in COMBINED_FEATURE_NAMES], dtype=np.float32)
    pred_vec = clf.predict(single_vec)
    assert pred_vec.shape == (1,)
    assert pred_vec[0] == pred_dict[0]


def test_random_forest_feature_scaling_toggle():
    """Verify scale_features=False (default) and scale_features=True operate properly."""
    df = _create_synthetic_feature_df(n_samples=16)
    X, y = separate_features_and_target(df)

    # Unscaled (default for Random Forest)
    clf_unscaled = VoiceRandomForest(scale_features=False, n_estimators=10, random_state=42)
    clf_unscaled.fit(X, y)
    assert clf_unscaled.scaler_ is None
    assert clf_unscaled.is_fitted
    preds = clf_unscaled.predict(X)
    assert len(preds) == 16

    # Scaled
    clf_scaled = VoiceRandomForest(scale_features=True, n_estimators=10, random_state=42)
    clf_scaled.fit(X, y)
    assert clf_scaled.scaler_ is not None
    assert clf_scaled.is_fitted
    preds_scaled = clf_scaled.predict(X)
    assert len(preds_scaled) == 16


def test_random_forest_dataset_loader_integration(tmp_path: Path):
    """End-to-end integration: load dataset via AudioDatasetLoader and fit VoiceRandomForest."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Create 4 synthetic audio files
    f1 = _generate_test_audio_file(audio_dir / "P001.wav", frequency=300.0)
    f2 = _generate_test_audio_file(audio_dir / "P002.wav", frequency=350.0)
    f3 = _generate_test_audio_file(audio_dir / "P003.wav", frequency=500.0)
    f4 = _generate_test_audio_file(audio_dir / "P004.wav", frequency=550.0)

    meta_df = pd.DataFrame([
        {"filename": f1.name, "participant_id": "P001", "target": "calm"},
        {"filename": f2.name, "participant_id": "P002", "target": "calm"},
        {"filename": f3.name, "participant_id": "P003", "target": "stressed"},
        {"filename": f4.name, "participant_id": "P004", "target": "stressed"},
    ])
    meta_path = tmp_path / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    # 1. Load dataset with AudioDatasetLoader
    dataset_df = load_voice_dataset(metadata=meta_path, audio_dir=audio_dir)
    assert len(dataset_df) == 4
    assert len(dataset_df.columns) == 43

    # 2. Fit VoiceRandomForest directly on dataset_df
    clf = VoiceRandomForest(n_estimators=15, random_state=42)
    clf.fit(dataset_df)

    assert clf.is_fitted
    assert set(clf.classes_) == {"calm", "stressed"}

    # 3. Predict on dataset_df
    predictions = clf.predict(dataset_df)
    assert len(predictions) == 4

    probabilities = clf.predict_proba(dataset_df)
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_random_forest_aliases_and_inheritance():
    """Verify class aliases and inheritance hierarchy."""
    assert VoiceRandomForestClassifier is VoiceRandomForest
    assert issubclass(VoiceRandomForest, BaseVoiceClassifier)
    assert issubclass(VoiceLogisticRegression, BaseVoiceClassifier)


# ===========================================================================
# Model Evaluation Metrics Tests (Commit 16)
# ===========================================================================

def test_calculate_voice_metrics_binary_arbitrary_labels():
    """Verify calculate_voice_metrics on binary string labels."""
    y_true = ["low_risk", "high_risk", "low_risk", "high_risk", "high_risk", "low_risk"]
    y_pred = ["low_risk", "high_risk", "high_risk", "high_risk", "high_risk", "low_risk"]

    metrics = calculate_voice_metrics(y_true, y_pred)

    assert isinstance(metrics, VoiceModelMetrics)
    assert np.isclose(metrics.accuracy, 5 / 6)
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert metrics.support == 6
    assert set(metrics.classes) == {"low_risk", "high_risk"}

    # Per-class metrics
    assert "low_risk" in metrics.per_class
    assert "high_risk" in metrics.per_class
    assert metrics.per_class["low_risk"]["support"] == 3
    assert metrics.per_class["high_risk"]["support"] == 3

    # Dictionary access and conversion
    metrics_dict = metrics.to_dict()
    assert isinstance(metrics_dict, dict)
    assert "accuracy" in metrics_dict
    assert "macro_f1" in metrics_dict
    assert "per_class" in metrics_dict
    assert metrics["accuracy"] == metrics.accuracy
    assert metrics["macro_f1"] == metrics.macro_f1
    assert "accuracy" in metrics
    assert metrics.get("support") == 6
    assert "accuracy=" in repr(metrics)


def test_calculate_voice_metrics_multiclass_arbitrary_labels():
    """Verify calculate_voice_metrics on multiclass arbitrary string targets."""
    y_true = ["healthy", "healthy", "mild", "mild", "severe", "severe"]
    y_pred = ["healthy", "mild", "mild", "mild", "severe", "healthy"]

    metrics = calculate_voice_metrics(y_true, y_pred)

    assert np.isclose(metrics.accuracy, 4 / 6)
    assert len(metrics.classes) == 3
    assert set(metrics.classes) == {"healthy", "mild", "severe"}

    # Macro F1 should equal the unweighted average of per-class F1
    per_class_f1s = [metrics.per_class[c]["f1"] for c in metrics.classes]
    assert np.isclose(metrics.macro_f1, np.mean(per_class_f1s))

    # Test with macro averaging strategy
    macro_metrics = calculate_voice_metrics(y_true, y_pred, average="macro")
    assert np.isclose(macro_metrics.f1, metrics.macro_f1)
    assert np.isclose(macro_metrics.precision, metrics.macro_precision)
    assert np.isclose(macro_metrics.recall, metrics.macro_recall)


def test_calculate_voice_metrics_numeric_labels():
    """Verify calculate_voice_metrics on numeric / integer targets."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])

    metrics = calculate_voice_metrics(y_true, y_pred)
    assert np.isclose(metrics.accuracy, 5 / 6)
    assert set(metrics.classes) == {0, 1, 2}
    assert metrics.support == 6


def test_calculate_voice_metrics_zero_division_safety():
    """Verify zero-division produces 0.0 scores safely without throwing exceptions or warnings."""
    # class_b is in y_true but never predicted
    y_true = ["class_a", "class_a", "class_b", "class_b"]
    y_pred = ["class_a", "class_a", "class_a", "class_a"]

    metrics = calculate_voice_metrics(y_true, y_pred)

    assert metrics.per_class["class_b"]["precision"] == 0.0
    assert metrics.per_class["class_b"]["recall"] == 0.0
    assert metrics.per_class["class_b"]["f1"] == 0.0
    assert np.isclose(metrics.accuracy, 0.5)
    assert metrics.macro_f1 < 1.0


def test_calculate_voice_metrics_input_validation():
    """Verify rigorous validation of ground truth and prediction inputs."""
    # None inputs
    with pytest.raises(EvaluationInputError, match="cannot be None"):
        calculate_voice_metrics(None, ["a", "b"])

    with pytest.raises(EvaluationInputError, match="cannot be None"):
        calculate_voice_metrics(["a", "b"], None)

    # Empty inputs
    with pytest.raises(EvaluationInputError, match="empty"):
        calculate_voice_metrics([], [])

    # Sample count mismatch
    with pytest.raises(EvaluationInputError, match="mismatch"):
        calculate_voice_metrics(["a", "b", "c"], ["a", "b"])

    # Null values in y_true
    with pytest.raises(EvaluationInputError, match="null/NaN"):
        calculate_voice_metrics(pd.Series(["a", None, "b"]), pd.Series(["a", "a", "b"]))

    # Null values in y_pred
    with pytest.raises(EvaluationInputError, match="null/NaN"):
        calculate_voice_metrics(pd.Series(["a", "b"]), pd.Series([np.nan, "b"]))


def test_evaluate_voice_model_with_logistic_regression():
    """Verify evaluate_voice_model works with VoiceLogisticRegression."""
    df_train = _create_synthetic_feature_df(n_samples=20, random_seed=42)
    df_test = _create_synthetic_feature_df(n_samples=10, random_seed=99)

    model = VoiceLogisticRegression(random_state=42)
    model.fit(df_train)

    # Evaluate on combined DataFrame
    metrics = evaluate_voice_model(model, df_test)
    assert isinstance(metrics, VoiceModelMetrics)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert metrics.support == 10

    # Evaluate on separated features and target
    X_test, y_test = separate_features_and_target(df_test)
    metrics_sep = evaluate_voice_model(model, X=X_test, y=y_test)
    assert metrics_sep.accuracy == metrics.accuracy
    assert metrics_sep.macro_f1 == metrics.macro_f1


def test_evaluate_voice_model_with_random_forest():
    """Verify evaluate_voice_model works with VoiceRandomForest."""
    df_train = _create_synthetic_feature_df(n_samples=24, random_seed=42)
    df_test = _create_synthetic_feature_df(n_samples=12, random_seed=88)

    model = VoiceRandomForest(n_estimators=20, random_state=42)
    model.fit(df_train)

    # Evaluate on combined DataFrame
    metrics = evaluate_voice_model(model, df_test)
    assert isinstance(metrics, VoiceModelMetrics)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert metrics.support == 12

    # Evaluate on separated features and target
    X_test, y_test = separate_features_and_target(df_test)
    metrics_sep = evaluate_voice_model(model, X=X_test, y=y_test)
    assert metrics_sep.accuracy == metrics.accuracy
    assert metrics_sep.macro_f1 == metrics.macro_f1


def test_evaluate_voice_model_error_handling():
    """Verify evaluate_voice_model error handling for unfitted or invalid models."""
    # None model
    with pytest.raises(EvaluationError, match="cannot be None"):
        evaluate_voice_model(None, pd.DataFrame({"feat_1": [1.0]}))

    # Model missing predict method
    class DummyNoPredict:
        pass

    with pytest.raises(EvaluationError, match="does not implement 'predict'"):
        evaluate_voice_model(DummyNoPredict(), pd.DataFrame({"feat_1": [1.0]}), y=np.array([1]))

    # Unfitted VoiceLogisticRegression
    unfitted_lr = VoiceLogisticRegression()
    with pytest.raises(ModelNotFittedError, match="not fitted yet"):
        evaluate_voice_model(unfitted_lr, pd.DataFrame({"feat_1": [1.0]}), y=np.array([1]))

    # Unfitted VoiceRandomForest
    unfitted_rf = VoiceRandomForest()
    with pytest.raises(ModelNotFittedError, match="not fitted yet"):
        evaluate_voice_model(unfitted_rf, pd.DataFrame({"feat_1": [1.0]}), y=np.array([1]))

    # Non-DataFrame without y
    dummy_fitted = VoiceLogisticRegression()
    df_train = _create_synthetic_feature_df(n_samples=10, random_seed=42)
    dummy_fitted.fit(df_train)
    with pytest.raises(EvaluationInputError, match="Target labels 'y' must be provided"):
        evaluate_voice_model(dummy_fitted, np.ones((5, 41)), y=None)


def test_voice_evaluation_aliases_and_exports():
    """Verify evaluation aliases and exports."""
    assert VoiceEvaluationMetrics is VoiceModelMetrics
    assert EvaluationMetrics is VoiceModelMetrics
    assert compute_voice_metrics is calculate_voice_metrics
    assert calculate_classification_metrics is calculate_voice_metrics
    assert issubclass(EvaluationError, ModelError)
    assert issubclass(EvaluationInputError, EvaluationError)


# ===========================================================================
# Confusion Matrix Tests (Commit 17)
# ===========================================================================

def test_compute_voice_confusion_matrix_binary_arbitrary_labels():
    """Verify compute_voice_confusion_matrix on binary string labels."""
    y_true = ["low_risk", "high_risk", "low_risk", "high_risk"]
    y_pred = ["low_risk", "high_risk", "high_risk", "high_risk"]

    cm = compute_voice_confusion_matrix(y_true, y_pred)

    assert isinstance(cm, VoiceConfusionMatrix)
    assert cm.matrix.shape == (2, 2)
    assert cm.total_samples == 4
    assert cm.correct_predictions == 3
    assert np.isclose(cm.accuracy, 0.75)
    assert set(cm.labels) == {"high_risk", "low_risk"}

    # Normalized matrix
    assert cm.normalized_matrix is not None
    assert cm.normalized_matrix.shape == (2, 2)
    assert np.all(cm.normalized_matrix >= 0.0)
    assert np.all(cm.normalized_matrix <= 1.0)
    assert np.allclose(cm.normalized_matrix.sum(axis=1), 1.0)

    # DataFrame representation
    df_raw = cm.to_dataframe(normalized=False)
    assert isinstance(df_raw, pd.DataFrame)
    assert df_raw.shape == (2, 2)
    assert list(df_raw.index) == [f"true_{lbl}" for lbl in cm.labels]
    assert list(df_raw.columns) == [f"pred_{lbl}" for lbl in cm.labels]

    df_norm = cm.to_dataframe(normalized=True)
    assert isinstance(df_norm, pd.DataFrame)
    assert np.allclose(df_norm.to_numpy(), cm.normalized_matrix)

    # Dictionary conversion & access
    d = cm.to_dict()
    assert isinstance(d, dict)
    assert "matrix" in d
    assert "labels" in d
    assert "normalized_matrix" in d
    assert "total_samples" in d
    assert cm["total_samples"] == 4
    assert "labels" in cm
    assert "classes=" in repr(cm)


def test_compute_voice_confusion_matrix_preserves_explicit_label_ordering():
    """Verify explicit class/label ordering is strictly preserved in rows and columns."""
    labels_order = ["severe", "mild", "moderate"]
    y_true = ["moderate", "severe", "mild", "severe"]
    y_pred = ["moderate", "severe", "moderate", "severe"]

    cm = compute_voice_confusion_matrix(y_true, y_pred, labels=labels_order)

    # Labels list must match the requested custom order exactly
    assert cm.labels == labels_order

    # Check row 0 corresponds to "severe" (2 true instances: 2 correct)
    assert cm.matrix[0, 0] == 2  # true severe, pred severe
    assert cm.matrix[0, 1] == 0  # true severe, pred mild
    assert cm.matrix[0, 2] == 0  # true severe, pred moderate

    # Check row 1 corresponds to "mild" (1 true instance: predicted as moderate)
    assert cm.matrix[1, 0] == 0  # true mild, pred severe
    assert cm.matrix[1, 1] == 0  # true mild, pred mild
    assert cm.matrix[1, 2] == 1  # true mild, pred moderate

    # Check row 2 corresponds to "moderate" (1 true instance: predicted as moderate)
    assert cm.matrix[2, 0] == 0  # true moderate, pred severe
    assert cm.matrix[2, 1] == 0  # true moderate, pred mild
    assert cm.matrix[2, 2] == 1  # true moderate, pred moderate

    # DataFrame reflects exact custom ordering
    df = cm.to_dataframe()
    assert list(df.index) == ["true_severe", "true_mild", "true_moderate"]
    assert list(df.columns) == ["pred_severe", "pred_mild", "pred_moderate"]


def test_compute_voice_confusion_matrix_multiclass():
    """Verify multiclass confusion matrix calculation and properties."""
    y_true = ["c1", "c1", "c2", "c2", "c3", "c3", "c3"]
    y_pred = ["c1", "c2", "c2", "c2", "c3", "c1", "c3"]

    cm = compute_voice_confusion_matrix(y_true, y_pred)
    assert cm.matrix.shape == (3, 3)
    assert cm.total_samples == 7
    assert cm.correct_predictions == 5
    assert np.isclose(cm.accuracy, 5 / 7)

    # Row sums equal true class supports: 2, 2, 3
    assert np.array_equal(cm.matrix.sum(axis=1), [2, 2, 3])


def test_compute_voice_confusion_matrix_numeric_labels():
    """Verify confusion matrix with integer labels."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 1, 2]

    cm = compute_voice_confusion_matrix(y_true, y_pred, labels=[2, 1, 0])
    assert cm.labels == [2, 1, 0]
    assert cm.matrix.shape == (3, 3)
    assert cm.correct_predictions == 5
    assert cm.total_samples == 6


def test_compute_voice_confusion_matrix_zero_sample_class():
    """Verify safe handling of a label that has zero occurrences in true and pred."""
    labels = ["present", "absent", "unobserved"]
    y_true = ["present", "present", "absent"]
    y_pred = ["present", "absent", "absent"]

    cm = compute_voice_confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    assert cm.matrix.shape == (3, 3)
    # Row for "unobserved" (index 2) must be all zeros without NaN
    assert np.all(cm.matrix[2, :] == 0)
    assert np.all(cm.normalized_matrix[2, :] == 0.0)
    assert not np.isnan(cm.normalized_matrix).any()


def test_compute_voice_confusion_matrix_input_validation():
    """Verify validation for empty, mismatched, or invalid inputs."""
    # None inputs
    with pytest.raises(EvaluationInputError, match="cannot be None"):
        compute_voice_confusion_matrix(None, ["a", "b"])

    with pytest.raises(EvaluationInputError, match="cannot be None"):
        compute_voice_confusion_matrix(["a", "b"], None)

    # Empty inputs
    with pytest.raises(EvaluationInputError, match="empty"):
        compute_voice_confusion_matrix([], [])

    # Sample count mismatch
    with pytest.raises(EvaluationInputError, match="mismatch"):
        compute_voice_confusion_matrix(["a", "b", "c"], ["a", "b"])

    # Empty explicit labels list
    with pytest.raises(EvaluationInputError, match="cannot be empty"):
        compute_voice_confusion_matrix(["a", "b"], ["a", "b"], labels=[])

    # Null values in y_true
    with pytest.raises(EvaluationInputError, match="null/NaN"):
        compute_voice_confusion_matrix(pd.Series(["a", None]), pd.Series(["a", "a"]))


def test_evaluate_voice_confusion_matrix_with_logistic_regression():
    """Verify evaluate_voice_confusion_matrix helper with VoiceLogisticRegression."""
    df_train = _create_synthetic_feature_df(n_samples=20, target_classes=("low", "high"), random_seed=42)
    df_test = _create_synthetic_feature_df(n_samples=10, target_classes=("low", "high"), random_seed=77)

    model = VoiceLogisticRegression(random_state=42)
    model.fit(df_train)

    # Evaluate on combined DataFrame
    cm = evaluate_voice_confusion_matrix(model, df_test)
    assert isinstance(cm, VoiceConfusionMatrix)
    assert cm.matrix.shape == (2, 2)
    assert cm.total_samples == 10
    # Labels should automatically match model.classes_ ordering
    assert cm.labels == list(model.classes_)

    # Evaluate on separated features and target
    X_test, y_test = separate_features_and_target(df_test)
    cm_sep = evaluate_voice_confusion_matrix(model, X=X_test, y=y_test)
    assert np.array_equal(cm_sep.matrix, cm.matrix)
    assert cm_sep.labels == cm.labels


def test_evaluate_voice_confusion_matrix_with_random_forest():
    """Verify evaluate_voice_confusion_matrix helper with VoiceRandomForest (multiclass)."""
    df_train = _create_synthetic_feature_df(
        n_samples=24,
        target_classes=("class_a", "class_b", "class_c"),
        random_seed=42,
    )
    df_test = _create_synthetic_feature_df(
        n_samples=12,
        target_classes=("class_a", "class_b", "class_c"),
        random_seed=99,
    )

    model = VoiceRandomForest(n_estimators=20, random_state=42)
    model.fit(df_train)

    # Preserves explicit custom label order
    custom_order = ["class_c", "class_a", "class_b"]
    cm = evaluate_voice_confusion_matrix(model, df_test, labels=custom_order)
    assert isinstance(cm, VoiceConfusionMatrix)
    assert cm.labels == custom_order
    assert cm.matrix.shape == (3, 3)
    assert cm.total_samples == 12


def test_evaluate_voice_confusion_matrix_error_handling():
    """Verify error handling in evaluate_voice_confusion_matrix."""
    # None model
    with pytest.raises(EvaluationError, match="cannot be None"):
        evaluate_voice_confusion_matrix(None, pd.DataFrame({"feat_1": [1.0]}))

    # Model missing predict
    class NoPredict:
        pass

    with pytest.raises(EvaluationError, match="does not implement 'predict'"):
        evaluate_voice_confusion_matrix(NoPredict(), pd.DataFrame({"feat_1": [1.0]}), y=np.array([1]))

    # Unfitted model
    unfitted_lr = VoiceLogisticRegression()
    with pytest.raises(ModelNotFittedError, match="not fitted yet"):
        evaluate_voice_confusion_matrix(unfitted_lr, pd.DataFrame({"feat_1": [1.0]}), y=np.array([1]))

    # Non-DataFrame without y
    dummy_fitted = VoiceLogisticRegression()
    df_train = _create_synthetic_feature_df(n_samples=10, random_seed=42)
    dummy_fitted.fit(df_train)
    with pytest.raises(EvaluationInputError, match="Target labels 'y' must be provided"):
        evaluate_voice_confusion_matrix(dummy_fitted, np.ones((5, 41)), y=None)


def test_voice_confusion_matrix_aliases_and_exports():
    """Verify confusion matrix aliases and module exports."""
    assert ConfusionMatrixResult is VoiceConfusionMatrix
    assert VoiceConfusionMatrixResult is VoiceConfusionMatrix
    assert calculate_voice_confusion_matrix is compute_voice_confusion_matrix
    assert compute_confusion_matrix is compute_voice_confusion_matrix
    assert evaluate_confusion_matrix is evaluate_voice_confusion_matrix


# ===========================================================================
# Model Serialization Tests (Commit 18)
# ===========================================================================

def test_save_and_load_voice_logistic_regression(tmp_path: Path):
    """Verify saving and loading VoiceLogisticRegression preserves fitted state, weights, and predictions."""
    df_train = _create_synthetic_feature_df(n_samples=24, target_classes=("low_risk", "high_risk"), random_seed=42)
    df_test = _create_synthetic_feature_df(n_samples=10, target_classes=("low_risk", "high_risk"), random_seed=99)

    model = VoiceLogisticRegression(random_state=42)
    model.fit(df_train)

    orig_preds = model.predict(df_test)
    orig_proba = model.predict_proba(df_test)

    # 1. Save model via instance method
    model_path = tmp_path / "voice_lr.joblib"
    saved_path = model.save(model_path, metadata={"author": "test_suite"})
    assert saved_path == model_path
    assert model_path.exists()
    assert model_path.stat().st_size > 0

    # 2. Load model via load_voice_model function
    loaded_fn = load_voice_model(model_path)
    assert isinstance(loaded_fn, VoiceLogisticRegression)
    assert loaded_fn.is_fitted
    assert loaded_fn.feature_names_ == model.feature_names_
    assert np.array_equal(loaded_fn.classes_, model.classes_)
    assert np.allclose(loaded_fn.coef_, model.coef_)
    assert np.allclose(loaded_fn.intercept_, model.intercept_)

    # Verify identical predictions
    assert np.array_equal(loaded_fn.predict(df_test), orig_preds)
    assert np.allclose(loaded_fn.predict_proba(df_test), orig_proba)

    # 3. Load model via class method
    loaded_cls = VoiceLogisticRegression.load(model_path)
    assert isinstance(loaded_cls, VoiceLogisticRegression)
    assert np.array_equal(loaded_cls.predict(df_test), orig_preds)


def test_save_and_load_voice_random_forest(tmp_path: Path):
    """Verify saving and loading VoiceRandomForest preserves multiclass state and importances."""
    df_train = _create_synthetic_feature_df(
        n_samples=30,
        target_classes=("mild", "moderate", "severe"),
        random_seed=42,
    )
    df_test = _create_synthetic_feature_df(
        n_samples=15,
        target_classes=("mild", "moderate", "severe"),
        random_seed=123,
    )

    model = VoiceRandomForest(n_estimators=25, random_state=42)
    model.fit(df_train)

    orig_preds = model.predict(df_test)
    orig_proba = model.predict_proba(df_test)
    orig_importances = model.feature_importances_

    # Save to a nested directory (tests parent directory auto-creation)
    nested_path = tmp_path / "models" / "rf_subdir" / "voice_rf.joblib"
    saved_path = save_voice_model(model, nested_path)
    assert saved_path == nested_path
    assert nested_path.exists()

    # Load and verify
    loaded_rf = VoiceRandomForest.load(nested_path)
    assert isinstance(loaded_rf, VoiceRandomForest)
    assert loaded_rf.is_fitted
    assert loaded_rf.feature_names_ == model.feature_names_
    assert list(loaded_rf.classes_) == list(model.classes_)
    assert np.allclose(loaded_rf.feature_importances_, orig_importances)

    # Verify identical outputs
    assert np.array_equal(loaded_rf.predict(df_test), orig_preds)
    assert np.allclose(loaded_rf.predict_proba(df_test), orig_proba)


def test_save_unfitted_model_raises_error(tmp_path: Path):
    """Verify saving an unfitted model raises ModelNotFittedError."""
    unfitted_lr = VoiceLogisticRegression()
    with pytest.raises(ModelNotFittedError, match="unfitted"):
        unfitted_lr.save(tmp_path / "unfitted_lr.joblib")

    unfitted_rf = VoiceRandomForest()
    with pytest.raises(ModelNotFittedError, match="unfitted"):
        save_voice_model(unfitted_rf, tmp_path / "unfitted_rf.joblib")


def test_save_voice_model_invalid_inputs(tmp_path: Path):
    """Verify validation of model and path arguments during save."""
    # None model
    with pytest.raises(ModelSerializationError, match="Cannot save None"):
        save_voice_model(None, tmp_path / "model.joblib")

    # Non-BaseVoiceClassifier object
    with pytest.raises(TypeError, match="Expected an instance of BaseVoiceClassifier"):
        save_voice_model("not_a_model", tmp_path / "model.joblib")

    # Path is directory
    df = _create_synthetic_feature_df(n_samples=10, random_seed=42)
    fitted_model = VoiceLogisticRegression().fit(df)
    with pytest.raises(ModelSerializationError, match="Invalid model destination path"):
        save_voice_model(fitted_model, tmp_path)


def test_load_voice_model_nonexistent_file(tmp_path: Path):
    """Verify loading from non-existent path raises ModelNotFoundError."""
    missing_path = tmp_path / "does_not_exist.joblib"
    with pytest.raises(ModelNotFoundError, match="Model file not found"):
        load_voice_model(missing_path)

    with pytest.raises(ModelNotFoundError, match="Model file not found"):
        VoiceRandomForest.load(missing_path)


def test_load_voice_model_empty_file(tmp_path: Path):
    """Verify loading a 0-byte file raises ModelCorruptError."""
    empty_file = tmp_path / "empty_model.joblib"
    empty_file.touch()

    with pytest.raises(ModelCorruptError, match="empty \\(0 bytes\\)"):
        load_voice_model(empty_file)


def test_load_voice_model_corrupt_data(tmp_path: Path):
    """Verify loading a corrupt file raises ModelCorruptError."""
    corrupt_file = tmp_path / "corrupt.joblib"
    corrupt_file.write_bytes(b"GARBAGE_NOT_A_VALID_PICKLE_1234567890")

    with pytest.raises(ModelCorruptError, match="Failed to deserialize"):
        load_voice_model(corrupt_file)


def test_load_voice_model_invalid_object(tmp_path: Path):
    """Verify loading a valid joblib file containing non-model data raises ModelCorruptError."""
    invalid_file = tmp_path / "not_a_model.joblib"
    joblib.dump({"some_random_key": 12345}, invalid_file)

    with pytest.raises(ModelCorruptError, match="invalid type"):
        load_voice_model(invalid_file)


def test_load_voice_model_expected_type_mismatch(tmp_path: Path):
    """Verify class method load enforces matching model type."""
    df = _create_synthetic_feature_df(n_samples=12, random_seed=42)
    rf_model = VoiceRandomForest(n_estimators=10, random_state=42).fit(df)

    rf_path = tmp_path / "rf_model.joblib"
    rf_model.save(rf_path)

    # Attempting to load a Random Forest via VoiceLogisticRegression.load
    with pytest.raises(ModelCorruptError, match="Expected model of type 'VoiceLogisticRegression'"):
        VoiceLogisticRegression.load(rf_path)


def test_voice_serialization_aliases_and_exports():
    """Verify serialization exceptions and helpers inheritance."""
    assert issubclass(ModelSerializationError, ModelError)
    assert issubclass(ModelNotFoundError, ModelError)
    assert issubclass(ModelCorruptError, ModelError)














