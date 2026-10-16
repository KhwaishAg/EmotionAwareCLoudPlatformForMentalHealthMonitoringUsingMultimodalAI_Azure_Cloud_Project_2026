"""
Comprehensive Pipeline Tests for Voice/Audio Modality (Commit 23).

Fills coverage gaps for the complete audio pipeline across Commits 1–13:
    - Audio recording, WAV ingestion across bit depths (8-bit, 16-bit, 32-bit, float32)
    - Multichannel audio ingestion and downmixing (mono, stereo, 4-channel)
    - Audio validation edge cases, boundary durations, and strict exception typing
    - Safe WAV saving with overwrite protection and directory auto-creation
    - Mono downmix arithmetic mean verification across channels
    - Resampling across diverse standard rates (8k, 22.05k, 32k, 44.1k, 48k to 16k)
    - Silence detection, boundary preservation, internal pause protection
    - Amplitude normalization determinism, custom targets, and silent signal safety
    - Integrated preprocessing pipeline with stage toggling and metadata integrity
    - Acoustic feature extraction edge cases (MFCC, pitch, energy, spectral)
    - Combined 41-feature extraction, deterministic ordering, and container support
    - Tabular dataset loading, error handling (skip vs raise), and feature separation
    - Full end-to-end pipeline: raw file -> validation -> preprocessing -> 41 features -> dataset

All tests use synthetic/generated audio only.
No private participant data, no fabricated real labels, no cloud dependencies.
"""

import io
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import scipy.io.wavfile as wavfile
import soundfile as sf

from src.ai_model.voice.config import (
    AUDIO_CONFIG,
    FEATURE_CONFIG,
    MODEL_CONFIG,
    AudioConfig,
    FeatureConfig,
)
from src.ai_model.voice.recording import (
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
    COMBINED_FEATURE_NAMES,
    ENERGY_FEATURE_NAMES,
    PITCH_FEATURE_NAMES,
    SPECTRAL_FEATURE_NAMES,
    FeatureExtractionError,
    VoiceFeatureExtractor,
    compute_energy_frames,
    compute_mfcc_frames,
    compute_pitch_frames,
    compute_spectral_frames,
    extract_combined,
    extract_combined_dict,
    extract_combined_features,
    extract_energy_dict,
    extract_energy_features,
    extract_mfcc_dict,
    extract_mfcc_features,
    extract_pitch_dict,
    extract_pitch_features,
    extract_spectral_dict,
    extract_spectral_features,
    get_combined_feature_names,
    get_mfcc_feature_names,
)
from src.ai_model.voice.dataset import (
    AudioDatasetLoader,
    DatasetAudioNotFoundError,
    DatasetError,
    DatasetExtractionError,
    DatasetLabelError,
    DatasetMetadataError,
    DatasetNotFoundError,
    load_audio_dataset,
    save_voice_dataset,
)
from src.ai_model.voice.model import separate_features_and_target


# ===========================================================================
# Helpers for Synthetic Audio Generation
# ===========================================================================

def make_sine_wave(
    duration_seconds: float = 5.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Generate a deterministic 1D float32 sine wave."""
    t = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    waveform = (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)
    return waveform


def make_stereo_sine(
    duration_seconds: float = 5.0,
    sample_rate: int = 16000,
    freq_left: float = 440.0,
    freq_right: float = 880.0,
) -> np.ndarray:
    """Generate a 2D float32 stereo waveform (samples, 2)."""
    left = make_sine_wave(duration_seconds, sample_rate, freq_left, amplitude=0.6)
    right = make_sine_wave(duration_seconds, sample_rate, freq_right, amplitude=0.4)
    return np.column_stack([left, right])


def write_canonical_wav(
    filepath: Path,
    data: np.ndarray,
    sample_rate: int = 16000,
    dtype: np.dtype = np.int16,
) -> Path:
    """Write canonical PCM WAV file using scipy.io.wavfile to avoid chunk warnings."""
    if dtype == np.int16:
        # Scale float in [-1.0, 1.0] to int16
        scaled = np.clip(data * 32767.0, -32768.0, 32767.0).astype(np.int16)
    elif dtype == np.int32:
        scaled = np.clip(data * 2147483647.0, -2147483648.0, 2147483647.0).astype(np.int32)
    elif dtype == np.uint8:
        scaled = np.clip((data * 127.0) + 128.0, 0.0, 255.0).astype(np.uint8)
    elif dtype == np.float32:
        scaled = data.astype(np.float32)
    else:
        scaled = data

    wavfile.write(str(filepath), sample_rate, scaled)
    return filepath


# ===========================================================================
# 1. WAV Ingestion Across Bit Depths & Multi-channel
# ===========================================================================

class TestAudioRecordingAndWavHandling:
    """Tests audio loading interfaces across diverse WAV bit depths and channel layouts."""

    def test_load_wav_int16_bit_depth(self, tmp_path: Path):
        """Verify 16-bit signed PCM WAV is loaded and scaled to [-1.0, 1.0] float32."""
        sine = make_sine_wave(duration_seconds=5.0, sample_rate=16000, amplitude=0.7)
        wav_file = tmp_path / "P001_int16.wav"
        write_canonical_wav(wav_file, sine, sample_rate=16000, dtype=np.int16)

        rec = load_wav_file(wav_file)
        assert rec.sample_rate == 16000
        assert rec.channels == 1
        assert rec.audio_data.dtype == np.float32
        assert np.all(rec.audio_data >= -1.0) and np.all(rec.audio_data <= 1.0)
        assert math.isclose(rec.duration_seconds, 5.0, abs_tol=0.01)
        assert rec.participant_id == "P001"
        assert rec.metadata["raw_dtype"] == "int16"

    def test_load_wav_int32_bit_depth(self, tmp_path: Path):
        """Verify 32-bit signed PCM WAV is loaded and scaled properly."""
        sine = make_sine_wave(duration_seconds=5.0, sample_rate=16000, amplitude=0.5)
        wav_file = tmp_path / "P042_int32.wav"
        write_canonical_wav(wav_file, sine, sample_rate=16000, dtype=np.int32)

        rec = load_wav_file(wav_file)
        assert rec.audio_data.dtype == np.float32
        assert rec.metadata["raw_dtype"] == "int32"
        assert np.max(np.abs(rec.audio_data)) <= 1.0
        assert rec.participant_id == "P042"

    def test_load_wav_uint8_bit_depth(self, tmp_path: Path):
        """Verify 8-bit unsigned PCM WAV (centered at 128) is normalized properly."""
        sine = make_sine_wave(duration_seconds=5.0, sample_rate=16000, amplitude=0.6)
        wav_file = tmp_path / "P100_uint8.wav"
        write_canonical_wav(wav_file, sine, sample_rate=16000, dtype=np.uint8)

        rec = load_wav_file(wav_file)
        assert rec.audio_data.dtype == np.float32
        assert rec.metadata["raw_dtype"] == "uint8"
        # Check that values are roughly centered around zero
        assert -1.0 <= np.min(rec.audio_data) < 0.0
        assert 0.0 < np.max(rec.audio_data) <= 1.0

    def test_load_wav_float32_bit_depth(self, tmp_path: Path):
        """Verify native float32 WAV is loaded without modification to amplitude."""
        sine = make_sine_wave(duration_seconds=5.0, sample_rate=16000, amplitude=0.85)
        wav_file = tmp_path / "float32.wav"
        write_canonical_wav(wav_file, sine, sample_rate=16000, dtype=np.float32)

        rec = load_wav_file(wav_file)
        assert rec.audio_data.dtype == np.float32
        assert np.isclose(np.max(np.abs(rec.audio_data)), 0.85, atol=1e-4)

    def test_load_multichannel_4_channels(self, tmp_path: Path):
        """Verify 4-channel audio file retains channel count and shape."""
        sine = make_sine_wave(duration_seconds=5.0, sample_rate=16000)
        multi = np.column_stack([sine, sine * 0.5, sine * 0.25, sine * 0.1])
        wav_file = tmp_path / "quad.wav"
        write_canonical_wav(wav_file, multi, sample_rate=16000)

        rec = load_wav_file(wav_file)
        assert rec.channels == 4
        assert rec.audio_data.shape == (len(sine), 4)
        assert rec.is_mono is False

    def test_save_wav_overwrite_protection(self, tmp_path: Path):
        """Verify save_wav_file respects overwrite=False and raises FileExistsError."""
        sine = make_sine_wave(duration_seconds=5.0, sample_rate=16000)
        rec = AudioRecording(audio_data=sine, sample_rate=16000, channels=1, duration_seconds=5.0)
        target = tmp_path / "existing.wav"
        target.write_bytes(b"dummy")

        with pytest.raises(FileExistsError):
            save_wav_file(rec, target, overwrite=False)

        # When overwrite=True, it succeeds
        saved_path = save_wav_file(rec, target, overwrite=True)
        assert saved_path.exists()
        reloaded = load_wav_file(saved_path)
        assert reloaded.sample_rate == 16000

    def test_save_wav_auto_creates_parent_directories(self, tmp_path: Path):
        """Verify save_wav_file creates intermediate subdirectories."""
        sine = make_sine_wave(duration_seconds=5.0, sample_rate=16000)
        rec = AudioRecording(audio_data=sine, sample_rate=16000, channels=1, duration_seconds=5.0)
        nested_target = tmp_path / "deep" / "nested" / "output.wav"

        saved = save_wav_file(rec, nested_target)
        assert saved.exists()
        assert nested_target.is_file()

    def test_recording_interface_polymorphic_ingest(self, tmp_path: Path):
        """Verify VoiceRecordingInterface can ingest both buffer and file sources."""
        interface = VoiceRecordingInterface()
        sine = make_sine_wave(duration_seconds=5.0, sample_rate=16000)
        wav_file = tmp_path / "P999_input.wav"
        write_canonical_wav(wav_file, sine, sample_rate=16000)

        # Local file source
        file_src = LocalFileInputSource(wav_file)
        rec1 = interface.ingest(file_src)
        assert rec1.participant_id == "P999"

        # Buffer source
        buf_bytes = wav_file.read_bytes()
        buf_src = BufferInputSource(buf_bytes, participant_id="P999")
        rec2 = interface.ingest(buf_src)
        assert rec2.participant_id == "P999"
        assert np.allclose(rec1.audio_data, rec2.audio_data, atol=1e-4)


# ===========================================================================
# 2. Audio Validation Edge Cases & Strict Exceptions
# ===========================================================================

class TestAudioValidationEdgeCases:
    """Tests edge cases, boundary durations, and typed exceptions during validation."""

    def test_validation_duration_boundary_conditions(self):
        """Verify exact boundary behavior for min (5.0s) and max (120.0s) duration."""
        sr = 16000

        # Exactly 5.0 seconds -> Valid
        rec_5s = AudioRecording(
            audio_data=make_sine_wave(5.0, sr), sample_rate=sr, channels=1, duration_seconds=5.0
        )
        res_5s = rec_5s.validate()
        assert res_5s.is_valid is True

        # Just under 5.0 seconds (4.9s) -> Invalid
        rec_4_9s = AudioRecording(
            audio_data=make_sine_wave(4.9, sr), sample_rate=sr, channels=1, duration_seconds=4.9
        )
        res_4_9s = rec_4_9s.validate()
        assert res_4_9s.is_valid is False
        assert any("duration" in err.lower() for err in res_4_9s.errors)

        # Strict mode raises AudioDurationError
        with pytest.raises(AudioDurationError):
            rec_4_9s.validate(strict=True)

        # Exceeding 120.0 seconds (120.1s) -> Invalid
        rec_121s = AudioRecording(
            audio_data=np.zeros(int(120.1 * sr), dtype=np.float32),
            sample_rate=sr,
            channels=1,
            duration_seconds=120.1,
        )
        res_121s = rec_121s.validate()
        assert res_121s.is_valid is False
        with pytest.raises(AudioDurationError):
            rec_121s.validate(strict=True)

    def test_validation_strict_exception_types(self):
        """Verify strict validation raises precise typed exception subclasses."""
        sr = 16000

        # 1. Non-finite data -> AudioDataError
        nan_data = make_sine_wave(5.0, sr)
        nan_data[100] = np.nan
        rec_nan = AudioRecording(audio_data=nan_data, sample_rate=sr, channels=1, duration_seconds=5.0)
        with pytest.raises(AudioDataError):
            rec_nan.validate(strict=True)

        # 2. Unsupported channel count (> 8) -> AudioChannelError
        rec_9ch = AudioRecording(
            audio_data=np.zeros((sr * 5, 9), dtype=np.float32),
            sample_rate=sr,
            channels=9,
            duration_seconds=5.0,
        )
        with pytest.raises(AudioChannelError):
            rec_9ch.validate(strict=True)

        # 3. Non-positive sample rate (<= 0) -> AudioSampleRateError
        rec_bad_sr = AudioRecording(
            audio_data=make_sine_wave(5.0, 16000),
            sample_rate=0,
            channels=1,
            duration_seconds=5.0,
        )
        with pytest.raises(AudioSampleRateError):
            rec_bad_sr.validate(strict=True)

    def test_validation_result_dict_and_attribute_access(self):
        """Verify ValidationResult provides both dict-like and attribute-like access."""
        rec = AudioRecording(
            audio_data=make_sine_wave(35.0, 16000),
            sample_rate=16000,
            channels=1,
            duration_seconds=35.0,
        )
        res = rec.validate()

        # Attribute access
        assert res.is_valid is True
        assert res.is_mono is True
        assert res.is_target_spec is True
        assert res.meets_recommended_duration is True

        # Dict key access
        assert res["is_valid"] is True
        assert res["is_mono"] is True
        assert res["is_target_spec"] is True
        assert res["meets_recommended_duration"] is True


# ===========================================================================
# 3. Mono Conversion and Resampling Pipeline
# ===========================================================================

class TestMonoConversionAndResampling:
    """Tests arithmetic downmixing and polyphase resampling across sample rates."""

    def test_mono_conversion_multi_channel_exact_mean(self):
        """Verify convert_to_mono computes arithmetic mean across columns."""
        # 3 channels with constant amplitudes 0.3, 0.6, 0.9 -> mean should be 0.6
        n_samples = 1000
        ch1 = np.full(n_samples, 0.3, dtype=np.float32)
        ch2 = np.full(n_samples, 0.6, dtype=np.float32)
        ch3 = np.full(n_samples, 0.9, dtype=np.float32)
        multi = np.column_stack([ch1, ch2, ch3])

        mono = convert_to_mono(multi)
        assert mono.ndim == 1
        assert len(mono) == n_samples
        assert np.allclose(mono, 0.6, atol=1e-5)

    def test_mono_conversion_2d_single_column(self):
        """Verify (N, 1) array squeezes cleanly to 1D."""
        arr = np.linspace(-0.5, 0.5, 500, dtype=np.float32).reshape(-1, 1)
        mono = convert_to_mono(arr)
        assert mono.ndim == 1
        assert mono.shape == (500,)
        assert np.allclose(mono, arr[:, 0])

    def test_mono_conversion_invalid_dimensions(self):
        """Verify 3D array raises AudioDataError."""
        arr_3d = np.zeros((10, 2, 2), dtype=np.float32)
        with pytest.raises(AudioDataError, match="Unsupported audio array dimension"):
            convert_to_mono(arr_3d)

    @pytest.mark.parametrize(
        "orig_sr",
        [8000, 22050, 32000, 44100, 48000],
    )
    def test_resample_audio_diverse_sample_rates(self, orig_sr: int):
        """Verify resampling preserves duration within 0.1% across varied sample rates."""
        duration = 5.0
        sine = make_sine_wave(duration_seconds=duration, sample_rate=orig_sr, frequency=440.0)

        resampled, was_resampled = resample_audio(sine, orig_sample_rate=orig_sr, target_sample_rate=16000)

        assert was_resampled is True
        assert resampled.dtype == np.float32
        expected_samples = int(duration * 16000)
        # Check duration matches within 0.1%
        assert abs(len(resampled) - expected_samples) <= int(0.001 * expected_samples) + 2

    def test_resample_audio_16k_passthrough_identity(self):
        """Verify 16 kHz input is not resampled and returns identical values."""
        sine = make_sine_wave(duration_seconds=5.0, sample_rate=16000)
        resampled, was_resampled = resample_audio(sine, 16000, 16000)
        assert was_resampled is False
        assert np.array_equal(resampled, sine)

    def test_resample_audio_non_positive_rate_raises(self):
        """Verify non-positive sample rates raise AudioSampleRateError."""
        sine = make_sine_wave(duration_seconds=5.0, sample_rate=16000)
        with pytest.raises(AudioSampleRateError):
            resample_audio(sine, orig_sample_rate=0, target_sample_rate=16000)
        with pytest.raises(AudioSampleRateError):
            resample_audio(sine, orig_sample_rate=16000, target_sample_rate=-16000)


# ===========================================================================
# 4. Silence Trimming and Amplitude Normalization
# ===========================================================================

class TestSilenceTrimmingAndNormalization:
    """Tests silence detection, internal pause preservation, and peak normalization."""

    def test_trim_silence_leading_and_trailing(self):
        """Verify trim_silence removes leading and trailing silent sections."""
        sr = 16000
        silence_1s = np.zeros(sr, dtype=np.float32)
        tone_4s = make_sine_wave(duration_seconds=4.0, sample_rate=sr, amplitude=0.8)
        audio = np.concatenate([silence_1s, tone_4s, silence_1s])

        trimmed, was_trimmed, lead, trail = trim_silence(audio, sample_rate=sr, top_db=25)

        assert was_trimmed is True
        assert lead > 0
        assert trail > 0
        # Result should be approximately 4.0s (min_duration_after_trim is 3.0s default)
        assert len(trimmed) < len(audio)
        assert len(trimmed) >= int(3.0 * sr)

    def test_trim_silence_preserves_internal_pause(self):
        """Verify trim_silence preserves interior silence between speech bursts."""
        sr = 16000
        tone1 = make_sine_wave(duration_seconds=2.0, sample_rate=sr, amplitude=0.7)
        pause = np.zeros(int(0.5 * sr), dtype=np.float32)
        tone2 = make_sine_wave(duration_seconds=2.0, sample_rate=sr, amplitude=0.7)
        audio = np.concatenate([tone1, pause, tone2])

        trimmed, was_trimmed, lead, trail = trim_silence(audio, sample_rate=sr)
        # Because leading and trailing are active tones, it shouldn't trim substantial samples
        assert abs(len(trimmed) - len(audio)) < 1000

    def test_trim_silence_completely_silent_returns_untrimmed(self):
        """Verify pure silence returns original array untrimmed safely without errors."""
        silent = np.zeros(16000 * 5, dtype=np.float32)
        trimmed, was_trimmed, lead, trail = trim_silence(silent, sample_rate=16000)
        assert was_trimmed is False
        assert len(trimmed) == len(silent)
        assert lead == 0 and trail == 0

    def test_normalize_amplitude_custom_targets(self):
        """Verify normalize_amplitude scales to arbitrary target peak in (0.0, 1.0]."""
        audio = make_sine_wave(duration_seconds=5.0, sample_rate=16000, amplitude=0.3)

        for target in [0.5, 0.75, 0.95, 1.0]:
            norm, orig_peak = normalize_amplitude(audio, target_peak=target)
            assert math.isclose(orig_peak, 0.3, abs_tol=1e-3)
            assert math.isclose(float(np.max(np.abs(norm))), target, abs_tol=1e-3)

    def test_normalize_amplitude_silent_audio_safety(self):
        """Verify normalize_amplitude on flatline silence returns zeros without error."""
        silent = np.zeros(1000, dtype=np.float32)
        norm, orig_peak = normalize_amplitude(silent, target_peak=0.95)
        assert orig_peak == 0.0
        assert np.array_equal(norm, silent)

    def test_normalize_amplitude_invalid_targets(self):
        """Verify invalid target peaks (<= 0 or > 1.0) raise AudioDataError."""
        audio = make_sine_wave(duration_seconds=5.0, sample_rate=16000)
        with pytest.raises(AudioDataError):
            normalize_amplitude(audio, target_peak=0.0)
        with pytest.raises(AudioDataError):
            normalize_amplitude(audio, target_peak=1.1)


# ===========================================================================
# 5. Integrated Preprocessing Pipeline Integration
# ===========================================================================

class TestPreprocessingPipelineIntegration:
    """Tests preprocessing pipeline stage toggles, input ingestion, and metadata."""

    def test_preprocess_audio_from_raw_bytes(self, tmp_path: Path):
        """Verify preprocess_audio directly accepts in-memory WAV bytes."""
        sine = make_sine_wave(duration_seconds=5.0, sample_rate=16000, amplitude=0.4)
        wav_path = tmp_path / "bytes_test.wav"
        write_canonical_wav(wav_path, sine, sample_rate=16000)
        raw_bytes = wav_path.read_bytes()

        preprocessed = preprocess_audio(raw_bytes)
        assert isinstance(preprocessed, PreprocessedAudio)
        assert preprocessed.sample_rate == 16000
        assert preprocessed.channels == 1
        assert math.isclose(preprocessed.peak_amplitude, 0.95, abs_tol=1e-3)

    def test_preprocess_audio_modular_controller(self):
        """Verify VoicePreprocessor modular methods can be called individually and cleanly."""
        sr = 44100
        stereo = make_stereo_sine(duration_seconds=5.0, sample_rate=sr)
        preprocessor = VoicePreprocessor()

        # 1. to_mono
        mono = preprocessor.to_mono(stereo)
        assert mono.ndim == 1

        # 2. resample
        resampled, was_resampled = preprocessor.resample(mono, orig_sample_rate=sr)
        assert was_resampled is True
        assert len(resampled) == int(5.0 * 16000)

        # 3. trim_silence
        trimmed, was_trimmed, lead, trail = preprocessor.trim_silence(resampled)
        assert len(trimmed) > 0

        # 4. normalize
        norm, orig_peak = preprocessor.normalize(trimmed, target_peak=0.80)
        assert math.isclose(float(np.max(np.abs(norm))), 0.80, abs_tol=1e-3)

        # 5. process with custom config
        custom_cfg = AudioConfig(normalization_peak=0.50)
        rec = AudioRecording(audio_data=mono, sample_rate=16000, channels=1, duration_seconds=5.0)
        res = preprocess_audio(rec, config=custom_cfg)
        assert res.metadata["normalization_target_peak"] == 0.50
        assert math.isclose(res.metadata["normalized_peak"], 0.50, abs_tol=1e-3)

    def test_preprocess_audio_participant_id_propagation(self):
        """Verify participant ID flows from AudioRecording to PreprocessedAudio and back."""
        sine = make_sine_wave(duration_seconds=5.0, sample_rate=16000)
        rec = AudioRecording(
            audio_data=sine,
            sample_rate=16000,
            channels=1,
            duration_seconds=5.0,
            participant_id="P777",
        )

        preprocessed = preprocess_audio(rec)
        assert preprocessed.participant_id == "P777"

        roundtrip_rec = preprocessed.to_recording()
        assert roundtrip_rec.participant_id == "P777"
        assert roundtrip_rec.channels == 1
        assert roundtrip_rec.sample_rate == 16000


# ===========================================================================
# 6. Acoustic Feature Extraction Edge Cases
# ===========================================================================

class TestAcousticFeatureExtractorsEdgeCases:
    """Tests edge cases, unvoiced audio, and frame-level outputs for all feature extractors."""

    def test_mfcc_custom_n_mfcc_and_stats(self):
        """Verify custom n_mfcc and statistics produce expected vector dimensions."""
        audio = make_sine_wave(duration_seconds=5.0, sample_rate=16000)

        # Custom n_mfcc=20 -> 40D (20 means + 20 stds)
        feat_40d = extract_mfcc_features(audio, n_mfcc=20)
        assert feat_40d.shape == (40,)
        assert feat_40d.dtype == np.float32

        # Default n_mfcc=13 -> 26D
        feat_26d = extract_mfcc_features(audio, n_mfcc=13)
        assert feat_26d.shape == (26,)

        # Names helper with custom stats
        names_40 = get_mfcc_feature_names(n_mfcc=20, stats=("mean", "std"))
        assert len(names_40) == 40
        assert names_40[0] == "mfcc_1_mean"
        assert names_40[-1] == "mfcc_20_std"

        names_13 = get_mfcc_feature_names(n_mfcc=13, stats=("mean",))
        assert len(names_13) == 13

    def test_pitch_voiced_harmonic_signal_accuracy(self):
        """Verify pitch extractor estimates the fundamental frequency of a pure tone."""
        # 220 Hz tone (A3 musical pitch)
        audio = make_sine_wave(duration_seconds=5.0, sample_rate=16000, frequency=220.0)
        pitch_dict = extract_pitch_dict(audio)

        assert "pitch_mean" in pitch_dict
        assert "pitch_std" in pitch_dict
        assert "pitch_min" in pitch_dict
        assert "pitch_max" in pitch_dict

        # Should be within +/- 10 Hz of 220 Hz
        assert 210.0 <= pitch_dict["pitch_mean"] <= 230.0
        assert pitch_dict["pitch_std"] < 5.0  # Standard deviation of steady tone is minimal

    def test_pitch_silent_audio_returns_finite_zeros(self):
        """Verify pitch extractor handles pure silence without NaN or exception."""
        silent = np.zeros(16000 * 5, dtype=np.float32)
        pitch_feats = extract_pitch_features(silent)
        assert pitch_feats.shape == (4,)
        assert np.all(pitch_feats == 0.0)
        assert np.all(np.isfinite(pitch_feats))

    def test_energy_features_properties(self):
        """Verify energy features are non-negative and range = max - min >= 0."""
        audio = make_sine_wave(duration_seconds=5.0, sample_rate=16000, amplitude=0.5)
        energy_dict = extract_energy_dict(audio)

        assert energy_dict["energy_mean"] > 0.0
        assert energy_dict["energy_std"] >= 0.0
        assert energy_dict["energy_range"] >= 0.0

        # Silent audio
        silent = np.zeros(16000 * 5, dtype=np.float32)
        silent_energy = extract_energy_dict(silent)
        assert silent_energy["energy_mean"] == 0.0
        assert silent_energy["energy_range"] == 0.0

    def test_spectral_descriptors_frequency_sensitivity(self):
        """Verify spectral centroid responds distinctly to low vs high frequencies."""
        sr = 16000
        low_tone = make_sine_wave(duration_seconds=5.0, sample_rate=sr, frequency=200.0)
        high_tone = make_sine_wave(duration_seconds=5.0, sample_rate=sr, frequency=3000.0)

        low_spec = extract_spectral_dict(low_tone)
        high_spec = extract_spectral_dict(high_tone)

        # High frequency tone MUST have significantly higher spectral centroid than low tone
        assert high_spec["spectral_centroid_mean"] > low_spec["spectral_centroid_mean"] + 1000.0
        # High tone should have higher zero crossing rate
        assert high_spec["zero_crossing_rate_mean"] > low_spec["zero_crossing_rate_mean"]

    def test_frame_level_feature_computations(self):
        """Verify frame-level extractor functions return 2D matrices."""
        audio = make_sine_wave(duration_seconds=5.0, sample_rate=16000)

        mfcc_frames = compute_mfcc_frames(audio)
        assert mfcc_frames.ndim == 2
        assert mfcc_frames.shape[0] == 13  # 13 coefficients

        f0, voiced_flag, voiced_probs = compute_pitch_frames(audio)
        assert f0.ndim == 1
        assert len(f0) > 0
        assert voiced_flag.dtype == bool

        energy_frames = compute_energy_frames(audio)
        assert energy_frames.ndim == 2
        assert energy_frames.shape[0] == 1

        spectral_frames = compute_spectral_frames(audio)
        assert isinstance(spectral_frames, dict)
        assert "spectral_centroid" in spectral_frames
        assert "spectral_bandwidth" in spectral_frames
        assert "spectral_rolloff" in spectral_frames
        assert "zero_crossing_rate" in spectral_frames


# ===========================================================================
# 7. Combined 41-Feature Extractor
# ===========================================================================

class TestCombined41FeatureExtractor:
    """Tests the unified 41-dimensional acoustic feature representation."""

    def test_combined_41_features_composition_and_order(self):
        """Verify exact 41-feature dimensionality and composition matching names."""
        audio = make_sine_wave(duration_seconds=5.0, sample_rate=16000)
        features = extract_combined_features(audio)

        assert features.shape == (41,)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))

        # Check names helper
        names = get_combined_feature_names()
        assert len(names) == 41
        assert tuple(names) == COMBINED_FEATURE_NAMES

        # Verify blocks:
        # Indices 0-25: MFCC (26)
        # Indices 26-29: Pitch (4)
        # Indices 30-32: Energy (3)
        # Indices 33-40: Spectral (8)
        assert names[0] == "mfcc_1_mean"
        assert names[25] == "mfcc_13_std"
        assert names[26] == "pitch_mean"
        assert names[29] == "pitch_max"
        assert names[30] == "energy_mean"
        assert names[32] == "energy_range"
        assert names[33] == "spectral_centroid_mean"
        assert names[40] == "zero_crossing_rate_std"

    def test_combined_dict_keys_match_constants(self):
        """Verify extract_combined_dict returns all 41 keys."""
        audio = make_sine_wave(duration_seconds=5.0, sample_rate=16000)
        c_dict = extract_combined_dict(audio)
        assert len(c_dict) == 41
        for name in COMBINED_FEATURE_NAMES:
            assert name in c_dict
            assert math.isfinite(c_dict[name])

    def test_voice_feature_extractor_class_methods(self):
        """Verify VoiceFeatureExtractor controller matches functional API outputs."""
        extractor = VoiceFeatureExtractor()
        audio = make_sine_wave(duration_seconds=5.0, sample_rate=16000)

        feat1 = extractor.extract_combined(audio)
        feat2 = extract_combined_features(audio)
        assert np.allclose(feat1, feat2)

        dict1 = extractor.extract_combined_dict(audio)
        dict2 = extract_combined_dict(audio)
        assert dict1 == dict2


# ===========================================================================
# 8. Tabular Dataset Loader Integration
# ===========================================================================

class TestAudioDatasetLoaderIntegration:
    """Tests loading multi-sample audio datasets into tabular DataFrames."""

    def test_load_audio_dataset_mixed_formats(self, tmp_path: Path):
        """Verify dataset loader handles mixed audio properties (stereo, 44.1k, 16k)."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        # Create 3 synthetic WAV files with different properties
        # File 1: 16k mono
        s1 = make_sine_wave(5.0, 16000, 300.0)
        write_canonical_wav(audio_dir / "P001_audio.wav", s1, 16000)

        # File 2: 44.1k stereo
        s2 = make_stereo_sine(5.0, 44100, 400.0, 600.0)
        write_canonical_wav(audio_dir / "P002_audio.wav", s2, 44100)

        # File 3: 48k mono
        s3 = make_sine_wave(5.0, 48000, 500.0)
        write_canonical_wav(audio_dir / "P003_audio.wav", s3, 48000)

        # Metadata DataFrame
        meta_df = pd.DataFrame([
            {"filename": "P001_audio.wav", "participant_id": "P001", "target": "stress_low"},
            {"filename": "P002_audio.wav", "participant_id": "P002", "target": "stress_high"},
            {"filename": "P003_audio.wav", "participant_id": "P003", "target": "stress_moderate"},
        ])
        meta_path = tmp_path / "metadata.csv"
        meta_df.to_csv(meta_path, index=False)

        # Load dataset
        loader = AudioDatasetLoader(audio_dir=audio_dir)
        df = loader.load(meta_path)

        assert len(df) == 3
        # 1 participant_id + 41 features + 1 target = 43 columns
        assert df.shape[1] == 43
        assert list(df["participant_id"]) == ["P001", "P002", "P003"]
        assert list(df["target"]) == ["stress_low", "stress_high", "stress_moderate"]
        assert not df[list(COMBINED_FEATURE_NAMES)].isna().any().any()

        # Separate features and target
        X, y = separate_features_and_target(df)
        assert X.shape == (3, 41)
        assert len(y) == 3

    def test_load_audio_dataset_on_error_skip_vs_raise(self, tmp_path: Path):
        """Verify on_error='skip' skips unreadable files while 'raise' throws."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        # 1 valid file
        s1 = make_sine_wave(5.0, 16000)
        write_canonical_wav(audio_dir / "valid.wav", s1, 16000)

        # 1 corrupt file
        corrupt_path = audio_dir / "corrupt.wav"
        corrupt_path.write_bytes(b"not a wav header")

        meta_df = pd.DataFrame([
            {"filename": "valid.wav", "participant_id": "P001", "target": "normal"},
            {"filename": "corrupt.wav", "participant_id": "P002", "target": "normal"},
        ])
        meta_path = tmp_path / "meta.csv"
        meta_df.to_csv(meta_path, index=False)

        # on_error='raise' raises DatasetExtractionError or DatasetError
        loader_raise = AudioDatasetLoader(audio_dir=audio_dir, on_error="raise")
        with pytest.raises(DatasetError):
            loader_raise.load(meta_path)

        # on_error='skip' skips the bad file and returns 1 row
        loader_skip = AudioDatasetLoader(audio_dir=audio_dir, on_error="skip")
        df_skipped = loader_skip.load(meta_path)
        assert len(df_skipped) == 1
        assert df_skipped.iloc[0]["participant_id"] == "P001"


# ===========================================================================
# 9. Complete End-to-End Pipeline
# ===========================================================================

class TestEndToEndAudioPipeline:
    """Tests the full chain: raw file -> validation -> preprocessing -> 41 features -> dataset row."""

    def test_full_pipeline_from_raw_unstandardized_file(self, tmp_path: Path):
        """
        Verify end-to-end processing of a raw non-standard audio file:
        - 44.1 kHz, stereo, leading/trailing silence, non-normalized amplitude.
        - Steps: validate -> preprocess -> extract 41 features -> assemble row.
        """
        sr = 44100
        # Create unstandardized audio: 0.5s silence + stereo sine + 0.5s silence
        silence = np.zeros((int(0.5 * sr), 2), dtype=np.float32)
        sine_stereo = make_stereo_sine(duration_seconds=5.0, sample_rate=sr)
        unstandardized = np.vstack([silence, sine_stereo, silence])

        raw_file = tmp_path / "P555_raw_sample.wav"
        write_canonical_wav(raw_file, unstandardized, sample_rate=sr)

        # Step 1: Validation
        val = validate_audio_file(raw_file)
        assert val.is_valid is True
        assert val.is_mono is False  # correctly detected as stereo
        assert val.is_target_sample_rate is False  # 44.1k != 16k target

        # Step 2: Preprocessing
        preprocessed = preprocess_audio(raw_file)
        assert preprocessed.sample_rate == 16000
        assert preprocessed.channels == 1
        assert preprocessed.participant_id == "P555"
        assert preprocessed.metadata["converted_to_mono"] is True
        assert preprocessed.metadata["was_resampled"] is True
        assert "silence_trimmed" in preprocessed.metadata
        assert math.isclose(preprocessed.peak_amplitude, 0.95, abs_tol=1e-3)

        # Step 3: Acoustic Feature Extraction
        features = extract_combined_features(preprocessed)
        assert features.shape == (41,)
        assert features.dtype == np.float32
        assert np.all(np.isfinite(features))

        # Step 4: Assemble Dataset Row
        feature_dict = extract_combined_dict(preprocessed)
        row = {"participant_id": preprocessed.participant_id, **feature_dict, "target": "custom_class"}
        df_row = pd.DataFrame([row])

        assert df_row.shape == (1, 43)
        assert df_row["participant_id"].iloc[0] == "P555"
        assert df_row["target"].iloc[0] == "custom_class"
        for col in COMBINED_FEATURE_NAMES:
            assert col in df_row.columns
            val = df_row[col].iloc[0]
            assert isinstance(val, (float, np.floating))
            assert math.isfinite(val)
