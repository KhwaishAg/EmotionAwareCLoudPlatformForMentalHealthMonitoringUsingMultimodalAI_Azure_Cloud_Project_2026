"""
Tests for Secure Participant Audio Upload Handling (Commit 20).

Covers:
    - Participant ID validation (anonymous alphanumeric formatting, rejection of traversal / special characters)
    - Filename sanitization and supported audio extension validation
    - Safe deterministic blob path generation (audio/<participant_id>/<filename>)
    - Path traversal prevention across participant IDs, filenames, and prefixes
    - Accidental overwrite protection (overwrite=False by default raises StorageUploadError)
    - Audio input validation for both files and bytes (WAV, MP3, FLAC, OGG, header inspection, empty data checks)
    - Protection against credential leakage in URLs, result objects, and metadata
    - Rejection of sensitive keywords in custom metadata
    - AudioUploadResult structure and serialization
    - Integration with AudioBlobStorage (Commit 19 reuse without duplication)
    - AudioBlobStorage participant upload convenience methods
    - Separation between Azure cloud code and ML modules
"""

import io
from pathlib import Path
from unittest.mock import MagicMock
import wave
import pytest

pytest.importorskip("azure.storage.blob", reason="optional Azure SDK not installed (not needed to run the app)")
pytest.importorskip("azure.ai.ml", reason="optional Azure SDK not installed (not needed to run the app)")

from src.azure.exceptions import (
    AzureStorageError,
    StorageFileNotFoundError,
    StorageUploadError,
    StorageValidationError,
)
from src.azure.storage import (
    AudioBlobStorage,
    validate_blob_name,
)
from src.azure.upload_handler import (
    DEFAULT_AUDIO_PREFIX,
    PARTICIPANT_ID_PATTERN,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_AUDIO_MIME_TYPES,
    AudioUploadResult,
    ParticipantAudioUploader,
    SecureAudioUploadHandler,
    SecureAudioUploader,
    detect_audio_format_from_bytes,
    generate_audio_blob_path,
    sanitize_audio_filename,
    sanitize_blob_url,
    sanitize_upload_metadata,
    validate_audio_bytes,
    validate_audio_file,
    validate_participant_id,
)


# ===========================================================================
# Helpers & Fixtures
# ===========================================================================

def create_synthetic_wav_bytes(duration_sec: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Create a valid in-memory PCM 16-bit mono WAV byte buffer."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n_samples = int(duration_sec * sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


@pytest.fixture
def sample_wav_file(tmp_path: Path) -> Path:
    """Create a temporary valid WAV file on disk."""
    wav_path = tmp_path / "test_recording.wav"
    wav_bytes = create_synthetic_wav_bytes(duration_sec=0.2)
    wav_path.write_bytes(wav_bytes)
    return wav_path


@pytest.fixture
def mock_service_client():
    """Mock Azure BlobServiceClient hierarchy for unit tests."""
    service_client = MagicMock()
    container_client = MagicMock()
    blob_client = MagicMock()

    service_client.get_container_client.return_value = container_client
    container_client.get_blob_client.return_value = blob_client

    blob_client.url = "https://mockstorage.blob.core.windows.net/audio-raw/audio/P001/test.wav"
    blob_client.exists.return_value = False

    return service_client


@pytest.fixture
def storage_service(mock_service_client):
    """AudioBlobStorage instance backed by mock_service_client."""
    return AudioBlobStorage(
        container_name="audio-raw",
        service_client=mock_service_client,
    )


@pytest.fixture
def upload_handler(storage_service):
    """SecureAudioUploadHandler instance backed by storage_service fixture."""
    return SecureAudioUploadHandler(storage=storage_service)


# ===========================================================================
# Participant ID Validation Tests
# ===========================================================================

def test_validate_participant_id_valid():
    """Verify validate_participant_id accepts safe anonymous alphanumeric IDs."""
    valid_ids = [
        "participant_001",
        "P123",
        "sub-042",
        "user_abc_99",
        "00123",
        "ANON-USER-1",
        "study_group_A_subject_12",
    ]
    for pid in valid_ids:
        assert validate_participant_id(pid) == pid


def test_validate_participant_id_invalid_types_and_empty():
    """Verify validate_participant_id rejects empty, non-string, or whitespace inputs."""
    with pytest.raises(StorageValidationError, match="must be a string"):
        validate_participant_id(None)  # type: ignore

    with pytest.raises(StorageValidationError, match="must be a string"):
        validate_participant_id(12345)  # type: ignore

    with pytest.raises(StorageValidationError, match="empty or whitespace"):
        validate_participant_id("")

    with pytest.raises(StorageValidationError, match="empty or whitespace"):
        validate_participant_id("    ")


def test_validate_participant_id_path_traversal_and_special_characters():
    """Verify validate_participant_id rejects path traversal and unsafe characters."""
    unsafe_ids = [
        "../p1",
        "..\\p1",
        "p1/p2",
        "p1\\p2",
        "../../secret",
        "p 1",  # spaces
        "p@123",
        "p#123",
        "p$123",
        "p*123",
        "p<script>",
        "p\x00123",  # null byte
    ]
    for uid in unsafe_ids:
        with pytest.raises(StorageValidationError, match="invalid|alphanumeric"):
            validate_participant_id(uid)


def test_validate_participant_id_max_length():
    """Verify validate_participant_id rejects excessively long identifiers."""
    long_id = "a" * 129
    with pytest.raises(StorageValidationError, match="maximum length"):
        validate_participant_id(long_id)


# ===========================================================================
# Audio Filename Sanitization Tests
# ===========================================================================

def test_sanitize_audio_filename_valid():
    """Verify sanitize_audio_filename accepts safe filenames with supported extensions."""
    valid_filenames = [
        "audio.wav",
        "recording_01.mp3",
        "voice_sample.flac",
        "clip.ogg",
        "speech.m4a",
        "session-12.aac",
        "sound.aiff",
    ]
    for fname in valid_filenames:
        assert sanitize_audio_filename(fname) == fname


def test_sanitize_audio_filename_invalid_extensions():
    """Verify sanitize_audio_filename rejects non-audio or unsupported extensions."""
    invalid_filenames = [
        "report.txt",
        "model.pkl",
        "script.py",
        "data.json",
        "malware.exe",
        "archive.zip",
        "recording",  # no extension
    ]
    for fname in invalid_filenames:
        with pytest.raises(StorageValidationError, match="Unsupported audio file extension"):
            sanitize_audio_filename(fname)


def test_sanitize_audio_filename_path_traversal():
    """Verify sanitize_audio_filename rejects filenames with path traversal or directory components."""
    unsafe_filenames = [
        "../../audio.wav",
        "..\\audio.wav",
        "subfolder/audio.wav",
        "subfolder\\audio.wav",
        "/etc/audio.wav",
        "C:\\audio.wav",
        "audio/../audio.wav",
    ]
    for fname in unsafe_filenames:
        with pytest.raises(StorageValidationError, match="separators|traversal"):
            sanitize_audio_filename(fname)


def test_sanitize_audio_filename_hidden_and_empty():
    """Verify sanitize_audio_filename rejects empty, hidden, or null byte names."""
    with pytest.raises(StorageValidationError, match="empty or whitespace"):
        sanitize_audio_filename("")

    with pytest.raises(StorageValidationError, match="stem cannot be empty"):
        sanitize_audio_filename(".wav")

    with pytest.raises(StorageValidationError, match="null bytes"):
        sanitize_audio_filename("audio\x00.wav")


# ===========================================================================
# Deterministic Path Generation Tests
# ===========================================================================

def test_generate_audio_blob_path_deterministic():
    """Verify generate_audio_blob_path creates audio/<participant_id>/<filename> deterministically."""
    path1 = generate_audio_blob_path("P001", "session_01.wav")
    path2 = generate_audio_blob_path("P001", "session_01.wav")
    assert path1 == "audio/P001/session_01.wav"
    assert path1 == path2


def test_generate_audio_blob_path_custom_prefix():
    """Verify generate_audio_blob_path respects custom prefix while maintaining validation."""
    path = generate_audio_blob_path("P002", "trial.flac", prefix="mental_health_voice")
    assert path == "mental_health_voice/P002/trial.flac"


def test_generate_audio_blob_path_prevents_traversal():
    """Verify generate_audio_blob_path prevents traversal in prefix or inputs."""
    with pytest.raises(StorageValidationError):
        generate_audio_blob_path("../../P001", "session.wav")

    with pytest.raises(StorageValidationError):
        generate_audio_blob_path("P001", "../../session.wav")

    with pytest.raises(StorageValidationError):
        generate_audio_blob_path("P001", "session.wav", prefix="../../audio")


# ===========================================================================
# Audio Format Detection & Validation Tests
# ===========================================================================

def test_detect_audio_format_from_bytes():
    """Verify detect_audio_format_from_bytes detects major audio signatures."""
    wav_bytes = create_synthetic_wav_bytes(0.05)
    assert detect_audio_format_from_bytes(wav_bytes) == "wav"

    flac_header = b"fLaC\x00\x00\x00\x22" + b"\x00" * 30
    assert detect_audio_format_from_bytes(flac_header) == "flac"

    ogg_header = b"OggS\x00\x02\x00\x00" + b"\x00" * 30
    assert detect_audio_format_from_bytes(ogg_header) == "ogg"

    mp3_id3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 20
    assert detect_audio_format_from_bytes(mp3_id3) == "mp3"

    aiff_header = b"FORM\x00\x00\x01\x00AIFFCOMM" + b"\x00" * 20
    assert detect_audio_format_from_bytes(aiff_header) == "aiff"

    # Non-audio bytes
    assert detect_audio_format_from_bytes(b"hello world this is text") is None
    assert detect_audio_format_from_bytes(b"\x00\x01\x02") is None


def test_validate_audio_bytes_valid():
    """Verify validate_audio_bytes accepts valid audio byte buffer."""
    wav_bytes = create_synthetic_wav_bytes(0.1)
    validated = validate_audio_bytes(wav_bytes, filename="recording.wav")
    assert validated == wav_bytes


def test_validate_audio_bytes_invalid():
    """Verify validate_audio_bytes rejects empty, non-bytes, or fake audio data."""
    with pytest.raises(StorageValidationError, match="must be bytes"):
        validate_audio_bytes("not bytes", filename="recording.wav")  # type: ignore

    with pytest.raises(StorageValidationError, match="cannot be empty"):
        validate_audio_bytes(b"", filename="recording.wav")

    with pytest.raises(StorageValidationError, match="does not contain a valid audio header"):
        validate_audio_bytes(b"some fake text payload", filename="recording.wav")


def test_validate_audio_file_valid(sample_wav_file: Path):
    """Verify validate_audio_file accepts valid local audio file."""
    validated_path = validate_audio_file(sample_wav_file)
    assert validated_path == sample_wav_file


def test_validate_audio_file_nonexistent(tmp_path: Path):
    """Verify validate_audio_file raises StorageFileNotFoundError on missing file."""
    with pytest.raises(StorageFileNotFoundError):
        validate_audio_file(tmp_path / "nonexistent.wav")


def test_validate_audio_file_empty(tmp_path: Path):
    """Verify validate_audio_file raises StorageValidationError on 0-byte file."""
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")
    with pytest.raises(StorageValidationError, match="is empty"):
        validate_audio_file(empty_file)


def test_validate_audio_file_invalid_content(tmp_path: Path):
    """Verify validate_audio_file rejects file with audio extension but corrupt/text content."""
    fake_file = tmp_path / "fake.wav"
    fake_file.write_text("Hello this is plain text, not a WAV file.")
    with pytest.raises(StorageValidationError, match="does not contain a valid audio header"):
        validate_audio_file(fake_file)


# ===========================================================================
# Metadata Sanitization & Credential Protection Tests
# ===========================================================================

def test_sanitize_blob_url_strips_sas_and_credentials():
    """Verify sanitize_blob_url removes query strings, signatures, and credentials."""
    raw_url_with_sas = (
        "https://myaccount.blob.core.windows.net/audio-raw/audio/P001/rec.wav"
        "?sig=SUPER_SECRET_SIGNATURE&se=2026-09-15T20:00:00Z&sp=rwdl"
    )
    clean_url = sanitize_blob_url(raw_url_with_sas)
    assert clean_url == "https://myaccount.blob.core.windows.net/audio-raw/audio/P001/rec.wav"
    assert "sig=" not in clean_url
    assert "SUPER_SECRET" not in clean_url


def test_sanitize_upload_metadata_valid():
    """Verify sanitize_upload_metadata adds safe fields and preserves non-sensitive metadata."""
    meta = sanitize_upload_metadata(
        metadata={"device": "Pixel7", "sampling_rate": "16000"},
        participant_id="P100",
        original_filename="sample.wav",
    )
    assert meta["participant_id"] == "P100"
    assert meta["original_filename"] == "sample.wav"
    assert meta["device"] == "Pixel7"
    assert meta["sampling_rate"] == "16000"
    assert "uploaded_at" in meta


def test_sanitize_upload_metadata_rejects_sensitive_keys():
    """Verify sanitize_upload_metadata raises StorageValidationError on sensitive keywords."""
    sensitive_keys = [
        "secret_token",
        "storage_account_key",
        "api_key",
        "connection_string",
        "auth_password",
        "sas_token",
    ]
    for key in sensitive_keys:
        with pytest.raises(StorageValidationError, match="contains sensitive term"):
            sanitize_upload_metadata(
                metadata={key: "sensitive_value"},
                participant_id="P100",
                original_filename="sample.wav",
            )


# ===========================================================================
# Secure Upload Operations (File & Bytes)
# ===========================================================================

def test_secure_upload_file_success(
    upload_handler: SecureAudioUploadHandler,
    sample_wav_file: Path,
):
    """Verify upload_file uploads valid audio file and returns clean AudioUploadResult."""
    result = upload_handler.upload_file(
        participant_id="P001",
        file_path=sample_wav_file,
        metadata={"session": "baseline"},
    )

    assert isinstance(result, AudioUploadResult)
    assert result.blob_name == "audio/P001/test_recording.wav"
    assert result.participant_id == "P001"
    assert result.container_name == "audio-raw"
    assert result.content_type == "audio/wav"
    assert result.size_bytes > 0
    assert "sig=" not in result.blob_url
    assert result.metadata["session"] == "baseline"
    assert result.metadata["participant_id"] == "P001"

    # Verify dictionary export
    res_dict = result.to_dict()
    assert res_dict["blob_name"] == "audio/P001/test_recording.wav"
    assert res_dict["participant_id"] == "P001"


def test_secure_upload_bytes_success(upload_handler: SecureAudioUploadHandler):
    """Verify upload_bytes uploads valid audio bytes buffer and returns clean AudioUploadResult."""
    wav_bytes = create_synthetic_wav_bytes(0.1)

    result = upload_handler.upload_bytes(
        participant_id="P002",
        data=wav_bytes,
        filename="in_memory.wav",
        metadata={"source": "microphone_stream"},
    )

    assert isinstance(result, AudioUploadResult)
    assert result.blob_name == "audio/P002/in_memory.wav"
    assert result.participant_id == "P002"
    assert result.size_bytes == len(wav_bytes)
    assert result.content_type == "audio/wav"
    assert result.metadata["source"] == "microphone_stream"


# ===========================================================================
# Overwrite Protection Tests
# ===========================================================================

def test_secure_upload_prevents_accidental_overwrite_by_default(
    upload_handler: SecureAudioUploadHandler,
    sample_wav_file: Path,
    mock_service_client,
):
    """Verify upload_file and upload_bytes reject overwriting existing blob when overwrite=False."""
    # Configure mock to indicate blob already exists
    blob_client = mock_service_client.get_container_client().get_blob_client()
    blob_client.exists.return_value = True

    with pytest.raises(StorageUploadError, match="already exists.*overwrite is False"):
        upload_handler.upload_file(
            participant_id="P001",
            file_path=sample_wav_file,
            overwrite=False,
        )

    wav_bytes = create_synthetic_wav_bytes(0.05)
    with pytest.raises(StorageUploadError, match="already exists.*overwrite is False"):
        upload_handler.upload_bytes(
            participant_id="P001",
            data=wav_bytes,
            filename="test_recording.wav",
            overwrite=False,
        )


def test_secure_upload_allows_explicit_overwrite(
    upload_handler: SecureAudioUploadHandler,
    sample_wav_file: Path,
    mock_service_client,
):
    """Verify upload_file succeeds when overwrite=True is explicitly set."""
    blob_client = mock_service_client.get_container_client().get_blob_client()
    blob_client.exists.return_value = True

    result = upload_handler.upload_file(
        participant_id="P001",
        file_path=sample_wav_file,
        overwrite=True,
    )
    assert result.blob_name == "audio/P001/test_recording.wav"


# ===========================================================================
# Convenience Dispatcher Tests (upload method)
# ===========================================================================

def test_secure_upload_dispatcher_with_file(
    upload_handler: SecureAudioUploadHandler,
    sample_wav_file: Path,
):
    """Verify generic upload() dispatches file paths properly."""
    result = upload_handler.upload(
        participant_id="P003",
        audio_source=sample_wav_file,
    )
    assert result.blob_name == "audio/P003/test_recording.wav"


def test_secure_upload_dispatcher_with_bytes(upload_handler: SecureAudioUploadHandler):
    """Verify generic upload() dispatches byte buffers properly."""
    wav_bytes = create_synthetic_wav_bytes(0.05)
    result = upload_handler.upload(
        participant_id="P003",
        audio_source=wav_bytes,
        filename="stream.wav",
    )
    assert result.blob_name == "audio/P003/stream.wav"


def test_secure_upload_dispatcher_invalid_inputs(upload_handler: SecureAudioUploadHandler):
    """Verify generic upload() rejects bytes without filename and unsupported types."""
    wav_bytes = create_synthetic_wav_bytes(0.05)
    with pytest.raises(StorageValidationError, match="Filename must be provided"):
        upload_handler.upload(
            participant_id="P003",
            audio_source=wav_bytes,
            filename=None,
        )

    with pytest.raises(StorageValidationError, match="Unsupported audio source type"):
        upload_handler.upload(
            participant_id="P003",
            audio_source=12345,  # type: ignore
        )


# ===========================================================================
# AudioBlobStorage Convenience Methods Integration Tests
# ===========================================================================

def test_storage_service_upload_participant_file(
    storage_service: AudioBlobStorage,
    sample_wav_file: Path,
):
    """Verify AudioBlobStorage.upload_participant_file delegates to SecureAudioUploadHandler."""
    result = storage_service.upload_participant_file(
        participant_id="P004",
        file_path=sample_wav_file,
    )
    assert isinstance(result, AudioUploadResult)
    assert result.blob_name == "audio/P004/test_recording.wav"
    assert result.participant_id == "P004"


def test_storage_service_upload_participant_bytes(storage_service: AudioBlobStorage):
    """Verify AudioBlobStorage.upload_participant_bytes delegates to SecureAudioUploadHandler."""
    wav_bytes = create_synthetic_wav_bytes(0.05)
    result = storage_service.upload_participant_bytes(
        participant_id="P004",
        data=wav_bytes,
        filename="bytes_audio.wav",
    )
    assert isinstance(result, AudioUploadResult)
    assert result.blob_name == "audio/P004/bytes_audio.wav"
    assert result.participant_id == "P004"


def test_storage_service_upload_participant_audio(
    storage_service: AudioBlobStorage,
    sample_wav_file: Path,
):
    """Verify AudioBlobStorage.upload_participant_audio convenience dispatcher works."""
    result = storage_service.upload_participant_audio(
        participant_id="P005",
        audio_source=sample_wav_file,
    )
    assert isinstance(result, AudioUploadResult)
    assert result.blob_name == "audio/P005/test_recording.wav"


# ===========================================================================
# Decoupling and Architectural Tests
# ===========================================================================

def test_azure_upload_handler_separate_from_ml():
    """Verify src/azure/upload_handler.py does not import ML models or scikit-learn."""
    import src.azure.upload_handler as uh_mod

    with open(uh_mod.__file__, "r", encoding="utf-8") as f:
        content = f.read()

    assert "sklearn" not in content
    assert "src.ai_model" not in content
    assert "joblib" not in content


def test_azure_upload_aliases_and_exports():
    """Verify src/azure/__init__.py correctly exports all upload handler symbols."""
    import src.azure as az

    assert hasattr(az, "SecureAudioUploadHandler")
    assert hasattr(az, "SecureAudioUploader")
    assert hasattr(az, "ParticipantAudioUploader")
    assert hasattr(az, "AudioUploadResult")
    assert hasattr(az, "validate_participant_id")
    assert hasattr(az, "sanitize_audio_filename")
    assert hasattr(az, "generate_audio_blob_path")
    assert hasattr(az, "validate_audio_file")
    assert hasattr(az, "validate_audio_bytes")
    assert hasattr(az, "SUPPORTED_AUDIO_EXTENSIONS")
    assert hasattr(az, "SUPPORTED_AUDIO_MIME_TYPES")
