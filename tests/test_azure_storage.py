"""
Tests for Azure Cloud Platform — Audio Blob Storage Integration.

Covers:
    - Validation of blob names and local file paths
    - Configuration resolution via environment variables (zero hardcoded secrets)
    - Upload operations (file and in-memory byte buffer)
    - Download operations (file and in-memory byte buffer)
    - Delete operations (with and without missing_ok)
    - Inspection operations (blob_exists, list_blobs, get_blob_url, get_blob_properties)
    - Container creation and lifecycle management
    - Custom storage exception hierarchy and error translation
    - Decoupling verification between Azure cloud code and ML modules

Note:
    All tests use unittest.mock fixtures and temporary files.
    No real Azure subscription, credentials, or network connectivity required.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from azure.core.exceptions import (
    ClientAuthenticationError,
    ResourceExistsError,
    ResourceNotFoundError,
)

from src.azure.config import (
    DEFAULT_AUDIO_CONTAINER,
    ENV_AZURE_STORAGE_ACCOUNT_KEY,
    ENV_AZURE_STORAGE_ACCOUNT_NAME,
    ENV_AZURE_STORAGE_CONNECTION_STRING,
    ENV_AZURE_STORAGE_CONTAINER_NAME,
    AzureStorageConfig,
)
from src.azure.exceptions import (
    AzureStorageError,
    StorageConfigurationError,
    StorageConnectionError,
    StorageDeleteError,
    StorageDownloadError,
    StorageFileNotFoundError,
    StorageUploadError,
    StorageValidationError,
)
from src.azure.storage import (
    AudioBlobStorage,
    AudioBlobStorageClient,
    AzureBlobStorage,
    validate_blob_name,
    validate_local_path_for_download,
    validate_local_path_for_upload,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def mock_service_client():
    """Mock Azure BlobServiceClient with container and blob client hierarchy."""
    service_client = MagicMock()
    container_client = MagicMock()
    blob_client = MagicMock()

    service_client.get_container_client.return_value = container_client
    container_client.get_blob_client.return_value = blob_client

    # Default blob properties
    blob_client.url = "https://mockaccount.blob.core.windows.net/audio-raw/test.wav"
    blob_client.exists.return_value = True

    return service_client


@pytest.fixture
def storage_service(mock_service_client):
    """AudioBlobStorage instance initialized with mock_service_client."""
    return AudioBlobStorage(
        container_name="audio-raw",
        service_client=mock_service_client,
    )


# ===========================================================================
# Validation Tests
# ===========================================================================

def test_validate_blob_name_valid():
    """Verify validate_blob_name accepts standard and hierarchical blob paths."""
    assert validate_blob_name("audio.wav") == "audio.wav"
    assert validate_blob_name("P001/recording.wav") == "P001/recording.wav"
    assert validate_blob_name("  dataset/P002/trial_01.wav  ") == "dataset/P002/trial_01.wav"
    assert validate_blob_name("audio_16k_mono.wav") == "audio_16k_mono.wav"


def test_validate_blob_name_invalid():
    """Verify validate_blob_name rejects empty names, leading slashes, and path traversal."""
    # Non-string
    with pytest.raises(StorageValidationError, match="must be a string"):
        validate_blob_name(123)  # type: ignore

    # Empty / whitespace
    with pytest.raises(StorageValidationError, match="empty or whitespace"):
        validate_blob_name("")

    with pytest.raises(StorageValidationError, match="empty or whitespace"):
        validate_blob_name("    ")

    # Leading slash
    with pytest.raises(StorageValidationError, match="cannot start with a slash"):
        validate_blob_name("/audio.wav")

    with pytest.raises(StorageValidationError, match="cannot start with a slash"):
        validate_blob_name("\\audio.wav")

    # Directory traversal
    with pytest.raises(StorageValidationError, match="Path traversal"):
        validate_blob_name("P001/../secret.wav")

    with pytest.raises(StorageValidationError, match="Path traversal"):
        validate_blob_name("..\\escape.wav")

    # Consecutive redundant slashes
    with pytest.raises(StorageValidationError, match="redundant consecutive slashes"):
        validate_blob_name("P001//audio.wav")


def test_validate_local_path_for_upload(tmp_path: Path):
    """Verify validate_local_path_for_upload validates file existence and type."""
    # None
    with pytest.raises(StorageValidationError, match="cannot be None"):
        validate_local_path_for_upload(None)  # type: ignore

    # Missing file
    missing = tmp_path / "nonexistent.wav"
    with pytest.raises(StorageFileNotFoundError, match="does not exist"):
        validate_local_path_for_upload(missing)

    # Directory instead of file
    directory = tmp_path / "audio_dir"
    directory.mkdir()
    with pytest.raises(StorageValidationError, match="not a regular file"):
        validate_local_path_for_upload(directory)

    # Valid file
    valid_file = tmp_path / "test.wav"
    valid_file.write_bytes(b"dummy")
    result = validate_local_path_for_upload(valid_file)
    assert result == valid_file


def test_validate_local_path_for_download(tmp_path: Path):
    """Verify validate_local_path_for_download checks destination and creates directories."""
    # None
    with pytest.raises(StorageValidationError, match="cannot be None"):
        validate_local_path_for_download(None)  # type: ignore

    # Target is an existing directory
    existing_dir = tmp_path / "dest_dir"
    existing_dir.mkdir()
    with pytest.raises(StorageValidationError, match="existing directory"):
        validate_local_path_for_download(existing_dir)

    # Target exists and overwrite=False
    existing_file = tmp_path / "exists.wav"
    existing_file.write_bytes(b"dummy")
    with pytest.raises(StorageValidationError, match="already exists and overwrite is False"):
        validate_local_path_for_download(existing_file, overwrite=False)

    # Target exists and overwrite=True
    res_overwrite = validate_local_path_for_download(existing_file, overwrite=True)
    assert res_overwrite == existing_file

    # Target in non-existent nested directory (auto-creates parent)
    nested = tmp_path / "sub1" / "sub2" / "out.wav"
    assert not nested.parent.exists()
    res_nested = validate_local_path_for_download(nested)
    assert res_nested == nested
    assert nested.parent.exists()


# ===========================================================================
# Configuration & Zero-Secret Tests
# ===========================================================================

def test_azure_storage_config_from_env(monkeypatch):
    """Verify loading configuration dynamically from environment variables."""
    monkeypatch.setenv(
        ENV_AZURE_STORAGE_CONNECTION_STRING,
        "DefaultEndpointsProtocol=https;AccountName=testacc;AccountKey=dGVzdGtleQ==;EndpointSuffix=core.windows.net",
    )
    monkeypatch.setenv(ENV_AZURE_STORAGE_CONTAINER_NAME, "custom-audio")

    config = AzureStorageConfig.from_env()
    assert config.connection_string.startswith("DefaultEndpointsProtocol=https")
    assert config.container_name == "custom-audio"
    assert config.build_connection_string() == config.connection_string


def test_azure_storage_config_account_credentials(monkeypatch):
    """Verify connection string construction from account name and key."""
    monkeypatch.delenv(ENV_AZURE_STORAGE_CONNECTION_STRING, raising=False)
    monkeypatch.setenv(ENV_AZURE_STORAGE_ACCOUNT_NAME, "myaccount")
    monkeypatch.setenv(ENV_AZURE_STORAGE_ACCOUNT_KEY, "mysecretkey")

    config = AzureStorageConfig.from_env()
    conn_str = config.build_connection_string()
    assert "AccountName=myaccount" in conn_str
    assert "AccountKey=mysecretkey" in conn_str
    assert config.container_name == DEFAULT_AUDIO_CONTAINER


def test_storage_init_missing_credentials_raises_config_error(monkeypatch):
    """Verify StorageConfigurationError is raised when no credentials exist in environment or args."""
    monkeypatch.delenv(ENV_AZURE_STORAGE_CONNECTION_STRING, raising=False)
    monkeypatch.delenv(ENV_AZURE_STORAGE_ACCOUNT_NAME, raising=False)
    monkeypatch.delenv(ENV_AZURE_STORAGE_ACCOUNT_KEY, raising=False)

    with pytest.raises(StorageConfigurationError, match="credentials not found"):
        AudioBlobStorage()


def test_storage_init_empty_container_name(mock_service_client):
    """Verify StorageConfigurationError is raised for empty container name."""
    with pytest.raises(StorageConfigurationError, match="Container name cannot be empty"):
        AudioBlobStorage(container_name="   ", service_client=mock_service_client)


# ===========================================================================
# Upload Tests
# ===========================================================================

def test_upload_file_success(tmp_path: Path, storage_service, mock_service_client):
    """Verify uploading a local audio file invokes azure-storage-blob correctly."""
    dummy_wav = tmp_path / "recording_01.wav"
    dummy_wav.write_bytes(b"RIFF1234WAVEfmt ")

    blob_name = storage_service.upload_file(
        local_path=dummy_wav,
        blob_name="P001/recording_01.wav",
        metadata={"participant_id": "P001"},
    )

    assert blob_name == "P001/recording_01.wav"
    container = mock_service_client.get_container_client("audio-raw")
    blob_client = container.get_blob_client("P001/recording_01.wav")

    blob_client.upload_blob.assert_called_once()
    call_kwargs = blob_client.upload_blob.call_args[1]
    assert call_kwargs["overwrite"] is False
    assert call_kwargs["content_settings"].content_type == "audio/wav"
    assert call_kwargs["metadata"] == {"participant_id": "P001"}


def test_upload_file_default_blob_name(tmp_path: Path, storage_service, mock_service_client):
    """Verify upload_file defaults blob_name to local filename when omitted."""
    dummy_wav = tmp_path / "sample.wav"
    dummy_wav.write_bytes(b"WAV_DATA")

    blob_name = storage_service.upload_file(local_path=dummy_wav)
    assert blob_name == "sample.wav"


def test_upload_file_already_exists_raises_upload_error(tmp_path: Path, storage_service, mock_service_client):
    """Verify ResourceExistsError is translated into StorageUploadError when overwrite=False."""
    dummy_wav = tmp_path / "existing.wav"
    dummy_wav.write_bytes(b"DATA")

    blob_client = mock_service_client.get_container_client().get_blob_client()
    blob_client.upload_blob.side_effect = ResourceExistsError("Blob already exists")

    with pytest.raises(StorageUploadError, match="already exists"):
        storage_service.upload_file(local_path=dummy_wav, blob_name="existing.wav", overwrite=False)


def test_upload_bytes_success(storage_service, mock_service_client):
    """Verify uploading in-memory bytes buffer."""
    raw_audio = b"RIFF5678WAVE"
    blob_name = storage_service.upload_bytes(
        data=raw_audio,
        blob_name="P002/buffer.wav",
        content_type="audio/wav",
    )

    assert blob_name == "P002/buffer.wav"
    blob_client = mock_service_client.get_container_client().get_blob_client("P002/buffer.wav")
    blob_client.upload_blob.assert_called_once()
    call_kwargs = blob_client.upload_blob.call_args[1]
    assert call_kwargs["data"] == raw_audio
    assert call_kwargs["content_settings"].content_type == "audio/wav"


def test_upload_bytes_invalid_type(storage_service):
    """Verify upload_bytes rejects non-byte inputs."""
    with pytest.raises(StorageValidationError, match="data must be bytes"):
        storage_service.upload_bytes(data="string_not_bytes", blob_name="test.wav")  # type: ignore


# ===========================================================================
# Download Tests
# ===========================================================================

def test_download_file_success(tmp_path: Path, storage_service, mock_service_client):
    """Verify downloading a blob streams data into a local file destination."""
    dest_file = tmp_path / "downloaded_test.wav"

    blob_client = mock_service_client.get_container_client().get_blob_client()
    mock_stream = MagicMock()

    def fake_readinto(fh):
        fh.write(b"DOWNLOADED_WAV_BYTES")
        return len(b"DOWNLOADED_WAV_BYTES")

    mock_stream.readinto.side_effect = fake_readinto
    blob_client.download_blob.return_value = mock_stream

    result_path = storage_service.download_file("P001/test.wav", dest_file)
    assert result_path == dest_file
    assert dest_file.exists()
    assert dest_file.read_bytes() == b"DOWNLOADED_WAV_BYTES"


def test_download_file_not_found(tmp_path: Path, storage_service, mock_service_client):
    """Verify ResourceNotFoundError is translated to StorageFileNotFoundError."""
    dest_file = tmp_path / "missing.wav"
    blob_client = mock_service_client.get_container_client().get_blob_client()
    blob_client.download_blob.side_effect = ResourceNotFoundError("Blob not found")

    with pytest.raises(StorageFileNotFoundError, match="not found in container"):
        storage_service.download_file("nonexistent.wav", dest_file)


def test_download_bytes_success(storage_service, mock_service_client):
    """Verify download_bytes returns in-memory byte buffer."""
    blob_client = mock_service_client.get_container_client().get_blob_client()
    mock_stream = MagicMock()
    mock_stream.readall.return_value = b"IN_MEMORY_AUDIO_BYTES"
    blob_client.download_blob.return_value = mock_stream

    data = storage_service.download_bytes("P003/sample.wav")
    assert data == b"IN_MEMORY_AUDIO_BYTES"


def test_download_bytes_not_found(storage_service, mock_service_client):
    """Verify download_bytes raises StorageFileNotFoundError when blob does not exist."""
    blob_client = mock_service_client.get_container_client().get_blob_client()
    blob_client.download_blob.side_effect = ResourceNotFoundError("Not found")

    with pytest.raises(StorageFileNotFoundError):
        storage_service.download_bytes("missing.wav")


# ===========================================================================
# Delete Tests
# ===========================================================================

def test_delete_file_success(storage_service, mock_service_client):
    """Verify deleting a blob succeeds."""
    blob_client = mock_service_client.get_container_client().get_blob_client()

    result = storage_service.delete_file("P001/old_audio.wav")
    assert result is True
    blob_client.delete_blob.assert_called_once()


def test_delete_file_not_found_missing_ok_false(storage_service, mock_service_client):
    """Verify delete_file raises StorageFileNotFoundError when blob does not exist and missing_ok=False."""
    blob_client = mock_service_client.get_container_client().get_blob_client()
    blob_client.delete_blob.side_effect = ResourceNotFoundError("Blob does not exist")

    with pytest.raises(StorageFileNotFoundError, match="not found in container"):
        storage_service.delete_file("ghost.wav", missing_ok=False)


def test_delete_file_not_found_missing_ok_true(storage_service, mock_service_client):
    """Verify delete_file returns False gracefully when missing_ok=True and blob not found."""
    blob_client = mock_service_client.get_container_client().get_blob_client()
    blob_client.delete_blob.side_effect = ResourceNotFoundError("Blob does not exist")

    result = storage_service.delete_file("ghost.wav", missing_ok=True)
    assert result is False


def test_delete_file_generic_error_raises_delete_error(storage_service, mock_service_client):
    """Verify delete failure raises StorageDeleteError."""
    blob_client = mock_service_client.get_container_client().get_blob_client()
    blob_client.delete_blob.side_effect = RuntimeError("Network timeout")

    with pytest.raises(StorageDeleteError, match="Failed to delete blob"):
        storage_service.delete_file("target.wav")


# ===========================================================================
# Inspection Tests (Exists, List, URL, Properties)
# ===========================================================================

def test_blob_exists(storage_service, mock_service_client):
    """Verify blob_exists calls exists() on blob client."""
    blob_client = mock_service_client.get_container_client().get_blob_client()
    blob_client.exists.return_value = True

    assert storage_service.blob_exists("P001/audio.wav") is True
    blob_client.exists.assert_called_once()


def test_list_blobs(storage_service, mock_service_client):
    """Verify list_blobs returns blob names matching prefix."""
    container = mock_service_client.get_container_client()
    b1 = MagicMock(); b1.name = "P001/sample1.wav"
    b2 = MagicMock(); b2.name = "P001/sample2.wav"
    container.list_blobs.return_value = [b1, b2]

    blobs = storage_service.list_blobs(prefix="P001/")
    assert blobs == ["P001/sample1.wav", "P001/sample2.wav"]
    container.list_blobs.assert_called_once_with(name_starts_with="P001/")


def test_get_blob_url(storage_service):
    """Verify get_blob_url returns correct blob URL."""
    url = storage_service.get_blob_url("audio.wav")
    assert "https://mockaccount.blob.core.windows.net" in url


def test_get_blob_properties(storage_service, mock_service_client):
    """Verify get_blob_properties extracts metadata and system properties."""
    blob_client = mock_service_client.get_container_client().get_blob_client()
    props = MagicMock()
    props.name = "P001/audio.wav"
    props.size = 102400
    props.content_settings.content_type = "audio/wav"
    props.last_modified = "2026-09-15T12:00:00Z"
    props.metadata = {"participant_id": "P001"}
    blob_client.get_blob_properties.return_value = props

    result = storage_service.get_blob_properties("P001/audio.wav")
    assert result["name"] == "P001/audio.wav"
    assert result["size"] == 102400
    assert result["content_type"] == "audio/wav"
    assert result["metadata"] == {"participant_id": "P001"}


def test_create_container_if_not_exists(storage_service, mock_service_client):
    """Verify create_container_if_not_exists creates container or catches ResourceExistsError."""
    container = mock_service_client.get_container_client()

    # Case 1: Successfully created
    container.create_container.return_value = None
    assert storage_service.create_container_if_not_exists() is True

    # Case 2: Already existed
    container.create_container.side_effect = ResourceExistsError("Already exists")
    assert storage_service.create_container_if_not_exists() is False


# ===========================================================================
# Architecture & Decoupling Tests
# ===========================================================================

def test_azure_code_separate_from_ml():
    """Verify that src.azure does not import ML packages (scikit-learn, joblib)."""
    import sys
    import src.azure.storage as storage_mod
    import src.azure.config as config_mod

    # Verify no scikit-learn or joblib imports inside storage or config modules
    assert not hasattr(storage_mod, "LogisticRegression")
    assert not hasattr(storage_mod, "RandomForestClassifier")
    assert not hasattr(config_mod, "joblib")


def test_azure_storage_aliases_and_exports():
    """Verify class aliases and exception hierarchy."""
    assert AzureBlobStorage is AudioBlobStorage
    assert AudioBlobStorageClient is AudioBlobStorage
    assert issubclass(StorageConfigurationError, AzureStorageError)
    assert issubclass(StorageValidationError, AzureStorageError)
    assert issubclass(StorageFileNotFoundError, AzureStorageError)
    assert issubclass(StorageUploadError, AzureStorageError)
    assert issubclass(StorageDownloadError, AzureStorageError)
    assert issubclass(StorageDeleteError, AzureStorageError)
    assert issubclass(StorageConnectionError, AzureStorageError)
