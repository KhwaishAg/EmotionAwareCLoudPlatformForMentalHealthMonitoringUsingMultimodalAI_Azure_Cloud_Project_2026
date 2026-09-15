"""
Azure Cloud Platform — Audio Blob Storage Service

Implements Azure Blob Storage integration for voice audio files and artifacts.
Adheres strictly to architectural requirements:
    - Uses azure-storage-blob SDK
    - Decoupled from ML and feature extraction modules
    - Zero hardcoded credentials policy (loads via environment variables or explicit config)
    - Rigorous validation of blob names and local file paths
    - Clear custom exceptions hierarchy
    - Full support for upload, download, delete, existence checking, and listing
"""

import io
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    ResourceExistsError,
    ResourceNotFoundError,
)
from azure.storage.blob import (
    BlobClient,
    BlobServiceClient,
    ContainerClient,
    ContentSettings,
)

from .config import (
    DEFAULT_AUDIO_CONTAINER,
    ENV_AZURE_STORAGE_CONNECTION_STRING,
    ENV_AZURE_STORAGE_CONTAINER_NAME,
    AzureStorageConfig,
)
from .exceptions import (
    AzureStorageError,
    StorageConfigurationError,
    StorageConnectionError,
    StorageDeleteError,
    StorageDownloadError,
    StorageFileNotFoundError,
    StorageUploadError,
    StorageValidationError,
)


# ===========================================================================
# Validation Utilities
# ===========================================================================

def validate_blob_name(blob_name: str) -> str:
    """
    Validate and sanitize an Azure Blob storage name.

    Rules:
        - Must be a non-empty string.
        - Cannot consist entirely of whitespace.
        - Cannot start with a forward or backward slash.
        - Cannot contain directory traversal segments ('..').

    Parameters:
        blob_name: The candidate blob identifier/path.

    Returns:
        The stripped, valid blob name.

    Raises:
        StorageValidationError: If the blob name violates naming constraints.
    """
    if not isinstance(blob_name, str):
        raise StorageValidationError(f"Blob name must be a string, got {type(blob_name).__name__}.")

    cleaned = blob_name.strip()
    if not cleaned:
        raise StorageValidationError("Blob name cannot be empty or whitespace.")

    if cleaned.startswith("/") or cleaned.startswith("\\"):
        raise StorageValidationError(f"Blob name cannot start with a slash: '{blob_name}'.")

    # Check for path traversal segments
    normalized_parts = cleaned.replace("\\", "/").split("/")
    if ".." in normalized_parts:
        raise StorageValidationError(f"Path traversal ('..') is forbidden in blob name: '{blob_name}'.")

    if any(p == "" for p in normalized_parts[:-1]):
        raise StorageValidationError(f"Blob name contains redundant consecutive slashes: '{blob_name}'.")

    return cleaned


def validate_local_path_for_upload(local_path: Union[str, Path]) -> Path:
    """
    Validate that a local file path exists and is a readable regular file.

    Parameters:
        local_path: Path to the local file to be uploaded.

    Returns:
        Resolved Path object.

    Raises:
        StorageFileNotFoundError: If the local file does not exist.
        StorageValidationError: If the path is a directory or otherwise invalid.
    """
    if local_path is None:
        raise StorageValidationError("Local file path cannot be None.")

    path = Path(local_path)
    if not path.exists():
        raise StorageFileNotFoundError(f"Local file to upload does not exist: '{local_path}'.")

    if not path.is_file():
        raise StorageValidationError(f"Local path is not a regular file: '{local_path}'.")

    return path


def validate_local_path_for_download(
    local_path: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """
    Validate that a target local download destination is valid and writable.

    Parameters:
        local_path: Target download destination path.
        overwrite: If False, prevents overwriting existing files.

    Returns:
        Resolved Path object with created parent directory.

    Raises:
        StorageValidationError: If target is an existing directory or file exists with overwrite=False.
    """
    if local_path is None:
        raise StorageValidationError("Local target path cannot be None.")

    path = Path(local_path)

    if path.is_dir():
        raise StorageValidationError(f"Target download path is an existing directory: '{local_path}'.")

    if path.exists() and not overwrite:
        raise StorageValidationError(
            f"Target download file already exists and overwrite is False: '{local_path}'."
        )

    # Automatically ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ===========================================================================
# Audio Blob Storage Service
# ===========================================================================

class AudioBlobStorage:
    """
    Azure Blob Storage manager for voice recordings and audio artifacts.

    Provides high-level upload, download, deletion, existence checking,
    and listing operations with strict validation and error translation.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
        config: Optional[AzureStorageConfig] = None,
        service_client: Optional[BlobServiceClient] = None,
    ):
        """
        Initialize AudioBlobStorage.

        Credentials are resolved in the following priority:
            1. Explicit service_client (used for dependency injection and testing)
            2. Explicit connection_string argument
            3. Explicit config object
            4. AZURE_STORAGE_CONNECTION_STRING environment variable
            5. AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY environment variables

        Parameters:
            connection_string: Optional Azure Storage connection string.
            container_name: Target container name (default: 'audio-raw').
            config: Optional AzureStorageConfig instance.
            service_client: Optional pre-constructed BlobServiceClient.

        Raises:
            StorageConfigurationError: If no valid credentials or connection settings are found.
            StorageConnectionError: If connection initialization fails.
        """
        self.config = config or AzureStorageConfig.from_env(container_name=container_name)

        # Resolve container name
        self.container_name = (
            container_name
            or self.config.container_name
            or os.getenv(ENV_AZURE_STORAGE_CONTAINER_NAME, DEFAULT_AUDIO_CONTAINER)
        )

        if not self.container_name or not self.container_name.strip():
            raise StorageConfigurationError("Container name cannot be empty.")

        # Resolve service client
        if service_client is not None:
            self.service_client = service_client
        else:
            resolved_conn = (
                connection_string
                or self.config.build_connection_string()
                or os.getenv(ENV_AZURE_STORAGE_CONNECTION_STRING)
            )

            if not resolved_conn:
                raise StorageConfigurationError(
                    "Azure Storage credentials not found. "
                    "Set the AZURE_STORAGE_CONNECTION_STRING environment variable, "
                    "or provide a connection_string or AzureStorageConfig explicitly."
                )

            try:
                self.service_client = BlobServiceClient.from_connection_string(resolved_conn)
            except (ValueError, ClientAuthenticationError) as e:
                raise StorageConfigurationError(f"Invalid Azure Storage connection string: {e}") from e
            except Exception as e:
                raise StorageConnectionError(f"Failed to initialize BlobServiceClient: {e}") from e

        self._container_client: Optional[ContainerClient] = None

    def get_container_client(self) -> ContainerClient:
        """
        Get or initialize the ContainerClient for the target container.

        Returns:
            ContainerClient instance.
        """
        if self._container_client is None:
            self._container_client = self.service_client.get_container_client(self.container_name)
        return self._container_client

    def get_blob_client(self, blob_name: str) -> BlobClient:
        """
        Get a BlobClient for a validated blob name.

        Parameters:
            blob_name: Name of the target blob.

        Returns:
            BlobClient instance.
        """
        valid_name = validate_blob_name(blob_name)
        return self.get_container_client().get_blob_client(valid_name)

    def create_container_if_not_exists(self) -> bool:
        """
        Create the target container if it does not already exist.

        Returns:
            True if container was created, False if it already existed.

        Raises:
            StorageConnectionError: If container creation fails due to network/auth error.
        """
        container = self.get_container_client()
        try:
            container.create_container()
            return True
        except ResourceExistsError:
            return False
        except Exception as e:
            raise StorageConnectionError(
                f"Failed to ensure container '{self.container_name}' exists: {e}"
            ) from e

    # -----------------------------------------------------------------------
    # Upload Operations
    # -----------------------------------------------------------------------

    def upload_file(
        self,
        local_path: Union[str, Path],
        blob_name: Optional[str] = None,
        overwrite: bool = False,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload a local audio file to Azure Blob Storage.

        Parameters:
            local_path: Local path to the file to upload.
            blob_name: Target blob name. If None, defaults to local_path.name.
            overwrite: Whether to overwrite existing blob (default: False).
            content_type: MIME type (e.g. 'audio/wav'). If None, inferred from filename.
            metadata: Optional dictionary of key-value metadata to attach to the blob.

        Returns:
            The stored blob name.

        Raises:
            StorageFileNotFoundError: If the local file does not exist.
            StorageValidationError: If path or blob name is invalid.
            StorageUploadError: If the upload operation fails.
        """
        path = validate_local_path_for_upload(local_path)
        target_name = validate_blob_name(blob_name or path.name)

        resolved_content_type = (
            content_type
            or mimetypes.guess_type(path.name)[0]
            or ("audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream")
        )

        content_settings = ContentSettings(content_type=resolved_content_type)
        blob_client = self.get_blob_client(target_name)

        try:
            with open(path, "rb") as file_data:
                blob_client.upload_blob(
                    data=file_data,
                    overwrite=overwrite,
                    content_settings=content_settings,
                    metadata=metadata,
                )
            return target_name
        except ResourceExistsError as e:
            raise StorageUploadError(
                f"Blob '{target_name}' already exists in container '{self.container_name}' "
                "and overwrite is False."
            ) from e
        except AzureStorageError:
            raise
        except Exception as e:
            raise StorageUploadError(
                f"Failed to upload '{local_path}' to blob '{target_name}': {e}"
            ) from e

    def upload_bytes(
        self,
        data: bytes,
        blob_name: str,
        overwrite: bool = False,
        content_type: str = "audio/wav",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload in-memory byte buffer (e.g., recorded audio from memory) to Azure Blob Storage.

        Parameters:
            data: Raw audio byte buffer.
            blob_name: Target blob name.
            overwrite: Whether to overwrite existing blob (default: False).
            content_type: MIME content type (default: 'audio/wav').
            metadata: Optional key-value metadata dictionary.

        Returns:
            The stored blob name.

        Raises:
            StorageValidationError: If data is not bytes or blob name is invalid.
            StorageUploadError: If the upload fails.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise StorageValidationError(f"data must be bytes or bytearray, got {type(data).__name__}.")

        target_name = validate_blob_name(blob_name)
        content_settings = ContentSettings(content_type=content_type)
        blob_client = self.get_blob_client(target_name)

        try:
            blob_client.upload_blob(
                data=bytes(data),
                overwrite=overwrite,
                content_settings=content_settings,
                metadata=metadata,
            )
            return target_name
        except ResourceExistsError as e:
            raise StorageUploadError(
                f"Blob '{target_name}' already exists in container '{self.container_name}' "
                "and overwrite is False."
            ) from e
        except Exception as e:
            raise StorageUploadError(
                f"Failed to upload bytes to blob '{target_name}': {e}"
            ) from e

    # -----------------------------------------------------------------------
    # Download Operations
    # -----------------------------------------------------------------------

    def download_file(
        self,
        blob_name: str,
        local_path: Union[str, Path],
        overwrite: bool = False,
    ) -> Path:
        """
        Download a remote blob to a local file destination.

        Parameters:
            blob_name: Name of the remote blob to download.
            local_path: Target local file destination.
            overwrite: Whether to overwrite local file if it already exists (default: False).

        Returns:
            Path object pointing to the downloaded local file.

        Raises:
            StorageFileNotFoundError: If the remote blob does not exist.
            StorageValidationError: If blob name or destination path is invalid.
            StorageDownloadError: If download fails.
        """
        target_name = validate_blob_name(blob_name)
        dest_path = validate_local_path_for_download(local_path, overwrite=overwrite)
        blob_client = self.get_blob_client(target_name)

        try:
            with open(dest_path, "wb") as file_handle:
                stream = blob_client.download_blob()
                stream.readinto(file_handle)
            return dest_path
        except ResourceNotFoundError as e:
            # Clean up partially created file if failed
            if dest_path.exists() and dest_path.stat().st_size == 0:
                dest_path.unlink(missing_ok=True)
            raise StorageFileNotFoundError(
                f"Blob '{target_name}' not found in container '{self.container_name}'."
            ) from e
        except AzureStorageError:
            raise
        except Exception as e:
            if dest_path.exists() and dest_path.stat().st_size == 0:
                dest_path.unlink(missing_ok=True)
            raise StorageDownloadError(
                f"Failed to download blob '{target_name}' to '{local_path}': {e}"
            ) from e

    def download_bytes(self, blob_name: str) -> bytes:
        """
        Download a remote blob directly into memory as bytes.

        Parameters:
            blob_name: Name of the remote blob to download.

        Returns:
            Raw audio byte content.

        Raises:
            StorageFileNotFoundError: If the remote blob does not exist.
            StorageDownloadError: If download fails.
        """
        target_name = validate_blob_name(blob_name)
        blob_client = self.get_blob_client(target_name)

        try:
            stream = blob_client.download_blob()
            return stream.readall()
        except ResourceNotFoundError as e:
            raise StorageFileNotFoundError(
                f"Blob '{target_name}' not found in container '{self.container_name}'."
            ) from e
        except Exception as e:
            raise StorageDownloadError(
                f"Failed to download blob '{target_name}' into memory: {e}"
            ) from e

    # -----------------------------------------------------------------------
    # Delete Operations
    # -----------------------------------------------------------------------

    def delete_file(
        self,
        blob_name: str,
        missing_ok: bool = False,
    ) -> bool:
        """
        Delete a blob from Azure Storage.

        Parameters:
            blob_name: Name of the blob to delete.
            missing_ok: If True, do not raise if the blob does not exist (default: False).

        Returns:
            True if the blob was deleted, False if missing_ok=True and blob was not found.

        Raises:
            StorageFileNotFoundError: If blob is not found and missing_ok is False.
            StorageDeleteError: If delete operation fails.
        """
        target_name = validate_blob_name(blob_name)
        blob_client = self.get_blob_client(target_name)

        try:
            blob_client.delete_blob()
            return True
        except ResourceNotFoundError as e:
            if missing_ok:
                return False
            raise StorageFileNotFoundError(
                f"Cannot delete blob '{target_name}': not found in container '{self.container_name}'."
            ) from e
        except Exception as e:
            raise StorageDeleteError(
                f"Failed to delete blob '{target_name}' from container '{self.container_name}': {e}"
            ) from e

    delete_blob = delete_file  # Convenient alias

    # -----------------------------------------------------------------------
    # Inspection Operations
    # -----------------------------------------------------------------------

    def blob_exists(self, blob_name: str) -> bool:
        """
        Check whether a blob exists in the container.

        Parameters:
            blob_name: Name of the blob to check.

        Returns:
            True if the blob exists, False otherwise.
        """
        target_name = validate_blob_name(blob_name)
        blob_client = self.get_blob_client(target_name)
        try:
            return bool(blob_client.exists())
        except Exception as e:
            raise StorageConnectionError(
                f"Failed to check existence of blob '{target_name}': {e}"
            ) from e

    def list_blobs(self, prefix: Optional[str] = None) -> List[str]:
        """
        List blob names in the container, optionally filtered by a prefix.

        Parameters:
            prefix: Optional prefix filter (e.g. 'P001/' for participant P001 audio).

        Returns:
            List of blob name strings.
        """
        container = self.get_container_client()
        try:
            blob_items = container.list_blobs(name_starts_with=prefix)
            return [b.name for b in blob_items]
        except Exception as e:
            raise StorageConnectionError(
                f"Failed to list blobs in container '{self.container_name}': {e}"
            ) from e

    def get_blob_url(self, blob_name: str) -> str:
        """
        Get the URI endpoint of a blob.

        Parameters:
            blob_name: Target blob name.

        Returns:
            String URL of the blob.
        """
        target_name = validate_blob_name(blob_name)
        return str(self.get_blob_client(target_name).url)

    def get_blob_properties(self, blob_name: str) -> Dict[str, Any]:
        """
        Get metadata and system properties for a blob.

        Parameters:
            blob_name: Target blob name.

        Returns:
            Dictionary containing size, content_type, last_modified, and metadata.

        Raises:
            StorageFileNotFoundError: If the blob does not exist.
        """
        target_name = validate_blob_name(blob_name)
        blob_client = self.get_blob_client(target_name)

        try:
            props = blob_client.get_blob_properties()
            return {
                "name": props.name,
                "size": props.size,
                "content_type": props.content_settings.content_type if props.content_settings else None,
                "last_modified": props.last_modified,
                "metadata": dict(props.metadata) if props.metadata else {},
            }
        except ResourceNotFoundError as e:
            raise StorageFileNotFoundError(
                f"Blob '{target_name}' not found in container '{self.container_name}'."
            ) from e
        except Exception as e:
            raise StorageConnectionError(
                f"Failed to retrieve properties for blob '{target_name}': {e}"
            ) from e


# Aliases for architectural naming conventions
AzureBlobStorage = AudioBlobStorage
AudioBlobStorageClient = AudioBlobStorage
