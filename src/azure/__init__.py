"""
Azure Cloud Platform Package

Provides cloud infrastructure integrations for the Multimodal Mental Health Platform.
This package is strictly decoupled from local feature extraction and model training.

Components:
    - AudioBlobStorage: Azure Blob Storage manager for voice recordings and artifacts.
    - AzureStorageConfig: Centralized configuration with zero hardcoded credentials.
    - Custom storage exceptions: Structured error handling for upload/download/delete operations.
"""

from .config import (
    DEFAULT_AUDIO_CONTAINER,
    DEFAULT_FEATURES_CONTAINER,
    DEFAULT_MODELS_CONTAINER,
    ENV_AZURE_STORAGE_ACCOUNT_KEY,
    ENV_AZURE_STORAGE_ACCOUNT_NAME,
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
from .storage import (
    AudioBlobStorage,
    AudioBlobStorageClient,
    AzureBlobStorage,
    validate_blob_name,
    validate_local_path_for_download,
    validate_local_path_for_upload,
)
from .upload_handler import (
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
    validate_audio_bytes,
    validate_audio_file,
    validate_participant_id,
)

__all__ = [
    # Storage Services
    "AudioBlobStorage",
    "AzureBlobStorage",
    "AudioBlobStorageClient",
    # Secure Audio Upload
    "SecureAudioUploadHandler",
    "SecureAudioUploader",
    "ParticipantAudioUploader",
    "AudioUploadResult",
    "generate_audio_blob_path",
    "validate_participant_id",
    "sanitize_audio_filename",
    "validate_audio_file",
    "validate_audio_bytes",
    "detect_audio_format_from_bytes",
    "sanitize_blob_url",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "SUPPORTED_AUDIO_MIME_TYPES",
    "DEFAULT_AUDIO_PREFIX",
    "PARTICIPANT_ID_PATTERN",
    # Validation Helpers
    "validate_blob_name",
    "validate_local_path_for_upload",
    "validate_local_path_for_download",
    # Configuration
    "AzureStorageConfig",
    "DEFAULT_AUDIO_CONTAINER",
    "DEFAULT_FEATURES_CONTAINER",
    "DEFAULT_MODELS_CONTAINER",
    "ENV_AZURE_STORAGE_CONNECTION_STRING",
    "ENV_AZURE_STORAGE_CONTAINER_NAME",
    "ENV_AZURE_STORAGE_ACCOUNT_NAME",
    "ENV_AZURE_STORAGE_ACCOUNT_KEY",
    # Exceptions
    "AzureStorageError",
    "StorageConfigurationError",
    "StorageConnectionError",
    "StorageDeleteError",
    "StorageDownloadError",
    "StorageFileNotFoundError",
    "StorageUploadError",
    "StorageValidationError",
]

