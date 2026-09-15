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

__all__ = [
    # Storage Services
    "AudioBlobStorage",
    "AzureBlobStorage",
    "AudioBlobStorageClient",
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
