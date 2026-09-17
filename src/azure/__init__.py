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
    DEFAULT_ML_COMPUTE_TARGET,
    DEFAULT_ML_ENVIRONMENT,
    DEFAULT_ML_EXPERIMENT_NAME,
    ENV_AZURE_CLIENT_ID,
    ENV_AZURE_CLIENT_SECRET,
    ENV_AZURE_ML_COMPUTE_NAME,
    ENV_AZURE_ML_ENVIRONMENT_NAME,
    ENV_AZURE_RESOURCE_GROUP,
    ENV_AZURE_STORAGE_ACCOUNT_KEY,
    ENV_AZURE_STORAGE_ACCOUNT_NAME,
    ENV_AZURE_STORAGE_CONNECTION_STRING,
    ENV_AZURE_STORAGE_CONTAINER_NAME,
    ENV_AZURE_SUBSCRIPTION_ID,
    ENV_AZURE_TENANT_ID,
    ENV_AZURE_WORKSPACE_NAME,
    AzureMLConfig,
    AzureStorageConfig,
)
from .exceptions import (
    AzureMLError,
    AzureMLConfigurationError,
    AzureMLJobError,
    AzureStorageError,
    StorageConfigurationError,
    StorageConnectionError,
    StorageDeleteError,
    StorageDownloadError,
    StorageFileNotFoundError,
    StorageUploadError,
    StorageValidationError,
)
from .ml_training import (
    DEFAULT_HYPERPARAMETERS,
    MODEL_TYPE_ALIASES,
    SUPPORTED_VOICE_MODELS,
    VoiceAzureMLTrainingManager,
    VoiceMLTrainingConfig,
    build_default_training_command,
    configure_voice_training_job,
    create_voice_training_job,
    validate_model_type,
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
    # Azure ML Training Configuration
    "VoiceMLTrainingConfig",
    "VoiceAzureMLTrainingManager",
    "configure_voice_training_job",
    "create_voice_training_job",
    "validate_model_type",
    "build_default_training_command",
    "SUPPORTED_VOICE_MODELS",
    "MODEL_TYPE_ALIASES",
    "DEFAULT_HYPERPARAMETERS",
    "DEFAULT_ML_COMPUTE_TARGET",
    "DEFAULT_ML_ENVIRONMENT",
    "DEFAULT_ML_EXPERIMENT_NAME",
    "AzureMLConfig",
    # Validation Helpers
    "validate_blob_name",
    "validate_local_path_for_upload",
    "validate_local_path_for_download",
    # Storage Configuration
    "AzureStorageConfig",
    "DEFAULT_AUDIO_CONTAINER",
    "DEFAULT_FEATURES_CONTAINER",
    "DEFAULT_MODELS_CONTAINER",
    "ENV_AZURE_STORAGE_CONNECTION_STRING",
    "ENV_AZURE_STORAGE_CONTAINER_NAME",
    "ENV_AZURE_STORAGE_ACCOUNT_NAME",
    "ENV_AZURE_STORAGE_ACCOUNT_KEY",
    # Azure ML Environment Variables
    "ENV_AZURE_SUBSCRIPTION_ID",
    "ENV_AZURE_RESOURCE_GROUP",
    "ENV_AZURE_WORKSPACE_NAME",
    "ENV_AZURE_TENANT_ID",
    "ENV_AZURE_CLIENT_ID",
    "ENV_AZURE_CLIENT_SECRET",
    "ENV_AZURE_ML_COMPUTE_NAME",
    "ENV_AZURE_ML_ENVIRONMENT_NAME",
    # Exceptions
    "AzureStorageError",
    "StorageConfigurationError",
    "StorageConnectionError",
    "StorageDeleteError",
    "StorageDownloadError",
    "StorageFileNotFoundError",
    "StorageUploadError",
    "StorageValidationError",
    "AzureMLError",
    "AzureMLConfigurationError",
    "AzureMLJobError",
]
