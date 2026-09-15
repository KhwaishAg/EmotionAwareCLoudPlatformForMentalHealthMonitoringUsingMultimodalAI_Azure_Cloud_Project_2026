"""
Azure Cloud Platform — Custom Storage Exceptions

Defines domain-specific exceptions for Azure Blob Storage operations.
Ensures errors from azure-storage-blob or invalid local paths/blob names
are translated into clear, actionable exceptions.
"""


class AzureStorageError(Exception):
    """Base exception for all Azure Storage operations."""
    pass


class StorageConfigurationError(AzureStorageError):
    """Raised when required connection settings, environment variables, or secrets are missing."""
    pass


class StorageValidationError(AzureStorageError, ValueError):
    """Raised when blob names, container names, or local file paths fail validation."""
    pass


class StorageFileNotFoundError(AzureStorageError, FileNotFoundError):
    """Raised when a requested remote blob or local file does not exist."""
    pass


class StorageConnectionError(AzureStorageError):
    """Raised when establishing a connection to Azure Storage fails."""
    pass


class StorageUploadError(AzureStorageError):
    """Raised when an upload operation to Azure Blob Storage fails."""
    pass


class StorageDownloadError(AzureStorageError):
    """Raised when a download operation from Azure Blob Storage fails."""
    pass


class StorageDeleteError(AzureStorageError):
    """Raised when a delete operation on Azure Blob Storage fails."""
    pass
