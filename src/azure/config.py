"""
Azure Cloud Platform — Storage Configuration

Centralized configuration for Azure Storage services.
Adheres strictly to the Zero Hardcoded Credentials policy:
all connection strings, account keys, and endpoints are loaded
from environment variables or runtime parameters.
"""

from dataclasses import dataclass
import os
from typing import Optional


# Environment variable keys
ENV_AZURE_STORAGE_CONNECTION_STRING = "AZURE_STORAGE_CONNECTION_STRING"
ENV_AZURE_STORAGE_CONTAINER_NAME = "AZURE_STORAGE_CONTAINER_NAME"
ENV_AZURE_STORAGE_ACCOUNT_NAME = "AZURE_STORAGE_ACCOUNT_NAME"
ENV_AZURE_STORAGE_ACCOUNT_KEY = "AZURE_STORAGE_ACCOUNT_KEY"

# Default container names following architecture planning
DEFAULT_AUDIO_CONTAINER = "audio-raw"
DEFAULT_FEATURES_CONTAINER = "features-processed"
DEFAULT_MODELS_CONTAINER = "model-artifacts"


@dataclass(frozen=True)
class AzureStorageConfig:
    """
    Configuration settings for Azure Blob Storage.

    Attributes:
        connection_string: Full Azure Storage connection string.
        account_name: Storage account name (used if connection_string is omitted).
        account_key: Storage account access key (used if connection_string is omitted).
        container_name: Target blob container name (default: 'audio-raw').
        endpoint_suffix: Azure cloud endpoint suffix (default: 'core.windows.net').
    """

    connection_string: Optional[str] = None
    account_name: Optional[str] = None
    account_key: Optional[str] = None
    container_name: str = DEFAULT_AUDIO_CONTAINER
    endpoint_suffix: str = "core.windows.net"

    @classmethod
    def from_env(
        cls,
        container_name: Optional[str] = None,
    ) -> "AzureStorageConfig":
        """
        Load Azure Storage configuration dynamically from environment variables.

        Parameters:
            container_name: Optional container name override. If None,
                            checks AZURE_STORAGE_CONTAINER_NAME, falling back to 'audio-raw'.

        Returns:
            Configured AzureStorageConfig instance.
        """
        resolved_container = (
            container_name
            or os.getenv(ENV_AZURE_STORAGE_CONTAINER_NAME)
            or DEFAULT_AUDIO_CONTAINER
        )

        return cls(
            connection_string=os.getenv(ENV_AZURE_STORAGE_CONNECTION_STRING),
            account_name=os.getenv(ENV_AZURE_STORAGE_ACCOUNT_NAME),
            account_key=os.getenv(ENV_AZURE_STORAGE_ACCOUNT_KEY),
            container_name=resolved_container,
        )

    def build_connection_string(self) -> Optional[str]:
        """
        Resolve or construct the connection string if account credentials are provided.

        Returns:
            A valid connection string, or None if insufficient settings.
        """
        if self.connection_string:
            return self.connection_string
        if self.account_name and self.account_key:
            return (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={self.account_name};"
                f"AccountKey={self.account_key};"
                f"EndpointSuffix={self.endpoint_suffix}"
            )
        return None
