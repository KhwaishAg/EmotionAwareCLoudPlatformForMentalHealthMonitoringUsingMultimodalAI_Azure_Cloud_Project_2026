"""
Azure Cloud Platform — Configuration

Centralized configuration for Azure Storage and Azure Machine Learning services.
Adheres strictly to the Zero Hardcoded Credentials policy:
all connection strings, account keys, and endpoints are loaded
from environment variables or runtime parameters.
"""

from dataclasses import dataclass
import os
from typing import Any, Optional



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


# ===========================================================================
# Azure Machine Learning Configuration
# ===========================================================================

# Environment variable keys for Azure ML
ENV_AZURE_SUBSCRIPTION_ID = "AZURE_SUBSCRIPTION_ID"
ENV_AZURE_RESOURCE_GROUP = "AZURE_RESOURCE_GROUP"
ENV_AZURE_WORKSPACE_NAME = "AZURE_WORKSPACE_NAME"
ENV_AZURE_TENANT_ID = "AZURE_TENANT_ID"
ENV_AZURE_CLIENT_ID = "AZURE_CLIENT_ID"
ENV_AZURE_CLIENT_SECRET = "AZURE_CLIENT_SECRET"
ENV_AZURE_ML_COMPUTE_NAME = "AZURE_ML_COMPUTE_NAME"
ENV_AZURE_ML_ENVIRONMENT_NAME = "AZURE_ML_ENVIRONMENT_NAME"

# Architectural defaults
DEFAULT_ML_COMPUTE_TARGET = "cpu-cluster"
DEFAULT_ML_ENVIRONMENT = "AzureML-sklearn-1.5-ubuntu22.04-py310-cpu:1"
DEFAULT_ML_EXPERIMENT_NAME = "voice-emotion-training"


@dataclass(frozen=True)
class AzureMLConfig:
    """
    Configuration settings for Azure Machine Learning workspace and jobs.

    Attributes:
        subscription_id: Azure subscription identifier.
        resource_group: Azure resource group containing the ML workspace.
        workspace_name: Azure Machine Learning workspace name.
        tenant_id: Optional Azure Active Directory tenant ID.
        client_id: Optional service principal client ID.
        client_secret: Optional service principal client secret.
        compute_target: Default compute target name (default: 'cpu-cluster').
        environment_name: Default curated or custom environment name.
        experiment_name: Default experiment tracking name.
    """

    subscription_id: Optional[str] = None
    resource_group: Optional[str] = None
    workspace_name: Optional[str] = None
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    compute_target: str = DEFAULT_ML_COMPUTE_TARGET
    environment_name: str = DEFAULT_ML_ENVIRONMENT
    experiment_name: str = DEFAULT_ML_EXPERIMENT_NAME

    @classmethod
    def from_env(
        cls,
        compute_target: Optional[str] = None,
        environment_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> "AzureMLConfig":
        """
        Load Azure ML configuration dynamically from environment variables.
        Zero hardcoded secrets policy: all credentials come from the environment.
        """
        resolved_compute = (
            compute_target
            or os.getenv(ENV_AZURE_ML_COMPUTE_NAME)
            or DEFAULT_ML_COMPUTE_TARGET
        )
        resolved_env = (
            environment_name
            or os.getenv(ENV_AZURE_ML_ENVIRONMENT_NAME)
            or DEFAULT_ML_ENVIRONMENT
        )
        resolved_exp = experiment_name or DEFAULT_ML_EXPERIMENT_NAME

        return cls(
            subscription_id=os.getenv(ENV_AZURE_SUBSCRIPTION_ID),
            resource_group=os.getenv(ENV_AZURE_RESOURCE_GROUP),
            workspace_name=os.getenv(ENV_AZURE_WORKSPACE_NAME),
            tenant_id=os.getenv(ENV_AZURE_TENANT_ID),
            client_id=os.getenv(ENV_AZURE_CLIENT_ID),
            client_secret=os.getenv(ENV_AZURE_CLIENT_SECRET),
            compute_target=resolved_compute,
            environment_name=resolved_env,
            experiment_name=resolved_exp,
        )

    def get_credential(self) -> Any:
        """
        Resolve an Azure Identity token credential safely without hardcoding secrets.

        Returns ClientSecretCredential if service principal env vars are present,
        otherwise falls back to DefaultAzureCredential.
        """
        from azure.identity import ClientSecretCredential, DefaultAzureCredential

        if self.tenant_id and self.client_id and self.client_secret:
            return ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret,
            )
        return DefaultAzureCredential()

    def get_ml_client(self, credential: Optional[Any] = None) -> Any:
        """
        Construct an azure.ai.ml.MLClient instance.

        Raises:
            StorageConfigurationError: If workspace, subscription, or resource group is missing.
        """
        from azure.ai.ml import MLClient
        from .exceptions import AzureMLConfigurationError

        if not self.subscription_id or not self.resource_group or not self.workspace_name:
            raise AzureMLConfigurationError(
                "Missing required Azure ML workspace settings. Ensure AZURE_SUBSCRIPTION_ID, "
                "AZURE_RESOURCE_GROUP, and AZURE_WORKSPACE_NAME environment variables are set."
            )

        cred = credential or self.get_credential()
        return MLClient(
            credential=cred,
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name,
        )

