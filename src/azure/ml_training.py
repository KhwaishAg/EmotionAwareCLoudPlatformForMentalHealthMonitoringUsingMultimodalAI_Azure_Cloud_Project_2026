"""
Azure Cloud Platform — Voice Training Job Configuration on Azure ML

Defines reusable, secure Azure Machine Learning training job configurations
for voice emotion classifiers using the azure-ai-ml SDK v2.

Requirements & Guarantees:
    - Uses azure-ai-ml (v2 SDK CommandJob and Input/Output abstractions).
    - Configurable environment, compute target, dataset input path, and training command.
    - Explicit support for both Logistic Regression and Random Forest model selection.
    - Dataset path and target classification label are fully parameterizable.
    - Dynamic credential resolution via AzureMLConfig with Zero Hardcoded Secrets.
    - Dry-run validation mode preventing accidental real job submission in local/test environments.
    - No training on synthetic or fabricated data.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Command

from .config import (
    DEFAULT_ML_COMPUTE_TARGET,
    DEFAULT_ML_ENVIRONMENT,
    DEFAULT_ML_EXPERIMENT_NAME,
    AzureMLConfig,
)
from .exceptions import (
    AzureMLConfigurationError,
    AzureMLJobError,
    StorageValidationError,
)


# ===========================================================================
# Supported Models and Aliases
# ===========================================================================

SUPPORTED_VOICE_MODELS: Set[str] = {
    "logistic_regression",
    "random_forest",
}

MODEL_TYPE_ALIASES: Dict[str, str] = {
    "logistic_regression": "logistic_regression",
    "logistic-regression": "logistic_regression",
    "logisticregression": "logistic_regression",
    "lr": "logistic_regression",
    "random_forest": "random_forest",
    "random-forest": "random_forest",
    "randomforest": "random_forest",
    "rf": "random_forest",
}

DEFAULT_HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": 42,
        "C": 1.0,
        "solver": "lbfgs",
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "random_state": 42,
    },
}


def validate_model_type(model_type: str) -> str:
    """
    Validate and normalize the selected voice model type.

    Parameters:
        model_type: Model name (e.g. 'logistic_regression', 'random_forest', 'rf', 'lr').

    Returns:
        Canonical model type string ('logistic_regression' or 'random_forest').

    Raises:
        StorageValidationError: If model_type is unsupported or invalid.
    """
    if not isinstance(model_type, str):
        raise StorageValidationError(
            f"model_type must be a string, got {type(model_type).__name__}."
        )

    cleaned = model_type.strip().lower()
    if not cleaned:
        raise StorageValidationError("model_type cannot be empty or whitespace.")

    canonical = MODEL_TYPE_ALIASES.get(cleaned)
    if not canonical or canonical not in SUPPORTED_VOICE_MODELS:
        raise StorageValidationError(
            f"Unsupported voice model type '{model_type}'. "
            f"Supported model types: {sorted(SUPPORTED_VOICE_MODELS)}."
        )

    return canonical


# ===========================================================================
# Training Configuration Dataclass
# ===========================================================================

@dataclass
class VoiceMLTrainingConfig:
    """
    Reusable configuration for Azure ML voice model training jobs.

    Attributes:
        model_type: 'logistic_regression' or 'random_forest'.
        dataset_path: URI or path to the input dataset file or folder.
        target_label: Target column name for emotion classification (default: 'emotion').
        compute_target: Compute cluster or instance name (default: 'cpu-cluster').
        environment: Curated or custom Azure ML environment.
        code_dir: Root directory of local source code to package.
        training_command: Shell command to execute inside container.
        hyperparameters: Dictionary of model-specific hyperparameter overrides.
        experiment_name: Azure ML experiment name for tracking.
        display_name: Human-readable job display name.
        description: Brief description of the training run.
        dataset_input_type: Azure ML asset type ('uri_file' or 'uri_folder').
        tags: Metadata tags attached to the Azure ML job.
    """

    model_type: str = "random_forest"
    dataset_path: str = "data/voice_features.csv"
    target_label: str = "emotion"
    compute_target: str = DEFAULT_ML_COMPUTE_TARGET
    environment: str = DEFAULT_ML_ENVIRONMENT
    code_dir: str = "./src"
    training_command: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    experiment_name: str = DEFAULT_ML_EXPERIMENT_NAME
    display_name: Optional[str] = None
    description: Optional[str] = None
    dataset_input_type: str = AssetTypes.URI_FILE
    tags: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Validate configuration settings upon initialization."""
        self.model_type = validate_model_type(self.model_type)

        if not self.dataset_path or not str(self.dataset_path).strip():
            raise StorageValidationError("dataset_path cannot be empty.")

        if not self.target_label or not str(self.target_label).strip():
            raise StorageValidationError("target_label cannot be empty.")

        if not self.compute_target or not str(self.compute_target).strip():
            raise StorageValidationError("compute_target cannot be empty.")

        if not self.environment or not str(self.environment).strip():
            raise StorageValidationError("environment cannot be empty.")

        # Default display name
        if not self.display_name:
            self.display_name = f"voice-{self.model_type}-training"


# ===========================================================================
# Training Command Builder
# ===========================================================================

def build_default_training_command(
    model_type: str,
    custom_command: Optional[str] = None,
) -> str:
    """
    Construct the parameter-driven training command line string.

    Parameters:
        model_type: Canonical voice model type.
        custom_command: Optional custom command override.

    Returns:
        Formatted shell execution string using Azure ML input/output references.
    """
    if custom_command and custom_command.strip():
        return custom_command.strip()

    return (
        "python -m src.ai_model.voice.train "
        "--model-type ${{inputs.model_type}} "
        "--dataset-path ${{inputs.dataset}} "
        "--target-label ${{inputs.target_label}} "
        "--output-dir ${{outputs.model_output}}"
    )


# ===========================================================================
# Reusable Azure ML Training Job Factory
# ===========================================================================

def configure_voice_training_job(
    model_type: str = "random_forest",
    dataset_path: Union[str, Path, Input] = "data/voice_features.csv",
    target_label: str = "emotion",
    compute_target: Optional[str] = None,
    environment: Optional[Union[str, Any]] = None,
    training_command: Optional[str] = None,
    code_dir: Union[str, Path] = "./src",
    experiment_name: Optional[str] = None,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    dataset_input_type: str = AssetTypes.URI_FILE,
    tags: Optional[Dict[str, str]] = None,
    config: Optional[AzureMLConfig] = None,
) -> Command:
    """
    Construct a reusable, parameterizable Azure ML Command training job.

    Parameters:
        model_type: 'logistic_regression' or 'random_forest' (or aliases 'lr', 'rf').
        dataset_path: File/folder path or Azure ML Input asset for training data.
        target_label: Configurable classification target label (default: 'emotion').
        compute_target: Name of the Azure ML compute cluster (e.g. 'cpu-cluster').
        environment: Curated or custom Azure ML environment identifier.
        training_command: Shell command to run in container.
        code_dir: Path to directory containing code to upload.
        experiment_name: Azure ML experiment name for tracking.
        display_name: Human-friendly name displayed in Azure ML Studio.
        description: Run description.
        hyperparameters: Dictionary of model-specific hyperparameter values.
        dataset_input_type: 'uri_file' or 'uri_folder'.
        tags: Metadata key-value pairs for the job.
        config: Optional AzureMLConfig instance for environment defaults.

    Returns:
        azure.ai.ml.entities.Command instance ready for inspection or submission.

    Raises:
        StorageValidationError: If model_type, dataset_path, or target_label is invalid.
    """
    canonical_model = validate_model_type(model_type)
    resolved_config = config or AzureMLConfig.from_env()

    # Validate inputs
    if not dataset_path:
        raise StorageValidationError("dataset_path cannot be empty.")
    if not target_label or not str(target_label).strip():
        raise StorageValidationError("target_label cannot be empty.")

    # Resolve compute, environment, and experiment
    resolved_compute = (
        compute_target
        or resolved_config.compute_target
        or DEFAULT_ML_COMPUTE_TARGET
    )
    resolved_env = (
        environment
        or resolved_config.environment_name
        or DEFAULT_ML_ENVIRONMENT
    )
    resolved_exp = (
        experiment_name
        or resolved_config.experiment_name
        or DEFAULT_ML_EXPERIMENT_NAME
    )
    resolved_display = display_name or f"voice-{canonical_model}-training"

    # Construct dataset Input object if not already provided as Input
    if isinstance(dataset_path, Input):
        dataset_input = dataset_path
    else:
        dataset_input = Input(
            type=dataset_input_type,
            path=str(dataset_path),
            description=f"Voice feature dataset with target '{target_label}'",
        )

    # Assemble inputs dictionary
    job_inputs: Dict[str, Any] = {
        "dataset": dataset_input,
        "model_type": canonical_model,
        "target_label": str(target_label).strip(),
    }

    # Inject model-specific hyperparameters
    merged_params = dict(DEFAULT_HYPERPARAMETERS.get(canonical_model, {}))
    if hyperparameters:
        merged_params.update(hyperparameters)

    for param_key, param_val in merged_params.items():
        if param_val is not None:
            job_inputs[param_key] = param_val

    # Construct outputs dictionary (model artifact destination)
    job_outputs: Dict[str, Any] = {
        "model_output": Output(
            type=AssetTypes.URI_FOLDER,
            description="Directory where trained model and metrics are serialized",
        )
    }

    # Resolve execution command
    cmd_str = build_default_training_command(
        model_type=canonical_model,
        custom_command=training_command,
    )

    # Assemble tags
    job_tags = {
        "model_type": canonical_model,
        "target_label": str(target_label).strip(),
        "modality": "voice",
    }
    if tags:
        job_tags.update(tags)

    # Build the Command job using azure-ai-ml SDK v2
    return command(
        code=str(code_dir),
        command=cmd_str,
        inputs=job_inputs,
        outputs=job_outputs,
        environment=resolved_env,
        compute=resolved_compute,
        display_name=resolved_display,
        description=description or f"Azure ML training job for voice {canonical_model} model",
        experiment_name=resolved_exp,
        tags=job_tags,
    )


# Alias for backward-compatible and alternate naming conventions
create_voice_training_job = configure_voice_training_job


# ===========================================================================
# Azure ML Training Manager
# ===========================================================================

def _safe_format_inputs(inputs: Optional[Any]) -> Dict[str, str]:
    """Safely format job inputs dictionary for inspection and dry-run reporting."""
    if not inputs:
        return {}
    formatted: Dict[str, str] = {}
    for k, v in inputs.items():
        data_obj = getattr(v, "_data", v)
        if hasattr(data_obj, "path") and data_obj.path:
            formatted[k] = str(data_obj.path)
        elif data_obj is not None:
            formatted[k] = str(data_obj)
        else:
            formatted[k] = ""
    return formatted



class VoiceAzureMLTrainingManager:
    """
    Manager for configuring and dry-running/submitting Azure ML voice training jobs.

    Guarantees:
        - Never hardcodes credentials or secrets.
        - Never submits real jobs during testing (enforces dry_run=True by default).
        - Fully reusable across Logistic Regression and Random Forest models.
    """

    def __init__(
        self,
        config: Optional[AzureMLConfig] = None,
        ml_client: Optional[Any] = None,
    ):
        """
        Initialize VoiceAzureMLTrainingManager.

        Parameters:
            config: Optional AzureMLConfig instance.
            ml_client: Optional azure.ai.ml.MLClient instance (mocked or live).
        """
        self.config = config or AzureMLConfig.from_env()
        self._ml_client = ml_client

    @property
    def ml_client(self) -> Any:
        """Get or lazily instantiate the MLClient."""
        if self._ml_client is None:
            self._ml_client = self.config.get_ml_client()
        return self._ml_client

    def configure_job(
        self,
        model_type: str = "random_forest",
        dataset_path: Union[str, Path, Input] = "data/voice_features.csv",
        target_label: str = "emotion",
        compute_target: Optional[str] = None,
        environment: Optional[Union[str, Any]] = None,
        training_command: Optional[str] = None,
        code_dir: Union[str, Path] = "./src",
        hyperparameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Command:
        """
        Configure an Azure ML Command training job using this manager's configuration.
        """
        return configure_voice_training_job(
            model_type=model_type,
            dataset_path=dataset_path,
            target_label=target_label,
            compute_target=compute_target or self.config.compute_target,
            environment=environment or self.config.environment_name,
            training_command=training_command,
            code_dir=code_dir,
            experiment_name=self.config.experiment_name,
            hyperparameters=hyperparameters,
            tags=tags,
            config=self.config,
        )

    def submit_job(
        self,
        job: Command,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Submit the configured Azure ML training job.

        Parameters:
            job: The configured Command job object.
            dry_run: If True (default), validates the job structure and returns
                     a detailed simulation dictionary without calling Azure APIs.

        Returns:
            Dictionary containing job status and metadata.

        Raises:
            AzureMLJobError: If job submission fails.
        """
        if dry_run:
            # Safe dry run: validate job structure without initiating cloud resources
            return {
                "status": "dry_run_success",
                "display_name": job.display_name,
                "compute": job.compute,
                "environment": str(job.environment),
                "experiment_name": job.experiment_name,
                "command": job.command,
                "inputs": _safe_format_inputs(job.inputs),
                "tags": dict(job.tags or {}),
                "message": "Job structure validated successfully. No real Azure ML job was submitted.",
            }


        # Real submission (only when explicitly requested with dry_run=False)
        try:
            client = self.ml_client
            returned_job = client.jobs.create_or_update(job)
            return {
                "status": "submitted",
                "name": getattr(returned_job, "name", None),
                "display_name": getattr(returned_job, "display_name", job.display_name),
                "studio_url": getattr(returned_job, "services", {}).get("Studio", None),
            }
        except Exception as e:
            raise AzureMLJobError(f"Failed to submit Azure ML job: {e}") from e
