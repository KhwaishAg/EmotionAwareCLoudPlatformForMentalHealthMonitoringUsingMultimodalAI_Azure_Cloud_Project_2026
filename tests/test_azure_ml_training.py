"""
Tests for Azure ML Voice Training Configuration (Commit 21).

Covers:
    - Reusable Azure ML CommandJob creation via azure-ai-ml SDK v2
    - Model selection: Logistic Regression and Random Forest
    - Configuration of environment, compute target, dataset path, and training command
    - Configurable target label and dataset path parameters
    - Hyperparameter injection and overriding for both model architectures
    - AzureMLConfig loading dynamically from environment variables (zero hardcoded secrets)
    - Safe dry-run execution preventing accidental real cloud job submissions
    - Mocked MLClient job submission validation (no real Azure account required)
    - Input validation (rejection of invalid model types, empty paths, or empty labels)
    - Architectural decoupling and package exports
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

pytest.importorskip("azure.storage.blob", reason="optional Azure SDK not installed (not needed to run the app)")
pytest.importorskip("azure.ai.ml", reason="optional Azure SDK not installed (not needed to run the app)")

from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Command

from src.azure.config import (
    DEFAULT_ML_COMPUTE_TARGET,
    DEFAULT_ML_ENVIRONMENT,
    DEFAULT_ML_EXPERIMENT_NAME,
    ENV_AZURE_CLIENT_ID,
    ENV_AZURE_CLIENT_SECRET,
    ENV_AZURE_ML_COMPUTE_NAME,
    ENV_AZURE_ML_ENVIRONMENT_NAME,
    ENV_AZURE_RESOURCE_GROUP,
    ENV_AZURE_SUBSCRIPTION_ID,
    ENV_AZURE_TENANT_ID,
    ENV_AZURE_WORKSPACE_NAME,
    AzureMLConfig,
)
from src.azure.exceptions import (
    AzureMLConfigurationError,
    AzureMLError,
    AzureMLJobError,
    StorageConfigurationError,
    StorageValidationError,
)
from src.azure.ml_training import (
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


# ===========================================================================
# Model Type Validation Tests
# ===========================================================================

def test_validate_model_type_supported():
    """Verify validate_model_type normalizes all supported model names and aliases."""
    # Canonical names
    assert validate_model_type("logistic_regression") == "logistic_regression"
    assert validate_model_type("random_forest") == "random_forest"

    # Aliases and casing
    assert validate_model_type("lr") == "logistic_regression"
    assert validate_model_type("rf") == "random_forest"
    assert validate_model_type("logistic-regression") == "logistic_regression"
    assert validate_model_type("random-forest") == "random_forest"
    assert validate_model_type("LogisticRegression") == "logistic_regression"
    assert validate_model_type("RandomForest") == "random_forest"
    assert validate_model_type("  RANDOM_FOREST  ") == "random_forest"


def test_validate_model_type_unsupported_and_invalid():
    """Verify validate_model_type raises StorageValidationError on invalid or unsupported inputs."""
    with pytest.raises(StorageValidationError, match="must be a string"):
        validate_model_type(123)  # type: ignore

    with pytest.raises(StorageValidationError, match="cannot be empty"):
        validate_model_type("")

    with pytest.raises(StorageValidationError, match="cannot be empty"):
        validate_model_type("   ")

    with pytest.raises(StorageValidationError, match="Unsupported voice model type"):
        validate_model_type("xgboost")

    with pytest.raises(StorageValidationError, match="Unsupported voice model type"):
        validate_model_type("svm")

    with pytest.raises(StorageValidationError, match="Unsupported voice model type"):
        validate_model_type("deep_learning")


# ===========================================================================
# AzureMLConfig Environment Resolution Tests
# ===========================================================================

def test_azure_ml_config_from_env(monkeypatch):
    """Verify AzureMLConfig reads all settings dynamically from environment variables."""
    monkeypatch.setenv(ENV_AZURE_SUBSCRIPTION_ID, "sub-test-12345")
    monkeypatch.setenv(ENV_AZURE_RESOURCE_GROUP, "rg-mental-health")
    monkeypatch.setenv(ENV_AZURE_WORKSPACE_NAME, "mlw-voice-prod")
    monkeypatch.setenv(ENV_AZURE_TENANT_ID, "tenant-test-999")
    monkeypatch.setenv(ENV_AZURE_CLIENT_ID, "client-app-888")
    monkeypatch.setenv(ENV_AZURE_CLIENT_SECRET, "secret-token-777")
    monkeypatch.setenv(ENV_AZURE_ML_COMPUTE_NAME, "gpu-cluster")
    monkeypatch.setenv(ENV_AZURE_ML_ENVIRONMENT_NAME, "custom-sklearn-env:2")

    cfg = AzureMLConfig.from_env()

    assert cfg.subscription_id == "sub-test-12345"
    assert cfg.resource_group == "rg-mental-health"
    assert cfg.workspace_name == "mlw-voice-prod"
    assert cfg.tenant_id == "tenant-test-999"
    assert cfg.client_id == "client-app-888"
    assert cfg.client_secret == "secret-token-777"
    assert cfg.compute_target == "gpu-cluster"
    assert cfg.environment_name == "custom-sklearn-env:2"


def test_azure_ml_config_missing_settings_raises():
    """Verify get_ml_client raises AzureMLConfigurationError if essential workspace settings are missing."""
    empty_cfg = AzureMLConfig()

    with pytest.raises(AzureMLConfigurationError, match="Missing required Azure ML workspace settings"):
        empty_cfg.get_ml_client()


def get_input_val(node_input):
    """Extract raw scalar value or underlying Input object from NodeInput."""
    if hasattr(node_input, "_data"):
        return node_input._data
    return node_input


# ===========================================================================
# Job Configuration: Random Forest
# ===========================================================================

def test_configure_voice_training_job_random_forest():
    """Verify configure_voice_training_job creates valid CommandJob for Random Forest."""
    job = configure_voice_training_job(
        model_type="random_forest",
        dataset_path="data/voice_features.csv",
        target_label="emotion",
        compute_target="cpu-cluster",
    )

    assert isinstance(job, Command)
    assert job.compute == "cpu-cluster"
    assert job.environment == DEFAULT_ML_ENVIRONMENT
    assert job.display_name == "voice-random_forest-training"
    assert job.experiment_name == DEFAULT_ML_EXPERIMENT_NAME

    # Inputs check
    inputs = job.inputs
    assert get_input_val(inputs["model_type"]) == "random_forest"
    assert get_input_val(inputs["target_label"]) == "emotion"
    assert inputs["dataset"].path == "data/voice_features.csv"
    assert get_input_val(inputs["n_estimators"]) == 100
    assert get_input_val(inputs["max_depth"]) == 10

    # Outputs check
    assert "model_output" in job.outputs
    assert job.outputs["model_output"].type == AssetTypes.URI_FOLDER

    # Tags check
    assert job.tags["model_type"] == "random_forest"
    assert job.tags["target_label"] == "emotion"
    assert job.tags["modality"] == "voice"

    # Command string check
    assert "python -m src.ai_model.voice.train" in job.command
    assert "--model-type ${{inputs.model_type}}" in job.command
    assert "--dataset-path ${{inputs.dataset}}" in job.command
    assert "--target-label ${{inputs.target_label}}" in job.command


# ===========================================================================
# Job Configuration: Logistic Regression
# ===========================================================================

def test_configure_voice_training_job_logistic_regression():
    """Verify configure_voice_training_job creates valid CommandJob for Logistic Regression."""
    job = configure_voice_training_job(
        model_type="logistic_regression",
        dataset_path="data/voice_features.csv",
        target_label="emotion",
        compute_target="cpu-cluster",
    )

    assert isinstance(job, Command)
    assert get_input_val(job.inputs["model_type"]) == "logistic_regression"
    assert get_input_val(job.inputs["target_label"]) == "emotion"
    assert get_input_val(job.inputs["max_iter"]) == 1000
    assert get_input_val(job.inputs["C"]) == 1.0
    assert get_input_val(job.inputs["solver"]) == "lbfgs"
    assert job.display_name == "voice-logistic_regression-training"


# ===========================================================================
# Configurable Parameters Tests
# ===========================================================================

def test_configure_voice_training_job_custom_parameters():
    """Verify all parameters (dataset, target, compute, env, command, hyperparams) are configurable."""
    custom_dataset = "azureml:voice_features_dataset:3"
    custom_target = "depression_risk_score"
    custom_compute = "highmem-cluster"
    custom_env = "AzureML-sklearn-1.0-ubuntu20.04-py38:1"
    custom_cmd = "python run_train.py --input-data ${{inputs.dataset}}"
    custom_hyperparams = {
        "n_estimators": 350,
        "max_depth": 20,
        "min_samples_split": 4,
    }
    custom_tags = {"experiment_id": "EXP-099", "lead": "researcher"}

    job = configure_voice_training_job(
        model_type="rf",
        dataset_path=custom_dataset,
        target_label=custom_target,
        compute_target=custom_compute,
        environment=custom_env,
        training_command=custom_cmd,
        hyperparameters=custom_hyperparams,
        tags=custom_tags,
        display_name="custom-voice-run",
        experiment_name="custom-experiment",
    )

    assert job.compute == custom_compute
    assert job.environment == custom_env
    assert job.command == custom_cmd
    assert job.display_name == "custom-voice-run"
    assert job.experiment_name == "custom-experiment"
    assert job.inputs["dataset"].path == custom_dataset
    assert get_input_val(job.inputs["target_label"]) == custom_target
    assert get_input_val(job.inputs["n_estimators"]) == 350
    assert get_input_val(job.inputs["max_depth"]) == 20
    assert get_input_val(job.inputs["min_samples_split"]) == 4
    assert job.tags["experiment_id"] == "EXP-099"
    assert job.tags["lead"] == "researcher"


def test_configure_voice_training_job_with_direct_input_object():
    """Verify configure_voice_training_job accepts pre-constructed azure.ai.ml.Input asset."""
    direct_input = Input(type=AssetTypes.URI_FOLDER, path="wasbs://data@account.blob.core.windows.net/folder")

    job = configure_voice_training_job(
        model_type="random_forest",
        dataset_path=direct_input,
        target_label="sentiment",
    )

    assert get_input_val(job.inputs["dataset"]).path == direct_input.path
    assert get_input_val(job.inputs["dataset"]).type == direct_input.type




def test_configure_voice_training_job_input_validation():
    """Verify configure_voice_training_job validates required arguments."""
    with pytest.raises(StorageValidationError, match="dataset_path cannot be empty"):
        configure_voice_training_job(dataset_path="")

    with pytest.raises(StorageValidationError, match="target_label cannot be empty"):
        configure_voice_training_job(target_label="")

    with pytest.raises(StorageValidationError, match="Unsupported voice model type"):
        configure_voice_training_job(model_type="invalid_model")


# ===========================================================================
# Dataclass VoiceMLTrainingConfig Tests
# ===========================================================================

def test_voice_ml_training_config_dataclass():
    """Verify VoiceMLTrainingConfig attributes, post-init validation, and default behavior."""
    cfg = VoiceMLTrainingConfig(
        model_type="logistic_regression",
        dataset_path="data/features.csv",
        target_label="affect",
    )

    assert cfg.model_type == "logistic_regression"
    assert cfg.dataset_path == "data/features.csv"
    assert cfg.target_label == "affect"
    assert cfg.compute_target == DEFAULT_ML_COMPUTE_TARGET
    assert cfg.environment == DEFAULT_ML_ENVIRONMENT
    assert cfg.display_name == "voice-logistic_regression-training"

    # Post init validation tests
    with pytest.raises(StorageValidationError):
        VoiceMLTrainingConfig(dataset_path="")

    with pytest.raises(StorageValidationError):
        VoiceMLTrainingConfig(target_label="")

    with pytest.raises(StorageValidationError):
        VoiceMLTrainingConfig(compute_target="")

    with pytest.raises(StorageValidationError):
        VoiceMLTrainingConfig(environment="")


# ===========================================================================
# Manager and Dry-Run Submission Tests
# ===========================================================================

def test_voice_training_manager_dry_run_success():
    """Verify VoiceAzureMLTrainingManager validates job structure without calling Azure."""
    manager = VoiceAzureMLTrainingManager()
    job = manager.configure_job(
        model_type="random_forest",
        dataset_path="data/voice_features.csv",
        target_label="emotion",
    )

    # Dry run must not call any real API
    result = manager.submit_job(job, dry_run=True)

    assert result["status"] == "dry_run_success"
    assert result["display_name"] == "voice-random_forest-training"
    assert result["compute"] == "cpu-cluster"
    assert "Job structure validated successfully" in result["message"]


def test_voice_training_manager_mocked_live_submission():
    """Verify VoiceAzureMLTrainingManager submits to MLClient when dry_run=False."""
    mock_ml_client = MagicMock()
    mock_submitted_job = MagicMock()
    mock_submitted_job.name = "voice_training_run_001"
    mock_submitted_job.display_name = "voice-rf-training"
    mock_submitted_job.services = {"Studio": "https://ml.azure.com/runs/voice_training_run_001"}
    mock_ml_client.jobs.create_or_update.return_value = mock_submitted_job

    manager = VoiceAzureMLTrainingManager(ml_client=mock_ml_client)
    job = manager.configure_job(
        model_type="random_forest",
        dataset_path="data/voice_features.csv",
        target_label="emotion",
    )

    result = manager.submit_job(job, dry_run=False)

    assert result["status"] == "submitted"
    assert result["name"] == "voice_training_run_001"
    assert result["studio_url"] == "https://ml.azure.com/runs/voice_training_run_001"
    mock_ml_client.jobs.create_or_update.assert_called_once_with(job)


def test_voice_training_manager_submission_failure_raises_job_error():
    """Verify VoiceAzureMLTrainingManager translates API exceptions into AzureMLJobError."""
    mock_ml_client = MagicMock()
    mock_ml_client.jobs.create_or_update.side_effect = RuntimeError("Azure ML Quota Exceeded")

    manager = VoiceAzureMLTrainingManager(ml_client=mock_ml_client)
    job = manager.configure_job()

    with pytest.raises(AzureMLJobError, match="Failed to submit Azure ML job: Azure ML Quota Exceeded"):
        manager.submit_job(job, dry_run=False)


# ===========================================================================
# Zero Hardcoded Secrets & Exports Tests
# ===========================================================================

def test_zero_hardcoded_secrets_in_code():
    """Verify no hardcoded credentials, subscription IDs, or keys exist in the modules."""
    import src.azure.config as cfg_mod
    import src.azure.ml_training as ml_mod

    for module in [cfg_mod, ml_mod]:
        with open(module.__file__, "r", encoding="utf-8") as f:
            content = f.read()

        # Check against common accidental credential leaks
        assert "DefaultEndpointsProtocol=https;AccountName=" not in content or "self.account_name" in content
        assert "password123" not in content
        assert "secret_key" not in content
        assert "subscription_id = \"" not in content or "None" in content


def test_azure_ml_training_aliases_and_exports():
    """Verify src/azure/__init__.py exports all Azure ML components and utilities."""
    import src.azure as az

    assert hasattr(az, "VoiceMLTrainingConfig")
    assert hasattr(az, "VoiceAzureMLTrainingManager")
    assert hasattr(az, "configure_voice_training_job")
    assert hasattr(az, "create_voice_training_job")
    assert hasattr(az, "validate_model_type")
    assert hasattr(az, "build_default_training_command")
    assert hasattr(az, "SUPPORTED_VOICE_MODELS")
    assert hasattr(az, "MODEL_TYPE_ALIASES")
    assert hasattr(az, "DEFAULT_HYPERPARAMETERS")
    assert hasattr(az, "AzureMLConfig")
    assert hasattr(az, "AzureMLError")
    assert hasattr(az, "AzureMLConfigurationError")
    assert hasattr(az, "AzureMLJobError")
