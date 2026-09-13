# Azure Cloud Architecture & Infrastructure Planning

## Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

This document details the planned Microsoft Azure cloud architecture, storage configurations, model lifecycle management, and security governance for the **Multimodal Student Stress/Risk Assessment and Early-Intervention Support System**.

> [!NOTE]
> **No Cloud Resources Created:**
> This document describes planned architectural designs. No Azure cloud resources, credentials, or automated deployment scripts are provisioned during this documentation pass.

---

## Planned Azure Service Roles

```
                                  +---------------------------------------+
                                  |         DEVELOPER ENVIRONMENT         |
                                  |  - Local Feature Extraction (Python)  |
                                  |  - Local Model Training & Validation  |
                                  +-------------------+-------------------+
                                                      |
                                                      v (CI/CD via GitHub Actions)
+-----------------------------------------------------------------------------------------+
|                                    MICROSOFT AZURE                                      |
|                                                                                         |
|  +---------------------------+   +---------------------------+   +-------------------+  |
|  |    Azure Blob Storage     |   |   Azure Machine Learning  |   | Azure App Service |  |
|  | - Encrypted Audio Blobs   |-->| - Experiment Tracking     |-->| - Containerized   |  |
|  | - Processed Feature Sets  |   | - Model Registry (MLflow) |   |   Backend REST    |  |
|  | - Anonymous Datasets      |   | - Automated Pipelines     |   |   API Engine      |  |
|  +---------------------------+   +---------------------------+   +---------+---------+  |
|                                                                            |            |
+----------------------------------------------------------------------------+------------+
                                                                             |
                                                                             v
                                                                     [Web Frontend UI]
```

### 1. Azure Blob Storage
- **Purpose:** Secure, scalable object storage for non-code assets.
- **Planned Containers:**
  - `audio-raw/`: Standardized participant audio files (`.wav`), organized strictly by anonymous participant ID (`P001/`, `P002/`).
  - `features-processed/`: Extracted acoustic feature tables (Parquet / CSV format) ready for model ingestion.
  - `model-artifacts/`: Serialized model checkpoints (e.g., `.joblib`, `.onnx`).

### 2. Azure Machine Learning (Azure ML)
- **Purpose:** Centralized, reproducible environment for training, tuning, and versioning modality models.
- **Capabilities:**
  - **Experiment Tracking:** Logging hyperparameter sweeps, metric curves, and confusion matrices across training runs.
  - **Model Registry:** Storing versioned model artifacts with lineage tracking from raw data to final estimator.
  - **Pipelines:** Orchestrating batch feature extraction and multi-modality retraining.

### 3. Azure App Service
- **Purpose:** Managed platform for deploying the containerized backend REST API service.
- **Capabilities:**
  - Auto-scaling based on HTTP request volume.
  - TLS/HTTPS termination by default.
  - Seamless environment variable and application setting integration.

### 4. GitHub Actions (CI/CD)
- **Purpose:** Continuous integration and deployment automation.
- **Workflow:** Running automated tests, linting code style, validating doc schemas, and triggering deployment to Azure App Service upon merged PRs to `main`.

---

## Note on Voice Processing & Azure Speech Services

> [!IMPORTANT]
> **Acoustic Extraction Is Local / Python-Based:**
> **Azure Speech Services is NOT required for the voice modality.**
>
> The core acoustic feature extraction pipeline (pitch tracking, MFCC computation, spectral centroid/rolloff, energy contouring) is implemented natively in Python using standard scientific audio libraries (e.g., Librosa, SoundFile, SciPy).
>
> Cloud-based speech APIs (such as Azure Speech-to-Text or Pronunciation Assessment) may only be explored in future work as optional experimental comparisons and are **not a dependency** of the core system.

---

## Security Governance & Secrets Management

To ensure academic research integrity and prevent data exposure, all contributors must adhere to the following policies:

### 1. Zero Hardcoded Credentials Policy
- **Never commit:** API keys, connection strings, Azure storage account keys, tenant IDs, or client secrets to source control.
- All secrets must be loaded dynamically at runtime via **environment variables** or Azure Key Vault references:
  ```python
  # Correct practice
  import os
  AZURE_STORAGE_CONNECTION = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
  ```

### 2. Participant Privacy in Cloud Storage
- Cloud storage containers must enforce **Storage Service Encryption (SSE)** at rest and secure transfer (HTTPS) in transit.
- Public blob access must remain strictly **disabled**.
- Access between services must utilize Azure Managed Identities (Entra ID) rather than persistent connection strings where possible.

### 3. Local Development vs. Cloud Deployment Boundary

| Environment | Data Storage | Model Execution | Intended Usage |
|---|---|---|---|
| **Local Development** | Local `scratch/` or mock fixtures (excluded in `.gitignore`) | Local CPU / Python virtual environment (`.venv`) | Rapid feature development, debugging, unit testing |
| **Cloud Deployment** | Azure Blob Storage (access-controlled) | Azure ML / Azure App Service containers | Production inference, team-wide benchmark training |
