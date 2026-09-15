# Azure ML questionnaire configuration

This directory contains Azure ML v2 configuration templates and supporting
scripts for the questionnaire modality training and inference workflow.

> **Azure cloud execution has NOT been performed.**
> No Azure resources have been created, no jobs submitted, no endpoints deployed,
> and no models registered. All configuration files use placeholders where
> real Azure resource values are required.

---

## Implementation status

### Locally implemented and validated

| Component | File | Status |
|---|---|---|
| Questionnaire preprocessing | `src/ai_model/questionnaire/` | ✅ Complete |
| Model training (LR + RF) | `src/ai_model/questionnaire/train_models.py` | ✅ Complete |
| Cross-validation evaluation | `src/ai_model/questionnaire/evaluate_models.py` | ✅ Complete |
| Explainability | `src/ai_model/questionnaire/explainability.py` | ✅ Complete |
| Prediction pipeline | `src/ai_model/questionnaire/predict.py` | ✅ Complete |
| Model artifacts | `models/questionnaire/` | ✅ Complete |
| Azure training adapter | `src/azure/questionnaire/train.py` | ✅ Complete |
| Questionnaire HTTP API | `src/backend/questionnaire/api.py` | ✅ Complete |
| API unit tests | `src/backend/questionnaire/test_api.py` | ✅ 4/4 PASS |

### Configured but not yet executed in Azure

| Component | File | Status |
|---|---|---|
| Azure ML workspace | `workspace.yml.template` | 🔧 Template only |
| Azure ML compute cluster | `compute.yml.template` | 🔧 Template only |
| Azure ML environment | `environment.yml` + `conda.yml` | 🔧 Template only |
| Azure ML training job | `job.yml.template` | 🔧 Template only |
| Azure ML model registration | `model.yml.template` | 🔧 Template only |
| Azure ML managed online endpoint | `endpoint.yml.template` | 🔧 Template only |
| Azure ML managed online deployment | `deployment.yml.template` | 🔧 Template only |
| Azure ML scoring adapter | `score.py` | 🔧 Ready; not deployed |

---

## Files

| File | Purpose |
|---|---|
| `workspace.yml.template` | Workspace name, resource group, and region. Copy locally and replace placeholders before `az ml workspace create`. |
| `compute.yml.template` | AML compute-cluster definition (Standard_DS3_v2, 0–2 nodes). Replace cluster name before creating. |
| `conda.yml` | **Authoritative** Azure ML conda dependency specification. Mirrors `requirements.txt` exactly. |
| `environment.yml` | Azure ML environment definition referencing `conda.yml`. |
| `requirements.txt` | Standalone pinned dependency record for local reproducibility inspection. |
| `job.yml.template` | Azure ML command job: mounts the approved private data asset and invokes `train.py`. |
| `train.py` | Azure ML training adapter. Accepts `--data` and `--output-dir`, stages input, delegates to `train_models.run_training`, `evaluate_models.run_evaluation`, and `explainability.run_explainability`, validates artifacts, copies output. |
| `model.yml.template` | Azure ML model asset registration for the selected Random Forest pipeline. |
| `model_registration.md` | Model registration documentation: selected model, artifacts, versioning, workflow steps, reproducibility, non-clinical limitation. |
| `endpoint.yml.template` | Azure ML managed online endpoint definition. |
| `deployment.yml.template` | Azure ML managed online deployment referencing `questionnaire-random-forest` and `score.py`. |
| `score.py` | Azure ML scoring adapter. Thin wrapper — all inference is delegated to the existing `predict_questionnaire` function; no model logic is duplicated. |
| `README.md` | This file. |

---

## Intended end-to-end flows

### Local prediction flow

```
Questionnaire response (dict Q1–Q20 required, Q21–Q25 optional)
        ↓
POST /predict/questionnaire  [src/backend/questionnaire/api.py]
        ↓
predict_questionnaire()  [src/ai_model/questionnaire/predict.py]
        ↓
Risk level + risk score + confidence + key factors + data quality
        ↓
Future multimodal backend / fusion integration
```

### Azure training flow (not yet executed)

```
Approved private questionnaire data asset (Azure ML datastore)
        ↓
Azure ML command job  [job.yml.template → train.py]
        ↓
train_models.run_training() + evaluate_models.run_evaluation() + explainability.run_explainability()
        ↓
Complete inference artifact set (.joblib models, feature names, training metadata, selected_model.json, explainability_metadata.json)
        ↓
Azure ML output path (artifact datastore)
        ↓
az ml model create  [model.yml.template]
        ↓
Registered questionnaire-random-forest model version
```

### Azure inference flow (not yet executed)

```
Registered questionnaire-random-forest model
        ↓
az ml online-endpoint create  [endpoint.yml.template]
az ml online-deployment create  [deployment.yml.template]
        ↓
Azure ML managed online endpoint
        ↓
HTTP POST  {"responses": {"Q1": "...", ..., "Q20": "..."}}
        ↓
score.py init() → MODELS_DIR redirected to Azure artifact path
score.py run() → predict_questionnaire()
        ↓
Fusion contract JSON response
```

---

## Required values before Azure execution

The following placeholder values must be replaced (outside version control) before any Azure operation:

| Placeholder | Used in | Description |
|---|---|---|
| `<azure-ml-workspace-name>` | `workspace.yml.template` | Azure ML workspace name |
| `<azure-resource-group>` | `workspace.yml.template` | Azure resource group name |
| `<azure-region>` | `workspace.yml.template` | Azure region (e.g. `eastus`) |
| `<questionnaire-compute-cluster-name>` | `compute.yml.template`, `job.yml.template` | AML compute cluster name |
| `<private-questionnaire-data-asset>` | `job.yml.template` | Azure ML data asset name for approved questionnaire CSV |
| `<artifact-datastore-name>` | `job.yml.template`, `model.yml.template` | Azure ML datastore for model artifacts |
| `<job-name>` | `model.yml.template` | Azure ML job run name (from Studio after job completes) |
| `<questionnaire-endpoint-name>` | `endpoint.yml.template`, `deployment.yml.template` | Azure ML endpoint name |
| `<questionnaire-random-forest-version>` | `deployment.yml.template` | Registered model version integer |

> Authenticate using Azure CLI (`az login`), managed identity, or another approved mechanism.
> **Never place subscription IDs, tenant IDs, API keys, tokens, or credentials in these files.**

---

## Data and privacy policy

- The questionnaire training CSV (`questionnaire_with_target.csv`) is excluded from Git (`.gitignore`).
- No participant-level data is committed to version control.
- The Azure data asset must contain approved real participant data only and must not be committed to Git.
- `score.py` and `api.py` do not log participant-level questionnaire responses.

---

## Non-clinical limitations

- Risk levels (low / moderate / high) are project-defined categories derived from Q4 self-report.
- Risk scores and confidence values are model outputs; they are not clinical assessments.
- Confidence is a model heuristic, not clinical certainty.
- The model was trained on 52 respondents; generalisation is limited.
- No clinical validation has been performed.
- Synthetic data was not used for model training or evaluation.
- Q4 and Timestamp are excluded from predictors in all contexts.

---

## Methodology boundary

The configuration in this directory references the existing training source
only.  It does not change the Q4-derived target, real-data policy, Logistic
Regression / Random Forest configuration, cross-validation methodology,
prediction pipeline, or fusion engine.  `train.py` and `score.py` are thin
adapters; the methodology source of truth remains
`src/ai_model/questionnaire/train_models.py` and
`src/ai_model/questionnaire/predict.py`.
