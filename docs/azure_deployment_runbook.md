# Azure deployment runbook (optional, NOT used, not executed)

> **This project is configured to run at zero cost locally; nothing here is required** (see `zero_cost_local_mode.md`).
> **If you want Azure within a small credit budget, use `azure_free_tier_deployment.md` instead** (free F1 web app; no ML endpoint/workspace).
> This runbook is preserved for reference only. Do not follow it unless you deliberately accept cloud charges.

> **Nothing in this document has been run.** No Azure resources exist for this project and the commands below have
> not been verified against a real subscription. They are the intended steps, derived from the repository's
> templates in `src/azure/questionnaire/`. Creating Azure ML compute, endpoints or App Service **incurs cost**; get
> approval first and delete resources when finished.

## What can and cannot be checked without Azure

| Verified offline (see `tests/test_azure_*.py`) | Requires real Azure |
|---|---|
| YAML templates parse, contain only `<placeholders>`, no secrets | Authentication, subscription/tenant permissions |
| `score.py` `init()`/`run()` against local model artifacts | Workspace, compute cluster, environment build |
| Blob/ML client wrappers, validation, error handling (mocked SDK) | Training job execution, model registration |
| Conda pins match `requirements.txt` | Endpoint/deployment creation, scoring over HTTPS |
| | App Service hosting of the FastAPI app |

## Prerequisites

* Azure subscription with permission to create resources in one resource group.
* Azure CLI with the ML extension: `az extension add -n ml`.
* An approved, private copy of `questionnaire_with_target.csv` (it is git-ignored and must never be committed).
* Sign in with `az login`. **Never put keys, tenant/subscription IDs or secrets in repository files.**

## Placeholders to replace (in local copies of the templates, not committed)

`<azure-ml-workspace-name>`, `<azure-resource-group>`, `<azure-region>`, `<questionnaire-compute-cluster-name>`,
`<private-questionnaire-data-asset>`, `<artifact-datastore-name>`, `<job-name>`, `<questionnaire-endpoint-name>`,
`<questionnaire-random-forest-version>`.

## Questionnaire model: train, register, deploy

```bash
cd src/azure/questionnaire
RG=<azure-resource-group>; WS=<azure-ml-workspace-name>

# 1. Workspace and compute (cost: compute scales to 0 when idle)
az ml workspace create -f workspace.yml --resource-group $RG
az ml compute create   -f compute.yml   --workspace-name $WS --resource-group $RG

# 2. Environment (conda.yml pins scikit-learn 1.8.0, the version the artifacts need)
az ml environment create -f environment.yml --workspace-name $WS --resource-group $RG

# 3. Upload the approved data as a private data asset, then train
az ml data create --name <private-questionnaire-data-asset> --type uri_folder --path <local-folder-with-csv> \
  --workspace-name $WS --resource-group $RG
az ml job create -f job.yml --workspace-name $WS --resource-group $RG

# 4. Register the model from the finished job's output path
az ml model create -f model.yml --workspace-name $WS --resource-group $RG

# 5. Endpoint + deployment (incurs hourly cost while running)
az ml online-endpoint create   -f endpoint.yml   --workspace-name $WS --resource-group $RG
az ml online-deployment create -f deployment.yml --all-traffic --workspace-name $WS --resource-group $RG

# 6. Smoke test, then tear down
az ml online-endpoint invoke -n <questionnaire-endpoint-name> --request-file request.json \
  --workspace-name $WS --resource-group $RG
az ml online-endpoint delete -n <questionnaire-endpoint-name> --workspace-name $WS --resource-group $RG --yes
```

(`workspace.yml`, `compute.yml`, `job.yml`, `model.yml`, `endpoint.yml`, `deployment.yml` are local, filled-in copies
of the matching `*.template` files.)

Known unverified points to check on first deployment: the managed-endpoint environment needs
`azureml-inference-server-http` (added to `conda.yml` but never built); the code snapshot (`code: ../../..`) uploads
the whole repository, so make sure no data, `venv/` or secrets are present (`.gitignore` entries help; consider an
`.amlignore`).

## Hosting the assessment API (recommended shape: models + fusion in one container)

The assessment API loads the model artifacts in-process (`models/questionnaire`, `models/behavioral`), so it can be
hosted on Azure App Service without calling the ML endpoint. Requirements: Python 3.11+ (scikit-learn 1.8.0), the
packages in `requirements.txt`, startup command
`python -m uvicorn src.backend.assessment.api:app --host 0.0.0.0 --port 8000`. Not yet containerised or deployed; no
Dockerfile or CI workflow exists in the repository. Before exposing it publicly add HTTPS, authentication and request
rate limiting; the API currently has none.

## Voice on Azure

Voice has no trained model. The offline-tested pieces are `src/azure/storage.py` (audio blobs),
`src/azure/upload_handler.py` and `src/azure/ml_training.py` (builds an Azure ML training job from environment
variables listed in the README). Training requires a real labeled voice dataset and an Azure ML workspace; neither
exists yet.
