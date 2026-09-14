# Azure ML questionnaire configuration

This directory contains Azure ML v2 configuration templates for the existing questionnaire training workflow. It does not create Azure resources, submit jobs, upload data, deploy an endpoint, or prove Azure execution. Azure execution has **not** been performed in this milestone.

## Files

- `workspace.yml.template` describes workspace name, resource group, and region. Copy it locally and replace its placeholders before creating a workspace.
- `compute.yml.template` is an AML compute-cluster reference. Replace the cluster name and review VM availability and quota before creating it.
- `requirements.txt` is a standalone, pinned dependency record for local inspection and reproducibility. It lists only packages used by the existing questionnaire ML pipeline.
- `conda.yml` is the authoritative Azure ML dependency specification and mirrors `requirements.txt` exactly.
- `environment.yml` defines the Azure ML environment using `conda.yml`; Azure ML uses this environment/conda pair rather than `requirements.txt` directly.
- `train.py` is the Azure ML command-line adapter. It accepts `--data` and `--output-dir`, stages the approved `questionnaire_with_target.csv` input in temporary storage, delegates to `train_models.run_training`, validates generated artifacts, and copies them to the supplied output directory.
- `job.yml.template` invokes `train.py` with the mounted private data asset and Azure ML model-artifact output. It identifies the repository-root code snapshot and environment/compute references.

## Required values before Azure execution

Supply a real Azure ML workspace name, resource group, region, compute cluster name, private Azure ML data-asset name/version, and artifact datastore name. The job template declares a versioned model-artifact output path under that datastore. Authenticate outside this repository using Azure CLI, managed identity, or another approved identity mechanism. Do not place subscription IDs, tenant IDs, keys, tokens, or credentials in these files.

Dependency and configuration files must remain aligned with the existing questionnaire pipeline. They must never contain secrets, credentials, participant records, or participant-level data.

The local `src/ai_model/questionnaire/train_models.py` script remains the methodology source of truth and is not modified. `train.py` is its minimal Azure adapter: it consumes the mounted `questionnaire_training_data`, stages the required input file, redirects artifact paths only in process, and copies artifacts to the Azure ML output. Respondent-level CSVs remain local/private and are excluded from Git. Actual Azure execution has not yet been performed.

## Methodology boundary

The configuration references the existing training source location only. It does not change the Q4-derived target, real-data policy, Logistic Regression/Random Forest configuration, cross-validation methodology, prediction pipeline, or fusion engine. The questionnaire data asset must contain approved real participant data only and must not be committed to Git.

## Intended future workflow

1. Copy workspace and compute templates outside version control and replace placeholders.
2. Create/select Azure resources with authenticated Azure CLI or Azure ML tooling.
3. Register `environment.yml` and an approved private questionnaire data asset.
4. Submit a completed job configuration only after replacing Azure resource/data placeholders and approving access to the private data asset.
