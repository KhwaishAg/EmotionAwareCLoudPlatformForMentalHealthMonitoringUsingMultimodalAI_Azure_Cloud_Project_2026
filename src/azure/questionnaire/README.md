# Azure ML questionnaire configuration

This directory contains Azure ML v2 configuration templates for the existing questionnaire training workflow. It does not create Azure resources, submit jobs, upload data, deploy an endpoint, or prove Azure execution. Azure execution has **not** been performed in this milestone.

## Files

- `workspace.yml.template` describes workspace name, resource group, and region. Copy it locally and replace its placeholders before creating a workspace.
- `compute.yml.template` is an AML compute-cluster reference. Replace the cluster name and review VM availability and quota before creating it.
- `conda.yml` lists only dependencies already used by the questionnaire pipeline.
- `environment.yml` defines an Azure ML environment using `conda.yml`.
- `job.yml.template` is a future Azure-specific training-entrypoint contract. It identifies the repository-root code snapshot, environment/compute references, private data input, and model/evaluation artifact outputs. Its command intentionally contains a placeholder and must not be submitted until that entrypoint exists.

## Required values before Azure execution

Supply a real Azure ML workspace name, resource group, region, compute cluster name, private Azure ML data-asset name/version, and artifact datastore name. The job template declares versioned model and evaluation output paths under that datastore. Authenticate outside this repository using Azure CLI, managed identity, or another approved identity mechanism. Do not place subscription IDs, tenant IDs, keys, tokens, or credentials in these files.

The local `src/ai_model/questionnaire/train_models.py` script resolves input and artifact paths from the repository root and does **not** consume the Azure ML mounted input declared in `job.yml.template`. Respondent-level CSVs remain local/private and are excluded from Git; therefore the template is deliberately not independently executable. A future, separately reviewed Azure training entrypoint must consume `questionnaire_training_data` and route generated models/results to the declared Azure ML outputs. This milestone intentionally does not implement that entrypoint.

## Methodology boundary

The configuration references the existing training source location only. It does not change the Q4-derived target, real-data policy, Logistic Regression/Random Forest configuration, cross-validation methodology, prediction pipeline, or fusion engine. The questionnaire data asset must contain approved real participant data only and must not be committed to Git.

## Intended future workflow

1. Copy workspace and compute templates outside version control and replace placeholders.
2. Create/select Azure resources with authenticated Azure CLI or Azure ML tooling.
3. Register `environment.yml` and an approved private questionnaire data asset.
4. Add and review the dedicated Azure training entrypoint before submitting a completed job configuration.
