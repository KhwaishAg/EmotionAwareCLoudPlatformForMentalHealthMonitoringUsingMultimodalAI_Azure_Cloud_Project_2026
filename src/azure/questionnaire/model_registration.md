# Azure ML Questionnaire Model Registration

> **Status:** Configuration template prepared. Azure model registration has **not** been performed.

---

## Selected model

**Random Forest** questionnaire stress-risk classifier (scikit-learn Pipeline).

Random Forest was selected over Logistic Regression based on 4-fold stratified cross-validation evidence (Random Forest mean balanced accuracy 0.5611 vs 0.5278 for Logistic Regression). This preliminary selection is not clinical validation.

---

## Primary artifact

| Artifact | Location | Description |
|---|---|---|
| `random_forest.joblib` | `models/questionnaire/random_forest.joblib` | Full sklearn Pipeline: SimpleImputer(median) → RandomForestClassifier |
| `logistic_regression.joblib` | `models/questionnaire/logistic_regression.joblib` | Retained as alternative; not selected |

## Supporting metadata artifacts

| File | Purpose |
|---|---|
| `models/questionnaire/questionnaire_feature_names.json` | Ordered list of 76 predictor feature names; loaded by inference to align input columns |
| `models/questionnaire/training_metadata.json` | Training configuration, class counts, random state, confirms `synthetic_data_used: false` |
| `models/questionnaire/selected_model.json` | Records which model is active; read by `predict.py` at inference time |
| `models/questionnaire/explainability_metadata.json` | Top-feature associations used in API `key_factors` response field |

---

## Versioning

The `model.yml.template` file registers the model as:

```
name:    questionnaire-random-forest
version: 1
type:    custom_model
```

Increment `version` for each retraining run. Azure ML preserves all registered versions, enabling rollback and lineage tracking. The training job output path (`path:` in the template) links the registered version to the exact job that produced it.

---

## Registration workflow (not yet executed)

1. Complete a successful Azure ML training job using `job.yml.template`.
2. Note the job name from the Azure ML Studio run history (e.g., `questionnaire-training-<run-id>`).
3. Copy `model.yml.template` to a local file (outside version control).
4. Replace `<artifact-datastore-name>` and `<job-name>` with real values.
5. Authenticate with Azure CLI:
   ```bash
   az login
   az account set --subscription <subscription-id>
   ```
6. Register the model:
   ```bash
   az ml model create -f model.yml \
     --workspace-name <workspace-name> \
     --resource-group <resource-group>
   ```
7. Verify registration in Azure ML Studio under **Models**.

> Do not place subscription IDs, tenant IDs, keys, or credentials in version-controlled files.

---

## Reproducibility

- `random_state = 42` in both models (recorded in `training_metadata.json`).
- `synthetic_data_used = false` — model was trained on 52 real questionnaire responses only.
- Methodology is unchanged from `src/ai_model/questionnaire/train_models.py`. The Azure adapter (`src/azure/questionnaire/train.py`) does not modify the training algorithm.
- Input CSV (`questionnaire_with_target.csv`) is excluded from Git (`.gitignore`) and must be supplied only as an approved private Azure ML data asset.

---

## Non-clinical limitation

This model is a **project-defined, non-clinical questionnaire stress classification tool**.

- It is not a clinical diagnostic instrument.
- Confidence scores are model heuristics, not clinical certainty.
- Risk levels (low / moderate / high) are project-defined categories derived from Q4 self-report.
- The model was trained on 52 respondents; generalisation is limited.
- No clinical validation has been performed.
