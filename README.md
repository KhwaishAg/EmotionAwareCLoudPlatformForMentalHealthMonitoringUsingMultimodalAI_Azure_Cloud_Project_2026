# Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI

## Project Overview

**Emotion Aware Cloud Platform For Mental Health Monitoring Using Multimodal AI** is an academic research and engineering project aimed at developing a **Multimodal Student Stress/Risk Assessment and Early-Intervention Support System**.

The platform is designed to provide a low-burden, explainable, and multi-perspective assessment of student stress and well-being indicators by integrating three complementary, non-invasive modalities:

1. **Questionnaire** — Subjective / self-reported perspective (academic stress, perceived pressure, emotional self-assessment)
2. **Voice** — Expressive / acoustic indicators derived from short standardized voice recordings
3. **Behavioral Information** — Contextual, lifestyle, and academic behavioral indicators (study patterns, sleep regularity, workload trends)

Outputs from these three modalities are synthesized through **multimodal fusion** to generate an estimated risk/stress level along with cross-modal consistency analysis, confidence estimates, factor explainability, and personalized supportive recommendations.

---

## Problem Statement

College students frequently experience acute and chronic stress stemming from academic workload, examinations, deadlines, and lifestyle adjustments. Conventional assessment methods face notable limitations:

- **Single-modality bias:** Relying solely on questionnaires introduces subjective reporting bias, social desirability bias, or retrospective recall errors.
- **High user burden:** Complex diagnostic interviews or invasive wearable sensor protocols can create friction, leading to low adherence.
- **Lack of explainability:** Black-box prediction outputs that state a binary outcome (e.g., "You are stressed") offer little actionable insight and foster distrust.
- **Inability to capture discordance:** When a student reports feeling "fine" on a questionnaire but exhibits significant acoustic or behavioral stress indicators, single-modality systems fail to detect the underlying tension.

This project addresses these challenges by combining three non-invasive modalities into an integrated, explainable assessment framework.

---

## Current Project Scope

The implementation scope is explicitly focused and bounded:

- **Included Modalities:** Exactly three modalities:
  1. Questionnaire
  2. Voice (short audio recordings)
  3. Behavioral information (academic & lifestyle context)
- **Target Audience:** College / university students.
- **Output Objectives:** Estimated stress/risk level, modality breakdown, modality consistency/disagreement, confidence metric, contributing factors (explainability), and supportive early-intervention recommendations.

### Out of Scope (Explicit Exclusions)

The following components are **NOT** part of the current implementation:
- Facial image or video analysis
- Facial emotion recognition (FER) / facial expression analysis
- Wearable or physiological sensors (e.g., smartwatches, ECG, PPG, galvanic skin response)
- Mandatory biometric hardware
- Separate natural-language text processing modality (beyond structured questionnaire responses)

---

## The Three Modalities

```
+-------------------+---------------------------------------------------------+
| Modality          | Description & Perspective                               |
+-------------------+---------------------------------------------------------+
| 1. Questionnaire  | Subjective / Self-Reported Perspective                  |
|                   | Captures perceived stress, academic load evaluation,     |
|                   | and self-reported emotional states via structured items.|
+-------------------+---------------------------------------------------------+
| 2. Voice          | Acoustic / Expressive Indicator Perspective              |
|                   | Captures acoustic properties (pitch, energy, spectral   |
|                   | dynamics, MFCCs) from a short 30-60s spoken response    |
|                   | to a standardized prompt.                               |
+-------------------+---------------------------------------------------------+
| 3. Behavioral     | Contextual / Lifestyle Perspective                      |
|                   | Captures academic habits, study duration, sleep trends,  |
|                   | deadline proximity, and routine regularity indicators.   |
+-------------------+---------------------------------------------------------+
```

---

## High-Level Architecture

The system follows a modular pipeline where each modality is analyzed by a specialized model before multimodal fusion and assessment generation:

```
Student
   |
   +---------------- Questionnaire
   |                       |
   |                       v
   |                 Questionnaire Model
   |
   +---------------- Voice (Short Recording)
   |                       |
   |                       v
   |                    Voice Model
   |
   +---------------- Behavioral Information
                           |
                           v
                    Behavioral Model
                           |
                           v
                    Multimodal Fusion
                           |
                           v
                    Overall Assessment
                           |
             +-------------+-------------+
             |             |             |
        Consistency    Confidence   Explainability
                                           |
                                           v
                               Personalized Supportive
                                   Recommendations
```

---

## Team Ownership & Branch Strategy

The project is developed by a 3-member team with clear modular responsibilities across dedicated feature branches:

| Member | Branch | Primary Responsibility | Scope Details |
|---|---|---|---|
| **Khwaish** (Member 1) | `feature/Khwaish` | **Questionnaire Modality** | Questionnaire design, data representation, questionnaire ML model |
| **Nandini** (Member 2) | `feature/Nandini` | **Voice Modality** | Audio preprocessing pipeline, acoustic feature extraction, voice ML model |
| **Netal** (Member 3) | `feature/Netal` | **Behavioral Modality** & Integration | Behavioral feature engineering, behavioral model, multimodal fusion, backend & deployment integration |

### Git Branch Structure

- `main` — Production-ready, stable releases.
- `develop` — Shared integration branch.
- `feature/Khwaish` — Questionnaire modality development.
- `feature/Nandini` — Voice modality development.
- `feature/Netal` — Behavioral modality, fusion, and integration development.

*Note: All current work for Member 2 remains strictly on `feature/Nandini`.*

---

## Planned Novelty

The project investigates several novel system-level characteristics:

1. **Low-Burden Multimodal Assessment:** Combines short audio (30–60 s), lightweight questionnaire items, and basic behavioral indicators without requiring wearable hardware or invasive monitoring.
2. **Tri-Perspective Synthesis:** Merges subjective self-report, objective acoustic signals, and contextual behavioral patterns for robust risk estimation.
3. **Cross-Modal Consistency & Disagreement Analysis:** Explicitly quantifies whether modalities agree (e.g., all indicate mild tension) or conflict (e.g., low self-reported stress but high vocal/behavioral tension), highlighting masked stress.
4. **Confidence-Aware Assessment:** Reports uncertainty/confidence scores alongside estimates so students and advisors understand output reliability.
5. **Factor Explainability:** Breaks down contributing factors per modality rather than delivering an opaque score.
6. **Non-Clinical Early-Intervention Support:** Maps assessment profiles to constructive, non-clinical recommendations (e.g., study pacing, sleep hygiene, breathing exercises, campus wellness resources).

*(Note: These novelty concepts represent planned research goals and will be iteratively implemented and validated.)*

---

## Non-Clinical Disclaimer

> [!IMPORTANT]
> **This platform is an academic research and early-intervention support system, NOT a medical diagnostic tool.**
>
> - The system does **not** diagnose depression, anxiety, or any psychiatric disorder.
> - The system does **not** provide medical or clinical advice.
> - The system does **not** replace licensed psychologists, psychiatrists, counselors, or medical professionals.
> - All outputs are framed strictly as **stress/risk indicators**, **estimated well-being indicators**, and **supportive recommendations** for proactive student self-care.

---

## Repository Directory Guide

```
EmotionAwareCloudPlatform/
|-- architecture/       # System architecture documentation and pipeline diagrams
|-- dataset/            # Data schemas, voice recording protocols, and ethical guidelines
|-- docs/               # Project documentation index, glossary, and ethical policies
|-- literature_survey/  # Literature reviews across questionnaire, voice, and behavioral modalities
|-- presentation/       # Presentation decks, milestone reviews, and demo assets
|-- references/         # Academic references, citation standards, and bibliography
|-- results/            # Experimental results, evaluation metrics, and benchmarking logs
|-- src/                # Source code
|   |-- ai_model/       # Modality models (Questionnaire, Voice, Behavioral) & fusion logic
|   |-- azure/          # Cloud infrastructure and deployment specifications
|   |-- backend/        # REST API and assessment orchestration service
|   +-- frontend/       # Student assessment web UI (served by the API)
+-- README.md           # Root project overview and architecture documentation (this file)
```

---

## Current Development Status

| Area | Status |
|---|---|
| Questionnaire model + prediction (`src/ai_model/questionnaire/`) | Implemented; trained on 52 real responses (small sample, no clinical validation) |
| Behavioral model + prediction (`src/ai_model/behavioral/`) | Implemented; **trained on synthetic data only** - accuracy figures are not real-world performance |
| Voice pipeline (`src/ai_model/voice/`) | Preprocessing, features, model code, reproducible training/evaluation CLI (`train_voice.py`) and safe inference implemented and tested on toy data; **no stress-risk voice model has been trained** (no stress-labeled audio exists). A separate, experimental *acted-emotion* model trained on CREMA-D is present (held-out accuracy 0.48 vs 0.17 chance; shown separately, never fused, not a stress measure). See [`docs/voice_model_training.md`](docs/voice_model_training.md) for the dataset required |
| Multimodal fusion (`src/ai_model/fusion/`) | Implemented and unit tested; contract documented in its README. Agreement is reported as `not_applicable` when only one source is used |
| Assessment API (`src/backend/assessment/`) | Implemented and integration tested locally |
| Web UI (`src/frontend/`, served at `/app/`) | Implemented: consent, questionnaire, habits, voice record/upload, per-source results, readable factor names, suggestions tailored to the student's own inputs, opt-in history and trend, evaluation panel |
| Optional free-tier Azure services (`src/cloud/`) | Speech-content voice signal (Azure AI Speech + Language F0), opt-in history via Event Grid -> Cosmos DB free tier, App Insights, optional Entra ID login and GitHub Actions CI/CD. Off unless configured; see [`docs/azure_free_services.md`](docs/azure_free_services.md) |
| Azure (`src/azure/`) | Azure ML / Blob templates and wrappers, tested **offline only** and not used (they would cost money). The app itself is deployed on App Service F1 with `scripts/azure_free_deploy.py` |

## Zero-cost, fully local

The whole project runs on your own computer for ₹0: no Azure subscription, no cloud resource, no internet needed after
`pip install`. The running app never imports Azure code and makes no network calls (enforced by
`tests/test_local_only.py`). The Azure templates/docs are preserved but not used. See
[`docs/zero_cost_local_mode.md`](docs/zero_cost_local_mode.md) for the Azure inventory, free local replacements, and
what genuinely cannot be provided without a paid service (public hosting, autoscaling).

**Free-tier hosting (deployed on App Service F1):** [`docs/azure_free_tier_deployment.md`](docs/azure_free_tier_deployment.md)
describes publishing the app on Azure App Service's free F1 plan (no credits used) with `scripts/package_for_azure.py` and
`scripts/azure_free_deploy.py` (dry-run by default, F1-only, access-password protected). The free add-on services (Speech, Language, Event Grid, Cosmos DB, App Insights, Entra ID, CI/CD) are set up with `scripts/azure_free_services.py`: see [`docs/azure_free_services.md`](docs/azure_free_services.md).

## Setup

Requires **Python 3.11+** (3.13 verified). The committed model artifacts were serialized with
scikit-learn 1.8.0, which does not support Python 3.10.

```bash
python -m venv venv
# Windows (Git Bash): source venv/Scripts/activate      PowerShell: .\venv\Scripts\Activate.ps1
# macOS/Linux:        source venv/bin/activate
pip install -r requirements.txt            # runtime only (free, no cloud SDK)
pip install -r requirements-dev.txt        # + test tools (optional)
```

No credentials are needed for local use. Azure-related code reads settings from environment variables only
(see "Azure" below); never commit secrets (`.env` is git-ignored).

## Run the app (one command)

```bash
python run_app.py            # checks the environment, serves http://127.0.0.1:8000 and opens your browser
python run_app.py --port 9000 --no-browser
```

The app is local-only (binds to 127.0.0.1). In the page you can fill in the questionnaire and daily habits, record or
upload a voice clip, and see the overall result, a card per source, contributing factors, general self-care suggestions
and a summary of how the models were evaluated. The interactive `/docs` page is off by default (it loads assets from a public CDN); enable it with `ENABLE_API_DOCS=1`. Equivalent manual start:
`python -m uvicorn src.backend.assessment.api:app --port 8000`.

**Privacy:** the service stores nothing (no database, no files kept; voice is analysed from a temporary file that is
deleted immediately), responses are sent with `Cache-Control: no-store`, and the UI is served with a strict
Content-Security-Policy. Consent and the non-clinical purpose are shown before anything is submitted.

| Route | Purpose |
|---|---|
| `POST /assessment` | Run any subset of questionnaire / behavioral / voice, fuse the usable results |
| `GET /health` | Liveness check |
| `GET /status` | Which model artifacts exist and whether the installed scikit-learn matches the artifacts |
| `GET /schema` | Questionnaire questions/options and behavioral field ranges (drives the UI) |
| `GET /evaluation` | File-derived summary of how each model was trained/evaluated, with caveats (see `docs/model_evaluation.md`) |

`POST /assessment` body (every modality optional; `assessment_payload.json` in the repo root is a complete example):

```bash
curl -s -X POST http://127.0.0.1:8000/assessment -H "Content-Type: application/json" -d @assessment_payload.json
```

```json
{
  "questionnaire": {"responses": {"Q1": "4th Year", "...": "...", "Q20": "Sometimes"}},
  "behavioral": {"responses": {"B1_sleep_hours": 5, "B2_study_consistency": 2, "B3_social_activity_freq": 1.5,
                               "B4_screen_time_hours": 9, "B5_routine_irregularity": 4}},
  "voice": {"audio_base64": "<base64 of a 5s+ mono/stereo WAV>"}
}
```

`voice` accepts either `audio_base64` (WAV, max 10 MB) or `features` (the 41 extracted acoustic features), not both.

Response: `assessment` (fused `risk_score`, `risk_level`, `confidence`, `agreement`, `key_factors`, `modalities_used`,
`normalized_weights`, `status`), `modality_details` (per-source score, level, confidence, completeness), `modality_status` (per modality: `available`, `not_provided`, `model_not_trained`,
`model_unavailable`, `invalid_input`, `prediction_failed`) and a non-clinical `disclaimer`.

Status codes: `200` for any valid request (including when no modality was usable: `assessment.status =
"fusion_unavailable"`, risk fields `null`); `422` for malformed bodies; `500` only if fusion itself fails
(generic message, details are logged server-side).

**Unavailable-modality policy:** a modality that is missing, untrained, invalid or failed is excluded from fusion
(never zero-filled), its reason is reported, and the remaining modalities' weights are re-normalized.
Voice is currently always `model_not_trained` unless a model is added to `models/voice/`.

## Model evaluation

See [`docs/model_evaluation.md`](docs/model_evaluation.md) for what each model was evaluated on and how far to trust it (questionnaire: 52 real responses, cross-validated; behavioral: synthetic data; voice: no model yet).

## Tests

```bash
python -m pytest tests src/backend/questionnaire/test_api.py
```

Everything runs offline; Azure clients are mocked. Tests that use the committed questionnaire/behavioral models are
skipped (with a reason) if the installed scikit-learn is not 1.8.0. Voice-contract tests use a throw-away model
trained on random synthetic features - they check wiring, not predictive quality.

## Azure

* **Optional and not used by the running app** (`pip install -r requirements-azure.txt` only if you ever deploy). `src/azure/` (voice): Blob Storage helper, secure upload handler and Azure ML training-job builder. Configuration
  comes from environment variables: `AZURE_STORAGE_CONNECTION_STRING` (or `AZURE_STORAGE_ACCOUNT_NAME` /
  `AZURE_STORAGE_ACCOUNT_KEY`), `AZURE_STORAGE_CONTAINER_NAME`, `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`,
  `AZURE_WORKSPACE_NAME`, `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_ML_COMPUTE_NAME`,
  `AZURE_ML_ENVIRONMENT_NAME`.
* `src/azure/questionnaire/`: Azure ML v2 YAML templates, `train.py` and `score.py` for the questionnaire model.
* Exact deployment steps and prerequisites: [`docs/azure_deployment_runbook.md`](docs/azure_deployment_runbook.md).
* **Status:** validated offline (templates parse, contain placeholders only, `score.py` works against local
  artifacts, Azure SDK calls are mocked). **No real Azure resource has been created, and no Azure deployment has been
  verified.**

## Known limitations

* The voice **stress-risk** modality has no trained model (no stress-labeled dataset). A separate CREMA-D acted-emotion model exists (`python -m src.ai_model.voice.train_emotion`); it is not a stress measure and is never fused; without Azure, voice input is reported as `model_not_trained`. When the free Azure AI Speech + Language services are configured and the student opts in, the *tone of what they said* is used as the voice input instead (a conservative, capped-confidence speech-content indicator, not an acoustic or clinical measure). Once labeled data exists, train with `python -m src.ai_model.voice.train_voice --dataset-kind real_student` (see `docs/voice_model_training.md`). Models trained on `synthetic_demo` data are refused at inference.
* The behavioral model is trained on synthetic data; its near-perfect test accuracy reflects clean synthetic classes.
* The questionnaire model was trained on 52 responses; generalisation is limited and nothing is clinically validated.
* Behavioral input requires all five fields (values may be `null`); partial dicts are rejected as `invalid_input`.
* Voice features must be complete (41 values); missing values are rejected rather than imputed.
* The UI records audio in the browser (WAV upload also works) but has no authentication and no data storage; deploy behind HTTPS and add
  authentication before any real use with students.
* Azure deployment (training job, endpoint, App Service hosting, CI/CD) is documented but unverified.
* This is a non-clinical research/demonstration system, not a diagnostic tool.
