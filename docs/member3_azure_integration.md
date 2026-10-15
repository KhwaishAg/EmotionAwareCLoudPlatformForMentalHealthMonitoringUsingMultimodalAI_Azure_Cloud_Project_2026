# Member 3 — Azure integration notes (behavioral, fusion, unified backend)

> **No Azure cloud resources have been created for this document.** This
> extends the existing project architecture (`src/azure/README.md`) and the
> questionnaire modality's Azure ML configuration (`src/azure/questionnaire/`)
> to cover the components Member 3 owns: behavioral ML, multimodal fusion,
> and the unified `/assessment` backend. It documents the intended
> architecture; it does not deploy anything.

---

## What already exists (Member 1, Netal) and is not duplicated here

Member 1 built a complete, unexecuted Azure ML v2 configuration for the
questionnaire modality: workspace/compute/environment templates, a training
job spec, model registration, a managed online endpoint + deployment spec,
and a thin `score.py` scoring adapter that delegates to
`predict_questionnaire()` without reimplementing any logic. See
`src/azure/questionnaire/README.md` for the full implementation-status table.

That pattern — **Azure ML hosts the trained model; a thin `score.py` adapter
delegates to the same `predict_*` function used locally** — is the pattern
this document follows for the remaining modalities, rather than inventing a
second, inconsistent architecture.

The project-level architecture doc (`src/azure/README.md`) already
establishes the service roles used below (Blob Storage, Azure ML, App
Service, GitHub Actions CI/CD) and explicitly rules out Azure Speech
Services for voice — acoustic feature extraction is local Python
(librosa/soundfile/scipy), which matches Nandini's actual `features.py`
implementation.

---

## How Member 3's components map onto the existing architecture

```
                                  DEVELOPER ENVIRONMENT
                          (local feature extraction, training, testing —
                           all of Member 3's work verified here today)
                                          |
                                          v  (CI/CD, not yet configured)
+----------------------------------------------------------------------------------+
|                                   MICROSOFT AZURE                                |
|                                                                                   |
|  Azure Blob Storage          Azure Machine Learning          Azure App Service   |
|  - model-artifacts/          - behavioral model registry     - Containerized     |
|    behavioral/*.joblib         (mirrors questionnaire's)       FastAPI backend   |
|  - model-artifacts/          - Managed online endpoints:       (unified          |
|    questionnaire/*.joblib      questionnaire, behavioral       /assessment       |
|                                 (voice: pending training)       endpoint)        |
+----------------------------------------------------------------------------------+
                                          |
                                          v
                                   [Web Frontend UI]
                              (not yet built — see plan)
```

Two deployment shapes are both consistent with what already exists; the
choice is a team decision to make once questionnaire's Azure execution is
actually performed, not something to guess at now:

**Shape A — Azure ML hosts every trained model, App Service hosts only the
fusion/orchestration layer.**
The unified `/assessment` FastAPI app (`src/backend/assessment/api.py`)
would call Azure ML managed online endpoints for questionnaire and
behavioral (each via its own `score.py`-style adapter, following Netal's
exact pattern) instead of importing `predict_questionnaire` /
`predict_behavioral` directly in-process. Fusion (`fuse_modalities`) still
runs inside the App Service container, since it is a lightweight pure
function with no model artifacts of its own.

**Shape B — App Service hosts everything (models + fusion) in one
container**, with Azure ML used only for training/experiment
tracking/model registry, not live inference. This is closer to what
`src/azure/README.md`'s original diagram implies (`Azure Machine Learning →
Azure App Service`, one arrow) and is the simpler shape to actually deploy
in the time available for this project.

Given the project's stated priority ("don't add Azure services just to say
Azure was used"), **Shape B is the more credible one to describe as the
intended default** for this submission: it is what the already-built
`src/backend/assessment/api.py` does today (in-process model calls, no
network hop to a separate endpoint), and moving to Shape A is a
documented future step rather than a today's-demo requirement.

---

## Behavioral modality — Azure ML configuration (not yet created)

Following the exact pattern in `src/azure/questionnaire/`, the equivalent
behavioral files, if produced, would be:

| File (not created) | Purpose |
|---|---|
| `src/azure/behavioral/job.yml.template` | Azure ML command job invoking `train_behavioral.py` |
| `src/azure/behavioral/model.yml.template` | Registers the trained `logistic_regression.joblib` as `behavioral-logistic-regression` |
| `src/azure/behavioral/endpoint.yml.template` | Managed online endpoint for behavioral inference |
| `src/azure/behavioral/deployment.yml.template` | Deployment referencing `score.py` |
| `src/azure/behavioral/score.py` | Thin adapter delegating to `predict_behavioral()`, identical structure to the questionnaire adapter |

These are deliberately **not created as empty template files** in this pass,
because Member 1's own README explicitly warns against unexecuted Azure
scaffolding being mistaken for actual deployment. The table above is the
specification; creating the templates is a 15-minute mechanical task
(copy `src/azure/questionnaire/*.template`, rename, adjust model name)
if the team decides to do it before submission.

## Voice modality — no Azure configuration yet

Voice has no trained model (`models/voice/` contains only `.gitkeep`) and
no Azure configuration exists for it, correctly, because there is nothing
to deploy yet. Once a real labeled voice dataset is available and a model
is trained (see `src/ai_model/voice/predict.py`'s docstring), the same
`score.py`-adapter pattern applies unchanged.

## Storage

- `model-artifacts/behavioral/` (Blob Storage) — planned location for the
  synthetic-data-trained `.joblib` and metadata files, mirroring
  `model-artifacts/questionnaire/` in the existing architecture doc.
- No participant-level behavioral self-report data requires storage beyond
  what the assessment API processes in-request; the behavioral training
  data is synthetic and does not require the same privacy controls as the
  questionnaire dataset (which does — see `src/azure/questionnaire/README.md`'s
  data and privacy policy section).

---

## What this document does not do

It does not create, provision, or authenticate to any Azure resource. It
does not duplicate or contradict Member 1's questionnaire Azure work. It
does not propose Azure Speech Services or any service not already justified
by an actual system need, per the project's own stated constraint.
