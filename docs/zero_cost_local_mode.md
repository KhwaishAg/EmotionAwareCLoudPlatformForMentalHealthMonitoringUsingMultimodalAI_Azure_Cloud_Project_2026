# Zero-cost local mode

The project runs completely on your own computer and costs ₹0. **No cloud resource has been created, no Azure
subscription is needed, and the running application contains no code path that calls Azure or any other online
service.** The Azure deployment templates and documentation are kept for reference only. An optional, credit-safe way to publish
the app on Azure's **free F1 web-app tier** is prepared (not deployed): see `azure_free_tier_deployment.md`.

## Run and demonstrate it (no account, no internet needed after installation)

```bash
python -m venv venv && source venv/Scripts/activate      # Python 3.11+ (3.13 verified)
pip install -r requirements.txt                          # free PyPI packages; no cloud SDK
python run_app.py                                        # http://127.0.0.1:8000
```

Optional: `pip install -r requirements-dev.txt` then `python -m pytest tests src/backend/questionnaire/test_api.py`.
Installing packages from PyPI is free. After installation the app works with the network disconnected.

## Where the project touches Azure (inventory)

| Item | What it is | Used when running the app? |
|---|---|---|
| `src/azure/storage.py`, `upload_handler.py`, `ml_training.py`, `config.py` | Library code (Blob storage, secure upload, Azure ML job builder), unit-tested with mocks | **No.** Nothing outside `src/azure` imports it; nothing runs it |
| `src/azure/questionnaire/*.yml.template`, `score.py`, `train.py` | Azure ML deployment templates and adapters (placeholders only, saved as `.template` so `az` cannot apply them) | **No.** Preserved for a future deployment |
| `requirements-azure.txt` (`azure-storage-blob`, `azure-ai-ml`, `azure-identity`) | Optional SDKs | **No.** Not in `requirements.txt`; their tests skip automatically when the SDKs are absent |
| `docs/azure_deployment_runbook.md`, `docs/member3_azure_integration.md`, `src/azure/README.md` | Documentation of the intended deployment | No |
| `AZURE_*` environment variables | Read only by `src/azure/*` | No: the runtime never reads them |
| FastAPI `/docs` (Swagger) page | Loads JavaScript from a public CDN | Disabled by default; set `ENABLE_API_DOCS=1` only if you want it (free, but fetches from the internet) |

Enforced by `tests/test_local_only.py`: it runs the whole app with every `azure` import blocked and all
non-loopback network access blocked (UI, schema, evaluation, status and a full assessment still succeed), scans the
runtime code for Azure imports, network libraries and `AZURE_*` settings, checks the frontend loads nothing from the
internet, checks the server binds to `127.0.0.1` only, and checks no real Azure identifiers are committed.

## Free local replacement for each Azure service in the design

| Azure service in the architecture | Local replacement (already implemented) |
|---|---|
| Blob Storage (audio, features, artifacts) | Local disk: models in `models/`, data in git-ignored `data/`; a voice upload is analysed from a temporary file that is deleted immediately and never stored |
| Azure ML training jobs and compute | Run on your CPU: `python -m src.ai_model.voice.train_voice`, `train_emotion`, and the questionnaire/behavioral training modules |
| Azure ML model registry / experiment tracking | Versioned artifacts plus JSON reports in `models/*/` and `results/` (e.g. `voice_training_report.json`) |
| Managed online endpoint (`score.py`) | The local FastAPI service (`POST /assessment`); `score.py` is verified to work against the local artifacts |
| App Service hosting, TLS | `python run_app.py` on `127.0.0.1` |
| Entra ID, Key Vault | Not needed: no accounts, no secrets, no keys locally |
| Azure Monitor | Console logs from uvicorn; user answers are never logged (tested) |
| CI/CD | Run `python -m pytest` locally |
| Azure Speech | Never used: acoustic features are extracted locally with librosa |

## Features that cannot be provided without a paid or online service

None of these is needed for the demonstration; they are listed for completeness.

* **A public URL / HTTPS access for other people.** Locally the app is reachable only from your own computer. (Free
  tunnelling tools exist but would expose personal data to third parties; not recommended and not set up.)
* **Always-on hosting and autoscaling** (App Service / managed endpoints): requires a hosted compute service.
* **Cloud-managed GPU or distributed training and a shared cloud model registry:** the models here train in
  seconds-to-minutes on a CPU, so this is not needed; sharing between team members can be done through Git or a shared folder.
* **A genuine stress-risk voice model:** unrelated to cloud: it needs a stress-labeled audio dataset, which no free
  or paid cloud service can supply.

## What was and was not done

No Azure CLI was installed, no `az login` was performed, no resource, compute, endpoint or storage was created, and no
Azure API was called. Only free local files were changed.
