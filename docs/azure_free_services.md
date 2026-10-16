# Free-tier Azure services (optional add-on to the F1 web app)

`scripts/azure_free_services.py` adds the cloud services below to the web app created by
`scripts/azure_free_deploy.py`. Every service uses a free tier with a hard cap, all of them live in the app's
resource group, and the existing teardown (`azure_free_deploy.py --teardown --execute`) removes everything.
With none of these configured, the app runs exactly as before, fully local (enforced by `tests/test_local_only.py`).

## What each service does in the app

| Service | Free tier / cap | Role in the project |
|---|---|---|
| **App Service** | F1 (free) | Hosts the FastAPI backend and the web UI (already deployed by `azure_free_deploy.py`). |
| **Azure AI Speech** | F0: 5 audio hours/month; over the limit requests fail, never bill | Transcribes the student's recording (short-audio REST API, up to 60 s). |
| **Azure AI Language** | F0: 5,000 text records/month | Sentiment and key phrases of the transcript, which become the **voice** input to fusion while no acoustic stress model is trained. |
| **Event Grid** | First 100,000 operations/month free | Each opted-in assessment publishes `EmotionAware.Assessment.Completed`; Event Grid delivers it to the app's `/events` webhook (event-driven pipeline, no Functions needed). |
| **Cosmos DB** | Free tier (1000 RU/s + 25 GB) with the account **limited to 1000 RU/s** | Stores the anonymised history summaries delivered by Event Grid; `/history` shows the student their trend; items expire after 180 days. |
| **Application Insights + Log Analytics** | 5 GB/month free; workspace **capped at 0.1 GB/day** | Codeless monitoring: requests, failures, latency, logs. |
| **Microsoft Entra ID** (optional) | Free | App Service built-in login with Microsoft accounts in your directory in place of the shared password; history is then tied to the signed-in user. |
| **GitHub Actions** (optional) | Free | `.github/workflows/ci-cd.yml` runs all tests on every push/PR and deploys `main` to App Service. |

Not used, because they would cost money: Azure Functions (needs a paid storage account), Blob Storage,
Azure ML workspaces/compute/endpoints, Container Registry, paid App Service plans. IoT Hub and Notification Hubs
are out of scope (no wearables; the project is non-clinical).

## Data flow

```
Browser ──POST /assessment──► App Service (FastAPI)
   │                              ├─ questionnaire + habits models (in-process)
   │  voice (opt-in) ────────────►├─ Azure AI Speech  → transcript
   │                              ├─ Azure AI Language → sentiment, key phrases → voice contract
   │                              ├─ fusion → result (returned to the browser)
   │  "save to history" (opt-in) ─┴─ Event Grid topic ──► POST /events?key=… ──► Cosmos DB
   └──GET/DELETE /history──────────────────────────────────────────────────────► Cosmos DB
   App Insights watches all of it.
```

## Privacy

* Speech analysis is **opt-in per check-in**. The audio goes to Azure AI Speech and the transcript to Azure AI
  Language for that request only; the app stores neither. The transcript is shown back to the student.
* History is **opt-in per check-in** and stores only: date, overall and per-source scores/levels, confidence,
  agreement level and the names of the top three factors. It never stores answers, habit values, audio or
  transcripts. The owner is a random ID generated in the browser (or, with Entra login, a hash of the user ID).
  Students can delete everything with "Delete my history"; items expire after 180 days.
* The `/events` webhook only accepts requests with the secret Event Grid was given, and only stores records with
  exactly the fields above.

## Run it

Prerequisites: the app is deployed (`azure_free_deploy.py`), you are signed in (`az login`), and
`az account show` names your subscription.

```powershell
python scripts/azure_free_services.py --subscription "<subscription name>" --app-name <app-name>            # dry run
python scripts/azure_free_services.py --subscription "<subscription name>" --app-name <app-name> --execute --confirm-free-tier
```

It creates the resources (skipping any that already exist), **reads back every SKU and cap and stops if anything
is not the free setting**, writes the keys into App Service settings through a temporary file (never printed),
rebuilds and deploys the app, and connects Event Grid to `/events`. Takes about 10-15 minutes (Cosmos DB is the
slow one).

Options:

* `--with-entra`: switch on Microsoft Entra ID login. Only accounts in your directory can sign in, so invite
  teammates and evaluators as guests (portal: Microsoft Entra ID > Users > Invite external user). Creating the app
  registration may ask you to `az login` again.
* `--with-cicd`: writes `dist/publish_profile.xml` (git-ignored). Add it as the repository secret
  `AZURE_WEBAPP_PUBLISH_PROFILE` and add the repository variable `AZURE_WEBAPP_NAME`, then delete the file.
  Pushes to `main` then deploy automatically after the tests pass.
* `--redeploy --execute`: only rebuild and deploy the code (use this after code changes; `azure_free_deploy.py`
  refuses to touch an existing resource group).
* `--cosmos-location southindia`: if the Cosmos DB free tier is not available in your main region.

Limits to know: a subscription can have only **one** Cosmos DB free-tier account and one F0 resource per
Cognitive Services kind. If one already exists, creation fails and the script stops (it never falls back to a
paid tier).

## Verified and not verified

Verified offline: request formats, the fusion contract for speech content, the webhook handshake/secret/record
validation, history read/delete, the cost guard (stops before writing settings or deploying), dry-run safety,
and that the app makes no network calls when nothing is configured (`tests/test_cloud_integration.py`,
`tests/test_azure_free_services_guard.py`, `tests/test_local_only.py`). **Not verified until the first real run:**
the live Azure responses, codeless App Insights on the F1 Python runtime, and Entra sign-in for your directory.
