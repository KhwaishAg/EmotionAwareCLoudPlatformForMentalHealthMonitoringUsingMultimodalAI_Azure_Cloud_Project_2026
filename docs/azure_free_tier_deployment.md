# Azure deployment on the free tier (optional)

> **Status: prepared, not deployed.** Nothing has been created in Azure. This is the credit-safe way to put the app
> online if you decide to. The project runs fully locally for free without any of this
> (`docs/zero_cost_local_mode.md`).

## The plan: one free web app, nothing else

Everything runs in **one Azure App Service on the F1 (Free) plan**, with the models loaded inside the app
(measured peak memory of the app: about 310 MB; free tier allows about 1 GB). It uses **no credits** because F1
is not billed. The deploy script creates exactly three things: a resource group, an F1 Linux plan, and a web app.

| Free tier limit (F1) | Effect on this project |
|---|---|
| ~60 CPU-minutes per day | Roughly 1 s of CPU per assessment, so plenty for demos; the site pauses until the next day if exceeded |
| ~1 GB RAM, ~1 GB disk | App needs ~0.3 GB RAM; packages ~0.5 GB disk (tight but fits; the bundle itself is ~6 MB) |
| No "Always On" | After idle the app sleeps; the first request takes ~10-60 s while it wakes and warms up |
| Shared compute, no SLA | Fine for a demo, not for production |
| HTTPS on `<app-name>.azurewebsites.net` | Included; custom domains/certificates are not |

## What NOT to create with 50 credits (these bill continuously, even when idle)

* **Azure ML managed online endpoints** (a VM runs per hour, 24/7): the template in `src/azure/questionnaire/` is
  intentionally **not** used.
* **Azure ML workspace + compute + environment builds:** a workspace creates storage, Key Vault and monitoring
  resources, and building a custom environment creates a Container Registry that bills daily. Training is not needed:
  the models are already trained and committed, and they train in minutes on your own CPU.
* Container Registry, AKS, databases, Log Analytics/Application Insights workspaces, VMs, paid App Service plans (B1 and up).

The deploy script physically cannot create any of these (see `tests/test_azure_free_tier_guard.py`).

## Before you start (you do this; never paste secrets into chat)

1. Install the Azure CLI and sign in: `az login`, then `az account set --subscription "<name>"`.
2. Confirm your subscription type and the **spending limit**: Azure free accounts/student subscriptions stop services
   rather than charge when credits run out, but verify in the portal under *Cost Management + Billing*. Create a
   **budget alert** at a small amount as a safety net.
3. Check the runtime exists: `az webapp list-runtimes --os linux` should list `PYTHON:3.12`.
4. Choose an access password (12+ characters) and set it only in your terminal: PowerShell
   `$env:DEPLOY_APP_PASSWORD = "<your password>"`. The site will require username `student` and that password,
   because a public address with no login would expose a form that handles personal answers. It is stored as an App
   Service setting, never in the repository.

## Deploy

```bash
python scripts/package_for_azure.py        # builds dist/azure_app.zip locally (nothing uploaded)
python scripts/azure_free_deploy.py --subscription "<name>" --location <region> --app-name <unique-name>
# review the printed commands (DRY RUN), then:
python scripts/azure_free_deploy.py --subscription "<name>" --location <region> --app-name <unique-name> --execute --confirm-free-tier
```

The script refuses to run if: the signed-in subscription is not the one named, the resource group already exists, the
password is missing/short, the bundle is missing, or after creating the plan it reads back anything other than `F1`.
It also forces HTTPS-only and turns "Always On" off (not available on F1).

Open `https://<app-name>.azurewebsites.net/` (first load can take a minute). If the first deployment fails because
the free disk fills during the package install, the fallback is to deploy locally-built Linux wheels instead of letting the
server install them; tell me and I will add that.

## Tear down (stops everything, nothing keeps running)

```bash
python scripts/azure_free_deploy.py --subscription "<name>" --location <region> --app-name <unique-name> --teardown --execute
```

## Privacy when public

The app stores nothing (answers and voice are processed in memory/temporary files and discarded) and sets no-cache
headers, but once it is on the internet it is a public service: keep the access password, share the URL only with
participants who have consented, and delete the site when the demonstration is over.

## What is verified and what is not

Verified locally: the bundle builds (62 files, ~6 MB, no data/audio/secrets/Azure code), the extracted bundle runs
the full assessment including the emotion model, the access password works, and every guard-rail is unit tested with a
fake `az`. **Not verified:** any real Azure behavior (runtime availability, the server-side package install on the 1 GB
free disk, cold-start time). Those only become known on the first real deployment.
