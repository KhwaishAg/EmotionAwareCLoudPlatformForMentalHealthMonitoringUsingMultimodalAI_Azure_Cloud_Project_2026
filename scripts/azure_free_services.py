"""Add the optional FREE-tier Azure services to an app deployed with scripts/azure_free_deploy.py.

DRY-RUN BY DEFAULT: prints every `az` command and runs nothing. Pass ``--execute --confirm-free-tier``
to create resources. Everything is created inside the app's existing resource group (``rg-<app-name>``),
so ``azure_free_deploy.py --teardown --execute`` still removes all of it in one step.

What it can create (and nothing else) - every one is a free tier with a hard cap:

    Azure AI Speech         F0   (5 audio hours/month; over the limit requests fail, never bill)
    Azure AI Language       F0   (5,000 text records/month; same)
    Azure Cosmos DB         free-tier account, total throughput LIMITED to 1000 RU/s (the free amount),
                                 one 400 RU/s shared database, items expire after 180 days
    Event Grid              custom topic (first 100,000 operations/month free) + webhook to the app's /events
    Log Analytics + App Insights   daily ingestion cap 0.1 GB (free allowance is 5 GB/month)
    Microsoft Entra ID      (optional, --with-entra) app registration + App Service built-in login: free
    GitHub Actions          (optional, --with-cicd) writes the publish profile for a GitHub secret: free

After creating, the script READS BACK the SKUs/limits and stops loudly if any is not the free setting.

Prerequisites: the app already exists (azure_free_deploy.py), you are signed in (`az login`) and
`az account show` names the subscription you pass with --subscription. Keys are read with `az` and written
straight into App Service settings through a temporary file; they are never printed.

    python scripts/azure_free_services.py --subscription "<name>" --app-name <app> --location centralindia
    python scripts/azure_free_services.py ... --execute --confirm-free-tier
    python scripts/azure_free_services.py ... --execute --confirm-free-tier --with-entra --with-cicd
    python scripts/azure_free_services.py ... --redeploy --execute      # just rebuild + deploy the code
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$")
LOCATION_RE = re.compile(r"^[a-z0-9]{3,30}$")

FREE_COGNITIVE_SKU = "F0"
COSMOS_THROUGHPUT_LIMIT = 1000   # free-tier allowance; the account refuses anything above it
COSMOS_DATABASE_THROUGHPUT = 400
COSMOS_TTL_SECONDS = 180 * 24 * 3600
LOG_DAILY_CAP_GB = 0.1
PROVIDERS = ("Microsoft.CognitiveServices", "Microsoft.DocumentDB", "Microsoft.EventGrid",
             "Microsoft.OperationalInsights", "Microsoft.Insights")
EVENT_SUBSCRIPTION = "history-writer"
#: Every `az` command group this script may call. Tests assert nothing outside this list appears.
ALLOWED_AZ_GROUPS = {
    ("provider",), ("config",), ("account",), ("group",), ("cognitiveservices",), ("cosmosdb",), ("resource",),
    ("eventgrid",), ("monitor",), ("webapp",), ("ad",),
}


def _resolve(command: List[str]) -> List[str]:
    if command and command[0] == "az":
        return [shutil.which("az") or "az", *command[1:]]
    return command


def default_run(command: List[str], **kwargs):
    return subprocess.run(_resolve(command), **kwargs)


@dataclass(frozen=True)
class Names:
    subscription: str
    app: str
    location: str
    cosmos_location: str

    @property
    def group(self) -> str:
        return f"rg-{self.app}"

    @property
    def speech(self) -> str:
        return f"{self.app}-speech"

    @property
    def language(self) -> str:
        return f"{self.app}-language"

    @property
    def cosmos(self) -> str:
        return f"{self.app}-cosmos"[:44]

    @property
    def topic(self) -> str:
        return f"{self.app}-events"

    @property
    def logs(self) -> str:
        return f"{self.app}-logs"

    @property
    def insights(self) -> str:
        return f"{self.app}-insights"

    @property
    def site(self) -> str:
        return f"https://{self.app}.azurewebsites.net"


def validate(names: Names) -> None:
    if not NAME_RE.match(names.app):
        raise ValueError("--app-name must be the existing app's name (lowercase letters, digits, hyphens).")
    for location in (names.location, names.cosmos_location):
        if not LOCATION_RE.match(location):
            raise ValueError("Locations must look like 'centralindia'.")
    if not names.subscription.strip():
        raise ValueError("--subscription is required.")


def _sub(names: Names) -> List[str]:
    return ["--subscription", names.subscription]


def create_commands(names: Names) -> List[List[str]]:
    """Resource creation, in order. Pure function (used for the dry run and the tests)."""
    validate(names)
    g, s = ["--resource-group", names.group], _sub(names)
    return [
        ["az", "config", "set", "extension.use_dynamic_install=yes_without_prompt"],
        *[["az", "provider", "register", "--namespace", ns, "--wait", *s] for ns in PROVIDERS],
        ["az", "cognitiveservices", "account", "create", "--name", names.speech, *g, "--location", names.location,
         "--kind", "SpeechServices", "--sku", FREE_COGNITIVE_SKU, "--yes", *s],
        ["az", "cognitiveservices", "account", "create", "--name", names.language, *g, "--location", names.location,
         "--kind", "TextAnalytics", "--sku", FREE_COGNITIVE_SKU, "--custom-domain", names.language, "--yes", *s],
        ["az", "cosmosdb", "create", "--name", names.cosmos, *g, "--enable-free-tier", "true",
         "--default-consistency-level", "Session",  # default periodic backup (2 copies) is included free
         "--locations", f"regionName={names.cosmos_location}", "failoverPriority=0", "isZoneRedundant=False", *s],
        ["az", "resource", "update", *g, "--name", names.cosmos, "--resource-type", "Microsoft.DocumentDB/databaseAccounts",
         "--set", "properties.capacity=" + json.dumps({"totalThroughputLimit": COSMOS_THROUGHPUT_LIMIT}), *s],
        ["az", "cosmosdb", "sql", "database", "create", "--account-name", names.cosmos, *g, "--name", "emotionaware",
         "--throughput", str(COSMOS_DATABASE_THROUGHPUT), *s],
        ["az", "cosmosdb", "sql", "container", "create", "--account-name", names.cosmos, *g, "--database-name",
         "emotionaware", "--name", "assessments", "--partition-key-path", "/owner", "--ttl", str(COSMOS_TTL_SECONDS), *s],
        ["az", "eventgrid", "topic", "create", "--name", names.topic, *g, "--location", names.location, *s],
        ["az", "monitor", "log-analytics", "workspace", "create", "--workspace-name", names.logs, *g,
         "--location", names.location, "--quota", str(LOG_DAILY_CAP_GB), "--retention-time", "30", *s],
        ["az", "monitor", "app-insights", "component", "create", "--app", names.insights, *g, "--location",
         names.location, "--workspace", names.logs, "--application-type", "web", *s],
    ]


def exists_command(command: List[str]) -> Optional[List[str]]:
    """The matching `show` command used to skip creation on re-runs (idempotent), or None."""
    joined = " ".join(command[:5])
    def value(flag: str) -> str:
        return command[command.index(flag) + 1]
    sub = command[command.index("--subscription"):command.index("--subscription") + 2] if "--subscription" in command else []
    g = ["--resource-group", value("--resource-group")] if "--resource-group" in command else []
    if joined.startswith("az cognitiveservices account create"):
        return ["az", "cognitiveservices", "account", "show", "--name", value("--name"), *g, *sub]
    if joined.startswith("az cosmosdb create"):
        return ["az", "cosmosdb", "show", "--name", value("--name"), *g, *sub]
    if joined.startswith("az cosmosdb sql database create"):
        return ["az", "cosmosdb", "sql", "database", "show", "--account-name", value("--account-name"), *g,
                "--name", value("--name"), *sub]
    if joined.startswith("az cosmosdb sql container create"):
        return ["az", "cosmosdb", "sql", "container", "show", "--account-name", value("--account-name"), *g,
                "--database-name", value("--database-name"), "--name", value("--name"), *sub]
    if joined.startswith("az eventgrid topic create"):
        return ["az", "eventgrid", "topic", "show", "--name", value("--name"), *g, *sub]
    if joined.startswith("az monitor log-analytics workspace"):
        return ["az", "monitor", "log-analytics", "workspace", "show", "--workspace-name", value("--workspace-name"), *g, *sub]
    if joined.startswith("az monitor app-insights component"):
        return ["az", "monitor", "app-insights", "component", "show", "--app", value("--app"), *g, *sub]
    return None


def deploy_commands(names: Names, bundle: Path) -> List[List[str]]:
    validate(names)
    return [
        [sys.executable, str(ROOT / "scripts" / "package_for_azure.py")],
        ["az", "webapp", "deploy", "--name", names.app, "--resource-group", names.group, "--src-path", str(bundle),
         "--type", "zip", *_sub(names)],
    ]


def subscription_command(names: Names, topic_id: str, secret: str) -> List[str]:
    return ["az", "eventgrid", "event-subscription", "create", "--name", EVENT_SUBSCRIPTION, "--source-resource-id",
            topic_id, "--endpoint", f"{names.site}/events?key={secret}", "--endpoint-type", "webhook",
            "--event-delivery-schema", "eventgridschema", "--max-delivery-attempts", "10", *_sub(names)]


def entra_commands(names: Names, client_id: str, tenant_id: str) -> List[List[str]]:
    g, s = ["--resource-group", names.group, "--name", names.app], _sub(names)
    return [
        ["az", "webapp", "auth", "microsoft", "update", *g, "--client-id", client_id,
         "--issuer", f"https://login.microsoftonline.com/{tenant_id}/v2.0", "--yes", *s],
        ["az", "webapp", "auth", "update", *g, "--enabled", "true", "--unauthenticated-client-action",
         "RedirectToLoginPage", "--redirect-provider", "azureactivedirectory",
         "--excluded-paths", json.dumps(["/events", "/health"]), *s],
        ["az", "webapp", "config", "appsettings", "delete", *g, "--setting-names", "APP_ACCESS_USER", "APP_ACCESS_PASSWORD", *s],
    ]


# ----------------------------------------------------------------------------- execution

class Runner:
    def __init__(self, run: Callable = default_run) -> None:
        self.run = run

    def az(self, command: List[str], capture: bool = True) -> str:
        done = self.run(command, capture_output=capture, text=True, check=True)
        return (done.stdout or "").strip() if capture else ""

    def ok(self, command: List[str]) -> bool:
        return self.run(command, capture_output=True, text=True, check=False).returncode == 0


def preflight(names: Names, runner: Runner) -> None:
    if shutil.which("az") is None:
        raise RuntimeError("Azure CLI ('az') is not installed or not on PATH.")
    account = json.loads(runner.az(["az", "account", "show", "-o", "json"]))
    if account.get("name") != names.subscription:
        raise RuntimeError(f"Signed-in subscription is {account.get('name')!r}, not {names.subscription!r}.")
    if not runner.ok(["az", "webapp", "show", "--name", names.app, "--resource-group", names.group, *_sub(names)]):
        raise RuntimeError(f"Web app {names.app!r} not found in {names.group!r}. Deploy it with azure_free_deploy.py first.")


def verify_free(names: Names, runner: Runner) -> List[str]:
    """Read every cost-relevant setting back; return a list of problems (empty = all free)."""
    g, s = ["--resource-group", names.group], _sub(names)
    problems = []
    for account in (names.speech, names.language):
        sku = runner.az(["az", "cognitiveservices", "account", "show", "--name", account, *g, "--query", "sku.name", "-o", "tsv", *s])
        if sku != FREE_COGNITIVE_SKU:
            problems.append(f"{account} SKU is {sku!r}, expected F0")
    cosmos = json.loads(runner.az(["az", "resource", "show", *g, "--name", names.cosmos, "--resource-type",
                                   "Microsoft.DocumentDB/databaseAccounts", "--query", "properties", "-o", "json", *s]) or "{}")
    if cosmos.get("enableFreeTier") is not True:
        problems.append("Cosmos DB free tier is not enabled")
    if (cosmos.get("capacity") or {}).get("totalThroughputLimit") != COSMOS_THROUGHPUT_LIMIT:
        problems.append("Cosmos DB total throughput is not limited to 1000 RU/s")
    quota = runner.az(["az", "monitor", "log-analytics", "workspace", "show", "--workspace-name", names.logs, *g,
                       "--query", "workspaceCapping.dailyQuotaGb", "-o", "tsv", *s])
    try:
        if float(quota) > LOG_DAILY_CAP_GB + 1e-9:
            problems.append(f"Log Analytics daily cap is {quota} GB, expected {LOG_DAILY_CAP_GB}")
    except ValueError:
        problems.append("Log Analytics daily cap could not be read")
    return problems


def read_settings(names: Names, runner: Runner) -> Dict[str, str]:
    g, s = ["--resource-group", names.group], _sub(names)
    existing = json.loads(runner.az(["az", "webapp", "config", "appsettings", "list", "--name", names.app, *g, "-o", "json", *s]) or "[]")
    current = {item["name"]: item.get("value") for item in existing}
    tsv = ["-o", "tsv"]
    return {
        "CLOUD_SPEECH_KEY": runner.az(["az", "cognitiveservices", "account", "keys", "list", "--name", names.speech, *g, "--query", "key1", *tsv, *s]),
        "CLOUD_SPEECH_REGION": names.location,
        "CLOUD_LANGUAGE_KEY": runner.az(["az", "cognitiveservices", "account", "keys", "list", "--name", names.language, *g, "--query", "key1", *tsv, *s]),
        "CLOUD_LANGUAGE_ENDPOINT": runner.az(["az", "cognitiveservices", "account", "show", "--name", names.language, *g, "--query", "properties.endpoint", *tsv, *s]),
        "CLOUD_COSMOS_ENDPOINT": runner.az(["az", "cosmosdb", "show", "--name", names.cosmos, *g, "--query", "documentEndpoint", *tsv, *s]),
        "CLOUD_COSMOS_KEY": runner.az(["az", "cosmosdb", "keys", "list", "--name", names.cosmos, *g, "--query", "primaryMasterKey", *tsv, *s]),
        "CLOUD_EVENTGRID_ENDPOINT": runner.az(["az", "eventgrid", "topic", "show", "--name", names.topic, *g, "--query", "endpoint", *tsv, *s]),
        "CLOUD_EVENTGRID_KEY": runner.az(["az", "eventgrid", "topic", "key", "list", "--name", names.topic, *g, "--query", "key1", *tsv, *s]),
        # keep the webhook secret stable across re-runs so the Event Grid subscription keeps working
        "CLOUD_EVENTS_SECRET": current.get("CLOUD_EVENTS_SECRET") or secrets.token_urlsafe(32),
        "APPLICATIONINSIGHTS_CONNECTION_STRING": runner.az(["az", "monitor", "app-insights", "component", "show", "--app", names.insights, *g, "--query", "connectionString", *tsv, *s]),
        "ApplicationInsightsAgent_EXTENSION_VERSION": "~3",   # codeless App Insights for Python on App Service
    }


def apply_settings(names: Names, values: Dict[str, str], runner: Runner) -> None:
    handle, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as target:
            json.dump(values, target)
        runner.az(["az", "webapp", "config", "appsettings", "set", "--name", names.app, "--resource-group", names.group,
                   "--settings", f"@{path}", *_sub(names)])
    finally:
        os.remove(path)


def wait_for_site(names: Names, timeout: float = 420) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(names.site + "/health", timeout=30) as response:  # noqa: S310
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(10)
    raise RuntimeError("The site did not answer /health in time; run with --redeploy later, then re-run this script.")


def deploy(names: Names, bundle: Path, runner: Runner, attempts: int = 3, pause: float = 60) -> None:
    package, upload = deploy_commands(names, bundle)
    print("  " + " ".join(package[:2]) + " ...")
    runner.run(package, check=True)
    for attempt in range(1, attempts + 1):
        print(f"  az webapp deploy (attempt {attempt}/{attempts}) ...")
        try:
            runner.run(upload, check=True)
            break
        except subprocess.CalledProcessError:
            # Changing app settings restarts the site; Kudu often answers 502 for a minute afterwards.
            if attempt == attempts:
                raise
            print(f"  deployment endpoint not ready yet; waiting {int(pause)} s and retrying ...")
            time.sleep(pause)
    print("  waiting for the site to start (cold start can take a few minutes) ...")
    wait_for_site(names)


def setup_entra(names: Names, runner: Runner) -> None:
    tenant = runner.az(["az", "account", "show", "--query", "tenantId", "-o", "tsv"])
    client_id = runner.az(["az", "ad", "app", "create", "--display-name", f"{names.app}-login", "--sign-in-audience",
                           "AzureADMyOrg", "--web-redirect-uris", f"{names.site}/.auth/login/aad/callback",
                           "--enable-id-token-issuance", "true", "--query", "appId", "-o", "tsv"])
    for command in entra_commands(names, client_id, tenant):
        runner.az(command)
    apply_settings(names, {"CLOUD_TRUST_EASY_AUTH": "1"}, runner)
    print("  Entra login is on. Only accounts in your directory can sign in: invite teammates/evaluators as guests in")
    print("  the portal (Microsoft Entra ID > Users > Invite external user). To undo: az webapp auth update --enabled false ...")


def setup_cicd(names: Names, runner: Runner) -> Path:
    g, s = ["--resource-group", names.group], _sub(names)
    runner.az(["az", "resource", "update", *g, "--name", "scm", "--namespace", "Microsoft.Web", "--resource-type",
               "basicPublishingCredentialsPolicies", "--parent", f"sites/{names.app}", "--set", "properties.allow=true", *s])
    profile = runner.az(["az", "webapp", "deployment", "list-publishing-profiles", "--name", names.app, *g, "--xml", *s])
    target = ROOT / "dist" / "publish_profile.xml"
    target.parent.mkdir(exist_ok=True)
    target.write_text(profile, encoding="utf-8")
    return target


def execute(names: Names, args, runner: Runner) -> None:
    preflight(names, runner)
    bundle = args.bundle
    if args.redeploy:
        deploy(names, bundle, runner)
        return
    commands = create_commands(names)
    for index, command in enumerate(commands, 1):
        check = exists_command(command)
        label = " ".join(command[1:4])
        if check and runner.ok(check):
            print(f"[{index}/{len(commands)}] {label}: already exists, skipped")
            continue
        print(f"[{index}/{len(commands)}] {label} ...")
        runner.az(command)
    problems = verify_free(names, runner)
    if problems:
        raise RuntimeError("COST GUARD FAILED - " + "; ".join(problems) +
                           ". Nothing else will be changed. Delete with: azure_free_deploy.py --teardown --execute")
    print("Verified free settings: Speech F0, Language F0, Cosmos free tier capped at 1000 RU/s, logs capped at 0.1 GB/day.")
    values = read_settings(names, runner)
    if getattr(args, "skip_deploy", False):
        print("Skipping settings + deploy (--skip-deploy); checking the site is up ...")
        wait_for_site(names)
    else:
        print("Writing app settings (keys are not printed) ...")
        apply_settings(names, values, runner)
        print("  letting the site restart after the settings change ...")
        time.sleep(45)
        print("Deploying the updated app ...")
        deploy(names, bundle, runner)
    topic_id = runner.az(["az", "eventgrid", "topic", "show", "--name", names.topic, "--resource-group", names.group,
                          "--query", "id", "-o", "tsv", *_sub(names)])
    print("Connecting Event Grid to the app's /events webhook (Event Grid validates the endpoint) ...")
    runner.az(subscription_command(names, topic_id, values["CLOUD_EVENTS_SECRET"]))
    if args.with_entra:
        print("Setting up Microsoft Entra ID login ...")
        setup_entra(names, runner)
    if args.with_cicd:
        path = setup_cicd(names, runner)
        print(f"Publish profile written to {path} (git-ignored). Add it to GitHub: Settings > Secrets and variables >")
        print("  Actions > New repository secret AZURE_WEBAPP_PUBLISH_PROFILE (paste the file), and a repository")
        print(f"  variable AZURE_WEBAPP_NAME = {names.app}. Then delete the file.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--subscription", required=True)
    parser.add_argument("--app-name", required=True)
    parser.add_argument("--location", default="centralindia")
    parser.add_argument("--cosmos-location", default=None, help="Cosmos DB region if the free tier is full in --location.")
    parser.add_argument("--bundle", type=Path, default=ROOT / "dist" / "azure_app.zip")
    parser.add_argument("--with-entra", action="store_true", help="Also switch on Microsoft Entra ID login (free).")
    parser.add_argument("--with-cicd", action="store_true", help="Also prepare GitHub Actions deployment (free).")
    parser.add_argument("--redeploy", action="store_true", help="Only rebuild the bundle and deploy the code.")
    parser.add_argument("--skip-deploy", action="store_true",
                        help="Resources, settings and code are already in place: only (re)connect Event Grid and the options.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-free-tier", action="store_true")
    args = parser.parse_args(argv)
    names = Names(args.subscription, args.app_name, args.location, args.cosmos_location or args.location)
    try:
        validate(names)
    except ValueError as error:
        print(f"Invalid arguments: {error}", file=sys.stderr)
        return 2
    if not args.execute:
        print("DRY RUN - nothing will be created. --execute would run:\n")
        plan = deploy_commands(names, args.bundle) if args.redeploy else create_commands(names)
        for command in plan:
            print("  " + " ".join(command))
        if not args.redeploy:
            print("  (then: read keys -> app settings, rebuild + deploy the app, Event Grid webhook subscription"
                  + (", Entra login" if args.with_entra else "") + (", GitHub publish profile" if args.with_cicd else "") + ")")
            print("\nAll tiers are fixed to free settings and verified after creation. Add --execute --confirm-free-tier to run.")
        return 0
    if not args.redeploy and not args.confirm_free_tier:
        print("Refusing to run: add --confirm-free-tier.", file=sys.stderr)
        return 2
    try:
        execute(names, args, Runner())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", None)
        print(f"Stopped: {error}" + (f"\n{stderr.strip()[-800:]}" if stderr else ""), file=sys.stderr)
        return 1
    print(f"\nDone. Open {names.site}/ - the voice section now offers Azure speech analysis and the page shows 'My history'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
