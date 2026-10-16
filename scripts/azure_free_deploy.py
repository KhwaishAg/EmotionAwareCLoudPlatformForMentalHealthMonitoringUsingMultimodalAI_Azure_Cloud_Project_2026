"""Deploy the app to Azure App Service on the FREE F1 plan only, with hard cost guard-rails.

DRY-RUN BY DEFAULT: prints the exact `az` commands and runs NOTHING. Nothing is created unless you pass
``--execute --confirm-free-tier``. The script can only ever create: one resource group, one **F1 (free)**
Linux App Service plan, and one web app. It has no option to choose another SKU and contains no commands for
paid services (Azure ML compute/endpoints, container registry, databases, monitoring workspaces, AKS ...).

Prerequisites (you do these, in your own terminal; credentials never pass through this script or the chat):
    az login
    az account set --subscription "<name>"
    set DEPLOY_APP_PASSWORD=<choose a password>      # used as the site's access password (stored as an app setting)

Typical use:
    python scripts/package_for_azure.py
    python scripts/azure_free_deploy.py --subscription "<name>" --location centralindia --app-name <unique-name>
    python scripts/azure_free_deploy.py ... --execute --confirm-free-tier
    python scripts/azure_free_deploy.py ... --teardown --execute      # deletes everything created here
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]

#: The ONLY SKU this script can create. F1 = free (no credits consumed).
ALLOWED_SKU = "F1"
PYTHON_RUNTIME = "PYTHON:3.12"
STARTUP_COMMAND = "python -m uvicorn src.backend.assessment.api:app --host 0.0.0.0 --port 8000"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$")
LOCATION_RE = re.compile(r"^[a-z0-9]{3,30}$")


@dataclass(frozen=True)
class Plan:
    subscription: str
    location: str
    app_name: str
    bundle: Path

    @property
    def resource_group(self) -> str:
        return f"rg-{self.app_name}"

    @property
    def plan_name(self) -> str:
        return f"plan-{self.app_name}-free"


def validate(plan: Plan) -> None:
    if not NAME_RE.match(plan.app_name):
        raise ValueError("--app-name must be 3-40 chars: lowercase letters, digits and hyphens (globally unique).")
    if not LOCATION_RE.match(plan.location):
        raise ValueError("--location must look like 'centralindia' or 'eastus'.")
    if not plan.subscription.strip():
        raise ValueError("--subscription is required so resources cannot land in the wrong subscription.")


def settings(password: Optional[str]) -> dict:
    """App settings. The access password is the only secret and is written to a temp file, never put on a command line."""
    values = {
        "SCM_DO_BUILD_DURING_DEPLOYMENT": "true",   # server installs requirements.txt from the bundle
        "WEBSITES_PORT": "8000",
        "ENABLE_ORYX_BUILD": "true",
        "APP_ACCESS_USER": "student",
    }
    if password:
        values["APP_ACCESS_PASSWORD"] = password
    return values


def build_commands(plan: Plan, settings_file: str = "<settings.json>") -> List[List[str]]:
    """Every command this script can run, in order. Pure function (testable); nothing is executed here."""
    validate(plan)
    sub = ["--subscription", plan.subscription]
    return [
        ["az", "group", "create", "--name", plan.resource_group, "--location", plan.location, *sub],
        ["az", "appservice", "plan", "create", "--name", plan.plan_name, "--resource-group", plan.resource_group,
         "--is-linux", "--sku", ALLOWED_SKU, *sub],
        ["az", "webapp", "create", "--name", plan.app_name, "--resource-group", plan.resource_group,
         "--plan", plan.plan_name, "--runtime", PYTHON_RUNTIME, *sub],
        ["az", "webapp", "update", "--name", plan.app_name, "--resource-group", plan.resource_group,
         "--https-only", "true", *sub],
        ["az", "webapp", "config", "appsettings", "set", "--name", plan.app_name, "--resource-group", plan.resource_group,
         "--settings", f"@{settings_file}", *sub],
        ["az", "webapp", "config", "set", "--name", plan.app_name, "--resource-group", plan.resource_group,
         "--startup-file", STARTUP_COMMAND, "--always-on", "false", *sub],
        ["az", "webapp", "deploy", "--name", plan.app_name, "--resource-group", plan.resource_group,
         "--src-path", str(plan.bundle), "--type", "zip", *sub],
    ]


def teardown_commands(plan: Plan) -> List[List[str]]:
    validate(plan)
    return [["az", "group", "delete", "--name", plan.resource_group, "--yes", "--subscription", plan.subscription]]


def _run(command, **kwargs):
    """subprocess.run that resolves `az` to its full path (az.cmd on Windows)."""
    if command and command[0] == "az":
        command = [shutil.which("az") or "az", *command[1:]]
    return subprocess.run(command, **kwargs)


def verify_plan_is_free(plan: Plan, run=_run) -> None:
    """After creation: read the plan back and fail loudly (so you can tear down) if it is not F1."""
    done = run(["az", "appservice", "plan", "show", "--name", plan.plan_name, "--resource-group", plan.resource_group,
                "--subscription", plan.subscription, "--query", "sku.name", "-o", "tsv"],
               capture_output=True, text=True, check=True)
    if done.stdout.strip() != ALLOWED_SKU:
        raise RuntimeError(f"Plan SKU is {done.stdout.strip()!r}, expected {ALLOWED_SKU!r}. Run with --teardown --execute now.")


def _az(args: List[str], run=_run) -> str:
    return run(["az", *args], capture_output=True, text=True, check=True).stdout.strip()


def preflight(plan: Plan, run=_run) -> None:
    if shutil.which("az") is None:
        raise RuntimeError("Azure CLI ('az') is not installed. Install it and run 'az login' first.")
    account = json.loads(_az(["account", "show", "-o", "json"], run))
    if account.get("state") != "Enabled":
        raise RuntimeError("The signed-in subscription is not enabled.")
    if account.get("name") != plan.subscription:
        raise RuntimeError(f"Signed-in subscription is {account.get('name')!r}, not {plan.subscription!r}. "
                           "Run 'az account set' or fix --subscription.")
    if not plan.bundle.is_file():
        raise RuntimeError(f"Bundle not found: {plan.bundle}. Run scripts/package_for_azure.py first.")
    existing = run(["az", "group", "exists", "--name", plan.resource_group, "--subscription", plan.subscription],
                   capture_output=True, text=True, check=True).stdout.strip()
    if existing == "true":
        raise RuntimeError(f"Resource group {plan.resource_group!r} already exists; refusing to modify it.")


def execute(plan: Plan, run=_run) -> str:
    preflight(plan, run)
    password = os.environ.get("DEPLOY_APP_PASSWORD")
    if not password or len(password) < 12:
        raise RuntimeError("Set DEPLOY_APP_PASSWORD (12+ characters) in your shell first: the public site needs an access password.")
    handle, name = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as target:
            json.dump(settings(password), target)
        commands = build_commands(plan, name)
        for index, command in enumerate(commands, 1):
            print(f"[{index}/{len(commands)}] {' '.join(command[:5])} ...")
            run(command, check=True)
            if command[:3] == ["az", "appservice", "plan"]:
                verify_plan_is_free(plan, run)
                print("    verified: plan SKU is F1 (free)")
    finally:
        os.remove(name)
    return f"https://{plan.app_name}.azurewebsites.net/"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--subscription", required=True, help="Subscription NAME (must match the signed-in one).")
    parser.add_argument("--location", required=True)
    parser.add_argument("--app-name", required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "dist" / "azure_app.zip")
    parser.add_argument("--execute", action="store_true", help="Actually run the commands (default: print only).")
    parser.add_argument("--confirm-free-tier", action="store_true", help="Required with --execute: you accept F1-only creation.")
    parser.add_argument("--teardown", action="store_true", help="Delete the resource group created by this script.")
    args = parser.parse_args(argv)

    plan = Plan(args.subscription, args.location, args.app_name, args.bundle)
    try:
        commands = teardown_commands(plan) if args.teardown else build_commands(plan)
    except ValueError as error:
        print(f"Invalid arguments: {error}", file=sys.stderr)
        return 2

    if not args.execute:
        print("DRY RUN - nothing will be created. Commands that --execute would run:\n")
        for command in commands:
            print("  " + " ".join(command))
        print(f"\nSKU is fixed to {ALLOWED_SKU} (free). Add --execute --confirm-free-tier to run for real.")
        return 0
    if not args.teardown and not args.confirm_free_tier:
        print("Refusing to run: add --confirm-free-tier to confirm F1-only creation.", file=sys.stderr)
        return 2
    try:
        if args.teardown:
            preflight_account = json.loads(_az(["account", "show", "-o", "json"]))
            if preflight_account.get("name") != plan.subscription:
                raise RuntimeError("Signed-in subscription does not match --subscription.")
            for command in commands:
                _run(command, check=True)
            print("Resource group deleted.")
        else:
            print("Site URL: " + execute(plan))
            print("Username is 'student'; the password is the DEPLOY_APP_PASSWORD you set. First load can take ~1 min (cold start).")
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
