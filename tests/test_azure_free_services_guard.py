"""Guard-rails for scripts/azure_free_services.py. Nothing here calls Azure: `az` is a recording fake."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("azure_free_services", ROOT / "scripts" / "azure_free_services.py")
svc = importlib.util.module_from_spec(spec)
sys.modules["azure_free_services"] = svc
spec.loader.exec_module(svc)

NAMES = svc.Names("My Sub", "emotion-demo-123", "centralindia", "centralindia")


def az_commands(commands):
    return [c for c in commands if c and c[0] == "az"]


def test_only_free_skus_and_caps_are_requested():
    commands = svc.create_commands(NAMES)
    cognitive = [c for c in commands if c[:4] == ["az", "cognitiveservices", "account", "create"]]
    assert len(cognitive) == 2 and all(c[c.index("--sku") + 1] == "F0" for c in cognitive)
    cosmos = next(c for c in commands if c[:3] == ["az", "cosmosdb", "create"])
    assert cosmos[cosmos.index("--enable-free-tier") + 1] == "true"
    assert "Continuous" not in " ".join(cosmos)  # continuous backup is billed
    assert any('properties.capacity={"totalThroughputLimit": 1000}' in c for c in commands)
    database = next(c for c in commands if c[:5] == ["az", "cosmosdb", "sql", "database", "create"])
    assert int(database[database.index("--throughput") + 1]) <= 1000
    container = next(c for c in commands if c[:5] == ["az", "cosmosdb", "sql", "container", "create"])
    assert "--throughput" not in container  # shares the database's free throughput
    logs = next(c for c in commands if c[:4] == ["az", "monitor", "log-analytics", "workspace"])
    assert float(logs[logs.index("--quota") + 1]) <= 0.1


def test_no_paid_service_command_exists_in_the_script():
    source = (ROOT / "scripts" / "azure_free_services.py").read_text(encoding="utf-8").lower()
    for forbidden in ("appservice plan create", '"plan", "create"', '"ml"', '"acr"', '"aks"', "functionapp", "storage account",
                      '"storage"', "sql server", '"vm"', "--sku s0", "--sku s1", "--sku b1", "serverless", "continuous7days"):
        assert forbidden not in source, forbidden


def test_every_command_uses_an_allowed_group_and_the_named_subscription():
    commands = az_commands(svc.create_commands(NAMES) + svc.deploy_commands(NAMES, Path("x.zip"))
                           + [svc.subscription_command(NAMES, "/topic/id", "s")] + svc.entra_commands(NAMES, "cid", "tid"))
    for command in commands:
        assert (command[1],) in svc.ALLOWED_AZ_GROUPS, command
        if command[1] not in ("config",):
            assert command[-2:] == ["--subscription", "My Sub"], command
    assert all("rg-emotion-demo-123" in c for c in commands if "--resource-group" in c)


def test_entra_keeps_the_webhook_reachable():
    update = next(c for c in svc.entra_commands(NAMES, "cid", "tid") if c[:4] == ["az", "webapp", "auth", "update"])
    assert "/events" in json.loads(update[update.index("--excluded-paths") + 1])


def test_invalid_names_rejected():
    with pytest.raises(ValueError):
        svc.create_commands(svc.Names("My Sub", "Bad Name", "centralindia", "centralindia"))


def test_dry_run_runs_nothing(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("dry run must not execute"))
    assert svc.main(["--subscription", "My Sub", "--app-name", "emotion-demo-123"]) == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_execute_requires_confirmation(capsys):
    assert svc.main(["--subscription", "My Sub", "--app-name", "emotion-demo-123", "--execute"]) == 2


class FakeAz:
    """Pretends to be az: records calls and returns free settings unless told otherwise."""

    def __init__(self, cosmos_free=True):
        self.calls, self.cosmos_free = [], cosmos_free

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        out, code = "", 0
        if command[:3] == ["az", "account", "show"]:
            out = json.dumps({"name": "My Sub", "state": "Enabled"})
        elif command[-6:-2] == ["--query", "sku.name", "-o", "tsv"] or "sku.name" in command:
            out = "F0"
        elif command[:3] == ["az", "resource", "show"] and "properties" in command:
            out = json.dumps({"enableFreeTier": self.cosmos_free, "capacity": {"totalThroughputLimit": 1000}})
        elif "workspaceCapping.dailyQuotaGb" in command:
            out = "0.1"
        elif command[:5] == ["az", "webapp", "config", "appsettings", "list"]:
            out = "[]"
        elif command[1:3] in (["cognitiveservices", "account"], ["cosmosdb", "show"], ["eventgrid", "topic"],
                              ["monitor", "log-analytics"], ["monitor", "app-insights"], ["cosmosdb", "sql"]) and "show" in command[:6]:
            code = 1  # nothing exists yet
            out = "x"
        else:
            out = "value"
        return subprocess.CompletedProcess(command, code, stdout=out, stderr="")


def test_cost_guard_stops_before_settings_or_deploy(monkeypatch):
    fake = FakeAz(cosmos_free=False)
    monkeypatch.setattr(svc.shutil, "which", lambda name: "az")
    args = type("A", (), {"redeploy": False, "bundle": Path("x.zip"), "with_entra": False, "with_cicd": False})
    with pytest.raises(RuntimeError, match="COST GUARD"):
        svc.execute(NAMES, args, svc.Runner(fake))
    assert not [c for c in fake.calls if c[:5] == ["az", "webapp", "config", "appsettings", "set"]]
    assert not [c for c in fake.calls if c[:3] == ["az", "webapp", "deploy"]]


def test_full_run_writes_settings_from_a_temp_file_and_subscribes_webhook(monkeypatch):
    fake = FakeAz()
    monkeypatch.setattr(svc.shutil, "which", lambda name: "az")
    monkeypatch.setattr(svc, "wait_for_site", lambda names: None)
    monkeypatch.setattr(svc.time, "sleep", lambda seconds: None)
    args = type("A", (), {"redeploy": False, "bundle": Path("x.zip"), "with_entra": False, "with_cicd": False})
    svc.execute(NAMES, args, svc.Runner(fake))
    settings_call = next(c for c in fake.calls if c[:5] == ["az", "webapp", "config", "appsettings", "set"])
    assert settings_call[settings_call.index("--settings") + 1].startswith("@")  # keys never on the command line
    sub = next(c for c in fake.calls if c[:3] == ["az", "eventgrid", "event-subscription"])
    assert sub[sub.index("--endpoint") + 1].startswith("https://emotion-demo-123.azurewebsites.net/events?key=")


def test_deploy_retries_after_a_transient_kudu_failure(monkeypatch):
    attempts = []

    def run(command, **kwargs):
        if command[:3] == ["az", "webapp", "deploy"]:
            attempts.append(1)
            if len(attempts) == 1:
                raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(svc, "wait_for_site", lambda names: None)
    monkeypatch.setattr(svc.time, "sleep", lambda seconds: None)
    svc.deploy(NAMES, Path("x.zip"), svc.Runner(run))
    assert len(attempts) == 2
