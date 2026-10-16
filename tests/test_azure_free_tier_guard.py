"""Cost guard-rails for the optional free-tier Azure deployment tooling.

Nothing here contacts Azure: `az` is replaced by a recording fake. These tests make sure the deploy script
can only ever create free (F1) resources, never leaks the access password onto a command line, and never
runs without explicit confirmation.
"""
import base64
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import azure_free_deploy as deploy  # noqa: E402
import package_for_azure as package  # noqa: E402
from src.backend.assessment import api as api_module  # noqa: E402

PLAN = deploy.Plan("My Sub", "centralindia", "emotion-demo-123", Path("dist/azure_app.zip"))
FORBIDDEN_SERVICES = ["ml workspace", "ml compute", "online-endpoint", "online-deployment", "acr ", "aks", "cosmosdb", "sql ",
                      "monitor", "log-analytics", "keyvault", "vm create", "functionapp", "storage account", "redis",
                      "--sku b", "--sku s", "--sku p", "--sku d", "--sku i"]


class TestOnlyFreeResources:
    def test_exactly_one_sku_and_it_is_free(self):
        assert deploy.ALLOWED_SKU == "F1"
        commands = deploy.build_commands(PLAN)
        skus = [c[c.index("--sku") + 1] for c in commands if "--sku" in c]
        assert skus == ["F1"]

    def test_only_expected_azure_operations(self):
        verbs = [" ".join(c[1:4]) for c in deploy.build_commands(PLAN)]
        assert verbs == ["group create --name", "appservice plan create", "webapp create --name", "webapp update --name",
                         "webapp config appsettings", "webapp config set", "webapp deploy --name"]

    def test_no_paid_service_commands_anywhere_in_the_script(self):
        source = (ROOT / "scripts" / "azure_free_deploy.py").read_text(encoding="utf-8").lower()
        commands = " ".join(" ".join(c) for c in deploy.build_commands(PLAN) + deploy.teardown_commands(PLAN)).lower()
        for item in FORBIDDEN_SERVICES:
            assert item not in commands, item
            assert not re.search(rf"\[\s*[\"']az[\"'],\s*[\"']{re.escape(item.strip())}", source), item

    def test_there_is_no_option_to_choose_a_sku(self):
        parser_source = (ROOT / "scripts" / "azure_free_deploy.py").read_text(encoding="utf-8")
        assert '"--sku"' not in parser_source.split("def main")[1]

    def test_always_on_is_not_requested_and_https_is_forced(self):
        flat = [" ".join(c) for c in deploy.build_commands(PLAN)]
        assert any("--always-on false" in c for c in flat) and any("--https-only true" in c for c in flat)

    def test_teardown_only_deletes_the_script_created_resource_group(self):
        (command,) = deploy.teardown_commands(PLAN)
        assert command[:4] == ["az", "group", "delete", "--name"] and command[4] == "rg-emotion-demo-123"

    @pytest.mark.parametrize("name", ["UPPER", "has space", "a", "-bad-", "x" * 50, "under_score"])
    def test_bad_names_rejected(self, name):
        with pytest.raises(ValueError):
            deploy.build_commands(deploy.Plan("S", "eastus", name, Path("b.zip")))

    def test_subscription_and_location_required_and_validated(self):
        with pytest.raises(ValueError):
            deploy.build_commands(deploy.Plan(" ", "eastus", "valid-name", Path("b.zip")))
        with pytest.raises(ValueError):
            deploy.build_commands(deploy.Plan("S", "East US!", "valid-name", Path("b.zip")))


class TestSecretsAndConfirmation:
    def test_password_never_appears_on_a_command_line(self):
        commands = deploy.build_commands(PLAN, "settings.json")
        assert "s3cret-password-123" not in " ".join(" ".join(c) for c in commands)
        assert all("APP_ACCESS_PASSWORD" not in " ".join(c) for c in commands)

    def test_settings_include_password_only_when_given(self):
        assert "APP_ACCESS_PASSWORD" not in deploy.settings(None)
        assert deploy.settings("x" * 12)["APP_ACCESS_PASSWORD"] == "x" * 12

    def test_dry_run_runs_nothing(self, monkeypatch, capsys):
        def boom(*a, **k):
            raise AssertionError("subprocess must not run in dry-run mode")

        monkeypatch.setattr(subprocess, "run", boom)
        code = deploy.main(["--subscription", "My Sub", "--location", "centralindia", "--app-name", "emotion-demo-123"])
        assert code == 0 and "DRY RUN" in capsys.readouterr().out

    def test_execute_without_confirmation_is_refused(self, monkeypatch, capsys):
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
        code = deploy.main(["--subscription", "S", "--location", "eastus", "--app-name", "valid-name", "--execute"])
        assert code == 2 and "--confirm-free-tier" in capsys.readouterr().err


class FakeAz:
    """Records every command and answers the few read queries the script makes."""

    def __init__(self, subscription="My Sub", state="Enabled", group_exists="false", sku="F1"):
        self.calls, self.subscription, self.state, self.group_exists, self.sku = [], subscription, state, group_exists, sku
        self.settings_seen = None

    def __call__(self, command, **kwargs):
        self.calls.append(list(command))
        if "@" in " ".join(command):
            path = next(part[1:] for part in command if part.startswith("@"))
            self.settings_seen = json.loads(Path(path).read_text())
        out = ""
        if command[:3] == ["az", "account", "show"]:
            out = json.dumps({"name": self.subscription, "state": self.state})
        elif command[:3] == ["az", "group", "exists"]:
            out = self.group_exists
        elif command[:4] == ["az", "appservice", "plan", "show"]:
            out = self.sku
        return SimpleNamespace(stdout=out, returncode=0)


@pytest.fixture
def ready(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"zip")
    monkeypatch.setattr(deploy.shutil, "which", lambda name: "az")
    monkeypatch.setenv("DEPLOY_APP_PASSWORD", "a-long-enough-password")
    return deploy.Plan("My Sub", "centralindia", "emotion-demo-123", bundle)


class TestExecutionPath:
    def test_full_run_issues_only_the_expected_commands(self, ready):
        fake = FakeAz()
        url = deploy.execute(ready, fake)
        assert url == "https://emotion-demo-123.azurewebsites.net/"
        creates = [c for c in fake.calls if c[:4] in (["az", "group", "create", "--name"], ["az", "appservice", "plan", "create"],
                                                   ["az", "webapp", "create", "--name"])]
        assert len(creates) == 3  # one resource group, one plan, one web app - nothing else is ever created
        assert fake.settings_seen["APP_ACCESS_PASSWORD"] == "a-long-enough-password"
        assert "a-long-enough-password" not in " ".join(" ".join(c) for c in fake.calls)

    def test_settings_file_is_deleted_afterwards(self, ready):
        fake = FakeAz()
        deploy.execute(ready, fake)
        path = next(part[1:] for call in fake.calls for part in call if part.startswith("@"))
        assert not Path(path).exists()

    def test_wrong_subscription_is_refused_before_anything_is_created(self, ready):
        fake = FakeAz(subscription="Someone Else")
        with pytest.raises(RuntimeError, match="not 'My Sub'"):
            deploy.execute(ready, fake)
        assert not any(c[1:3] == ["group", "create"] for c in fake.calls)

    def test_existing_resource_group_is_never_modified(self, ready):
        fake = FakeAz(group_exists="true")
        with pytest.raises(RuntimeError, match="already exists"):
            deploy.execute(ready, fake)
        assert not any(c[1:3] == ["group", "create"] for c in fake.calls)

    def test_missing_cli_bundle_and_password_are_refused(self, ready, monkeypatch):
        monkeypatch.setattr(deploy.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="not installed"):
            deploy.execute(ready, FakeAz())
        monkeypatch.setattr(deploy.shutil, "which", lambda name: "az")
        with pytest.raises(RuntimeError, match="Bundle not found"):
            deploy.execute(deploy.Plan("My Sub", "eastus", "valid-name", Path("nope.zip")), FakeAz())
        monkeypatch.setenv("DEPLOY_APP_PASSWORD", "short")
        with pytest.raises(RuntimeError, match="DEPLOY_APP_PASSWORD"):
            deploy.execute(ready, FakeAz())

    def test_non_free_plan_is_detected_and_stops(self, ready):
        fake = FakeAz(sku="B1")
        with pytest.raises(RuntimeError, match="expected 'F1'"):
            deploy.execute(ready, fake)
        assert not any(c[1:3] == ["webapp", "create"] for c in fake.calls)  # stopped right after the plan was created

    def test_disabled_subscription_refused(self, ready):
        with pytest.raises(RuntimeError, match="not enabled"):
            deploy.execute(ready, FakeAz(state="Disabled"))


class TestBundle:
    def test_deploy_requirements_drop_training_test_and_azure_packages(self):
        text = package.deploy_requirements((ROOT / "requirements.txt").read_text(encoding="utf-8"))
        names = {re.split(r"[<>=!~ ]", line)[0].lower() for line in text.split()}
        assert {"fastapi", "uvicorn", "scikit-learn", "librosa"} <= names
        assert not names & {"matplotlib", "pytest", "azure-ai-ml", "azure-storage-blob", "pyyaml", "httpx"}
        assert "scikit-learn==1.8.0" in text

    def test_bundle_contents(self, tmp_path):
        summary = package.build_bundle(tmp_path / "b.zip")
        names = zipfile.ZipFile(tmp_path / "b.zip").namelist()
        assert "requirements.txt" in names and "src/backend/assessment/api.py" in names
        assert "models/voice_emotion/voice_emotion_random_forest.joblib" in names
        assert "models/questionnaire/random_forest.joblib" in names
        assert not [n for n in names if n.startswith(("data/", "tests/", "venv/", "docs/", "results/", "src/azure/"))]
        assert not [n for n in names if n.endswith((".wav", ".csv", ".env", ".pem", ".key", ".template", ".pyc"))]
        assert summary["size_mb"] < package.MAX_BUNDLE_MB

    def test_missing_model_file_fails_clearly(self, tmp_path, monkeypatch):
        monkeypatch.setitem(package.MODEL_FILES, "models/behavioral", {"does_not_exist.joblib"})
        assert package.main(["--output", str(tmp_path / "b.zip")]) == 2


class TestOptionalAccessPassword:
    @pytest.fixture(autouse=True)
    def client(self):
        self.client = TestClient(api_module.app)

    @staticmethod
    def basic(user, password):
        return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}

    def test_open_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("APP_ACCESS_USER", raising=False)
        monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
        assert self.client.get("/app/").status_code == 200

    def test_protected_when_configured(self, monkeypatch):
        monkeypatch.setenv("APP_ACCESS_USER", "student")
        monkeypatch.setenv("APP_ACCESS_PASSWORD", "correct-horse-battery")
        denied = self.client.get("/app/")
        assert denied.status_code == 401 and "Basic" in denied.headers["www-authenticate"]
        assert denied.headers["cache-control"] == "no-store"
        assert self.client.post("/assessment", json={}).status_code == 401
        assert self.client.get("/status").status_code == 401
        assert self.client.get("/app/", headers=self.basic("student", "wrong")).status_code == 401
        assert self.client.get("/app/", headers=self.basic("other", "correct-horse-battery")).status_code == 401
        assert self.client.get("/app/", headers={"Authorization": "Basic !!!"}).status_code == 401
        assert self.client.get("/app/", headers=self.basic("student", "correct-horse-battery")).status_code == 200
        assert self.client.post("/assessment", json={}, headers=self.basic("student", "correct-horse-battery")).status_code == 200

    def test_health_stays_open_for_host_probes(self, monkeypatch):
        monkeypatch.setenv("APP_ACCESS_USER", "student")
        monkeypatch.setenv("APP_ACCESS_PASSWORD", "correct-horse-battery")
        assert self.client.get("/health").json() == {"status": "ok"}

    def test_only_one_of_the_two_variables_does_not_lock_or_half_protect(self, monkeypatch):
        monkeypatch.setenv("APP_ACCESS_USER", "student")
        monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)
        assert self.client.get("/app/").status_code == 200
