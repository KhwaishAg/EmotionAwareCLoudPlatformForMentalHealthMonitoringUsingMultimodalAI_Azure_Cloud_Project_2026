"""Zero-cost guarantees: the runnable app is fully local and needs no cloud service.

These tests make the "no cloud spending" requirement enforceable. Azure code under
``src/azure`` and the deployment templates are preserved but are never imported or
used by the running application.

The optional free-tier integrations in ``src/cloud`` (Azure AI Speech/Language F0, Event Grid,
Cosmos DB free tier) are the one exception: they are switched off unless ``CLOUD_*`` settings
exist, all their network traffic goes through ``src/cloud/http.py``, and with no settings the
app still makes no network calls at all (checked below with the network blocked).
"""
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME_FILES = [p for p in SRC.rglob("*.py") if "azure" not in p.relative_to(SRC).parts and "__pycache__" not in p.parts]
CLOUD_DIR = SRC / "cloud"
LOCAL_RUNTIME_FILES = [p for p in RUNTIME_FILES if CLOUD_DIR not in p.parents]

OFFLINE_RUN = textwrap.dedent(
    """
    import json, socket, sys, warnings
    warnings.filterwarnings("ignore")
    attempts = []

    class BlockAzure:
        def find_spec(self, name, path=None, target=None):
            if name == "azure" or name.startswith("azure."):
                attempts.append(("azure-import", name))
                raise ImportError("BLOCKED " + name)

    sys.meta_path.insert(0, BlockAzure())
    LOOP = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "", None}
    real_connect, real_create, real_gai = socket.socket.connect, socket.create_connection, socket.getaddrinfo
    host = lambda a: a[0] if isinstance(a, tuple) and a else str(a)

    def connect(self, address):
        if host(address) not in LOOP:
            attempts.append(("network", str(address)))
            raise OSError("BLOCKED")
        return real_connect(self, address)

    def create(address, *a, **k):
        if host(address) not in LOOP:
            attempts.append(("network", str(address)))
            raise OSError("BLOCKED")
        return real_create(address, *a, **k)

    def gai(h, *a, **k):
        if h not in LOOP:
            attempts.append(("dns", str(h)))
            raise OSError("BLOCKED")
        return real_gai(h, *a, **k)

    socket.socket.connect, socket.create_connection, socket.getaddrinfo = connect, create, gai
    from fastapi.testclient import TestClient
    from src.backend.assessment.api import app
    c = TestClient(app)
    payload = json.load(open("assessment_payload.json"))
    out = {
        "codes": [c.get(p).status_code for p in ("/health", "/app/", "/app/app.js", "/schema", "/evaluation", "/status")],
        "assessment": c.post("/assessment", json=payload).status_code,
        "azure_loaded": sorted(m for m in sys.modules if m == "azure" or m.startswith("azure.")),
        "attempts": attempts,
    }
    print("RESULT" + json.dumps(out))
    """
)


def test_app_runs_with_azure_imports_and_external_network_blocked():
    import os

    env = {k: v for k, v in os.environ.items() if not k.startswith("CLOUD_")}  # default install: nothing configured
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONWARNINGS="ignore")
    done = subprocess.run([sys.executable, "-c", OFFLINE_RUN], cwd=ROOT, capture_output=True, text=True, env=env, timeout=240)
    line = next((l for l in done.stdout.splitlines() if l.startswith("RESULT")), None)
    assert line, f"offline run failed:\n{done.stderr[-1500:]}"
    result = json.loads(line[len("RESULT"):])
    assert result["codes"] == [200] * 6
    assert result["assessment"] == 200
    assert result["azure_loaded"] == []
    assert result["attempts"] == []


def test_runtime_code_never_imports_azure():
    pattern = re.compile(r"^\s*(?:import|from)\s+(?:azure|src\.azure)\b", re.MULTILINE)
    offenders = [str(p.relative_to(ROOT)) for p in RUNTIME_FILES if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_runtime_code_has_no_network_libraries_or_azure_settings():
    pattern = re.compile(r"\b(requests|urllib|http\.client|aiohttp|httpx|urlopen|boto3|openai|AZURE_[A-Z_]+)\b")
    offenders = {str(p.relative_to(ROOT)): sorted(set(pattern.findall(p.read_text(encoding="utf-8")))) for p in LOCAL_RUNTIME_FILES}
    assert {k: v for k, v in offenders.items() if v} == {}


def test_optional_cloud_code_has_a_single_network_choke_point():
    """Only src/cloud/http.py may open connections; the rest of src/cloud builds requests through it."""
    pattern = re.compile(r"\b(requests|urllib\.request|urlopen|http\.client|aiohttp|httpx|socket|boto3|openai)\b")
    offenders = [str(p.relative_to(ROOT)) for p in CLOUD_DIR.glob("*.py")
                 if p.name != "http.py" and pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
    assert not re.search(r"^\s*(?:import|from)\s+azure\b", "\n".join(p.read_text(encoding="utf-8") for p in CLOUD_DIR.glob("*.py")), re.M)


def test_cloud_features_are_off_without_settings(monkeypatch):
    from src.cloud.config import load_settings

    for name in list(__import__("os").environ):
        if name.startswith("CLOUD_"):
            monkeypatch.delenv(name)
    settings = load_settings()
    assert not (settings.speech_content_enabled or settings.history_enabled or settings.events_enabled)


def test_frontend_loads_nothing_from_the_internet():
    for path in (ROOT / "src" / "frontend").glob("*"):
        if path.suffix in {".html", ".js", ".css"}:
            assert not re.search(r"https?://", path.read_text(encoding="utf-8")), path.name


def test_server_binds_to_loopback_only():
    launcher = (ROOT / "run_app.py").read_text(encoding="utf-8")
    assert 'host="127.0.0.1"' in launcher and "0.0.0.0" not in launcher


def test_interactive_docs_cdn_page_is_off_by_default():
    from fastapi.testclient import TestClient

    from src.backend.assessment.api import app

    assert TestClient(app).get("/docs").status_code == 404


def test_required_dependencies_contain_no_cloud_sdk():
    for name in ("requirements.txt", "requirements-dev.txt"):
        lines = [l.split("#")[0].strip().lower() for l in (ROOT / name).read_text(encoding="utf-8").splitlines()]
        assert not [l for l in lines if l.startswith(("azure", "boto", "google-cloud"))], name
    optional = (ROOT / "requirements-azure.txt").read_text(encoding="utf-8")
    assert "OPTIONAL" in optional and "azure-ai-ml" in optional


def test_deployment_templates_are_preserved_but_cannot_be_applied_as_is():
    folder = ROOT / "src" / "azure" / "questionnaire"
    templates = sorted(folder.glob("*.yml.template"))
    assert len(templates) >= 6  # workspace, compute, job, model, endpoint, deployment are all still there
    assert not [p for p in folder.glob("*.yml") if p.name not in {"conda.yml", "environment.yml"}]  # nothing deployable
    assert (ROOT / "docs" / "azure_deployment_runbook.md").exists()


def test_no_filled_in_azure_identifiers_are_committed():
    patterns = [r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", r"AccountKey=", r"\.database\.windows\.net",
                r"\.blob\.core\.windows\.net/[a-z]", r"(?<!>)\.azurewebsites\.net"]
    files = list((ROOT / "src" / "azure").rglob("*")) + list((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]
    for path in files:
        if path.is_file() and path.suffix in {".md", ".template", ".yml", ".py", ".txt"} and "tests" not in path.parts:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns:
                if path.name in {"config.py"} and pattern == r"AccountKey=":
                    continue  # documented connection-string format example, no value
                assert not re.search(pattern, text), f"{path.relative_to(ROOT)} matches {pattern}"
