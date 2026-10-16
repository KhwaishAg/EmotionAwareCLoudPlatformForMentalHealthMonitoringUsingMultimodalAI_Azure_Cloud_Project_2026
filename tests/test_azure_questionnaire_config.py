"""Local (no Azure access) checks of the questionnaire Azure ML configuration.

These verify that templates parse, contain placeholders instead of real
resource values, and that score.py works against the local model artifacts.
They do NOT prove that anything deploys or runs in Azure.
"""
import json
import re
from pathlib import Path

import pytest
import yaml

import src.ai_model.questionnaire.predict as q_predict
from tests.test_assessment_api import QUESTIONNAIRE, SKLEARN_OK, SKLEARN_REASON

AZURE_DIR = Path(__file__).resolve().parents[1] / "src" / "azure" / "questionnaire"
TEMPLATES = sorted(AZURE_DIR.glob("*.yml.template")) + [AZURE_DIR / "environment.yml", AZURE_DIR / "conda.yml"]
SECRET_PATTERNS = [r"AccountKey=", r"sig=", r"client_secret", r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"]


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_template_is_valid_yaml_without_secrets(path):
    text = path.read_text(encoding="utf-8")
    assert yaml.safe_load(text)
    for pattern in SECRET_PATTERNS:
        assert not re.search(pattern, text, re.IGNORECASE), f"{path.name} matches secret pattern {pattern}"


def test_resource_names_are_placeholders():
    text = (AZURE_DIR / "endpoint.yml.template").read_text(encoding="utf-8")
    assert re.search(r"^name: <[^>]+>", text, re.MULTILINE)
    assert re.search(r"^name: <[^>]+>", (AZURE_DIR / "workspace.yml.template").read_text(encoding="utf-8"), re.MULTILINE)


def test_deployment_references_existing_scoring_script_and_code_root():
    deployment = yaml.safe_load((AZURE_DIR / "deployment.yml.template").read_text(encoding="utf-8"))
    code_root = (AZURE_DIR / deployment["code_configuration"]["code"]).resolve()
    assert (code_root / deployment["code_configuration"]["scoring_script"]).is_file()


def test_environment_pins_match_between_conda_and_requirements():
    conda = yaml.safe_load((AZURE_DIR / "conda.yml").read_text(encoding="utf-8"))
    pins = {item.split("==")[0]: item.split("==")[1] for item in conda["dependencies"][-1]["pip"] if "==" in item}
    requirements = (AZURE_DIR / "requirements.txt").read_text(encoding="utf-8")
    for name, version in pins.items():
        assert f"{name}=={version}" in requirements
    assert any("azureml-inference-server-http" in item for item in conda["dependencies"][-1]["pip"])


@pytest.mark.skipif(not SKLEARN_OK, reason=SKLEARN_REASON)
class TestScoringAdapterLocally:
    @pytest.fixture
    def score(self, monkeypatch):
        import importlib.util

        spec = importlib.util.spec_from_file_location("questionnaire_score", AZURE_DIR / "score.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setenv("AZUREML_MODEL_DIR", str(Path(q_predict.MODELS_DIR)))
        module.init()
        return module

    def test_run_returns_fusion_contract(self, score):
        result = json.loads(score.run(json.dumps({"responses": QUESTIONNAIRE})))
        assert result["modality"] == "questionnaire"
        assert 0.0 <= result["risk_score"] <= 1.0

    @pytest.mark.parametrize("body, code", [("not json", 400), ("{}", 400), ('{"responses": []}', 400),
                                            ('{"responses": {"Q1": "x"}}', 422)])
    def test_run_rejects_bad_requests(self, score, body, code):
        assert json.loads(score.run(body))["status_code"] == code

    def test_init_fails_clearly_when_artifacts_missing(self, tmp_path, monkeypatch):
        import importlib.util

        spec = importlib.util.spec_from_file_location("questionnaire_score_missing", AZURE_DIR / "score.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = q_predict.MODELS_DIR
        monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path))
        try:
            with pytest.raises(RuntimeError, match="required artifacts not found"):
                module.init()
        finally:
            q_predict.MODELS_DIR = original
