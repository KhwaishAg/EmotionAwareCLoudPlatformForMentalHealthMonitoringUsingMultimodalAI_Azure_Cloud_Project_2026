"""Integration tests for the unified /assessment API.

Questionnaire and behavioral tests use the real committed model artifacts and
therefore need the scikit-learn version they were trained with (1.8.0,
Python >= 3.11); they are skipped with a clear reason otherwise. Voice tests use
a test-only model trained on random synthetic data (wiring checks only).
"""
import base64

import pytest
from fastapi.testclient import TestClient

from src.backend.assessment import api as api_module
from src.ai_model.voice import predict as voice_predict
from tests.voice_fixtures import full_feature_dict, make_test_voice_model, write_test_wav

client = TestClient(api_module.app)

try:
    import sklearn

    SKLEARN_OK = sklearn.__version__ == api_module.EXPECTED_SKLEARN_VERSION
    SKLEARN_REASON = f"needs scikit-learn {api_module.EXPECTED_SKLEARN_VERSION}, found {sklearn.__version__}"
except ImportError:  # pragma: no cover
    SKLEARN_OK, SKLEARN_REASON = False, "scikit-learn not installed"

needs_artifacts = pytest.mark.skipif(not SKLEARN_OK, reason=SKLEARN_REASON)

QUESTIONNAIRE = {
    "Q1": "4th Year", "Q2": "Regular academic semester; Working on academic projects", "Q3": "4–6 hours",
    "Q4": "4 — Very", "Q5": "Often", "Q6": "Sometimes", "Q7": "Very difficult", "Q8": "Often", "Q9": "Sometimes",
    "Q10": "Academic workload; Projects", "Q11": "Anxious; Tired", "Q12": "4", "Q13": "3",
    "Q14": "15–30 minutes", "Q15": "Sometimes", "Q16": "Within an hour", "Q17": "Often", "Q18": "Sometimes",
    "Q19": "Rarely", "Q20": "Sometimes",
}
BEHAVIORAL = {
    "B1_sleep_hours": 5.0, "B2_study_consistency": 2.0, "B3_social_activity_freq": 1.5,
    "B4_screen_time_hours": 9.0, "B5_routine_irregularity": 4.0,
}


def post(body):
    return client.post("/assessment", json=body)


@pytest.fixture
def no_voice_model(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_predict, "MODELS_DIR", tmp_path)


@pytest.fixture
def voice_model(monkeypatch):
    model = make_test_voice_model()
    monkeypatch.setattr(voice_predict, "load_selected_voice_model", lambda: (model, "test_model"))
    return model


class TestServiceEndpoints:
    def test_health(self):
        assert client.get("/health").json() == {"status": "ok"}

    def test_status_reports_availability_without_leaking_paths(self):
        body = client.get("/status").json()
        assert set(body["models"]) == {"questionnaire", "behavioral", "voice"}
        assert "environment" in body
        assert "C:\\" not in str(body) and "/Users/" not in str(body)

    def test_schema_describes_inputs(self):
        body = client.get("/schema").json()
        assert len(body["questionnaire_items"]) == 20
        assert all(item["options"] for item in body["questionnaire_items"] if item["type"] != "numeric")
        assert set(body["behavioral_fields"]) == set(BEHAVIORAL)

    def test_frontend_is_served(self):
        response = client.get("/app/")
        assert response.status_code == 200
        assert "not a medical tool" in response.text.lower()


@needs_artifacts
class TestRealModels:
    def test_questionnaire_and_behavioral_are_fused(self, no_voice_model):
        body = post({"questionnaire": {"responses": QUESTIONNAIRE}, "behavioral": {"responses": BEHAVIORAL}}).json()
        assessment = body["assessment"]
        assert body["modality_status"] == {"questionnaire": "available", "behavioral": "available", "voice": "not_provided"}
        assert assessment["status"] == "available"
        assert sorted(assessment["modalities_used"]) == ["behavioral", "questionnaire"]
        assert 0.0 <= assessment["risk_score"] <= 1.0
        assert sum(assessment["normalized_weights"].values()) == pytest.approx(1.0)
        assert "diagnosis" in body["disclaimer"]

    def test_questionnaire_only(self):
        body = post({"questionnaire": {"responses": QUESTIONNAIRE}}).json()
        assert body["assessment"]["modalities_used"] == ["questionnaire"]

    def test_behavioral_only(self):
        body = post({"behavioral": {"responses": BEHAVIORAL}}).json()
        assert body["assessment"]["modalities_used"] == ["behavioral"]

    def test_healthier_habits_score_lower_than_poor_habits(self):
        healthy = {"B1_sleep_hours": 8.0, "B2_study_consistency": 5.0, "B3_social_activity_freq": 4.0,
                   "B4_screen_time_hours": 2.0, "B5_routine_irregularity": 1.0}
        low = post({"behavioral": {"responses": healthy}}).json()["assessment"]["risk_score"]
        high = post({"behavioral": {"responses": BEHAVIORAL}}).json()["assessment"]["risk_score"]
        assert low < high

    def test_out_of_range_behavioral_value_marked_invalid(self):
        bad = dict(BEHAVIORAL, B1_sleep_hours=99)
        body = post({"behavioral": {"responses": bad}}).json()
        assert body["modality_status"]["behavioral"] == "invalid_input"
        assert body["assessment"]["status"] == "fusion_unavailable"

    def test_unknown_questionnaire_answer_does_not_crash(self):
        weird = dict(QUESTIONNAIRE, Q5="Banana")
        response = post({"questionnaire": {"responses": weird}})
        assert response.status_code == 200


class TestRequestValidation:
    def test_empty_request_is_valid_but_unavailable(self):
        body = post({}).json()
        assert body["assessment"]["status"] == "fusion_unavailable"
        assert body["assessment"]["risk_score"] is None
        assert set(body["modality_status"].values()) == {"not_provided"}

    @pytest.mark.parametrize("body", [
        {"questionnaire": {"responses": {"Q1": "x"}}},
        {"behavioral": {"responses": {"bogus": 1}}},
        {"unknown_modality": {}},
        {"voice": {}},
        {"voice": {"features": {}, "audio_base64": "AAAA"}},
        {"questionnaire": "text"},
    ])
    def test_malformed_requests_get_422(self, body):
        assert post(body).status_code == 422

    def test_oversized_audio_rejected(self):
        huge = "A" * ((api_module.MAX_AUDIO_BYTES * 4) // 3 + 100)
        assert post({"voice": {"audio_base64": huge}}).status_code == 422


class TestVoiceHandling:
    def test_untrained_voice_is_unavailable_not_an_error(self, no_voice_model):
        response = post({"voice": {"features": full_feature_dict()}})
        assert response.status_code == 200
        body = response.json()
        assert body["modality_status"]["voice"] == "model_not_trained"
        assert body["assessment"]["status"] == "fusion_unavailable"

    @needs_artifacts
    def test_untrained_voice_does_not_block_other_modalities(self, no_voice_model):
        body = post({"questionnaire": {"responses": QUESTIONNAIRE}, "voice": {"features": full_feature_dict()}}).json()
        assert body["modality_status"]["voice"] == "model_not_trained"
        assert body["assessment"]["modalities_used"] == ["questionnaire"]

    def test_voice_features_with_model(self, voice_model):
        body = post({"voice": {"features": full_feature_dict()}}).json()
        assert body["modality_status"]["voice"] == "available"
        assert body["assessment"]["modalities_used"] == ["voice"]

    @needs_artifacts
    def test_voice_fuses_with_other_modalities(self, voice_model):
        body = post({"questionnaire": {"responses": QUESTIONNAIRE}, "behavioral": {"responses": BEHAVIORAL},
                     "voice": {"features": full_feature_dict()}}).json()
        assert sorted(body["assessment"]["modalities_used"]) == ["behavioral", "questionnaire", "voice"]

    def test_voice_audio_upload(self, voice_model, tmp_path):
        wav = write_test_wav(tmp_path / "clip.wav")
        encoded = base64.b64encode(wav.read_bytes()).decode()
        body = post({"voice": {"audio_base64": encoded}}).json()
        assert body["modality_status"]["voice"] == "available"

    def test_temp_audio_file_is_deleted(self, voice_model, tmp_path, monkeypatch):
        seen = []
        original = api_module.predict_voice_contract_from_audio

        def spy(path):
            seen.append(path)
            return original(path)

        monkeypatch.setattr(api_module, "predict_voice_contract_from_audio", spy)
        encoded = base64.b64encode(write_test_wav(tmp_path / "clip.wav").read_bytes()).decode()
        post({"voice": {"audio_base64": encoded}})
        import os
        assert seen and not os.path.exists(seen[0])

    def test_invalid_base64_marked_invalid(self, voice_model):
        body = post({"voice": {"audio_base64": "!!!not base64!!!"}}).json()
        assert body["modality_status"]["voice"] == "invalid_input"

    def test_corrupt_audio_marked_invalid(self, voice_model):
        encoded = base64.b64encode(b"definitely not a wav").decode()
        body = post({"voice": {"audio_base64": encoded}}).json()
        assert body["modality_status"]["voice"] == "invalid_input"

    def test_unknown_voice_feature_marked_invalid(self, voice_model):
        body = post({"voice": {"features": {"made_up": 1.0}}}).json()
        assert body["modality_status"]["voice"] == "invalid_input"


class TestFailureIsolation:
    def test_unexpected_predictor_error_is_contained(self, monkeypatch):
        def boom(_):
            raise RuntimeError("secret internal detail")

        monkeypatch.setattr(api_module, "predict_questionnaire", boom)
        response = post({"questionnaire": {"responses": QUESTIONNAIRE}})
        assert response.status_code == 200
        assert response.json()["modality_status"]["questionnaire"] == "prediction_failed"
        assert "secret internal detail" not in response.text

    def test_missing_model_files_reported(self, monkeypatch):
        def missing(_):
            raise FileNotFoundError("C:\\private\\models\\x.joblib")

        monkeypatch.setattr(api_module, "predict_behavioral", missing)
        body = post({"behavioral": {"responses": BEHAVIORAL}}).json()
        assert body["modality_status"]["behavioral"] == "model_unavailable"
        assert "private" not in str(body)

    def test_fusion_contract_violation_returns_generic_500(self, monkeypatch):
        monkeypatch.setattr(api_module, "predict_behavioral", lambda _: {"modality": "behavioral"})
        response = post({"behavioral": {"responses": BEHAVIORAL}})
        assert response.status_code == 500
        assert response.json() == {"detail": "Assessment could not be completed."}

    def test_one_failing_modality_does_not_stop_the_other(self, monkeypatch):
        monkeypatch.setattr(api_module, "predict_questionnaire", lambda _: (_ for _ in ()).throw(RuntimeError("x")))
        good = {
            "modality": "behavioral", "risk_score": 0.4, "risk_level": "moderate", "confidence": 0.7,
            "available_features": 5, "total_features": 5, "key_factors": [],
            "data_quality": {"missing_features": 0, "completeness": 1.0},
        }
        monkeypatch.setattr(api_module, "predict_behavioral", lambda _: good)
        body = post({"questionnaire": {"responses": QUESTIONNAIRE}, "behavioral": {"responses": BEHAVIORAL}}).json()
        assert body["modality_status"] == {"questionnaire": "prediction_failed", "behavioral": "available", "voice": "not_provided"}
        assert body["assessment"]["modalities_used"] == ["behavioral"]


class TestVoiceModelProvenanceInApi:
    def test_corrupt_voice_artifact_reports_model_unavailable(self, tmp_path, monkeypatch):
        (tmp_path / "voice_model.joblib").write_bytes(b"garbage")
        monkeypatch.setattr(voice_predict, "MODELS_DIR", tmp_path)
        body = post({"voice": {"features": full_feature_dict()}}).json()
        assert body["modality_status"]["voice"] == "model_unavailable"
        assert "garbage" not in str(body)

    def test_inference_failure_reports_prediction_failed(self, monkeypatch):
        class Broken:
            def predict_proba_dict(self, _):
                raise RuntimeError("boom")

        monkeypatch.setattr(voice_predict, "load_selected_voice_model", lambda: (Broken(), "broken"))
        body = post({"voice": {"features": full_feature_dict()}}).json()
        assert body["modality_status"]["voice"] == "prediction_failed"

    def test_status_summary_without_model(self):
        assert api_module._voice_model_summary(False) == {"trained": False}

    def test_status_summary_from_training_report(self, tmp_path, monkeypatch):
        import json

        report = {"dataset_kind": "synthetic_demo", "held_out_metrics": {"macro_f1": 0.5},
                  "split": {"held_out_test": {"participants": 7}}}
        (tmp_path / voice_predict.TRAINING_REPORT_FILENAME).write_text(json.dumps(report))
        monkeypatch.setattr(voice_predict, "MODELS_DIR", tmp_path)
        summary = api_module._voice_model_summary(True)
        assert summary["served"] is False and summary["held_out_participants"] == 7
        assert "not clinical" in summary["note"]

    def test_status_summary_without_report_flags_unknown_provenance(self, tmp_path, monkeypatch):
        monkeypatch.setattr(voice_predict, "MODELS_DIR", tmp_path)
        assert "unknown" in api_module._voice_model_summary(True)["provenance"]


class TestDashboardAndPrivacy:
    @needs_artifacts
    def test_modality_details_returned_per_source(self):
        body = post({"questionnaire": {"responses": QUESTIONNAIRE}, "behavioral": {"responses": BEHAVIORAL}}).json()
        assert set(body["modality_details"]) == {"questionnaire", "behavioral"}
        for detail in body["modality_details"].values():
            assert set(detail) == {"risk_score", "risk_level", "confidence", "completeness"}
            assert detail["risk_level"] in ("low", "moderate", "high")

    def test_no_details_for_unavailable_modalities(self, no_voice_model):
        body = post({"voice": {"features": full_feature_dict()}}).json()
        assert body["modality_details"] == {}

    def test_evaluation_summary_is_honest(self):
        body = client.get("/evaluation").json()
        assert body["behavioral"]["data"] == "synthetic"
        assert "synthetic" in body["behavioral"]["caveat"].lower()
        assert body["questionnaire"]["respondents"] == 52
        assert "not clinically validated" in body["questionnaire"]["caveat"].lower()
        assert "clinical" in body["disclaimer"]

    def test_responses_are_not_cacheable_and_ui_has_strict_csp(self):
        api_response = client.get("/health")
        assert api_response.headers["cache-control"] == "no-store"
        assert api_response.headers["x-content-type-options"] == "nosniff"
        ui = client.get("/app/")
        csp = ui.headers["content-security-policy"]
        assert "script-src 'self'" in csp and "unsafe-inline" not in csp
        assert "content-security-policy" not in client.get("/docs").headers  # Swagger UI needs its CDN assets

    def test_ui_assets_are_served_and_ui_has_no_inline_script(self):
        page = client.get("/app/").text
        assert '<script src="app.js">' in page and "<script>" not in page
        assert client.get("/app/app.js").status_code == 200
        assert client.get("/app/style.css").status_code == 200

    def test_ui_states_privacy_and_non_clinical_purpose(self):
        page = client.get("/app/").text.lower()
        for phrase in ("not a medical tool", "not stored", "consent", "deleted immediately"):
            assert phrase in page

    def test_ui_never_uses_innerhtml_with_server_data(self):
        assert "innerhtml" not in client.get("/app/app.js").text.lower()

    def test_user_answers_are_not_logged(self, caplog):
        secret = {"Q1": "SECRET-ANSWER-123"}
        caplog.set_level("DEBUG")
        post({"questionnaire": {"responses": dict(QUESTIONNAIRE, **secret)}})
        assert "SECRET-ANSWER-123" not in caplog.text
