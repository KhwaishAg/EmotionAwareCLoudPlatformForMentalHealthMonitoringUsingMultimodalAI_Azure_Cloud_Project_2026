"""Optional free-tier Azure integrations (src/cloud). No test touches the network: urlopen is faked."""
import base64
import io
import json
import uuid

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.ai_model.fusion.fusion_engine import fuse_modalities, validate_modality_result
from src.backend.assessment import api as api_module
from src.cloud import cosmos, events, history, http, language, speech, voice_content
from src.cloud.config import load_settings

PAYLOAD = json.load(open("assessment_payload.json", encoding="utf-8"))
SPEECH_ENV = {"CLOUD_SPEECH_KEY": "k", "CLOUD_SPEECH_REGION": "centralindia", "CLOUD_LANGUAGE_KEY": "k",
              "CLOUD_LANGUAGE_ENDPOINT": "https://lang.example.test/"}
COSMOS_ENV = {"CLOUD_COSMOS_ENDPOINT": "https://db.example.test:443/", "CLOUD_COSMOS_KEY": base64.b64encode(b"secret").decode()}
EVENTS_ENV = {"CLOUD_EVENTGRID_ENDPOINT": "https://topic.example.test/api/events", "CLOUD_EVENTGRID_KEY": "k",
              "CLOUD_EVENTS_SECRET": "hook-secret"}
SENTIMENT = {"label": "negative", "scores": {"positive": 0.05, "neutral": 0.15, "negative": 0.8}}
TRANSCRIPT = "I have three deadlines this week and I am barely sleeping and I feel like I am falling behind everyone"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith(("CLOUD_", "APP_ACCESS_")):
            monkeypatch.delenv(name)


def set_env(monkeypatch, *groups):
    for group in groups:
        for key, value in group.items():
            monkeypatch.setenv(key, value)


def wav_base64(seconds=6.0, rate=44100, channels=2):
    t = np.linspace(0, seconds, int(seconds * rate), endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    buffer = io.BytesIO()
    sf.write(buffer, np.stack([tone] * channels, axis=1), rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode()


class FakeResponse:
    def __init__(self, payload, status=200):
        self.status, self._raw = status, json.dumps(payload).encode() if payload is not None else b""

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_net(monkeypatch):
    calls, replies = [], []

    def urlopen(req, timeout=None):
        calls.append(req)
        reply = replies.pop(0) if replies else None
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply)

    monkeypatch.setattr(http.urllib.request, "urlopen", urlopen)
    return calls, replies


# ---------------------------------------------------------------- voice content contract

class TestVoiceContent:
    def test_contract_is_fusion_valid_and_conservative(self):
        contract = voice_content.build_contract(TRANSCRIPT, SENTIMENT, ["deadlines"])
        validate_modality_result(contract)
        assert contract["risk_score"] == pytest.approx(0.8 + 0.25 * 0.15)
        assert contract["risk_level"] == "high"
        assert contract["confidence"] <= voice_content.MAX_CONFIDENCE
        fused = fuse_modalities(voice=contract)
        assert fused["status"] == "available"

    def test_positive_speech_is_low_and_short_speech_is_less_confident(self):
        happy = {"label": "positive", "scores": {"positive": 0.9, "neutral": 0.1, "negative": 0.0}}
        long_one = voice_content.build_contract(TRANSCRIPT, happy)
        short_one = voice_content.build_contract("I am fine really", happy)
        assert long_one["risk_level"] == "low"
        assert short_one["confidence"] < long_one["confidence"]

    def test_too_little_speech_is_rejected(self):
        with pytest.raises(voice_content.NoSpeechError):
            voice_content.build_contract("um ok", SENTIMENT)

    def test_audio_is_converted_to_16k_mono_and_trimmed(self, tmp_path):
        path = tmp_path / "in.wav"
        path.write_bytes(base64.b64decode(wav_base64(seconds=70)))
        data, rate = sf.read(io.BytesIO(speech.to_pcm16_wav(path)))
        assert rate == 16000 and data.ndim == 1 and len(data) <= 59 * 16000


# ---------------------------------------------------------------- REST clients

class TestClients:
    def test_transcribe_parses_detailed_result(self, fake_net):
        calls, replies = fake_net
        replies.append({"RecognitionStatus": "Success", "NBest": [{"Display": "Hello there.", "Confidence": 0.9}]})
        out = speech.transcribe(b"wav", "key", "centralindia")
        assert out == {"text": "Hello there.", "confidence": 0.9}
        assert calls[0].full_url.startswith("https://centralindia.stt.speech.microsoft.com/")
        assert calls[0].get_header("Ocp-apim-subscription-key") == "key"

    def test_transcribe_no_match_is_empty(self, fake_net):
        fake_net[1].append({"RecognitionStatus": "NoMatch"})
        assert speech.transcribe(b"wav", "k", "r")["text"] == ""

    def test_sentiment_and_key_phrases(self, fake_net):
        calls, replies = fake_net
        replies.append({"results": {"documents": [{"sentiment": "negative", "confidenceScores": SENTIMENT["scores"]}]}})
        replies.append({"results": {"documents": [{"keyPhrases": ["deadlines", "sleep"]}]}})
        assert language.sentiment("text", "k", "https://lang.example.test/") == SENTIMENT
        assert language.key_phrases("text", "k", "https://lang.example.test/") == ["deadlines", "sleep"]
        body = json.loads(calls[0].data)
        assert body["kind"] == "SentimentAnalysis" and calls[0].full_url.endswith("api-version=2023-04-01")

    def test_http_errors_hide_details(self, fake_net):
        fake_net[1].append(http.urllib.error.URLError("boom"))
        with pytest.raises(http.CloudServiceError) as error:
            speech.transcribe(b"wav", "secret-key", "r")
        assert "secret-key" not in str(error.value)

    def test_cosmos_signs_requests_with_partition_key(self, fake_net):
        calls, replies = fake_net
        replies.append({"Documents": [{"id": "1", "owner": "anon-x", "_rid": "r"}]})
        container = cosmos.CosmosContainer("https://db.example.test:443/", COSMOS_ENV["CLOUD_COSMOS_KEY"], "d", "c")
        rows = container.query("SELECT * FROM c", [], "anon-x")
        assert rows[0]["id"] == "1"
        req = calls[0]
        assert req.full_url == "https://db.example.test:443/dbs/d/colls/c/docs"
        assert req.get_header("Authorization").startswith("type%3Dmaster%26ver%3D1.0%26sig%3D")
        assert json.loads(req.get_header("X-ms-documentdb-partitionkey")) == ["anon-x"]

    def test_event_grid_publish_shape(self, fake_net):
        calls, _ = fake_net
        events.publish("https://topic.example.test/api/events", "k", events.ASSESSMENT_COMPLETED, "/a", {"x": 1})
        sent = json.loads(calls[0].data)
        assert sent[0]["eventType"] == events.ASSESSMENT_COMPLETED and sent[0]["data"] == {"x": 1}
        assert calls[0].get_header("Aeg-sas-key") == "k"


# ---------------------------------------------------------------- history helpers

class TestHistory:
    def test_owner_from_anonymous_id_or_trusted_login(self, monkeypatch):
        anon = str(uuid.uuid4())
        assert history.resolve_owner(anon, None, load_settings()) == "anon-" + anon
        assert history.resolve_owner("not-a-uuid", None, load_settings()) is None
        # identity header ignored unless Easy Auth is declared trusted
        assert history.resolve_owner(anon, "user-123", load_settings()) == "anon-" + anon
        monkeypatch.setenv("CLOUD_TRUST_EASY_AUTH", "1")
        assert history.resolve_owner(anon, "user-123", load_settings()).startswith("user-")

    def test_record_contains_no_raw_inputs(self):
        fused = TestClient(api_module.app).post("/assessment", json=PAYLOAD).json()["assessment"]
        record = history.validate_record(history.build_record("anon-" + str(uuid.uuid4()), fused))
        text = json.dumps(record)
        assert "4th Year" not in text and "B1_sleep_hours\": 5" not in text
        assert set(record) == history.RECORD_FIELDS

    def test_trend(self):
        rows = [{"risk_score": s} for s in (0.8, 0.4, 0.5, 0.45)]
        assert history.summarise_trend(rows)["direction"] == "rising"
        assert history.summarise_trend(rows[::-1][:2])["direction"] in ("falling", "stable", "rising")
        assert history.summarise_trend(rows[:1])["direction"] == "not_enough_data"


# ---------------------------------------------------------------- API integration

@pytest.fixture
def client():
    return TestClient(api_module.app)


def voice_payload(cloud_speech=True):
    return {"voice": {"audio_base64": wav_base64(), "cloud_speech": cloud_speech}}


class TestApiSpeech:
    def test_speech_content_feeds_fusion_when_opted_in(self, client, monkeypatch):
        set_env(monkeypatch, SPEECH_ENV)
        monkeypatch.setattr(api_module, "assess_speech_content",
                            lambda path, settings: voice_content.build_contract(TRANSCRIPT, SENTIMENT, ["deadlines"]))
        data = client.post("/assessment", json={**PAYLOAD, **voice_payload()}).json()
        assert data["modality_status"]["voice"] == "available"
        assert "voice" in data["assessment"]["modalities_used"]
        detail = data["modality_details"]["voice"]
        assert detail["source"] == "azure_speech_language" and detail["transcript"] == TRANSCRIPT
        assert data["assessment"]["agreement"]["sources"] == 3

    def test_not_used_without_opt_in_or_settings(self, client, monkeypatch):
        called = []
        monkeypatch.setattr(api_module, "assess_speech_content", lambda *a: called.append(1))
        assert client.post("/assessment", json=voice_payload()).json()["modality_status"]["voice"] == "model_not_trained"
        set_env(monkeypatch, SPEECH_ENV)
        assert client.post("/assessment", json=voice_payload(False)).json()["modality_status"]["voice"] == "model_not_trained"
        assert called == []

    def test_service_failure_is_reported_not_raised(self, client, monkeypatch):
        set_env(monkeypatch, SPEECH_ENV)

        def fail(path, settings):
            raise http.CloudServiceError("speech", 503)

        monkeypatch.setattr(api_module, "assess_speech_content", fail)
        data = client.post("/assessment", json={**PAYLOAD, **voice_payload()}).json()
        assert data["modality_status"]["voice"] == "speech_service_unavailable"
        assert data["assessment"]["status"] == "available"  # the other sources still work

    def test_status_reports_cloud_flags_without_secrets(self, client, monkeypatch):
        set_env(monkeypatch, SPEECH_ENV, COSMOS_ENV)
        body = client.get("/status").json()
        assert body["cloud"]["speech_content"] and body["cloud"]["history"] and not body["cloud"]["events"]
        assert "lang.example.test" not in json.dumps(body)


class TestApiHistory:
    def test_saved_directly_when_no_event_grid(self, client, monkeypatch):
        set_env(monkeypatch, COSMOS_ENV)
        saved = []
        monkeypatch.setattr(history, "save", lambda record, settings: saved.append(record))
        data = client.post("/assessment", json={**PAYLOAD, "history": {"anon_id": str(uuid.uuid4())}}).json()
        assert data["history_status"] == "saved" and saved[0]["risk_level"] == data["assessment"]["risk_level"]

    def test_queued_through_event_grid(self, client, monkeypatch):
        set_env(monkeypatch, COSMOS_ENV, EVENTS_ENV)
        published = []
        monkeypatch.setattr(events, "publish", lambda *a: published.append(a))
        data = client.post("/assessment", json={**PAYLOAD, "history": {"anon_id": str(uuid.uuid4())}}).json()
        assert data["history_status"] == "queued" and published[0][2] == events.ASSESSMENT_COMPLETED

    def test_opt_in_states(self, client, monkeypatch):
        assert client.post("/assessment", json=PAYLOAD).json()["history_status"] is None
        assert client.post("/assessment", json={**PAYLOAD, "history": {"anon_id": str(uuid.uuid4())}}).json()["history_status"] == "unavailable"
        set_env(monkeypatch, COSMOS_ENV)
        assert client.post("/assessment", json={**PAYLOAD, "history": {"anon_id": "bad"}}).json()["history_status"] == "invalid_id"

    def test_webhook_secret_validation_and_storage(self, client, monkeypatch):
        set_env(monkeypatch, COSMOS_ENV, EVENTS_ENV)
        saved = []
        monkeypatch.setattr(history, "save", lambda record, settings: saved.append(record))
        assert client.post("/events?key=wrong", json=[]).status_code == 401
        handshake = [{"eventType": events.SUBSCRIPTION_VALIDATION, "data": {"validationCode": "abc"}}]
        assert client.post("/events?key=hook-secret", json=handshake).json() == {"validationResponse": "abc"}
        fused = client.post("/assessment", json=PAYLOAD).json()["assessment"]
        record = history.build_record("anon-" + str(uuid.uuid4()), fused)
        good = [{"eventType": events.ASSESSMENT_COMPLETED, "data": record},
                {"eventType": events.ASSESSMENT_COMPLETED, "data": {**record, "answers": "leak"}}]
        assert client.post("/events?key=hook-secret", json=good).json() == {"stored": 1}
        assert saved == [record]

    def test_webhook_bypasses_shared_password_but_nothing_else_does(self, client, monkeypatch):
        set_env(monkeypatch, EVENTS_ENV, {"APP_ACCESS_USER": "student", "APP_ACCESS_PASSWORD": "x" * 12})
        assert client.get("/status").status_code == 401
        assert client.post("/events?key=hook-secret", json=[]).status_code == 200

    def test_read_and_delete_history(self, client, monkeypatch):
        anon = str(uuid.uuid4())
        assert client.get("/history", params={"anon_id": anon}).status_code == 404  # disabled by default
        set_env(monkeypatch, COSMOS_ENV)
        rows = [{"id": str(uuid.uuid4()), "created_at": "2026-10-16T10:00:00Z", "risk_score": 0.7, "risk_level": "high"},
                {"id": str(uuid.uuid4()), "created_at": "2026-10-15T10:00:00Z", "risk_score": 0.3, "risk_level": "low"}]
        seen = []
        monkeypatch.setattr(history, "list_for", lambda owner, settings: seen.append(owner) or rows)
        monkeypatch.setattr(history, "delete_all", lambda owner, settings: 2)
        body = client.get("/history", params={"anon_id": anon}).json()
        assert seen == ["anon-" + anon] and body["trend"]["direction"] == "rising"
        assert client.delete("/history", params={"anon_id": anon}).json() == {"deleted": 2}
        assert client.get("/history", params={"anon_id": "../etc"}).status_code == 400


def test_deploy_bundle_ships_the_cloud_package():
    import importlib.util
    import pathlib
    import zipfile

    spec = importlib.util.spec_from_file_location("package_for_azure", pathlib.Path("scripts/package_for_azure.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = pathlib.Path("dist/test_bundle.zip")
    try:
        module.build_bundle(out)
        names = zipfile.ZipFile(out).namelist()
        assert "src/cloud/voice_content.py" in names and "src/cloud/http.py" in names
    finally:
        out.unlink(missing_ok=True)
