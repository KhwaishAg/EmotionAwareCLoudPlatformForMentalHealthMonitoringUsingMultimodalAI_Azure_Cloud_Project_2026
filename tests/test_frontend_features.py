"""The redesigned page must keep every feature added by the optional Azure work, and the personal explanations
must cover the speech-tone result and the per-stressor tips."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.ai_model.fusion.explanations import (
    STRESSOR_TIPS,
    build_recommendations,
    describe_feature,
    personal_insights,
    voice_insights,
)
from src.backend.assessment import api as api_module

FRONTEND = Path(__file__).resolve().parents[1] / "src" / "frontend"
HTML = (FRONTEND / "index.html").read_text(encoding="utf-8")
JS = (FRONTEND / "app.js").read_text(encoding="utf-8")
client = TestClient(api_module.app)

SPEECH_VOICE = {"source": "azure_speech_language", "risk_level": "high", "risk_score": 0.82, "sentiment": "negative",
                "transcript": "I have been so overwhelmed with everything this month", "key_phrases": ["overwhelmed"]}


class TestCloudFeaturesSurviveTheRedesign:
    @pytest.mark.parametrize("element_id", ["privacy-where", "privacy-tail", "cloud-speech-note", "history-note", "voice-help", "cloud-speech-opt",
                                             "use-cloud-speech", "history-opt", "save-history", "history-card", "h-trend", "h-rows",
                                             "h-refresh", "h-delete", "h-status", "h-id-note"])
    def test_page_has_element(self, element_id):
        assert f'id="{element_id}"' in HTML

    @pytest.mark.parametrize("element_id", ["cloud-speech-note", "history-note", "cloud-speech-opt", "history-opt", "history-card"])
    def test_optional_features_are_hidden_until_switched_on(self, element_id):
        assert re.search(rf'id="{element_id}"[^>]*\shidden', HTML)

    @pytest.mark.parametrize("token", ["cloud_speech", "body.history", '"/history"', "anon_id", "checkin-anon-id", "azure_speech_language",
                                       "no_speech_detected", "speech_service_unavailable", "history_status", "d.cloud", "speech_content", "signed_in_history"])
    def test_script_keeps_feature(self, token):
        assert token in JS

    def test_hosted_privacy_wording_is_kept(self):
        assert "Microsoft Azure App Service" in JS

    def test_consent_text_still_states_the_essentials(self):
        text = HTML.lower()
        for phrase in ("not stored", "deleted immediately", "not a medical tool", "consent"):
            assert phrase in text

    def test_history_card_is_served_and_status_exposes_cloud_flags(self):
        assert client.get("/app/").status_code == 200
        assert set(client.get("/status").json()["cloud"]) >= {"hosted", "speech_content", "history", "events", "signed_in_history"}

    def test_new_api_fields_do_not_break_history_and_cloud_responses(self):
        body = client.post("/assessment", json={}).json()
        assert {"insights", "recommendations", "history_status", "modality_details"} <= set(body)


class TestSpeechToneAndStressorTips:
    def test_speech_factors_have_readable_labels(self):
        assert describe_feature("speech_negative_tone")["label"] == "Negative tone in what you said (voice)"
        assert describe_feature("speech_negative_tone")["themes"] == ["speech_tone"]
        assert "_" not in describe_feature("speech_positive_tone")["label"]

    def test_strained_speech_creates_a_tone_insight_without_the_transcript(self):
        (insight,) = voice_insights(SPEECH_VOICE)
        assert insight["theme"] == "speech_tone" and "negative" in insight["evidence"]
        assert "overwhelmed with everything" not in insight["evidence"]  # the transcript is never copied into an insight

    @pytest.mark.parametrize("voice", [None, {}, {**SPEECH_VOICE, "risk_level": "low"}, {**SPEECH_VOICE, "source": "acoustic"}])
    def test_no_tone_insight_otherwise(self, voice):
        assert voice_insights(voice) == []

    def test_tone_recommendation_is_offered(self):
        insights = personal_insights(None, None, SPEECH_VOICE)
        recs = build_recommendations(insights, {"risk_level": "moderate", "key_factors": [], "consistency": {"pattern": "none"}})
        assert recs[0]["theme"] == "speech_tone" and "someone you trust" in recs[0]["advice"]

    def test_selected_stressors_bring_their_specific_tips(self):
        insights = personal_insights({"Q10": "Exams; Financial concerns"}, None)
        recs = {r["theme"]: r for r in build_recommendations(insights, {"risk_level": "moderate", "key_factors": [], "consistency": {"pattern": "none"}})}
        assert STRESSOR_TIPS["Exams"] in recs["exams"]["tips"]
        assert STRESSOR_TIPS["Financial_concerns"] in recs["financial"]["tips"]

    def test_tips_are_capped_and_absent_when_nothing_matches(self):
        many = "Exams; Coding or technical assessments; Assignments; Projects; Academic workload"
        recs = build_recommendations(personal_insights({"Q10": many}, None), {"risk_level": "moderate", "key_factors": [], "consistency": {"pattern": "none"}})
        assert all(len(r.get("tips", [])) <= 2 for r in recs)
        plain = build_recommendations(personal_insights(None, {"B1_sleep_hours": 4, "B2_study_consistency": 5, "B3_social_activity_freq": 5,
                                                              "B4_screen_time_hours": 1, "B5_routine_irregularity": 1}),
                                      {"risk_level": "moderate", "key_factors": [], "consistency": {"pattern": "none"}})
        assert plain[0]["tips"] == []

    def test_every_stressor_tip_is_real_text_without_em_dashes(self):
        assert all(tip and "—" not in tip for tip in STRESSOR_TIPS.values())

    def test_api_returns_tips_and_a_tone_recommendation_through_the_real_endpoint(self, monkeypatch):
        monkeypatch.setattr(api_module, "_run_voice", lambda payload: ({
            "modality": "voice", "risk_score": 0.82, "risk_level": "high", "confidence": 0.7, "available_features": 2, "total_features": 2,
            "data_quality": {"missing_features": 0, "completeness": 1.0},
            "key_factors": [{"feature": "speech_negative_tone", "contribution": 0.8, "direction": "increases_risk"}],
            "source": "azure_speech_language", "sentiment": "negative", "transcript": "I feel overwhelmed", "key_phrases": [], "caveat": "x"}, None))
        body = client.post("/assessment", json={"voice": {"features": {}}}).json()
        assert body["assessment"]["status"] == "available"
        assert "speech_tone" in [r["theme"] for r in body["recommendations"]]
        assert body["assessment"]["key_factors"][0]["label"] == "Negative tone in what you said (voice)"
        assert body["modality_details"]["voice"]["transcript"] == "I feel overwhelmed"
