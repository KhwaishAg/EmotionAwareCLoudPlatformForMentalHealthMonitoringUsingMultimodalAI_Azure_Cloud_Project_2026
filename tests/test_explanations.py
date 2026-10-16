"""Readable factor names, personal insights and personalized recommendations."""
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.ai_model.fusion.explanations import (
    ADVICE,
    MAX_RECOMMENDATIONS,
    behavioral_insights,
    build_recommendations,
    describe_feature,
    label_key_factors,
    personal_insights,
    questionnaire_insights,
    theme_importance,
)
from src.ai_model.voice.features import COMBINED_FEATURE_NAMES
from src.backend.assessment import api as api_module
from tests.test_assessment_api import BEHAVIORAL, QUESTIONNAIRE, needs_artifacts

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(api_module.app)

HEALTHY_HABITS = {"B1_sleep_hours": 8.0, "B2_study_consistency": 5.0, "B3_social_activity_freq": 4.0,
                  "B4_screen_time_hours": 2.0, "B5_routine_irregularity": 1.0}
CALM_QUESTIONNAIRE = dict(QUESTIONNAIRE, Q3="2–4 hours", Q4="1 — Not at all", Q5="Never", Q6="Never", Q7="Not difficult at all",
                          Q8="Never", Q9="Never", Q10="Nothing", Q11="Calm; Happy", Q12="1", Q13="5", Q14="Less than 5 minutes",
                          Q15="Never", Q16="Within a few minutes", Q17="Never", Q18="Never", Q19="Never", Q20="Never")


def assessment(level="moderate", factors=(), pattern="none"):
    return {"risk_level": level, "key_factors": list(factors), "consistency": {"pattern": pattern}}


class TestReadableLabels:
    def all_feature_names(self):
        q = json.loads((ROOT / "models/questionnaire/questionnaire_feature_names.json").read_text())["feature_names"]
        b = json.loads((ROOT / "models/behavioral/behavioral_feature_names.json").read_text())["feature_names"]
        return q + b + list(COMBINED_FEATURE_NAMES)

    def test_every_real_feature_has_a_human_label(self):
        for name in self.all_feature_names():
            label = describe_feature(name)["label"]
            assert label and label != name, name
            assert "_" not in label and "encoded" not in label.lower(), (name, label)

    @pytest.mark.parametrize("name,expected", [
        ("Q9_encoded", "Stress affecting your sleep"),
        ("B1_sleep_hours", "Hours of sleep per night"),
        ("Q10_Lack_of_sleep", "Stressor you selected: lack of sleep"),
        ("Q11_Anxious", "Feeling anxious recently"),
    ])
    def test_known_labels(self, name, expected):
        assert describe_feature(name)["label"] == expected

    def test_engineered_scores_are_explained_not_just_renamed(self):
        label = describe_feature("stress_impact_dimension")["label"]
        assert "relaxing" in label and "sleep" in label
        assert set(describe_feature("stress_impact_dimension")["themes"]) == {"relaxation", "focus", "sleep"}

    def test_unknown_names_still_get_a_readable_fallback(self):
        assert describe_feature("some_new_feature_encoded")["label"] == "Some new feature"

    def test_label_key_factors_keeps_the_raw_name_and_adds_a_label(self):
        out = label_key_factors([{"feature": "Q9_encoded", "contribution": 0.5, "direction": "model_associated", "modality": "questionnaire"}])
        assert out[0]["label"] == "Stress affecting your sleep" and out[0]["feature"] == "Q9_encoded"


class TestPersonalInsights:
    def test_sleep_insight_only_when_sleep_is_adverse(self):
        assert "sleep" in [i["theme"] for i in personal_insights(None, dict(HEALTHY_HABITS, B1_sleep_hours=4.5))]
        assert "sleep" not in [i["theme"] for i in personal_insights(None, HEALTHY_HABITS)]

    def test_healthy_user_has_no_insights(self):
        assert personal_insights(CALM_QUESTIONNAIRE, HEALTHY_HABITS) == []

    @pytest.mark.parametrize("hours,expected", [(4.0, 1.0), (5.5, 0.75), (6.5, 0.5), (7.0, None), (9.0, None)])
    def test_sleep_hours_severity(self, hours, expected):
        found = [i for i in behavioral_insights({"B1_sleep_hours": hours}) if i["theme"] == "sleep"]
        assert (found[0]["severity"] if found else None) == expected

    def test_screen_time_and_routine_and_social(self):
        themes = {i["theme"] for i in behavioral_insights(dict(HEALTHY_HABITS, B4_screen_time_hours=9, B5_routine_irregularity=5,
                                                                B2_study_consistency=1, B3_social_activity_freq=1))}
        assert themes == {"screen_time", "routine", "social"}

    def test_questionnaire_sleep_answer_drives_sleep_theme(self):
        bad = personal_insights(dict(CALM_QUESTIONNAIRE, Q9="Always"), None)
        assert [i["theme"] for i in bad] == ["sleep"]
        assert "Always" in bad[0]["evidence"][0]  # the person's own answer is quoted back

    def test_sleep_evidence_from_both_sources_is_merged(self):
        merged = personal_insights(dict(CALM_QUESTIONNAIRE, Q9="Often"), dict(HEALTHY_HABITS, B1_sleep_hours=5.0))
        sleep = next(i for i in merged if i["theme"] == "sleep")
        assert set(sleep["sources"]) == {"questionnaire", "behavioral"} and len(sleep["evidence"]) == 2

    def test_selected_stressors_create_themes(self):
        themes = {i["theme"] for i in personal_insights(dict(CALM_QUESTIONNAIRE, Q10="Financial concerns; Exams; Fear of not getting a job"), None)}
        assert {"financial", "exams", "career"} <= themes

    def test_negative_feelings_raise_emotional_theme_but_positive_ones_do_not(self):
        assert "emotional" in [i["theme"] for i in personal_insights(dict(CALM_QUESTIONNAIRE, Q11="Anxious; Sad; Tired"), None)]
        assert "emotional" not in [i["theme"] for i in personal_insights(dict(CALM_QUESTIONNAIRE, Q11="Calm; Happy; Motivated"), None)]

    def test_results_are_sorted_strongest_first_and_thresholded(self):
        insights = personal_insights(dict(CALM_QUESTIONNAIRE, Q9="Always", Q8="Often"), None)
        severities = [i["severity"] for i in insights]
        assert severities == sorted(severities, reverse=True) and min(severities) >= 0.5

    def test_missing_or_odd_answers_do_not_crash(self):
        assert personal_insights({"Q9": None}, {"B1_sleep_hours": None, "B4_screen_time_hours": "lots"}) == []


class TestRecommendations:
    def test_advice_follows_the_persons_answers_not_just_global_importance(self):
        sleep_important = [{"feature": "B1_sleep_hours", "contribution": 0.9, "direction": "decreases_risk"}]
        poor = build_recommendations(personal_insights(None, dict(HEALTHY_HABITS, B1_sleep_hours=4)), assessment(factors=sleep_important))
        fine = build_recommendations(personal_insights(None, dict(HEALTHY_HABITS, B4_screen_time_hours=9)), assessment(factors=sleep_important))
        assert "sleep" in [r["theme"] for r in poor]
        assert "sleep" not in [r["theme"] for r in fine]   # sleep is an important model feature here, but THIS user sleeps fine
        assert "screen_time" in [r["theme"] for r in fine]

    def test_different_people_get_different_recommendations(self):
        a = build_recommendations(personal_insights(None, dict(HEALTHY_HABITS, B1_sleep_hours=4)), assessment())
        b = build_recommendations(personal_insights(None, dict(HEALTHY_HABITS, B4_screen_time_hours=10, B3_social_activity_freq=1)), assessment())
        assert [r["theme"] for r in a] != [r["theme"] for r in b]

    def test_each_recommendation_says_why_using_the_persons_own_evidence(self):
        recs = build_recommendations(personal_insights(None, dict(HEALTHY_HABITS, B1_sleep_hours=4)), assessment())
        sleep = next(r for r in recs if r["theme"] == "sleep")
        assert sleep["advice"] == ADVICE["sleep"] and "4 hours of sleep" in sleep["because"][0]

    def test_model_importance_breaks_ties_between_equally_severe_areas(self):
        insights = personal_insights(None, dict(HEALTHY_HABITS, B1_sleep_hours=4, B4_screen_time_hours=10))  # both severity 1.0
        screen_important = [{"feature": "B4_screen_time_hours", "contribution": 0.9, "direction": "increases_risk"}]
        sleep_important = [{"feature": "B1_sleep_hours", "contribution": 0.9, "direction": "decreases_risk"}]
        assert build_recommendations(insights, assessment(factors=screen_important))[0]["theme"] == "screen_time"
        assert build_recommendations(insights, assessment(factors=sleep_important))[0]["theme"] == "sleep"

    def test_severity_beats_model_importance(self):
        insights = personal_insights(None, dict(HEALTHY_HABITS, B1_sleep_hours=4, B4_screen_time_hours=5.5))  # sleep 1.0 vs screen 0.5
        factors = [{"feature": "B4_screen_time_hours", "contribution": 1.0, "direction": "increases_risk"}]
        assert build_recommendations(insights, assessment(factors=factors))[0]["theme"] == "sleep"

    def test_at_most_four_specific_recommendations(self):
        everything = personal_insights(dict(CALM_QUESTIONNAIRE, Q5="Always", Q6="Always", Q7="Extremely difficult", Q8="Always", Q9="Always",
                                            Q12="5", Q13="1", Q17="Always", Q18="Always", Q19="Always", Q20="Always"),
                                       dict(B1_sleep_hours=3, B2_study_consistency=1, B3_social_activity_freq=1, B4_screen_time_hours=12, B5_routine_irregularity=5))
        specific = [r for r in build_recommendations(everything, assessment("high")) if r["theme"] not in {"support"}]
        assert len(specific) == MAX_RECOMMENDATIONS

    def test_masked_stress_recommendation_only_when_flagged(self):
        flagged = build_recommendations([], assessment("moderate", pattern="masked_stress_possible"))
        assert flagged[0]["theme"] == "masked_stress"
        assert "masked_stress" not in [r["theme"] for r in build_recommendations([], assessment("moderate"))]
        higher = build_recommendations([], assessment("moderate", pattern="self_report_higher"))
        assert higher[0]["theme"] == "self_report_higher"

    def test_support_line_depends_on_overall_level(self):
        assert not [r for r in build_recommendations([], assessment("low")) if r["theme"] == "support"]
        high = [r for r in build_recommendations([], assessment("high")) if r["theme"] == "support"][0]
        assert "crisis" in high["advice"]
        assert "counselling" in [r for r in build_recommendations([], assessment("moderate")) if r["theme"] == "support"][0]["advice"]

    def test_fallback_when_nothing_stands_out(self):
        low = build_recommendations([], assessment("low"))
        assert low[0]["theme"] == "general" and "No strong stress signals" in low[0]["advice"]

    def test_no_recommendations_without_an_assessment(self):
        assert build_recommendations([], {"risk_level": None, "key_factors": []}) == []

    def test_theme_importance_sums_matching_factors_and_caps_at_one(self):
        factors = [{"feature": "stress_impact_dimension", "contribution": 0.7}, {"feature": "Q9_encoded", "contribution": 0.6}]
        importance = theme_importance(factors)
        assert importance["sleep"] == 1.0 and importance["focus"] == pytest.approx(0.7)

    def test_all_advice_is_non_clinical(self):
        text = " ".join(ADVICE.values()).lower()
        assert not re.search(r"\b(diagnos|disorder|depress(ed|ion)|medicat|prescri|therap(y|ist) for)\b", text)


@needs_artifacts
class TestApiIntegration:
    def post(self, body):
        return client.post("/assessment", json=body).json()

    def test_response_has_readable_labels_insights_and_recommendations(self):
        body = self.post({"questionnaire": {"responses": QUESTIONNAIRE}, "behavioral": {"responses": BEHAVIORAL}})
        for factor in body["assessment"]["key_factors"]:
            assert "_" not in factor["label"] and "encoded" not in factor["label"].lower()
        assert body["insights"] and body["recommendations"]
        assert all(r["advice"] and "advice" in r for r in body["recommendations"])

    def test_recommendations_change_with_the_users_answers(self):
        poor = self.post({"behavioral": {"responses": dict(HEALTHY_HABITS, B1_sleep_hours=4)}})
        healthy = self.post({"behavioral": {"responses": HEALTHY_HABITS}})
        assert "sleep" in [r["theme"] for r in poor["recommendations"]]
        assert "sleep" not in [r["theme"] for r in healthy["recommendations"]]

    def test_single_source_reports_agreement_not_applicable(self):
        body = self.post({"behavioral": {"responses": BEHAVIORAL}})["assessment"]
        assert body["agreement"]["level"] == "not_applicable" and body["consistency"]["status"] == "single_source"

    def test_two_real_sources_can_show_masked_stress(self):
        body = self.post({"questionnaire": {"responses": CALM_QUESTIONNAIRE}, "behavioral": {"responses": BEHAVIORAL}})
        assert body["assessment"]["consistency"]["pattern"] == "masked_stress_possible"
        assert body["recommendations"][0]["theme"] == "masked_stress"
        only_questionnaire = self.post({"questionnaire": {"responses": CALM_QUESTIONNAIRE}})
        assert only_questionnaire["assessment"]["consistency"]["pattern"] == "none"

    def test_no_insights_when_nothing_could_be_assessed(self):
        body = self.post({})
        assert body["insights"] == [] and body["recommendations"] == []

    def test_invalid_input_does_not_produce_explanations_for_it(self):
        body = self.post({"behavioral": {"responses": dict(BEHAVIORAL, B1_sleep_hours=99)}})
        assert body["insights"] == []

    def test_user_answers_are_not_logged_when_explaining(self, caplog):
        caplog.set_level("DEBUG")
        self.post({"questionnaire": {"responses": dict(QUESTIONNAIRE, Q1="4th Year")}, "behavioral": {"responses": BEHAVIORAL}})
        assert "non-academic screen time" not in caplog.text


class TestFrontendUsesServerExplanations:
    def test_no_hardcoded_advice_left_in_the_ui(self):
        source = client.get("/app/app.js").text
        assert "ADVICE" not in source and "Prioritise sleep, food" not in source
        assert "r-advice" in source and "data.recommendations" in source and "data.insights" in source

    def test_ui_shows_readable_labels_and_not_applicable_agreement(self):
        source = client.get("/app/app.js").text
        assert "f.label" in source and "not applicable" in source and "masked_stress_possible" in source
        assert "innerhtml" not in source.lower()

    def test_page_has_personal_sections(self):
        page = client.get("/app/").text
        for text in ("What stood out in your answers", "Suggestions for you", "Do your sources agree?", "the same for everyone"):
            assert text in page
