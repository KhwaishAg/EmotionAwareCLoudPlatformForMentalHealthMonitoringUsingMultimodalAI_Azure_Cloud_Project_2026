"""User-facing explanations: readable factor names, personal insights and personalized recommendations.

Why this exists
    The model's ``key_factors`` are MODEL-WIDE importances: the same few features for every user, under raw names such
    as ``Q9_encoded`` or ``stress_impact_dimension``. They cannot, on their own, say what matters for THIS person. So:

    1. ``describe_feature`` gives every feature a plain-language label (and the themes it relates to).
    2. ``personal_insights`` reads the person's OWN answers and finds the areas that look adverse (sleep, workload,
       focus, screen time ...), with the evidence quoted back ("You said stress affects your sleep: 'Often'").
    3. ``build_recommendations`` ranks those areas by how adverse the answers are, weighted by how much the model relies
       on that area (the fused ``key_factors``), and attaches advice for each. Sleep advice therefore appears only when
       sleep answers are actually adverse - not just because sleep is an important feature in general.
    4. If the consistency analysis flags a possible masked-stress pattern it is added, worded as something to reflect
       on, not a finding.

Everything is rule-based and deterministic, advice is general self-care (non-clinical), and thresholds below are
project-defined. Nothing here changes any model output.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Minimum severity (0-1) for an answer pattern to count as an adverse signal worth advice.
INSIGHT_THRESHOLD = 0.5
MAX_RECOMMENDATIONS = 4
#: Weight of model importance when ranking (severity keeps the larger share, so advice follows the person's answers).
IMPORTANCE_WEIGHT = 0.3

# ---------------------------------------------------------------------------
# 1. Readable feature labels
# ---------------------------------------------------------------------------

_NEGATIVE_EMOTIONS = ("Anxious", "Frustrated", "Irritated", "Lonely", "Overwhelmed", "Sad", "Tired", "Worried")

_BASE: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "Q3_encoded": ("Daily hours spent on study or preparation", ("workload",)),
    "Q5_encoded": ("Feeling overwhelmed by your responsibilities", ("workload",)),
    "Q6_encoded": ("Feeling unable to keep up with everything", ("workload",)),
    "Q7_encoded": ("Difficulty relaxing after work or study", ("relaxation",)),
    "Q8_encoded": ("Difficulty concentrating because of stress", ("focus",)),
    "Q9_encoded": ("Stress affecting your sleep", ("sleep",)),
    "Q12_encoded": ("Feeling emotionally overwhelmed right now", ("emotional",)),
    "Q13_encoded": ("How capable you feel of handling your responsibilities", ("coping",)),
    "Q13_stress_aligned": ("Feeling less able to cope with your responsibilities", ("coping",)),
    "Q14_encoded": ("How long stress stays noticeable", ("recovery",)),
    "Q14_varies": ("How long stress lasts varies a lot", ("recovery",)),
    "Q15_encoded": ("Sudden episodes of intense stress", ("emotional",)),
    "Q16_encoded": ("How long it takes to feel normal again after stress", ("recovery",)),
    "Q16_varies": ("Recovery time after stress varies a lot", ("recovery",)),
    "Q17_encoded": ("Difficulty starting or finishing tasks when stressed", ("task_management",)),
    "Q18_encoded": ("Postponing tasks under pressure", ("task_management",)),
    "Q19_encoded": ("Repeatedly checking or redoing work when stressed", ("checking_decisions",)),
    "Q20_encoded": ("Taking longer than usual to make decisions when stressed", ("checking_decisions",)),
    "Q1_year_encoded": ("Year of study", ()),
    # Engineered scores (what they combine is defined in questionnaire/feature_engineering.py)
    "overwhelm_dimension": ("Feeling overwhelmed and unable to keep up (combined score)", ("workload",)),
    "stress_impact_dimension": ("How much stress affects your relaxing, focus and sleep (combined score)",
                                ("relaxation", "focus", "sleep")),
    "emotional_strain_dimension": ("Emotional overwhelm", ("emotional",)),
    "stress_persistence_dimension": ("How long stress stays noticeable", ("recovery",)),
    "recovery_difficulty_dimension": ("Difficulty recovering after stress", ("recovery",)),
    "stress_behavior_dimension": ("Stress-related work habits: delaying, checking, slow decisions (combined score)",
                                  ("task_management", "checking_decisions")),
    "academic_preparation_load": ("Time spent on academic preparation", ("workload",)),
    # Voice: tone of what was said (optional speech-content analysis)
    "speech_negative_tone": ("Negative tone in what you said (voice)", ("speech_tone",)),
    "speech_positive_tone": ("Positive tone in what you said (voice)", ()),
    # Daily habits
    "B1_sleep_hours": ("Hours of sleep per night", ("sleep",)),
    "B2_study_consistency": ("Consistency of your study routine", ("routine",)),
    "B3_social_activity_freq": ("How often you socialise", ("social",)),
    "B4_screen_time_hours": ("Non-academic screen time per day", ("screen_time",)),
    "B5_routine_irregularity": ("How irregular your daily routine is", ("routine",)),
}

_STRESSOR_THEMES = {
    "Academic_performance_grades": ("grades and academic performance", ("workload",)),
    "Academic_workload": ("academic workload", ("workload",)),
    "Assignments": ("assignments", ("workload",)),
    "Coding_technical_assessments": ("coding or technical assessments", ("exams",)),
    "Competition_with_peers": ("competition with peers", ("social",)),
    "Difficulty_balancing_academics_and_personal_life": ("balancing academics and personal life", ("workload",)),
    "Exams": ("exams", ("exams",)),
    "Family_expectations": ("family expectations", ("family",)),
    "Fear_of_not_getting_a_job": ("fear of not getting a job", ("career",)),
    "Financial_concerns": ("financial concerns", ("financial",)),
    "Internship_job_applications": ("internship or job applications", ("career",)),
    "Lack_of_sleep": ("lack of sleep", ("sleep",)),
    "Placement_preparation": ("placement preparation", ("career",)),
    "Projects": ("projects", ("workload",)),
    "Relationship_social_concerns": ("relationship or social concerns", ("social",)),
    "Time_management": ("time management", ("task_management",)),
    "Uncertainty_about_career": ("uncertainty about your career", ("career",)),
}
#: Specific tips for individual stressors a person selected (shown under the matching theme's suggestion).
STRESSOR_TIPS = {
    "Exams": "Exams: plan revision in short blocks with breaks, and do one timed practice paper instead of re-reading.",
    "Academic_workload": "Workload: list this week's tasks, pick the top three, and ask early about extensions you may need.",
    "Assignments": "Assignments: start with a 25-minute first draft; finishing a rough version lowers the pressure.",
    "Projects": "Projects: agree on clear owners and small weekly milestones with your teammates.",
    "Placement_preparation": "Placements: set a fixed daily prep slot (for example 1 hour) so it doesn't spill into everything else.",
    "Internship_job_applications": "Applications: batch them on one or two days a week rather than checking constantly.",
    "Fear_of_not_getting_a_job": "Career worries: one conversation with your placement cell or a senior can make the path feel more concrete.",
    "Uncertainty_about_career": "Career uncertainty: write down two options you would be happy with; it reduces all-or-nothing thinking.",
    "Coding_technical_assessments": "Coding tests: practise a few timed problems a day; consistency beats long cramming sessions.",
    "Time_management": "Time management: block your day in a calendar, including breaks and sleep, and review it each evening.",
    "Lack_of_sleep": "Sleep: keep a fixed wake-up time and stop screens 30 minutes before bed.",
    "Financial_concerns": "Money worries: your college's student support or scholarship office can explain options confidentially.",
    "Family_expectations": "Family expectations: sharing a realistic plan with your family often eases the pressure on both sides.",
    "Competition_with_peers": "Peer comparison: compare with your own progress last month, not with others' highlights.",
    "Relationship_social_concerns": "Relationships: talk to someone you trust; you don't have to sort it out alone.",
    "Difficulty_balancing_academics_and_personal_life": "Balance: protect one small non-academic activity every day.",
    "Academic_performance_grades": "Grades: ask a faculty member which one or two changes would help most.",
}
MAX_TIPS_PER_RECOMMENDATION = 2

_POSITIVE_EMOTIONS = {"Calm", "Confident", "Happy", "Motivated", "Neutral"}

_VOICE_PARTS = {
    "mfcc": "Voice timbre, MFCC {n}", "pitch": "Pitch", "energy": "Loudness",
    "spectral_centroid": "Voice brightness", "spectral_bandwidth": "Spread of voice frequencies",
    "spectral_rolloff": "High-frequency content of the voice", "zero_crossing_rate": "Noisiness of the voice signal",
}
_VOICE_STATS = {"mean": "average", "std": "variation", "min": "lowest", "max": "highest", "range": "range"}


def _humanize(name: str) -> str:
    text = re.sub(r"_encoded$", "", name).replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def describe_feature(name: str) -> Dict[str, Any]:
    """Plain-language label and related themes for any model feature name (never returns a raw code name)."""
    if name in _BASE:
        label, themes = _BASE[name]
        return {"label": label, "themes": list(themes)}
    match = re.match(r"^Q10_(.+)$", name)
    if match:
        key = match.group(1)
        if key in _STRESSOR_THEMES:
            text, themes = _STRESSOR_THEMES[key]
            return {"label": f"Stressor you selected: {text}", "themes": list(themes)}
        return {"label": "No stressors selected" if key == "nothing" else "Other stressor you selected", "themes": []}
    match = re.match(r"^Q11_(.+)$", name)
    if match:
        key = match.group(1)
        themes = ["emotional"] if key in _NEGATIVE_EMOTIONS else []
        return {"label": f"Feeling {key.lower()} recently" if key != "Other" else "Other recent feeling", "themes": themes}
    match = re.match(r"^Q2_(.+)$", name)
    if match:
        return {"label": "Current situation: " + _humanize(match.group(1)).lower(), "themes": []}
    match = re.match(r"^Q25_(.+)$", name)
    if match:
        return {"label": "Main concern: " + _humanize(match.group(1)).lower(), "themes": []}
    match = re.match(r"^(mfcc_(\d+)|pitch|energy|spectral_centroid|spectral_bandwidth|spectral_rolloff|zero_crossing_rate)_(mean|std|min|max|range)$", name)
    if match:
        base = match.group(1).split("_")[0] if match.group(1).startswith("mfcc") else match.group(1)
        template = _VOICE_PARTS[base]
        label = template.format(n=match.group(2)) if base == "mfcc" else template
        return {"label": f"{label} ({_VOICE_STATS[match.group(3)]})", "themes": ["voice"]}
    return {"label": _humanize(name), "themes": []}


def label_key_factors(factors: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Copy of the fused key factors with a readable ``label`` added (the raw ``feature`` code is kept for developers)."""
    return [{**dict(factor), "label": describe_feature(str(factor["feature"]))["label"]} for factor in factors]


# ---------------------------------------------------------------------------
# 2. Personal insights from the user's own answers
# ---------------------------------------------------------------------------

def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _selected(answer: Any) -> List[str]:
    return [part.strip() for part in str(answer).split(";") if part.strip()] if answer else []


def _insight(theme: str, source: str, severity: float, evidence: str) -> Dict[str, Any]:
    return {"theme": theme, "source": source, "severity": round(_clip(severity), 3), "evidence": evidence}


def questionnaire_insights(responses: Mapping[str, Any]) -> List[Dict[str, Any]]:
    from src.ai_model.questionnaire.predict import _ordinal_value  # same answer encoding the model uses

    def code(question: str) -> Optional[float]:
        value = _ordinal_value(question, responses.get(question))
        return None if value != value else float(value)  # NaN -> None

    def said(question: str) -> str:
        return f"“{responses.get(question)}”"

    found: List[Dict[str, Any]] = []
    q9, q8, q7 = code("Q9"), code("Q8"), code("Q7")
    if q9 is not None:
        found.append(_insight("sleep", "questionnaire", q9 / 4, f"You said stress affects your sleep: {said('Q9')}."))
    if q8 is not None:
        found.append(_insight("focus", "questionnaire", q8 / 4, f"You said stress makes it hard to concentrate: {said('Q8')}."))
    if q7 is not None:
        found.append(_insight("relaxation", "questionnaire", q7 / 4, f"Relaxing after work or study is: {said('Q7')}."))
    q5, q6, q3 = code("Q5"), code("Q6"), code("Q3")
    overwhelm = [v for v in (q5, q6) if v is not None]
    if overwhelm:
        found.append(_insight("workload", "questionnaire", (sum(overwhelm) / len(overwhelm)) / 4,
                              f"You feel overwhelmed {said('Q5')} and unable to keep up {said('Q6')}."))
    if q3 is not None and q3 >= 3:
        found.append(_insight("workload", "questionnaire", 0.5 + 0.25 * (q3 - 3), f"You spend {said('Q3')} a day on study or preparation."))
    q12, q13 = code("Q12"), code("Q13")
    emotional = []
    if q12 is not None:
        emotional.append(_insight("emotional", "questionnaire", (q12 - 1) / 4, f"You rated feeling emotionally overwhelmed {int(q12)} out of 5."))
    feelings = [f for f in _selected(responses.get("Q11")) if f in _NEGATIVE_EMOTIONS]
    if len(feelings) >= 2:
        emotional.append(_insight("emotional", "questionnaire", 0.5 if len(feelings) == 2 else 0.75,
                                  "Recent feelings you chose: " + ", ".join(f.lower() for f in feelings) + "."))
    found.extend(emotional)
    if q13 is not None:
        found.append(_insight("coping", "questionnaire", (5 - q13) / 4, f"You rated feeling capable of handling your responsibilities {int(q13)} out of 5."))
    q14, q16 = code("Q14"), code("Q16")
    if q14 is not None:
        found.append(_insight("recovery", "questionnaire", q14 / 6, f"Stress usually stays noticeable for: {said('Q14')}."))
    if q16 is not None:
        found.append(_insight("recovery", "questionnaire", q16 / 5, f"Feeling normal again after stress takes: {said('Q16')}."))
    q17, q18, q19, q20 = (code(q) for q in ("Q17", "Q18", "Q19", "Q20"))
    delay = [v for v in (q17, q18) if v is not None]
    if delay:
        found.append(_insight("task_management", "questionnaire", (sum(delay) / len(delay)) / 4,
                              f"Under stress, starting or finishing tasks is hard {said('Q17')} and you postpone {said('Q18')}."))
    checking = [v for v in (q19, q20) if v is not None]
    if checking:
        found.append(_insight("checking_decisions", "questionnaire", (sum(checking) / len(checking)) / 4,
                              f"Under stress you recheck work {said('Q19')} and decide slowly {said('Q20')}."))
    stressors = _selected(responses.get("Q10"))
    by_theme: Dict[str, List[str]] = {}
    keys_by_theme: Dict[str, List[str]] = {}
    for item in stressors:
        key = re.sub(r"[^A-Za-z0-9]+", "_", item).strip("_")
        if key in _STRESSOR_THEMES:
            text, themes = _STRESSOR_THEMES[key]
            for theme in themes:
                by_theme.setdefault(theme, []).append(text)
                keys_by_theme.setdefault(theme, []).append(key)
    for theme, items in by_theme.items():
        entry = _insight(theme, "questionnaire", min(0.9, 0.6 + 0.1 * (len(items) - 1)),
                         "Stressors you selected: " + ", ".join(items) + ".")
        entry["stressors"] = keys_by_theme[theme]
        found.append(entry)
    if "Lonely" in _selected(responses.get("Q11")):
        found.append(_insight("social", "questionnaire", 0.6, "You said you have been feeling lonely recently."))
    return found


def behavioral_insights(responses: Mapping[str, Any]) -> List[Dict[str, Any]]:
    def number(key: str) -> Optional[float]:
        value = responses.get(key)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    found: List[Dict[str, Any]] = []
    sleep = number("B1_sleep_hours")
    if sleep is not None and sleep < 7:
        found.append(_insight("sleep", "behavioral", 1.0 if sleep < 5 else 0.75 if sleep < 6 else 0.5,
                              f"You reported about {sleep:g} hours of sleep a night."))
    consistency, irregular = number("B2_study_consistency"), number("B5_routine_irregularity")
    routine = []
    if consistency is not None and consistency <= 2:
        routine.append((_clip((3 - consistency) / 2), f"You rated your study routine consistency {consistency:g} out of 5."))
    if irregular is not None and irregular >= 4:
        routine.append((_clip((irregular - 3) / 2), f"You rated your daily routine irregularity {irregular:g} out of 5."))
    if routine:
        best = max(routine, key=lambda item: item[0])
        found.append(_insight("routine", "behavioral", best[0], " ".join(text for _, text in routine)))
    social = number("B3_social_activity_freq")
    if social is not None and social <= 2:
        found.append(_insight("social", "behavioral", _clip((3 - social) / 2), f"You rated how often you socialise {social:g} out of 5."))
    screen = number("B4_screen_time_hours")
    if screen is not None and screen > 5:
        found.append(_insight("screen_time", "behavioral", 1.0 if screen > 8 else 0.75 if screen > 6 else 0.5,
                              f"You reported about {screen:g} hours of non-academic screen time a day."))
    return found


def voice_insights(voice: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Tone of what was said, when the optional speech-content analysis supplied the voice result.

    The transcript itself is never copied into an insight; only the overall tone label is used.
    """
    if not voice or voice.get("source") != "azure_speech_language" or voice.get("risk_level") not in ("moderate", "high"):
        return []
    severity = max(INSIGHT_THRESHOLD, float(voice.get("risk_score", 0.0)))
    tone = voice.get("sentiment") or "strained"
    return [_insight("speech_tone", "voice", severity, f"What you said in your recording came across as {tone}.")]


def personal_insights(questionnaire: Optional[Mapping[str, Any]] = None, behavioral: Optional[Mapping[str, Any]] = None,
                      voice: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Adverse areas found in the person's own answers, merged per theme (strongest first)."""
    raw: List[Dict[str, Any]] = []
    if questionnaire:
        raw.extend(questionnaire_insights(questionnaire))
    if behavioral:
        raw.extend(behavioral_insights(behavioral))
    raw.extend(voice_insights(voice))
    merged: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        if item["severity"] < INSIGHT_THRESHOLD:
            continue
        current = merged.setdefault(item["theme"], {"theme": item["theme"], "severity": 0.0, "sources": [], "evidence": [], "stressors": []})
        for stressor in item.get("stressors", []):
            if stressor not in current["stressors"]:
                current["stressors"].append(stressor)
        current["severity"] = max(current["severity"], item["severity"])
        if item["source"] not in current["sources"]:
            current["sources"].append(item["source"])
        current["evidence"].append(item["evidence"])
    return sorted(merged.values(), key=lambda item: item["severity"], reverse=True)


# ---------------------------------------------------------------------------
# 3. Recommendations
# ---------------------------------------------------------------------------

THEME_TITLES = {
    "sleep": "Sleep", "focus": "Concentration", "workload": "Workload", "relaxation": "Winding down",
    "emotional": "Emotional strain", "coping": "Confidence in coping", "recovery": "Recovering from stress",
    "task_management": "Starting and finishing tasks", "checking_decisions": "Checking and deciding",
    "screen_time": "Screen time", "routine": "Daily routine", "social": "Connection with others",
    "speech_tone": "Tone of what you said", "exams": "Exams and assessments", "career": "Career worries", "financial": "Money worries", "family": "Family expectations",
}

ADVICE = {
    "sleep": "Protect your sleep: keep a consistent bedtime and wake-up time and wind down without screens for 30 minutes before bed. Most adults need roughly 7–9 hours.",
    "focus": "When concentration slips, work in short focused blocks (for example 25 minutes on, 5 off) and silence notifications while you study.",
    "workload": "Write down everything due this week, pick the top three and break each into small steps. If the load is truly too much, talk to an instructor or advisor early about options.",
    "relaxation": "Build a short wind-down ritual after studying (a walk, stretching, music or slow breathing) so your mind gets a clear signal that work is over.",
    "emotional": "Name what you are feeling and share it with someone you trust. A few minutes of slow breathing (in for 4 seconds, out for 6) can ease intense moments.",
    "coping": "Confidence dips under pressure. List what you have already completed, and ask a mentor or classmate for help with one thing this week.",
    "recovery": "Stress that lingers is easier to manage with regular recovery: movement, time outdoors, time with friends, and one non-academic activity you enjoy every day.",
    "task_management": "If starting or finishing tasks is hard, begin with a 10-minute “just start” step, set small deadlines and reward completion. A study partner can help.",
    "checking_decisions": "Repeated checking and slow decisions often come from wanting to avoid mistakes. Set a “good enough” standard and a time limit for reviewing or deciding.",
    "screen_time": "Your non-academic screen time is high. Try phone-free blocks while studying and in the hour before bed, and swap one scrolling session for something offline.",
    "routine": "A steadier routine helps: fix your wake-up, meal and study-start times (even at weekends) and plan tomorrow the evening before.",
    "social": "Make time for people: a short call, a meal or a study session with friends or classmates. Feeling connected buffers stress.",
    "speech_tone": "What you said sounded strained. Saying it out loud to someone you trust can help as much as writing it down.",
    "exams": "For exams and assessments, work backwards from the dates, mix short practice tests with revision, and build rest days into the plan.",
    "career": "Career worries are common. Break them into steps: update your CV, book a session with your placement or career cell, and apply to a few roles each week.",
    "financial": "Money worries weigh heavily. Find out what your institution offers (scholarships, fee instalments, student support) and talk to the right office early.",
    "family": "Family expectations can feel heavy. Consider an honest conversation about what you can realistically manage, or ask a counsellor or mentor to help you prepare for it.",
}
MASKED_ADVICE = ("Your own answers look lower than your daily-habit signals. That can happen when stress is hidden or has become normal. "
                 "It may help to check whether your habits (sleep, routine, screen time) match how you describe feeling, and to talk to someone you trust.")
SELF_REPORT_HIGHER_ADVICE = ("You describe more stress than your daily-habit signals show. Your own experience matters: "
                             "consider sharing how you feel with someone you trust, even if your routines look fine.")
SUPPORT = {
    "moderate": "If these feelings continue for more than a couple of weeks, consider talking to your campus counselling service.",
    "high": "Consider talking to someone you trust or your campus counselling service soon. If you feel unsafe or in crisis, contact local emergency services or a crisis helpline right away.",
}
LOW_FALLBACK = "No strong stress signals stood out in your answers. Keep the routines that are working and check in again if things change."
NO_SPECIFIC_FALLBACK = "Nothing specific stood out in your answers, so these are general suggestions: keep regular sleep, take short breaks and talk to someone you trust if things feel heavy."


def theme_importance(key_factors: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """How much the fused model relies on each theme (sum of the matching key-factor contributions, capped at 1)."""
    totals: Dict[str, float] = {}
    for factor in key_factors:
        for theme in describe_feature(str(factor.get("feature", "")))["themes"]:
            totals[theme] = totals.get(theme, 0.0) + float(factor.get("contribution", 0.0))
    return {theme: min(1.0, value) for theme, value in totals.items()}


def build_recommendations(insights: Sequence[Mapping[str, Any]], assessment: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Ranked, personalized recommendations. Each one says why it is shown (the person's own answers)."""
    level = assessment.get("risk_level")
    importance = theme_importance(assessment.get("key_factors", []))
    ranked = sorted(insights, key=lambda item: item["severity"] * ((1 - IMPORTANCE_WEIGHT) + IMPORTANCE_WEIGHT * importance.get(item["theme"], 0.0)),
                    reverse=True)
    recommendations: List[Dict[str, Any]] = []
    for item in ranked:
        if item["theme"] not in ADVICE:
            continue
        recommendations.append({
            "theme": item["theme"], "title": THEME_TITLES[item["theme"]], "advice": ADVICE[item["theme"]],
            "because": list(item["evidence"]), "severity": round(item["severity"], 2),
            "model_influence": round(importance.get(item["theme"], 0.0), 2),
            "tips": [STRESSOR_TIPS[key] for key in item.get("stressors", []) if key in STRESSOR_TIPS][:MAX_TIPS_PER_RECOMMENDATION],
        })
        if len(recommendations) >= MAX_RECOMMENDATIONS:
            break
    consistency = assessment.get("consistency") or {}
    if consistency.get("pattern") == "masked_stress_possible":
        recommendations.insert(0, {"theme": "masked_stress", "title": "Self-report vs daily habits", "advice": MASKED_ADVICE,
                                   "because": ["Your questionnaire result is noticeably lower than your daily-habit result."],
                                   "severity": None, "model_influence": None})
    elif consistency.get("pattern") == "self_report_higher":
        recommendations.insert(0, {"theme": "self_report_higher", "title": "Self-report vs daily habits", "advice": SELF_REPORT_HIGHER_ADVICE,
                                   "because": ["Your questionnaire result is noticeably higher than your daily-habit result."],
                                   "severity": None, "model_influence": None})
    if not recommendations and level is not None:
        recommendations.append({"theme": "general", "title": "Overall", "advice": LOW_FALLBACK if level == "low" else NO_SPECIFIC_FALLBACK,
                                "because": [], "severity": None, "model_influence": None})
    if level in SUPPORT:
        recommendations.append({"theme": "support", "title": "Support", "advice": SUPPORT[level], "because": [f"Your overall estimate is {level}."],
                                "severity": None, "model_influence": None})
    return recommendations
