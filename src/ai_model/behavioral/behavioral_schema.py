"""Schema and synthetic-data definitions for the behavioral stress-risk modality.

Only self-reportable behavioral signals are used here (no passive
sensor/wearable data), because not every student has a smartwatch or
similar device. Five features, each a simple self-report:

  B1 sleep_hours            - average hours of sleep per night (self-estimated)
  B2 study_consistency      - self-rated consistency of daily study routine (1-5)
  B3 social_activity_freq   - self-rated frequency of social interaction (1-5)
  B4 screen_time_hours      - average non-academic screen time per day (hours)
  B5 routine_irregularity   - self-rated irregularity of daily routine (1-5,
                              5 = highly irregular)

This mirrors the same "modality-independent contract" pattern used by
the questionnaire modality (src/ai_model/questionnaire/predict.py):
raw response in, standardized fusion contract out, non-clinical framing
throughout.
"""
from __future__ import annotations

from typing import Any, Dict

REQUIRED_BEHAVIORAL_FIELDS = (
    "B1_sleep_hours",
    "B2_study_consistency",
    "B3_social_activity_freq",
    "B4_screen_time_hours",
    "B5_routine_irregularity",
)

FEATURE_LABELS = {
    "B1_sleep_hours": "sleep_hours",
    "B2_study_consistency": "study_consistency",
    "B3_social_activity_freq": "social_activity_freq",
    "B4_screen_time_hours": "screen_time_hours",
    "B5_routine_irregularity": "routine_irregularity",
}

# Valid ranges used for both synthetic generation and input validation.
FIELD_RANGES = {
    "B1_sleep_hours": (0.0, 12.0),
    "B2_study_consistency": (1.0, 5.0),
    "B3_social_activity_freq": (1.0, 5.0),
    "B4_screen_time_hours": (0.0, 16.0),
    "B5_routine_irregularity": (1.0, 5.0),
}

# Direction each feature contributes to risk, for key-factor annotation.
# These are project-defined associations for explainability display only,
# not causal claims.
STRESS_DIRECTION = {
    "B1_sleep_hours": "decreases_risk",         # more sleep -> lower risk
    "B2_study_consistency": "decreases_risk",   # more consistency -> lower risk
    "B3_social_activity_freq": "decreases_risk",  # more social activity -> lower risk
    "B4_screen_time_hours": "increases_risk",   # more non-academic screen time -> higher risk
    "B5_routine_irregularity": "increases_risk",  # more irregularity -> higher risk
}


def validate_behavioral_input(response: Dict[str, Any]) -> None:
    """Validate the raw behavioral response shape before preprocessing."""
    if not isinstance(response, dict):
        raise TypeError("Behavioral response must be a Python dictionary.")
    missing = [field for field in REQUIRED_BEHAVIORAL_FIELDS if field not in response]
    if missing:
        raise ValueError("Missing required behavioral fields: " + ", ".join(missing))
    invalid = [key for key in response if key not in REQUIRED_BEHAVIORAL_FIELDS]
    if invalid:
        raise ValueError("Unsupported behavioral fields: " + ", ".join(map(str, invalid)))
    for field, (low, high) in FIELD_RANGES.items():
        value = response[field]
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field} must be numeric.")
        if not (low <= float(value) <= high):
            raise ValueError(f"{field} must be between {low} and {high}.")
