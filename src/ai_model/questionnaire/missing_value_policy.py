"""Missing-value handling policy for the questionnaire modality.

This module defines the policy for how missing values should be treated in the
questionnaire dataset. It is intentionally declarative only:

- it does not impute any values,
- it does not modify any CSV file,
- it does not delete rows or columns,
- it does not train any ML model,
- it does not create targets or stress scores.

The policy is structured so later preprocessing and model-specific pipeline
modules can consume it as a machine-readable specification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from .questionnaire_schema import QUESTIONNAIRE_SCHEMA
except ImportError:  # pragma: no cover - fallback for direct script execution
    from questionnaire_schema import QUESTIONNAIRE_SCHEMA


CORE_STRESS_QUESTIONS: Tuple[str, ...] = ("Q4", "Q5", "Q6", "Q7", "Q8", "Q9")
STRUCTURED_FEATURE_QUESTIONS: Tuple[str, ...] = (
    "Q10",
    "Q11",
    "Q12",
    "Q13",
    "Q14",
    "Q15",
    "Q16",
    "Q17",
    "Q18",
    "Q19",
    "Q20",
)
OPTIONAL_CONTEXT_QUESTIONS: Tuple[str, ...] = ("Q21", "Q25")
OPTIONAL_TEXT_QUESTIONS: Tuple[str, ...] = ("Q22", "Q23")
EXCLUDED_QUESTIONS: Tuple[str, ...] = ("Timestamp", "Q22", "Q23", "Q24")
QUESTIONS_REQUIRING_TRAINING_TIME_IMPUTATION: Tuple[str, ...] = CORE_STRESS_QUESTIONS + STRUCTURED_FEATURE_QUESTIONS
OPTIONAL_QUESTIONS: Tuple[str, ...] = OPTIONAL_CONTEXT_QUESTIONS + OPTIONAL_TEXT_QUESTIONS


def _validate_policy_questions() -> None:
    """Validate that the policy references only known questionnaire questions."""

    known_question_ids = set(QUESTIONNAIRE_SCHEMA.keys())
    all_policy_questions = (
        CORE_STRESS_QUESTIONS
        + STRUCTURED_FEATURE_QUESTIONS
        + OPTIONAL_CONTEXT_QUESTIONS
        + OPTIONAL_TEXT_QUESTIONS
        + EXCLUDED_QUESTIONS
        + QUESTIONS_REQUIRING_TRAINING_TIME_IMPUTATION
        + OPTIONAL_QUESTIONS
    )

    invalid_questions = sorted(set(all_policy_questions) - known_question_ids)
    if invalid_questions:
        raise ValueError(
            "Missing-value policy contains unknown questionnaire question IDs: "
            + ", ".join(invalid_questions)
        )


def get_missing_value_policy() -> Dict[str, Any]:
    """Return the missing-value policy as a machine-readable dictionary.

    Returns:
        A dictionary describing which questions are core stress indicators,
        structured features, optional context/text questions, excluded questions,
        and which questions may require training-time imputation handling.
    """

    _validate_policy_questions()

    return {
        "policy_name": "Questionnaire missing-value handling policy",
        "core_stress_questions": list(CORE_STRESS_QUESTIONS),
        "structured_feature_questions": list(STRUCTURED_FEATURE_QUESTIONS),
        "optional_context_questions": list(OPTIONAL_CONTEXT_QUESTIONS),
        "optional_text_questions": list(OPTIONAL_TEXT_QUESTIONS),
        "excluded_questions": list(EXCLUDED_QUESTIONS),
        "questions_requiring_training_time_imputation": list(QUESTIONS_REQUIRING_TRAINING_TIME_IMPUTATION),
        "optional_questions": list(OPTIONAL_QUESTIONS),
        "rules": {
            "Q4-Q9": {
                "status": "core_stress_indicators",
                "policy": "Keep them. Do not manually impute values in this step. Any required imputation must later be learned only from training data inside the ML preprocessing pipeline.",
            },
            "Q10-Q20": {
                "status": "structured_questionnaire_features",
                "policy": "Keep them. Their small amount of missingness should later be handled by preprocessing. Do not globally calculate or apply imputation values now.",
            },
            "Q21": {
                "status": "optional_contextual_information",
                "policy": "Keep as contextual information. Do not use it as a required feature of the first structured ML model because approximately 48% of responses are missing.",
            },
            "Q22-Q23": {
                "status": "optional_text_information",
                "policy": "Keep as optional free-text information. Exclude them from the first structured numerical ML model. Missing values should remain missing/empty until a future text-processing stage.",
            },
            "Q24": {
                "status": "excluded_metadata",
                "policy": "Exclude from all ML features because it is research participation metadata.",
            },
            "Q25": {
                "status": "optional_contextual_information",
                "policy": "Keep as optional contextual information. Later preprocessing may represent missing values as an explicit 'Unknown' category.",
            },
            "Timestamp": {
                "status": "metadata_only",
                "policy": "Keep as metadata. Never use it as a stress predictor.",
            },
            "high_missingness_rows": {
                "row_17": {
                    "missing_answers": 12,
                    "retain": True,
                    "policy": "Do not automatically remove this row. Retain it in the cleaned dataset. Later model-specific preprocessing may determine whether a particular sample is usable.",
                },
                "row_42": {
                    "missing_answers": 14,
                    "retain": True,
                    "policy": "Do not automatically remove this row. Retain it in the cleaned dataset. Later model-specific preprocessing may determine whether a particular sample is usable.",
                },
            },
        },
    }


def get_questions_requiring_imputation() -> List[str]:
    """Return the questions that may require training-time imputation handling.

    Returns:
        A list of question IDs that should be considered for imprinting or
        missing-value handling later in the training pipeline, rather than now.
    """

    _validate_policy_questions()
    return list(QUESTIONS_REQUIRING_TRAINING_TIME_IMPUTATION)


def get_excluded_questions() -> List[str]:
    """Return the questions that are excluded from the ML feature pipeline.

    Returns:
        A list of question IDs that should not be used in the first structured ML
        feature set or should be treated as metadata-only.
    """

    _validate_policy_questions()
    return list(EXCLUDED_QUESTIONS)


def get_optional_questions() -> List[str]:
    """Return optional questions that should stay available but not be required.

    Returns:
        A list combining optional contextual and optional text questions.
    """

    _validate_policy_questions()
    return list(OPTIONAL_QUESTIONS)


def main() -> None:
    """Print the missing-value policy in a readable, machine-readable form."""

    policy = get_missing_value_policy()
    print("QUESTIONNAIRE MISSING-VALUE POLICY")
    print("=" * 36)
    for key in [
        "core_stress_questions",
        "structured_feature_questions",
        "optional_context_questions",
        "optional_text_questions",
        "excluded_questions",
        "questions_requiring_training_time_imputation",
        "optional_questions",
    ]:
        print(f"{key}: {policy[key]}")

    print("\nRules:")
    for rule_key, rule_value in policy["rules"].items():
        if isinstance(rule_value, dict) and "policy" in rule_value:
            print(f"- {rule_key}: {rule_value['policy']}")
        elif isinstance(rule_value, dict):
            print(f"- {rule_key}:")
            for nested_key, nested_value in rule_value.items():
                if isinstance(nested_value, dict):
                    print(f"  * {nested_key}: {nested_value['policy']}")


if __name__ == "__main__":
    main()
