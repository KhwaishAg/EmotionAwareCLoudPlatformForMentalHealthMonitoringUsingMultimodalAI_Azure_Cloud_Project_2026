"""Formal metadata schema registry for the questionnaire modality.

This module defines a reusable, question-level schema for the current questionnaire
CSV in this repository. It intentionally does not modify the raw CSV, assign a
final target variable, or perform any preprocessing or feature engineering.

The registry is designed so later preprocessing and feature-engineering modules
can map the canonical question IDs (Q1..Q25) to the actual Google Forms column
names already present in the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class QuestionnaireColumnSchema:
    """Metadata describing one questionnaire column.

    Attributes:
        question_id: Canonical questionnaire identifier such as "Q1".
        column_name: Exact raw CSV column name currently present in the dataset.
        column_pattern: A pattern/string useful for later column matching logic.
        description: Human-readable description of the question.
        data_type: Data category such as "categorical", "multi-select categorical",
            "ordered categorical", or "free text".
        question_group: Semantically grouped area such as "context", "stress",
            "emotional", or "recovery".
        ml_role: High-level role in a later ML pipeline.
        encoding_strategy: How the field should be encoded during later preprocessing.
        direction: Optional orientation semantics, such as whether higher values mean
            greater coping capability or greater stress intensity.
        overlap_note: Optional note for potentially overlapping or special-case fields.
        reason: Optional explanation of why the column is excluded or treated
            specially.
        include_in_structured_ml: Whether this question should be included in the
            first structured numerical ML feature set.
        requires_special_handling: Whether later preprocessing should apply special
            logic beyond a standard one-to-one mapping.
    """

    question_id: str
    column_name: str
    column_pattern: str
    description: str
    data_type: str
    question_group: str
    ml_role: str
    encoding_strategy: str
    direction: Optional[str] = None
    overlap_note: Optional[str] = None
    reason: Optional[str] = None
    include_in_structured_ml: bool = True
    requires_special_handling: bool = False
    canonical_options: Optional[List[str]] = None


QUESTIONNAIRE_SCHEMA: Dict[str, QuestionnaireColumnSchema] = {
    "Timestamp": QuestionnaireColumnSchema(
        question_id="Timestamp",
        column_name="Timestamp",
        column_pattern="Timestamp",
        description="Submission timestamp recorded by the Google Form.",
        data_type="datetime",
        question_group="metadata",
        ml_role="exclude",
        encoding_strategy="none",
        direction=None,
        overlap_note=None,
        reason="submission metadata",
        include_in_structured_ml=False,
        requires_special_handling=False,
    ),
    "Q1": QuestionnaireColumnSchema(
        question_id="Q1",
        column_name="Q1. What is your current year of study?  ",
        column_pattern="Q1. What is your current year of study?",
        description="Current year of study.",
        data_type="categorical",
        question_group="context",
        ml_role="context feature",
        encoding_strategy="category encoding",
        direction=None,
        overlap_note=None,
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=False,
    ),
    "Q2": QuestionnaireColumnSchema(
        question_id="Q2",
        column_name="Q2. What is your current academic situation?  ",
        column_pattern="Q2. What is your current academic situation?",
        description="Current academic situation.",
        data_type="multi-select categorical",
        question_group="context",
        ml_role="context feature",
        encoding_strategy="multi-hot",
        direction=None,
        overlap_note=None,
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=True,
    ),
    "Q3": QuestionnaireColumnSchema(
        question_id="Q3",
        column_name="Q3. Approximately how many hours per day do you spend on academic/professional preparation?  ",
        column_pattern="Q3. Approximately how many hours per day do you spend on academic/professional preparation?",
        description="Hours per day spent on academic/professional preparation.",
        data_type="ordered categorical",
        question_group="context",
        ml_role="feature",
        encoding_strategy="ordinal encoding",
        direction=None,
        overlap_note=None,
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=False,
    ),
    "Q4": QuestionnaireColumnSchema(
        question_id="Q4",
        column_name="Q4. How stressed have you felt during the past week?  ",
        column_pattern="Q4. How stressed have you felt during the past week?",
        description="Perceived stress during the past week.",
        data_type="ordered categorical / Likert",
        question_group="stress",
        ml_role="core stress feature",
        encoding_strategy="ordinal encoding",
        direction="higher values indicate greater stress intensity",
        overlap_note=None,
        reason="Do not define as the final target yet.",
        include_in_structured_ml=True,
        requires_special_handling=False,
    ),
    "Q5": QuestionnaireColumnSchema(
        question_id="Q5",
        column_name="Q5. How often have you felt overwhelmed by your academic or professional responsibilities during the past week?  ",
        column_pattern="Q5. How often have you felt overwhelmed by your academic or professional responsibilities during the past week?",
        description="Frequency of feeling overwhelmed by academic/professional responsibilities.",
        data_type="ordered categorical",
        question_group="stress",
        ml_role="core stress feature",
        encoding_strategy="ordinal encoding",
        direction="higher values indicate greater stress intensity",
        overlap_note=None,
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=False,
    ),
    "Q6": QuestionnaireColumnSchema(
        question_id="Q6",
        column_name="Q6. How often have you felt that you were unable to keep up with everything you needed to do?  ",
        column_pattern="Q6. How often have you felt that you were unable to keep up with everything you needed to do?",
        description="Perceived inability to keep up with responsibilities.",
        data_type="ordered categorical",
        question_group="stress",
        ml_role="core stress feature",
        encoding_strategy="ordinal encoding",
        direction="higher values indicate greater stress intensity",
        overlap_note=None,
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=False,
    ),
    "Q7": QuestionnaireColumnSchema(
        question_id="Q7",
        column_name="Q7. How difficult has it been for you to relax after completing your academic or professional work?  ",
        column_pattern="Q7. How difficult has it been for you to relax after completing your academic or professional work?",
        description="Difficulty relaxing after work.",
        data_type="ordered categorical",
        question_group="stress",
        ml_role="core stress feature",
        encoding_strategy="ordinal encoding",
        direction="higher values indicate greater stress intensity",
        overlap_note=None,
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=False,
    ),
    "Q8": QuestionnaireColumnSchema(
        question_id="Q8",
        column_name="Q8. How often have you found it difficult to concentrate because of stress?  ",
        column_pattern="Q8. How often have you found it difficult to concentrate because of stress?",
        description="Stress-related concentration difficulty.",
        data_type="ordered categorical",
        question_group="stress",
        ml_role="core stress feature",
        encoding_strategy="ordinal encoding",
        direction="higher values indicate greater stress intensity",
        overlap_note=None,
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=False,
    ),
    "Q9": QuestionnaireColumnSchema(
        question_id="Q9",
        column_name="Q9. How often has stress affected your sleep?  ",
        column_pattern="Q9. How often has stress affected your sleep?",
        description="Degree to which stress has affected sleep.",
        data_type="ordered categorical",
        question_group="stress",
        ml_role="core stress feature",
        encoding_strategy="ordinal encoding",
        direction="higher values indicate greater stress intensity",
        overlap_note=None,
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=False,
    ),
    "Q10": QuestionnaireColumnSchema(
        question_id="Q10",
        column_name="Q10. What factors are currently contributing to your stress?  ",
        column_pattern="Q10. What factors are currently contributing to your stress?",
        description="Factors currently contributing to stress.",
        data_type="multi-select categorical",
        question_group="context",
        ml_role="contextual/explanatory feature",
        encoding_strategy="multi-hot",
        direction=None,
        overlap_note=None,
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=True,
        canonical_options=[
            "Academic performance/grades",
            "Academic workload",
            "Assignments",
            "Coding/technical assessments",
            "Competition with peers",
            "Difficulty balancing academics and personal life",
            "Exams",
            "Family expectations",
            "Fear of not getting a job",
            "Financial concerns",
            "Internship/job applications",
            "Lack of sleep",
            "Placement preparation",
            "Projects",
            "Relationship/social concerns",
            "Time management",
            "Uncertainty about career",
            "Other",
        ],
    ),
    "Q11": QuestionnaireColumnSchema(
        question_id="Q11",
        column_name="Q11. Which emotions have you experienced most frequently during the past week?  ",
        column_pattern="Q11. Which emotions have you experienced most frequently during the past week?",
        description="Emotions experienced most frequently during the past week.",
        data_type="multi-select categorical",
        question_group="emotional",
        ml_role="emotional feature",
        encoding_strategy="multi-hot",
        direction=None,
        overlap_note=None,
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=True,
        canonical_options=[
            "Anxious",
            "Calm",
            "Confident",
            "Frustrated",
            "Happy",
            "Irritated",
            "Lonely",
            "Motivated",
            "Neutral",
            "Other",
            "Overwhelmed",
            "Sad",
            "Tired",
            "Worried",
        ],
    ),
    "Q12": QuestionnaireColumnSchema(
        question_id="Q12",
        column_name="Q12. How emotionally overwhelmed do you feel right now?  ",
        column_pattern="Q12. How emotionally overwhelmed do you feel right now?",
        description="Current emotional overwhelm.",
        data_type="1–5 ordered numeric scale",
        question_group="emotional",
        ml_role="emotional feature",
        encoding_strategy="numeric passthrough",
        direction="higher values indicate greater emotional overwhelm",
        overlap_note=None,
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=False,
    ),
    "Q13": QuestionnaireColumnSchema(
        question_id="Q13",
        column_name="Q13. How capable do you currently feel of handling your responsibilities?  ",
        column_pattern="Q13. How capable do you currently feel of handling your responsibilities?",
        description="Perceived capability to handle responsibilities.",
        data_type="1–5 ordered numeric scale",
        question_group="coping",
        ml_role="coping feature",
        encoding_strategy="numeric passthrough",
        direction="higher values indicate greater coping capability; opposite to stress intensity",
        overlap_note=None,
        reason="Higher values indicate greater coping capability, so the direction is opposite to stress intensity.",
        include_in_structured_ml=True,
        requires_special_handling=True,
    ),
    "Q14": QuestionnaireColumnSchema(
        question_id="Q14",
        column_name="Q14. When you experience significant stress, how long does it typically remain noticeable?  ",
        column_pattern="Q14. When you experience significant stress, how long does it typically remain noticeable?",
        description="Duration of significant stress.",
        data_type="ordered categorical with special \"It varies considerably\" response",
        question_group="stress/recovery",
        ml_role="stress/recovery feature",
        encoding_strategy="ordinal encoding with special handling",
        direction=None,
        overlap_note='The response "It varies considerably" must not be treated as simply the highest ordinal category.',
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=True,
    ),
    "Q15": QuestionnaireColumnSchema(
        question_id="Q15",
        column_name="Q15. How often do you experience sudden periods of intense stress that later return to normal?  ",
        column_pattern="Q15. How often do you experience sudden periods of intense stress that later return to normal?",
        description="Frequency of sudden intense stress returning to normal.",
        data_type="ordered categorical",
        question_group="stress/recovery",
        ml_role="stress/recovery feature",
        encoding_strategy="ordinal encoding",
        direction=None,
        overlap_note=None,
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=False,
    ),
    "Q16": QuestionnaireColumnSchema(
        question_id="Q16",
        column_name="Q16. After a stressful event, how quickly do you usually feel back to your normal state?  ",
        column_pattern="Q16. After a stressful event, how quickly do you usually feel back to your normal state?",
        description="Recovery time after a stressful event.",
        data_type="ordered categorical with special \"It varies considerably\" response",
        question_group="recovery",
        ml_role="recovery feature",
        encoding_strategy="ordinal encoding with special handling",
        direction=None,
        overlap_note='The response "It varies considerably" must not be treated as simply the highest ordinal category.',
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=True,
    ),
    "Q17": QuestionnaireColumnSchema(
        question_id="Q17",
        column_name="Q17. When you are stressed, how often do you have difficulty starting or completing tasks? ",
        column_pattern="Q17. When you are stressed, how often do you have difficulty starting or completing tasks?",
        description="Difficulty starting/completing tasks under stress.",
        data_type="ordered categorical",
        question_group="behavioral",
        ml_role="self-reported behavioral feature",
        encoding_strategy="ordinal encoding",
        direction="higher values indicate more frequent behavioral difficulty",
        overlap_note="Potentially overlapping with the separate behavioral modality.",
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=True,
    ),
    "Q18": QuestionnaireColumnSchema(
        question_id="Q18",
        column_name="Q18. When you are under pressure, how often do you postpone tasks even when you know they are important?  ",
        column_pattern="Q18. When you are under pressure, how often do you postpone tasks even when you know they are important?",
        description="Postponing important tasks under pressure.",
        data_type="ordered categorical",
        question_group="behavioral",
        ml_role="self-reported behavioral feature",
        encoding_strategy="ordinal encoding",
        direction="higher values indicate more frequent task postponement",
        overlap_note="Potentially overlapping with the separate behavioral modality.",
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=True,
    ),
    "Q19": QuestionnaireColumnSchema(
        question_id="Q19",
        column_name="Q19. When stressed, how often do you repeatedly check, change, or redo your work? ",
        column_pattern="Q19. When stressed, how often do you repeatedly check, change, or redo your work?",
        description="Repeatedly checking, changing, or redoing work under stress.",
        data_type="ordered categorical",
        question_group="behavioral",
        ml_role="self-reported behavioral feature",
        encoding_strategy="ordinal encoding",
        direction="higher values indicate more frequent rework behavior",
        overlap_note="Potentially overlapping with the separate behavioral modality.",
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=True,
    ),
    "Q20": QuestionnaireColumnSchema(
        question_id="Q20",
        column_name="Q20. When stressed, how often do you find yourself taking longer than usual to make decisions?  ",
        column_pattern="Q20. When stressed, how often do you find yourself taking longer than usual to make decisions?",
        description="Taking longer to make decisions under stress.",
        data_type="ordered categorical",
        question_group="behavioral",
        ml_role="self-reported behavioral feature",
        encoding_strategy="ordinal encoding",
        direction="higher values indicate more frequent decision delay",
        overlap_note="Potentially overlapping with the separate behavioral modality.",
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=True,
    ),
    "Q21": QuestionnaireColumnSchema(
        question_id="Q21",
        column_name="Q21. What is currently causing you the most stress?  ",
        column_pattern="Q21. What is currently causing you the most stress?",
        description="Current main stressor.",
        data_type="categorical/free-response contextual variable",
        question_group="context",
        ml_role="context feature",
        encoding_strategy="normalization / categorical mapping",
        direction=None,
        overlap_note=None,
        reason="Will require normalization during preprocessing.",
        include_in_structured_ml=True,
        requires_special_handling=True,
    ),
    "Q22": QuestionnaireColumnSchema(
        question_id="Q22",
        column_name="Q22. Describe how you usually feel when this stress becomes difficult to manage.  ",
        column_pattern="Q22. Describe how you usually feel when this stress becomes difficult to manage?",
        description="Description of feelings when stress becomes difficult to manage.",
        data_type="free text",
        question_group="text",
        ml_role="optional text/explanatory feature",
        encoding_strategy="excluded from first structured numerical ML model",
        direction=None,
        overlap_note=None,
        reason="Free text; exclude from first structured numerical ML model.",
        include_in_structured_ml=False,
        requires_special_handling=True,
    ),
    "Q23": QuestionnaireColumnSchema(
        question_id="Q23",
        column_name="Q23. What usually helps you feel better or recover after experiencing stress?  ",
        column_pattern="Q23. What usually helps you feel better or recover after experiencing stress?",
        description="Recovery strategies.",
        data_type="free text",
        question_group="text",
        ml_role="optional text/explanatory feature",
        encoding_strategy="excluded from first structured numerical ML model",
        direction=None,
        overlap_note=None,
        reason="Free text; exclude from first structured numerical ML model.",
        include_in_structured_ml=False,
        requires_special_handling=True,
    ),
    "Q24": QuestionnaireColumnSchema(
        question_id="Q24",
        column_name="Q24. Would you be willing to participate in a follow-up session for this research project? ",
        column_pattern="Q24. Would you be willing to participate in a follow-up session for this research project?",
        description="Willingness to participate in follow-up.",
        data_type="binary research metadata",
        question_group="metadata",
        ml_role="exclude",
        encoding_strategy="none",
        direction=None,
        overlap_note=None,
        reason="binary research metadata; must not be used as a stress feature",
        include_in_structured_ml=False,
        requires_special_handling=False,
    ),
    "Q25": QuestionnaireColumnSchema(
        question_id="Q25",
        column_name="Q25. Which type of situation most strongly affects your stress?   ",
        column_pattern="Q25. Which type of situation most strongly affects your stress?",
        description="Type of situation most strongly affecting stress.",
        data_type="categorical",
        question_group="context",
        ml_role="context feature",
        encoding_strategy="category encoding",
        direction=None,
        overlap_note=None,
        reason=None,
        include_in_structured_ml=True,
        requires_special_handling=False,
    ),
}


def get_full_schema() -> Dict[str, QuestionnaireColumnSchema]:
    """Return the complete questionnaire schema registry.

    Returns:
        A dictionary keyed by canonical question ID (e.g. "Q1", "Q2", ...).
    """

    return dict(QUESTIONNAIRE_SCHEMA)


def get_question_schema(question_id: str) -> QuestionnaireColumnSchema:
    """Return the schema metadata for one canonical question.

    Args:
        question_id: Canonical question identifier such as "Q1".

    Returns:
        The metadata object for that question.

    Raises:
        KeyError: If the question ID is not in the registry.
    """

    try:
        return QUESTIONNAIRE_SCHEMA[question_id]
    except KeyError as exc:
        raise KeyError(f"Unsupported question_id: {question_id!r}") from exc


def get_structured_ml_features() -> List[QuestionnaireColumnSchema]:
    """Return questions intended for the first structured ML feature set.

    This helper excludes questionnaire columns that are metadata, free text, or
    otherwise intentionally omitted from the first structured numerical model.
    """

    return [schema for schema in QUESTIONNAIRE_SCHEMA.values() if schema.include_in_structured_ml]


def get_excluded_questions() -> List[QuestionnaireColumnSchema]:
    """Return questions that are excluded from the structured ML feature set."""

    return [schema for schema in QUESTIONNAIRE_SCHEMA.values() if not schema.include_in_structured_ml]


def get_questions_requiring_special_handling() -> List[QuestionnaireColumnSchema]:
    """Return questions needing non-trivial preprocessing or encoding logic."""

    return [schema for schema in QUESTIONNAIRE_SCHEMA.values() if schema.requires_special_handling]


__all__ = [
    "QuestionnaireColumnSchema",
    "QUESTIONNAIRE_SCHEMA",
    "get_full_schema",
    "get_question_schema",
    "get_structured_ml_features",
    "get_excluded_questions",
    "get_questions_requiring_special_handling",
]
