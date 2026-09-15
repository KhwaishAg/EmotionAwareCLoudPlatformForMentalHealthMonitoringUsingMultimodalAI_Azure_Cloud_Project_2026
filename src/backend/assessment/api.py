"""Unified multimodal assessment API.

POST /assessment accepts whichever raw modality inputs the student actually
provided (questionnaire responses, behavioral self-report, voice recording)
and returns one fused assessment. Every modality is optional; each is run in
its own try/except so a missing or failing modality is marked unavailable
instead of crashing the whole request -- this is the actual "integration"
contribution of Member 3's work, not a side effect.

This module wraps, but does not modify, the already-validated components:
    - src.ai_model.questionnaire.predict.predict_questionnaire
    - src.ai_model.behavioral.predict.predict_behavioral
    - src.ai_model.voice.predict.predict_voice   (raises VoiceModelNotTrainedError
      today -- no trained voice model exists yet; see that module's docstring)
    - src.ai_model.fusion.fusion_engine.fuse_modalities
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator

from src.ai_model.questionnaire.predict import REQUIRED_QUESTION_FIELDS, predict_questionnaire
from src.ai_model.behavioral.predict import predict_behavioral
from src.ai_model.behavioral.behavioral_schema import REQUIRED_BEHAVIORAL_FIELDS
from src.ai_model.fusion.fusion_engine import fuse_modalities

try:
    from src.ai_model.voice.predict import predict_voice, VoiceModelNotTrainedError
except ImportError:  # pragma: no cover - voice package not importable in this environment
    predict_voice = None  # type: ignore[assignment]
    VoiceModelNotTrainedError = Exception  # type: ignore[assignment,misc]

logger = logging.getLogger("assessment_api")

app = FastAPI(title="Multimodal Student Assessment API", version="1.0.0")


# ---------------------------------------------------------------------------
# Request schema -- every modality is optional.
# ---------------------------------------------------------------------------

class QuestionnaireInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    responses: dict[str, Any]

    @field_validator("responses")
    @classmethod
    def validate_required_fields(cls, responses: dict[str, Any]) -> dict[str, Any]:
        missing = [field for field in REQUIRED_QUESTION_FIELDS if field not in responses]
        if missing:
            raise ValueError("Missing required questionnaire fields: " + ", ".join(missing))
        return responses


class BehavioralInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    responses: dict[str, Any]

    @field_validator("responses")
    @classmethod
    def validate_required_fields(cls, responses: dict[str, Any]) -> dict[str, Any]:
        unsupported = [key for key in responses if key not in REQUIRED_BEHAVIORAL_FIELDS]
        if unsupported:
            raise ValueError("Unsupported behavioral fields: " + ", ".join(unsupported))
        return responses


class VoiceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Placeholder shape: acoustic feature dict. Once voice recording upload
    # is wired up, this becomes raw audio bytes/base64 and the API extracts
    # features itself via src.ai_model.voice.features. Left minimal on
    # purpose since no trained model consumes this yet.
    features: dict[str, Any]


class AssessmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questionnaire: Optional[QuestionnaireInput] = None
    behavioral: Optional[BehavioralInput] = None
    voice: Optional[VoiceInput] = None


# ---------------------------------------------------------------------------
# Per-modality runners -- each swallows its own failures and reports why.
# ---------------------------------------------------------------------------

def _run_questionnaire(payload: Optional[QuestionnaireInput]) -> tuple[Optional[dict], Optional[str]]:
    if payload is None:
        return None, "not_provided"
    try:
        return predict_questionnaire(payload.responses), None
    except FileNotFoundError:
        logger.error("Questionnaire model artifacts unavailable.")
        return None, "model_unavailable"
    except (TypeError, ValueError) as error:
        logger.warning("Questionnaire input rejected: %s", error)
        return None, "invalid_input"
    except Exception:
        logger.exception("Unexpected questionnaire prediction failure.")
        return None, "prediction_failed"


def _run_behavioral(payload: Optional[BehavioralInput]) -> tuple[Optional[dict], Optional[str]]:
    if payload is None:
        return None, "not_provided"
    try:
        return predict_behavioral(payload.responses), None
    except FileNotFoundError:
        logger.error("Behavioral model artifacts unavailable.")
        return None, "model_unavailable"
    except (TypeError, ValueError) as error:
        logger.warning("Behavioral input rejected: %s", error)
        return None, "invalid_input"
    except Exception:
        logger.exception("Unexpected behavioral prediction failure.")
        return None, "prediction_failed"


def _run_voice(payload: Optional[VoiceInput]) -> tuple[Optional[dict], Optional[str]]:
    if payload is None:
        return None, "not_provided"
    if predict_voice is None:
        return None, "module_unavailable"
    try:
        return predict_voice(payload.features), None
    except VoiceModelNotTrainedError:
        # Expected today: no trained voice model exists yet. This is not a
        # bug -- it is the correct, honest state of an optional modality
        # that has infrastructure but no dataset/training run yet.
        logger.info("Voice modality skipped: no trained model available yet.")
        return None, "model_not_trained"
    except (TypeError, ValueError) as error:
        logger.warning("Voice input rejected: %s", error)
        return None, "invalid_input"
    except Exception:
        logger.exception("Unexpected voice prediction failure.")
        return None, "prediction_failed"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/assessment")
def assess(request: AssessmentRequest) -> dict:
    """Run every provided modality, fuse the usable results, and return one assessment.

    Never raises for a missing or failing individual modality -- that is
    reflected in `modality_status` and in the fusion result's own
    `modalities_used` / `status` fields. A 422 is only returned when the
    request body itself is unparseable (handled by FastAPI/pydantic before
    this function runs); a 500 is reserved for the fusion step itself
    failing, which should not happen given fuse_modalities' own contract
    validation.
    """
    questionnaire_result, questionnaire_status = _run_questionnaire(request.questionnaire)
    behavioral_result, behavioral_status = _run_behavioral(request.behavioral)
    voice_result, voice_status = _run_voice(request.voice)

    try:
        fused = fuse_modalities(
            questionnaire=questionnaire_result,
            voice=voice_result,
            behavioral=behavioral_result,
        )
    except ValueError as error:
        # A modality result failed fuse_modalities' own strict contract
        # validation -- this indicates an internal bug in one of the
        # predictors above, not a client error, so surface it as a 500
        # rather than misleadingly blaming the request.
        logger.exception("Fusion contract validation failed unexpectedly.")
        raise HTTPException(status_code=500, detail=f"Fusion could not be completed: {error}") from error

    return {
        "assessment": fused,
        "modality_status": {
            "questionnaire": questionnaire_status or "available",
            "behavioral": behavioral_status or "available",
            "voice": voice_status or "available",
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
