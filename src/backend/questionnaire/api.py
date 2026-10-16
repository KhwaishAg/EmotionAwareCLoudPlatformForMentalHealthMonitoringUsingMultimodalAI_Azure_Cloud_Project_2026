"""Minimal HTTP adapter for the existing questionnaire prediction contract."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator

from src.ai_model.questionnaire.predict import REQUIRED_QUESTION_FIELDS, predict_questionnaire


class QuestionnaireRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    responses: dict[str, Any]

    @field_validator("responses")
    @classmethod
    def validate_required_fields(cls, responses: dict[str, Any]) -> dict[str, Any]:
        missing = [field for field in REQUIRED_QUESTION_FIELDS if field not in responses]
        if missing:
            raise ValueError("Missing required questionnaire fields: " + ", ".join(missing))
        return responses


app = FastAPI(title="Questionnaire Prediction API", version="1.0.0")


@app.post("/predict/questionnaire")
def predict(request: QuestionnaireRequest) -> dict:
    """Return the existing non-clinical questionnaire fusion contract as JSON."""
    try:
        return predict_questionnaire(request.responses)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Questionnaire model artifacts are unavailable.")
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error))
    except Exception:
        raise HTTPException(status_code=500, detail="Questionnaire prediction could not be completed.")
