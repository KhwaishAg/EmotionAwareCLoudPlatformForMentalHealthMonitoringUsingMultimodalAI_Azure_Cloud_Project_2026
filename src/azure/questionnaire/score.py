"""Azure ML managed online endpoint scoring adapter for the questionnaire modality.

This module is the ``scoring_script`` entry point required by Azure ML managed
online deployments.  It is a thin adapter only:

- ``init()`` loads the model artifacts from the Azure ML model directory once
  at container startup.
- ``run()`` delegates all inference to the existing
  ``src.ai_model.questionnaire.predict`` module without reimplementing any
  model or preprocessing logic.

Architecture note
-----------------
The Azure ML runtime copies the registered ``questionnaire-random-forest``
model artifacts into the directory returned by
``os.environ["AZUREML_MODEL_DIR"]``.  This adapter redirects
``predict.MODELS_DIR`` to that runtime path so that ``predict_questionnaire``
loads the Azure-hosted artifacts instead of the local repository path.

No participant-level questionnaire responses are logged.
No model training, retraining, or preprocessing refitting is performed here.
This model is a non-clinical project-defined assessment tool.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrap — make the repository package importable inside the Azure ML
# container, where the code snapshot is copied to the working directory.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
# When Azure ML copies the code snapshot, the project root lands at:
#   <container_working_dir>/  (containing src/, models/, etc.)
# Resolve upward until we find the ``src`` package directory.
for _candidate in [_THIS_FILE.parents[3], _THIS_FILE.parents[2], Path.cwd()]:
    if (_candidate / "src").is_dir():
        _PROJECT_ROOT = _candidate
        break
else:
    _PROJECT_ROOT = _THIS_FILE.parents[3]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ai_model.questionnaire import predict as _predict_module
from src.ai_model.questionnaire.predict import (
    REQUIRED_QUESTION_FIELDS,
    predict_questionnaire,
)


def init() -> None:
    """Load model artifacts from the Azure ML model directory.

    Azure ML calls this once when the deployment container starts.
    The ``AZUREML_MODEL_DIR`` environment variable points to the directory
    where the registered model artifacts were downloaded. We redirect the
    ``predict`` module's ``MODELS_DIR`` to that path so that subsequent
    ``predict_questionnaire`` calls load the Azure-registered artifacts.
    """
    model_dir_env = os.environ.get("AZUREML_MODEL_DIR")
    if model_dir_env:
        azure_models_dir = Path(model_dir_env)
        # Search candidate directories in priority order for selected_model.json
        candidate_dirs = [
            azure_models_dir,
            azure_models_dir / "models",
            azure_models_dir / "models" / "questionnaire",
            azure_models_dir / "questionnaire",
        ]
        resolved_dir = None
        for candidate in candidate_dirs:
            if candidate.is_dir() and (candidate / "selected_model.json").is_file():
                resolved_dir = candidate
                break

        if not resolved_dir:
            matches = list(azure_models_dir.rglob("selected_model.json"))
            if matches:
                resolved_dir = matches[0].parent

        if resolved_dir:
            _predict_module.MODELS_DIR = resolved_dir
        elif azure_models_dir.is_dir():
            _predict_module.MODELS_DIR = azure_models_dir

        logger.info("Questionnaire scoring adapter: MODELS_DIR = %s", _predict_module.MODELS_DIR)
    else:
        # Fallback for local testing without an Azure ML container.
        logger.warning(
            "AZUREML_MODEL_DIR not set; using default local MODELS_DIR: %s",
            _predict_module.MODELS_DIR,
        )

    # Validate that required artifacts are present at startup.
    required = [
        "selected_model.json",
        "questionnaire_feature_names.json",
        "explainability_metadata.json",
        "random_forest.joblib",
    ]
    missing = [name for name in required if not (_predict_module.MODELS_DIR / name).is_file()]
    if missing:
        raise RuntimeError(
            "Questionnaire scoring adapter: required artifacts not found at "
            f"{_predict_module.MODELS_DIR}: {', '.join(missing)}"
        )
    logger.info("Questionnaire scoring adapter initialised successfully.")


def run(raw_data: str) -> str:
    """Score one questionnaire response and return the fusion contract as JSON.

    Azure ML calls this for each inference request.  ``raw_data`` is the
    request body as a UTF-8 string.

    Expected input format::

        {"responses": {"Q1": "...", "Q2": "...", ..., "Q20": "..."}}

    Q1–Q20 are required.  Q21–Q25 and Timestamp are optional.

    Returns the questionnaire modality fusion contract::

        {
            "modality": "questionnaire",
            "risk_score": <float 0-1>,
            "risk_level": "<low|moderate|high>",
            "confidence": <float 0-1>,
            "available_features": <int>,
            "total_features": <int>,
            "key_factors": [...],
            "data_quality": {"missing_features": <int>, "completeness": <float>}
        }

    Non-clinical disclaimer: risk_level and risk_score are project-defined
    categories; confidence is a model heuristic, not clinical certainty.
    """
    try:
        payload = json.loads(raw_data)
    except (json.JSONDecodeError, TypeError) as exc:
        return json.dumps({"error": f"Invalid JSON input: {exc}", "status_code": 400})

    if not isinstance(payload, dict) or "responses" not in payload:
        return json.dumps(
            {"error": "Request body must be JSON with a 'responses' key.", "status_code": 400}
        )

    responses = payload["responses"]
    if not isinstance(responses, dict):
        return json.dumps({"error": "'responses' must be a JSON object.", "status_code": 400})

    missing_fields = [f for f in REQUIRED_QUESTION_FIELDS if f not in responses]
    if missing_fields:
        return json.dumps(
            {
                "error": "Missing required questionnaire fields: " + ", ".join(missing_fields),
                "status_code": 422,
            }
        )

    try:
        result = predict_questionnaire(responses)
    except FileNotFoundError:
        logger.error("Questionnaire model artifacts unavailable.")
        return json.dumps(
            {"error": "Questionnaire model artifacts are unavailable.", "status_code": 503}
        )
    except (TypeError, ValueError) as exc:
        logger.warning("Questionnaire input validation error: %s", exc)
        return json.dumps({"error": str(exc), "status_code": 422})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected questionnaire scoring error.")
        return json.dumps(
            {"error": "Questionnaire prediction could not be completed.", "status_code": 500}
        )

    return json.dumps(result)
