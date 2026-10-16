"""Unified multimodal assessment API.

POST /assessment accepts whichever raw modality inputs the student actually
provided (questionnaire responses, behavioral self-report, voice recording or
pre-extracted voice features) and returns one fused assessment. Every modality
is optional; each is run in its own try/except so a missing or failing
modality is marked unavailable instead of crashing the whole request.

Unavailable-modality policy (explicit, not silent):
    * A modality that was not provided, has no trained model, or failed is
      left out of fusion -- it is never zero-filled or guessed.
    * The reason is reported per modality in ``modality_status``.
    * Fusion re-normalizes weights over the modalities actually used and lists
      them in ``assessment.modalities_used``.
    * If no modality is usable, ``assessment.status`` is ``fusion_unavailable``
      and the risk fields are ``null`` (HTTP 200: the request was valid, there
      was just nothing to assess).

This module wraps, but does not modify, the validated components:
    - src.ai_model.questionnaire.predict.predict_questionnaire
    - src.ai_model.behavioral.predict.predict_behavioral
    - src.ai_model.voice.predict.predict_voice_from_features /
      predict_voice_contract_from_audio  (VoiceModelNotTrainedError until a
      voice model is trained)
    - src.ai_model.fusion.fusion_engine.fuse_modalities

Non-clinical: outputs are stress/risk indicators for early-intervention
support, not a diagnosis.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import secrets
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from src.ai_model.behavioral.behavioral_schema import FIELD_RANGES, REQUIRED_BEHAVIORAL_FIELDS
from src.ai_model.behavioral.predict import predict_behavioral
from src.ai_model.fusion.fusion_engine import fuse_modalities
from src.ai_model.questionnaire.predict import REQUIRED_QUESTION_FIELDS, predict_questionnaire
from src.cloud import events as cloud_events
from src.cloud import history as cloud_history
from src.cloud.config import is_hosted, load_settings
from src.cloud.http import CloudServiceError
from src.cloud.voice_content import NoSpeechError
from src.cloud.voice_content import assess_audio as assess_speech_content

try:
    from src.ai_model.voice.predict import (
        PredictionAudioError,
        PredictionFeatureError,
        VoiceModelLoadError,
        VoiceModelNotTrainedError,
        predict_voice_contract_from_audio,
        predict_voice_from_features,
        read_voice_training_report,
    )
except ImportError:  # pragma: no cover - voice dependencies not importable in this environment
    predict_voice_from_features = None  # type: ignore[assignment]
    predict_voice_contract_from_audio = None  # type: ignore[assignment]
    VoiceModelNotTrainedError = type("VoiceModelNotTrainedError", (Exception,), {})  # type: ignore[assignment,misc]
    PredictionAudioError = type("PredictionAudioError", (Exception,), {})  # type: ignore[assignment,misc]
    PredictionFeatureError = type("PredictionFeatureError", (Exception,), {})  # type: ignore[assignment,misc]
    VoiceModelLoadError = type("VoiceModelLoadError", (Exception,), {})  # type: ignore[assignment,misc]
    read_voice_training_report = lambda: None  # type: ignore[assignment]  # noqa: E731

try:
    from src.ai_model.voice.emotion import (
        EmotionModelLoadError,
        EmotionModelNotTrainedError,
        predict_emotion_from_audio,
        predict_emotion_from_features,
    )
except ImportError:  # pragma: no cover
    predict_emotion_from_audio = None  # type: ignore[assignment]
    predict_emotion_from_features = None  # type: ignore[assignment]
    EmotionModelNotTrainedError = type("EmotionModelNotTrainedError", (Exception,), {})  # type: ignore[assignment,misc]
    EmotionModelLoadError = type("EmotionModelLoadError", (Exception,), {})  # type: ignore[assignment,misc]

logger = logging.getLogger("assessment_api")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_DIR = PROJECT_ROOT / "src" / "frontend"

# The shipped questionnaire/behavioral artifacts were serialized with this
# scikit-learn version (it needs Python >= 3.11); other versions may fail to
# load or predict.
EXPECTED_SKLEARN_VERSION = "1.8.0"
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # decoded WAV upload limit

DISCLAIMER = (
    "Non-clinical research/demonstration tool. Results are stress/risk indicators "
    "for early-intervention support, not a medical or psychological diagnosis."
)

# Zero-cost, fully local by default: the interactive API docs page loads its assets from a public CDN,
# so it is off unless explicitly enabled (ENABLE_API_DOCS=1). The app itself makes no network calls.
_DOCS_ENABLED = os.environ.get("ENABLE_API_DOCS") == "1"
app = FastAPI(
    title="Multimodal Student Assessment API",
    version="1.2.0",
    docs_url="/docs" if _DOCS_ENABLED else None,
    redoc_url=None,
)

# Privacy: nothing the user submits is stored by this service, and responses must
# not be cached by browsers or proxies. The UI is served with a strict CSP.
_UI_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; media-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)


def _access_credentials() -> Optional[tuple[str, str]]:
    """Optional shared password for public hosting. Off unless BOTH env vars are set (never in code)."""
    user, password = os.environ.get("APP_ACCESS_USER"), os.environ.get("APP_ACCESS_PASSWORD")
    return (user, password) if user and password else None


def _is_authorized(request: Request) -> bool:
    credentials = _access_credentials()
    if credentials is None or request.url.path == "/health":  # /health lets a host probe liveness; it exposes nothing
        return True
    if request.url.path == "/events":  # Event Grid webhook: protected by its own shared secret instead
        return True
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        return False
    try:
        user, _, password = base64.b64decode(header[6:], validate=True).decode("utf-8").partition(":")
    except (binascii.Error, ValueError):
        return False
    return secrets.compare_digest(user.encode(), credentials[0].encode()) and secrets.compare_digest(
        password.encode(), credentials[1].encode()
    )


@app.middleware("http")
async def privacy_headers(request: Request, call_next):
    if _is_authorized(request):
        response = await call_next(request)
    else:
        response = Response("Authentication required.", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Student Stress Check-in"'})
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "microphone=(self), camera=(), geolocation=()"
    if request.url.path == "/" or request.url.path.startswith("/app"):
        response.headers["Content-Security-Policy"] = _UI_CSP
    return response


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
    """Voice input: either a base64-encoded WAV recording or extracted features."""

    model_config = ConfigDict(extra="forbid")
    audio_base64: Optional[str] = None
    features: Optional[dict[str, Any]] = None
    # Opt-in: while no acoustic stress model is trained, send the recording to Azure AI Speech + Language
    # (when configured) and use the tone of what was said as the voice signal.
    cloud_speech: bool = False

    @model_validator(mode="after")
    def exactly_one_source(self) -> "VoiceInput":
        if (self.audio_base64 is None) == (self.features is None):
            raise ValueError("Provide exactly one of 'audio_base64' or 'features'.")
        if self.audio_base64 is not None and len(self.audio_base64) > (MAX_AUDIO_BYTES * 4) // 3 + 8:
            raise ValueError("Audio upload is too large.")
        return self


class HistoryOptIn(BaseModel):
    """Opt-in to saving an anonymised summary. anon_id is a random UUID the browser generated."""

    model_config = ConfigDict(extra="forbid")
    anon_id: Optional[str] = None


class AssessmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questionnaire: Optional[QuestionnaireInput] = None
    behavioral: Optional[BehavioralInput] = None
    voice: Optional[VoiceInput] = None
    history: Optional[HistoryOptIn] = None


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


@contextmanager
def _temp_wav(audio_base64: str):
    """Decode an uploaded WAV to a private temp file for the duration of the block; always deleted."""
    try:
        raw = base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("audio_base64 is not valid base64.") from error
    if not raw or len(raw) > MAX_AUDIO_BYTES:
        raise ValueError("Audio is empty or exceeds the upload size limit.")
    handle, name = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(handle, "wb") as target:
            target.write(raw)
        yield name
    finally:
        try:
            os.remove(name)
        except OSError:  # pragma: no cover - best-effort cleanup
            logger.warning("Could not remove temporary audio file.")


def _predict_voice_audio(audio_base64: str) -> dict:
    with _temp_wav(audio_base64) as path:
        return predict_voice_contract_from_audio(path)


def _run_voice(payload: Optional[VoiceInput]) -> tuple[Optional[dict], Optional[str]]:
    if payload is None:
        return None, "not_provided"
    if predict_voice_from_features is None:
        return None, "module_unavailable"
    try:
        if payload.features is not None:
            return predict_voice_from_features(payload.features), None
        return _predict_voice_audio(payload.audio_base64 or ""), None
    except VoiceModelNotTrainedError:
        if payload.audio_base64 is not None and payload.cloud_speech and load_settings().speech_content_enabled:
            return _run_speech_content(payload.audio_base64)
        # Expected until a voice model is trained: honest "unavailable", not a bug.
        logger.info("Voice modality skipped: no trained model available yet.")
        return None, "model_not_trained"
    except VoiceModelLoadError:
        # Artifact present but unusable (corrupt, wrong feature set, demo-only model).
        logger.error("Voice model artifact could not be used.")
        return None, "model_unavailable"
    except (PredictionAudioError, PredictionFeatureError, TypeError, ValueError) as error:
        logger.warning("Voice input rejected: %s", error)
        return None, "invalid_input"
    except Exception:
        logger.exception("Unexpected voice prediction failure.")
        return None, "prediction_failed"


def _run_speech_content(audio_base64: str) -> tuple[Optional[dict], Optional[str]]:
    """Voice via Azure AI Speech (transcript) + AI Language (sentiment). Audio/transcript are not stored."""
    try:
        with _temp_wav(audio_base64) as path:
            return assess_speech_content(path, load_settings()), None
    except NoSpeechError:
        return None, "no_speech_detected"
    except CloudServiceError as error:
        logger.warning("Speech-content analysis unavailable: %s", error)
        return None, "speech_service_unavailable"
    except (TypeError, ValueError, RuntimeError) as error:
        logger.warning("Speech-content input rejected: %s", error)
        return None, "invalid_input"
    except Exception:
        logger.exception("Unexpected speech-content failure.")
        return None, "prediction_failed"


def _run_voice_emotion(payload: Optional[VoiceInput]) -> tuple[Optional[dict], Optional[str]]:
    """Experimental acted-emotion category. Separate from stress risk: never fused, never in modality_status."""
    if payload is None:
        return None, None
    if predict_emotion_from_audio is None:
        return None, "module_unavailable"
    try:
        if payload.features is not None:
            return predict_emotion_from_features(payload.features), "available"
        with _temp_wav(payload.audio_base64 or "") as path:
            return predict_emotion_from_audio(path), "available"
    except EmotionModelNotTrainedError:
        return None, "model_not_trained"
    except EmotionModelLoadError:
        logger.error("Emotion model artifact could not be used.")
        return None, "model_unavailable"
    except (PredictionAudioError, PredictionFeatureError, TypeError, ValueError) as error:
        logger.warning("Emotion input rejected: %s", error)
        return None, "invalid_input"
    except Exception:
        logger.exception("Unexpected emotion prediction failure.")
        return None, "prediction_failed"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

VOICE_DETAIL_EXTRAS = ("source", "sentiment", "transcript", "key_phrases", "caveat")


def _principal(http_request: Optional[Request]) -> Optional[str]:
    return http_request.headers.get("x-ms-client-principal-id") if http_request is not None else None


def _save_history(opt_in: Optional[HistoryOptIn], fused: dict, http_request: Optional[Request]) -> Optional[str]:
    """Opt-in only. Event Grid (-> /events -> Cosmos) when configured, else straight to Cosmos."""
    if opt_in is None:
        return None
    settings = load_settings()
    if not settings.history_enabled:
        return "unavailable"
    if fused.get("status") != "available":
        return "skipped"
    owner = cloud_history.resolve_owner(opt_in.anon_id, _principal(http_request), settings)
    if owner is None:
        return "invalid_id"
    record = cloud_history.build_record(owner, fused)
    try:
        if settings.events_enabled:
            cloud_events.publish(settings.eventgrid_endpoint or "", settings.eventgrid_key or "",
                                 cloud_events.ASSESSMENT_COMPLETED, "/assessments", record)
            return "queued"
        cloud_history.save(record, settings)
        return "saved"
    except CloudServiceError as error:
        logger.warning("History could not be saved: %s", error)
        return "failed"


@app.post("/assessment")
def assess(request: AssessmentRequest, http_request: Request) -> dict:
    """Run every provided modality, fuse the usable results, and return one assessment.

    Never raises for a missing or failing individual modality -- that is
    reflected in ``modality_status`` and in the fusion result's own
    ``modalities_used`` / ``status`` fields. A 422 is returned when the request
    body itself is invalid; a 500 is reserved for the fusion step failing its
    own contract validation (an internal bug, never a client error).
    """
    questionnaire_result, questionnaire_status = _run_questionnaire(request.questionnaire)
    behavioral_result, behavioral_status = _run_behavioral(request.behavioral)
    voice_result, voice_status = _run_voice(request.voice)
    emotion_result, emotion_status = _run_voice_emotion(request.voice)

    try:
        fused = fuse_modalities(
            questionnaire=questionnaire_result,
            voice=voice_result,
            behavioral=behavioral_result,
        )
    except ValueError:
        # A modality result failed the fusion contract: an internal predictor
        # bug. Log details server-side; return a generic message to the client.
        logger.exception("Fusion contract validation failed unexpectedly.")
        raise HTTPException(status_code=500, detail="Assessment could not be completed.") from None

    details = {
        name: {
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "confidence": result["confidence"],
            "completeness": result["data_quality"]["completeness"],
            **{key: result[key] for key in VOICE_DETAIL_EXTRAS if name == "voice" and key in result},
        }
        for name, result in (
            ("questionnaire", questionnaire_result),
            ("behavioral", behavioral_result),
            ("voice", voice_result),
        )
        if result is not None
    }
    return {
        "assessment": fused,
        "modality_details": details,
        # Experimental, acted-speech emotion category: separate from stress risk and never fused.
        "voice_emotion": None if emotion_status is None else {"status": emotion_status, "result": emotion_result},
        "modality_status": {
            "questionnaire": questionnaire_status or "available",
            "behavioral": behavioral_status or "available",
            "voice": voice_status or "available",
        },
        "history_status": _save_history(request.history, fused, http_request),
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Opt-in history (Cosmos DB) and the Event Grid webhook that fills it
# ---------------------------------------------------------------------------

@app.post("/events")
async def events_webhook(http_request: Request) -> Any:
    """Event Grid delivery endpoint. Requires ?key=<CLOUD_EVENTS_SECRET>; stores validated summaries."""
    settings = load_settings()
    supplied = http_request.query_params.get("key", "")
    if not settings.events_secret or not secrets.compare_digest(supplied.encode(), settings.events_secret.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized.")
    try:
        events = await http_request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON.") from None
    if not isinstance(events, list):
        raise HTTPException(status_code=400, detail="Expected a list of events.")
    stored = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("eventType") == cloud_events.SUBSCRIPTION_VALIDATION:
            return {"validationResponse": (event.get("data") or {}).get("validationCode")}
        if event.get("eventType") != cloud_events.ASSESSMENT_COMPLETED:
            continue
        try:
            record = cloud_history.validate_record(event.get("data"))
        except (TypeError, ValueError):
            logger.warning("Ignored malformed assessment event.")
            continue
        if not settings.cosmos_enabled:
            raise HTTPException(status_code=503, detail="History storage is not configured.")
        try:
            cloud_history.save(record, settings)
        except CloudServiceError as error:
            logger.warning("Event could not be stored: %s", error)
            raise HTTPException(status_code=503, detail="Storage unavailable; retry.") from None  # Event Grid retries
        stored += 1
    return {"stored": stored}


def _history_owner(anon_id: Optional[str], http_request: Request) -> str:
    settings = load_settings()
    if not settings.history_enabled:
        raise HTTPException(status_code=404, detail="History is not enabled on this deployment.")
    owner = cloud_history.resolve_owner(anon_id, _principal(http_request), settings)
    if owner is None:
        raise HTTPException(status_code=400, detail="A valid anonymous ID is required.")
    return owner


@app.get("/history")
def get_history(http_request: Request, anon_id: Optional[str] = None) -> dict:
    owner = _history_owner(anon_id, http_request)
    try:
        items = cloud_history.list_for(owner, load_settings())
    except CloudServiceError as error:
        logger.warning("History could not be read: %s", error)
        raise HTTPException(status_code=503, detail="History is temporarily unavailable.") from None
    return {"items": items, "trend": cloud_history.summarise_trend(items)}


@app.delete("/history")
def delete_history(http_request: Request, anon_id: Optional[str] = None) -> dict:
    owner = _history_owner(anon_id, http_request)
    try:
        deleted = cloud_history.delete_all(owner, load_settings())
    except CloudServiceError as error:
        logger.warning("History could not be deleted: %s", error)
        raise HTTPException(status_code=503, detail="History is temporarily unavailable.") from None
    return {"deleted": deleted}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _artifact_present(*relative: str) -> bool:
    return all((PROJECT_ROOT / part).exists() for part in relative)


def _voice_model_summary(present: bool) -> dict:
    """Provenance of the voice model (from its training report); never paths or raw data."""
    if not present:
        return {"trained": False}
    report = read_voice_training_report()
    if not report:
        return {"trained": True, "provenance": "unknown (no training report found)"}
    held_out = report.get("held_out_metrics") or {}
    split = (report.get("split") or {}).get("held_out_test") or {}
    return {
        "trained": True,
        "dataset_kind": report.get("dataset_kind"),
        "served": report.get("dataset_kind") != "synthetic_demo",
        "held_out_macro_f1": held_out.get("macro_f1"),
        "held_out_participants": split.get("participants"),
        "note": "Held-out figures describe the training dataset only; not clinical or real-world accuracy.",
    }


def _cloud_summary() -> dict:
    """Which optional free-tier Azure features are switched on (never keys or endpoints)."""
    settings = load_settings()
    return {
        "hosted": is_hosted(),
        "speech_content": settings.speech_content_enabled,
        "history": settings.history_enabled,
        "events": settings.events_enabled,
        "signed_in_history": settings.trust_easy_auth,
    }


@app.get("/status")
def status() -> dict:
    """Diagnostics: model artifact availability and environment compatibility.

    Reports only presence/versions -- never paths outside the project, secrets
    or user data.
    """
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except ImportError:  # pragma: no cover
        sklearn_version = None
    voice_trained = _artifact_present("models/voice/selected_model.json") or _artifact_present(
        "models/voice/voice_model.joblib"
    )
    return {
        "status": "ok",
        "models": {
            "questionnaire": _artifact_present("models/questionnaire/selected_model.json"),
            "behavioral": _artifact_present("models/behavioral/logistic_regression.joblib"),
            "voice": voice_trained,
        },
        "environment": {
            "scikit_learn": sklearn_version,
            "expected_scikit_learn": EXPECTED_SKLEARN_VERSION,
            "compatible": sklearn_version == EXPECTED_SKLEARN_VERSION,
        },
        "voice_model": _voice_model_summary(voice_trained),
        "voice_emotion_model": {"trained": _artifact_present("models/voice_emotion/selected_model.json")},
        "cloud": _cloud_summary(),
        "note": "Behavioral model was trained on synthetic data; voice model is untrained unless 'voice' is true.",
    }


def _read_json(relative: str) -> Optional[dict]:
    path = PROJECT_ROOT / relative
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _emotion_summary() -> dict:
    report = _read_json("models/voice_emotion/voice_training_report.json")
    caveat = ("Experimental acted-speech emotion categories; not a stress measure, never used in the overall "
              "assessment, not validated on students.")
    if not report:
        return {"trained": False, "caveat": caveat}
    held = report.get("held_out_metrics") or {}
    split = (report.get("split") or {}).get("held_out_test") or {}
    return {
        "trained": True,
        "dataset": (report.get("dataset") or {}).get("name"),
        "classes": report.get("classes"),
        "held_out_accuracy": held.get("accuracy"),
        "held_out_macro_f1": held.get("macro_f1"),
        "held_out_speakers": split.get("participants"),
        "chance_accuracy": 1 / len(report["classes"]) if report.get("classes") else None,
        "caveat": caveat,
    }


@app.get("/evaluation")
def evaluation() -> dict:
    """Honest, file-derived summary of how each model was trained and evaluated.

    Figures describe the training datasets only and are never clinical or
    real-world accuracy.
    """
    questionnaire_meta = _read_json("models/questionnaire/training_metadata.json") or {}
    questionnaire_sel = _read_json("models/questionnaire/selected_model.json") or {}
    behavioral_meta = _read_json("models/behavioral/training_metadata.json") or {}
    voice_summary = _voice_model_summary(
        _artifact_present("models/voice/selected_model.json") or _artifact_present("models/voice/voice_model.joblib")
    )
    return {
        "questionnaire": {
            "data": "real responses",
            "respondents": questionnaire_meta.get("dataset_rows"),
            "class_counts": questionnaire_meta.get("class_counts"),
            "model": questionnaire_sel.get("selected_model"),
            "cross_validated_macro_f1": questionnaire_sel.get("macro_f1_mean"),
            "cross_validated_balanced_accuracy": questionnaire_sel.get("balanced_accuracy_mean"),
            "caveat": "Small sample (cross-validation); the target is derived from one self-reported stress item. Not clinically validated.",
        },
        "behavioral": {
            "data": "synthetic" if behavioral_meta.get("synthetic_data") else "unknown",
            "model": behavioral_meta.get("model"),
            "test_accuracy": behavioral_meta.get("test_accuracy"),
            "caveat": behavioral_meta.get("notice")
            or "Trained on synthetic data; the score reflects clean synthetic classes, not real-world performance.",
        },
        "voice": voice_summary
        if voice_summary.get("trained")
        else {"trained": False, "caveat": "No voice model yet: a labeled audio dataset is required (see docs/voice_model_training.md)."},
        "voice_emotion": _emotion_summary(),
        "disclaimer": DISCLAIMER,
    }


def _questionnaire_items() -> list[dict]:
    """Question text, answer type and allowed options, derived from the model's own encoders."""
    from src.ai_model.questionnaire.categorical_encoder import Q1_MAPPING, Q2_CANONICAL_OPTIONS
    from src.ai_model.questionnaire.ordinal_encoder import QUESTION_CATEGORY_MAPPINGS, SPECIAL_RESPONSE_TEXT
    from src.ai_model.questionnaire.questionnaire_schema import QUESTIONNAIRE_SCHEMA

    items = []
    for question_id in REQUIRED_QUESTION_FIELDS:
        meta = QUESTIONNAIRE_SCHEMA[question_id]
        kind, options = "single", []
        if question_id == "Q1":
            options = list(Q1_MAPPING)
        elif question_id == "Q2":
            kind, options = "multi", list(Q2_CANONICAL_OPTIONS)
        elif question_id in ("Q10", "Q11"):
            kind, options = "multi", list(meta.canonical_options or [])
        elif question_id in QUESTION_CATEGORY_MAPPINGS:
            options = list(QUESTION_CATEGORY_MAPPINGS[question_id])
            if question_id in ("Q14", "Q16"):
                options.append(SPECIAL_RESPONSE_TEXT)
        text = meta.column_name.strip()
        items.append({"id": question_id, "text": text.split(". ", 1)[-1], "type": kind, "options": options})
    return items


@app.get("/schema")
def schema() -> dict:
    """Input schema hints used by the bundled web UI."""
    return {
        "questionnaire_fields": list(REQUIRED_QUESTION_FIELDS),
        "questionnaire_items": _questionnaire_items(),
        "behavioral_fields": {name: {"min": low, "max": high} for name, (low, high) in FIELD_RANGES.items()},
        "max_audio_bytes": MAX_AUDIO_BYTES,
    }


if (FRONTEND_DIR / "index.html").exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/app/")
