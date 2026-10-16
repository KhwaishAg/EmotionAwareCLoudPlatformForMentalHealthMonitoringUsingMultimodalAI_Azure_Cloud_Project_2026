"""Turn "what the student said" (Azure Speech transcript + Azure Language sentiment) into a voice contract.

This is a *speech-content* indicator, used for the voice modality only while no acoustic stress model
is trained. It is deliberately conservative:

* risk_score = negative + 0.25 * neutral (positive speech -> 0, neutral -> 0.25, negative -> 1);
* confidence is capped at MAX_CONFIDENCE and scaled down for short answers, so it can never outweigh
  the questionnaire in fusion;
* it describes the tone of the words, not the voice itself, and is not a clinical measure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.cloud import language, speech
from src.cloud.config import CloudSettings

MAX_CONFIDENCE = 0.6
MIN_WORDS = 3
FULL_CONFIDENCE_WORDS = 30
NEUTRAL_WEIGHT = 0.25
SOURCE = "azure_speech_language"


class NoSpeechError(ValueError):
    """The recording contained too little recognisable speech."""


def _level(score: float) -> str:
    return "low" if score < .34 else "moderate" if score < .67 else "high"


def build_contract(transcript: str, sentiment: Dict[str, Any], phrases: Optional[list] = None) -> Dict[str, Any]:
    words = len(transcript.split())
    if words < MIN_WORDS:
        raise NoSpeechError("Not enough recognisable speech in the recording.")
    scores = sentiment["scores"]
    positive, neutral, negative = (float(scores[k]) for k in ("positive", "neutral", "negative"))
    risk = min(1.0, max(0.0, negative + NEUTRAL_WEIGHT * neutral))
    length_factor = min(1.0, words / FULL_CONFIDENCE_WORDS)
    confidence = round(MAX_CONFIDENCE * max(positive, neutral, negative) * length_factor, 4)
    return {
        "modality": "voice",
        "risk_score": round(risk, 4),
        "risk_level": _level(risk),
        "confidence": confidence,
        "available_features": 2,
        "total_features": 2,
        "data_quality": {"missing_features": 0, "completeness": 1.0},
        "key_factors": [
            {"feature": "speech_negative_tone", "contribution": round(negative, 4), "direction": "increases_risk"},
            {"feature": "speech_positive_tone", "contribution": round(positive, 4), "direction": "decreases_risk"},
        ],
        "source": SOURCE,
        "sentiment": sentiment.get("label"),
        "transcript": transcript,
        "transcript_words": words,
        "key_phrases": list(phrases or []),
        "caveat": "Based on the tone of the words you said (Azure AI Speech + Language), not on vocal acoustics. "
                  "Not a clinical measure.",
    }


def assess_audio(path: Union[str, Path], settings: CloudSettings) -> Dict[str, Any]:
    """Audio file -> transcript -> sentiment/key phrases -> voice fusion contract."""
    wav = speech.to_pcm16_wav(path)
    heard = speech.transcribe(wav, settings.speech_key or "", settings.speech_region or "")
    transcript = str(heard["text"])
    if len(transcript.split()) < MIN_WORDS:
        raise NoSpeechError("Not enough recognisable speech in the recording.")
    mood = language.sentiment(transcript, settings.language_key or "", settings.language_endpoint or "")
    try:
        phrases = language.key_phrases(transcript, settings.language_key or "", settings.language_endpoint or "")
    except Exception:  # key phrases are a nice-to-have; sentiment alone is enough for the contract
        phrases = []
    return build_contract(transcript, mood, phrases)
