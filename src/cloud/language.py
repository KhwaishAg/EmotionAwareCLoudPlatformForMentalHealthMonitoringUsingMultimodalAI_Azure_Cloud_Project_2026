"""Azure AI Language (F0): sentiment analysis and key-phrase extraction for one short text."""
from __future__ import annotations

from typing import Dict, List

from src.cloud.http import CloudServiceError, json_body, request

API_VERSION = "2023-04-01"
MAX_CHARS = 5000  # service limit is 5,120 characters per document; F0 counts each 1,000 characters as one record


def _analyze(kind: str, text: str, key: str, endpoint: str, language: str) -> Dict:
    url = endpoint.rstrip("/") + f"/language/:analyze-text?api-version={API_VERSION}"
    body = {"kind": kind, "analysisInput": {"documents": [{"id": "1", "language": language, "text": text[:MAX_CHARS]}]}}
    if kind == "SentimentAnalysis":
        body["parameters"] = {"opinionMining": False}
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"}
    _, payload = request("language", "POST", url, headers, json_body(body))
    documents = ((payload or {}).get("results") or {}).get("documents") or []
    if not documents:
        raise CloudServiceError("language", 200, f"{kind} returned no document")
    return documents[0]


def sentiment(text: str, key: str, endpoint: str, language: str = "en") -> Dict[str, object]:
    """Return {"label": positive|neutral|negative|mixed, "scores": {positive, neutral, negative}}."""
    document = _analyze("SentimentAnalysis", text, key, endpoint, language)
    scores = document.get("confidenceScores") or {}
    return {
        "label": str(document.get("sentiment") or "neutral"),
        "scores": {name: float(scores.get(name) or 0.0) for name in ("positive", "neutral", "negative")},
    }


def key_phrases(text: str, key: str, endpoint: str, language: str = "en") -> List[str]:
    document = _analyze("KeyPhraseExtraction", text, key, endpoint, language)
    return [str(phrase) for phrase in (document.get("keyPhrases") or [])][:8]
