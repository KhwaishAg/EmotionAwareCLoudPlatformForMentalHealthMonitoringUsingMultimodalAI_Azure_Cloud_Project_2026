"""Azure AI Speech (F0) speech-to-text via the short-audio REST API (up to 60 s per request)."""
from __future__ import annotations

import io
from math import gcd
from pathlib import Path
from typing import Dict, Union

import numpy as np

from src.cloud.http import CloudServiceError, request

TARGET_RATE = 16000
MAX_SECONDS = 59.0  # the short-audio endpoint accepts at most 60 s
DEFAULT_LANGUAGE = "en-IN"


def to_pcm16_wav(path: Union[str, Path]) -> bytes:
    """Read any WAV the app accepts and return 16 kHz mono 16-bit PCM WAV bytes, trimmed to MAX_SECONDS."""
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != TARGET_RATE:
        divisor = gcd(int(rate), TARGET_RATE)
        mono = resample_poly(mono, TARGET_RATE // divisor, int(rate) // divisor).astype("float32")
    mono = np.clip(mono[: int(MAX_SECONDS * TARGET_RATE)], -1.0, 1.0)
    buffer = io.BytesIO()
    sf.write(buffer, mono, TARGET_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def transcribe(wav_bytes: bytes, key: str, region: str, language: str = DEFAULT_LANGUAGE) -> Dict[str, object]:
    """Return {"text": str, "confidence": float}. Empty text means no speech was recognised."""
    url = (
        f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
        f"?language={language}&format=detailed&profanity=masked"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": f"audio/wav; codecs=audio/pcm; samplerate={TARGET_RATE}",
        "Accept": "application/json",
    }
    _, payload = request("speech", "POST", url, headers, wav_bytes, timeout=45)
    payload = payload or {}
    status = payload.get("RecognitionStatus")
    if status in ("NoMatch", "InitialSilenceTimeout", "BabbleTimeout"):
        return {"text": "", "confidence": 0.0}
    if status != "Success":
        raise CloudServiceError("speech", 200, f"RecognitionStatus={status}")
    best = (payload.get("NBest") or [{}])[0]
    text = str(best.get("Display") or payload.get("DisplayText") or "").strip()
    return {"text": text, "confidence": float(best.get("Confidence") or 0.0)}
