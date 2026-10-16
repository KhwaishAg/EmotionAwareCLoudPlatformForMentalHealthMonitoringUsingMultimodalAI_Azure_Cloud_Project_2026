"""Opt-in assessment history: what is stored, who owns it, and the trend shown back to the student.

Stored per assessment (and nothing else): owner ID, time, overall and per-source scores/levels,
confidence, agreement level and the names of the top factors. Never questionnaire answers, habit
values, audio or transcripts. Documents expire automatically (container TTL, set by the setup script).
"""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.cloud.config import CloudSettings
from src.cloud.cosmos import CosmosContainer

ANON_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
OWNER_PATTERN = re.compile(r"^(anon|user)-[0-9a-f-]{32,64}$")
HISTORY_LIMIT = 20
RECORD_FIELDS = {"id", "owner", "created_at", "risk_score", "risk_level", "confidence", "agreement_level",
                 "modalities_used", "modality_scores", "top_factors", "schema_version"}


def resolve_owner(anon_id: Optional[str], principal_id: Optional[str], settings: CloudSettings) -> Optional[str]:
    """Signed-in Entra user (hashed) when Easy Auth is trusted; otherwise the browser's random anonymous ID."""
    if principal_id and settings.trust_easy_auth:
        return "user-" + hashlib.sha256(principal_id.encode("utf-8")).hexdigest()[:48]
    if anon_id and ANON_ID_PATTERN.match(anon_id.lower()):
        return "anon-" + anon_id.lower()
    return None


def build_record(owner: str, assessment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "owner": owner,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "risk_score": assessment.get("risk_score"),
        "risk_level": assessment.get("risk_level"),
        "confidence": assessment.get("confidence"),
        "agreement_level": (assessment.get("agreement") or {}).get("level"),
        "modalities_used": list(assessment.get("modalities_used") or []),
        "modality_scores": dict(assessment.get("modality_scores") or {}),
        "top_factors": [f.get("feature") for f in (assessment.get("key_factors") or [])[:3]],
        "schema_version": 1,
    }


def validate_record(record: Any) -> Dict[str, Any]:
    """Accept only well-formed summaries from the events webhook (defence in depth)."""
    if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
        raise ValueError("Unexpected history record shape.")
    if not OWNER_PATTERN.match(str(record["owner"])):
        raise ValueError("Invalid owner.")
    uuid.UUID(str(record["id"]))
    score = record["risk_score"]
    if score is not None and not (isinstance(score, (int, float)) and 0 <= score <= 1):
        raise ValueError("Invalid risk score.")
    return record


def summarise_trend(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """items are newest first. Compare the latest score with the average of up to five earlier ones."""
    scored = [i for i in items if isinstance(i.get("risk_score"), (int, float))]
    if len(scored) < 2:
        return {"direction": "not_enough_data", "count": len(scored)}
    latest = scored[0]["risk_score"]
    earlier = [i["risk_score"] for i in scored[1:6]]
    baseline = sum(earlier) / len(earlier)
    change = latest - baseline
    direction = "rising" if change > 0.1 else "falling" if change < -0.1 else "stable"
    return {"direction": direction, "count": len(scored), "latest": round(latest, 4),
            "previous_average": round(baseline, 4), "change": round(change, 4)}


def _container(settings: CloudSettings) -> CosmosContainer:
    return CosmosContainer(settings.cosmos_endpoint or "", settings.cosmos_key or "",
                           settings.cosmos_database, settings.cosmos_container)


def save(record: Dict[str, Any], settings: CloudSettings) -> None:
    _container(settings).upsert(record, record["owner"])


def list_for(owner: str, settings: CloudSettings) -> List[Dict[str, Any]]:
    rows = _container(settings).query(
        f"SELECT * FROM c WHERE c.owner = @owner ORDER BY c.created_at DESC OFFSET 0 LIMIT {int(HISTORY_LIMIT)}",
        [{"name": "@owner", "value": owner}], owner)
    return [{k: v for k, v in row.items() if k in RECORD_FIELDS and k != "owner"} for row in rows]


def delete_all(owner: str, settings: CloudSettings) -> int:
    container = _container(settings)
    ids = container.query("SELECT c.id FROM c WHERE c.owner = @owner", [{"name": "@owner", "value": owner}], owner)
    for row in ids:
        container.delete(str(row["id"]), owner)
    return len(ids)
