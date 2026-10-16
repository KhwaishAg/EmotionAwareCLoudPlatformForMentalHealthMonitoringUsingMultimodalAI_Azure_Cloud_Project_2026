"""Event Grid custom topic: publish "assessment completed" events (Event Grid schema)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from src.cloud.http import json_body, request

ASSESSMENT_COMPLETED = "EmotionAware.Assessment.Completed"
SUBSCRIPTION_VALIDATION = "Microsoft.EventGrid.SubscriptionValidationEvent"


def publish(endpoint: str, key: str, event_type: str, subject: str, data: Dict[str, Any]) -> None:
    event = {
        "id": str(uuid.uuid4()),
        "eventType": event_type,
        "subject": subject,
        "eventTime": datetime.now(timezone.utc).isoformat(),
        "data": data,
        "dataVersion": "1.0",
    }
    headers = {"aeg-sas-key": key, "Content-Type": "application/json"}
    request("eventgrid", "POST", endpoint, headers, json_body([event]))
