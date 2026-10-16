"""Reads optional cloud settings from the environment. Nothing is enabled by default."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _get(name: str) -> Optional[str]:
    value = os.environ.get(name, "").strip()
    return value or None


@dataclass(frozen=True)
class CloudSettings:
    speech_key: Optional[str]
    speech_region: Optional[str]
    language_key: Optional[str]
    language_endpoint: Optional[str]
    cosmos_endpoint: Optional[str]
    cosmos_key: Optional[str]
    cosmos_database: str
    cosmos_container: str
    eventgrid_endpoint: Optional[str]
    eventgrid_key: Optional[str]
    events_secret: Optional[str]
    trust_easy_auth: bool

    @property
    def speech_content_enabled(self) -> bool:
        return bool(self.speech_key and self.speech_region and self.language_key and self.language_endpoint)

    @property
    def cosmos_enabled(self) -> bool:
        return bool(self.cosmos_endpoint and self.cosmos_key)

    @property
    def events_enabled(self) -> bool:
        return bool(self.eventgrid_endpoint and self.eventgrid_key and self.events_secret)

    @property
    def history_enabled(self) -> bool:
        # Reading/deleting history needs Cosmos; saving goes through Event Grid when it is configured.
        return self.cosmos_enabled


def load_settings() -> CloudSettings:
    """Read settings on every call so tests (and App Service setting changes) take effect without restarts."""
    return CloudSettings(
        speech_key=_get("CLOUD_SPEECH_KEY"),
        speech_region=_get("CLOUD_SPEECH_REGION"),
        language_key=_get("CLOUD_LANGUAGE_KEY"),
        language_endpoint=_get("CLOUD_LANGUAGE_ENDPOINT"),
        cosmos_endpoint=_get("CLOUD_COSMOS_ENDPOINT"),
        cosmos_key=_get("CLOUD_COSMOS_KEY"),
        cosmos_database=_get("CLOUD_COSMOS_DATABASE") or "emotionaware",
        cosmos_container=_get("CLOUD_COSMOS_CONTAINER") or "assessments",
        eventgrid_endpoint=_get("CLOUD_EVENTGRID_ENDPOINT"),
        eventgrid_key=_get("CLOUD_EVENTGRID_KEY"),
        events_secret=_get("CLOUD_EVENTS_SECRET"),
        # Only trust the Easy Auth identity header when Entra login is actually switched on for the site;
        # otherwise a client could send the header itself.
        trust_easy_auth=_get("CLOUD_TRUST_EASY_AUTH") == "1",
    )


def is_hosted() -> bool:
    """True when running on Azure App Service (the platform sets WEBSITE_SITE_NAME)."""
    return bool(_get("WEBSITE_SITE_NAME"))
