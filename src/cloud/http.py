"""Minimal JSON-over-HTTPS helper on the standard library (no SDK dependency)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

DEFAULT_TIMEOUT = 20


class CloudServiceError(RuntimeError):
    """A cloud call failed. Messages never include keys or user content."""

    def __init__(self, service: str, status: Optional[int], detail: str = "") -> None:
        self.service, self.status = service, status
        super().__init__(f"{service} request failed (HTTP {status})" + (f": {detail}" if detail else ""))


def request(
    service: str,
    method: str,
    url: str,
    headers: Dict[str, str],
    body: Optional[bytes] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[int, Any]:
    """Send one request; return (status, parsed JSON or None). Raises CloudServiceError on failure."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - https URLs built from settings
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        code = error.code
        try:
            payload = json.loads(error.read() or b"{}")
            detail = str(payload.get("error", {}).get("code") or payload.get("code") or "")[:80]
        except (ValueError, AttributeError):
            detail = ""
        raise CloudServiceError(service, code, detail) from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise CloudServiceError(service, None, type(error).__name__) from None
    if not raw:
        return status, None
    try:
        return status, json.loads(raw)
    except ValueError:
        raise CloudServiceError(service, status, "non-JSON response") from None


def json_body(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")
