from __future__ import annotations

import asyncio
from typing import Any, Dict

from fastapi import APIRouter, Response, status

# Router mounted next to others in create_app(); prefix aligns with frontend calls.
router = APIRouter(prefix="/api/v1", tags=["Settings"])

# Module-level in-memory store guarded by an asyncio lock.
_lock: asyncio.Lock = asyncio.Lock()
_settings_store: Dict[str, Any] = {}


def _default_settings() -> Dict[str, Any]:
    """
    Stable defaults mirroring frontend fallbacks in apiService.fetchSettings().

    Returns shape:
    {
      "apiKeys": { ... },
      "defaults": { ... },
      "ui": { ... }
    }
    """
    # Note: timezone default is computed client-side in the frontend fallback.
    # Here we return a deterministic default; clients can override.
    return {
        "apiKeys": {
            "openai": "",
            "anthropic": "",
            "google": "",
            "cohere": "",
            "openrouter": "",
        },
        "defaults": {
            "defaultLLM": "gpt-4",
            "defaultScenario": "standard",
            "defaultAgent": "basic",
            "defaultMetrics": ["revenue", "profit", "costs"],
            "autoSave": True,
            "notifications": True,
        },
        "ui": {
            "theme": "system",
            "language": "en",
            "timezone": "UTC",
        },
    }


@router.get("/settings")
async def get_settings() -> Dict[str, Any]:
    """
    GET /api/v1/settings

    Returns:
    {
      "settings": { ... },
      "message": Optional[str]
    }
    """
    async with _lock:
        stored = _settings_store.get("settings")
        msg = _settings_store.get("message")

    settings = stored if isinstance(stored, dict) else _default_settings()
    resp: Dict[str, Any] = {"settings": settings}
    if isinstance(msg, str) and msg:
        resp["message"] = msg
    return resp


@router.post("/settings")
async def save_settings(payload: Dict[str, Any]) -> Response:
    """
    POST /api/v1/settings

    Accepts the same shape used by the frontend client:
    - Either the raw settings object
    - Or an envelope { "settings": { ... }, "message": Optional[str] }

    Stores in-memory; returns 204 No Content.
    """
    # Accept both raw settings and enveloped payloads
    if "settings" in payload and isinstance(payload.get("settings"), dict):
        new_settings = payload["settings"]
        maybe_message = payload.get("message")
    else:
        new_settings = payload
        maybe_message = None

    if not isinstance(new_settings, dict):
        # Coerce to empty defaults to keep behavior stable
        new_settings = _default_settings()

    async with _lock:
        _settings_store["settings"] = new_settings
        if isinstance(maybe_message, str):
            _settings_store["message"] = maybe_message

    return Response(status_code=status.HTTP_204_NO_CONTENT)