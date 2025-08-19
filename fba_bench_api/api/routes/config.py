from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

# NOTE: Existing endpoints remain for backward compatibility.
# This patch adds GET /config and PATCH /config for runtime overrides.

router = APIRouter(prefix="/api/v1/config", tags=["Configuration"])

# In-memory overrides with async lock (ready to swap with DB/SQLAlchemy)
_overrides: dict = {}
_lock = asyncio.Lock()

class RuntimeConfig(BaseModel):
    enable_observability: Optional[bool] = Field(None, description="Enable OpenTelemetry/observability features")
    profile: Optional[str] = Field(None, description="Active configuration profile")
    telemetry_endpoint: Optional[str] = Field(None, description="OTLP endpoint")

    class Config:
        json_schema_extra = {
            "example": {
                "enable_observability": True,
                "profile": "development",
                "telemetry_endpoint": "http://localhost:4318"
            }
        }

class ConfigResponse(RuntimeConfig):
    # Effective, merged configuration
    pass

def _env_config() -> dict:
    return {
        "enable_observability": (os.getenv("ENABLE_OBSERVABILITY", "").lower() in {"1","true","yes"}),
        "profile": os.getenv("PROFILE", None),
        "telemetry_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", None),
    }

def _now() -> datetime:
    return datetime.utcnow()

@router.get("", response_model=ConfigResponse, description="Get effective merged configuration (env + runtime overrides)")
async def get_config():
    async with _lock:
        merged = {**_env_config(), **_overrides}
        return ConfigResponse(**merged)

@router.patch("", response_model=ConfigResponse, description="Patch runtime configuration overrides (non-persistent for now)")
async def patch_config(patch: RuntimeConfig):
    async with _lock:
        updates = patch.model_dump(exclude_unset=True)
        # Validate values (if any constraints needed add here)
        _overrides.update(updates)
        merged = {**_env_config(), **_overrides}
        return ConfigResponse(**merged)