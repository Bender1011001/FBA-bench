from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    # Pydantic v2
    from pydantic import BaseModel, Field, ValidationError
except Exception:  # pragma: no cover - pydantic is a core dep in this repo
    # Soft fallback to avoid hard import failure in edge environments
    class BaseModel:  # type: ignore
        def model_dump(self, *_, **__):  # minimal surface
            return self.__dict__

    def Field(*_, **__):  # type: ignore
        return None  # no-op

from .registry import register_metric

logger = logging.getLogger(__name__)


# -----------------------------
# Schemas
# -----------------------------
class TechnicalPerformanceInput(BaseModel):
    # Minimal RunResult-like fields we care about
    status: str = Field(description="success|failed|timeout|error")
    duration_ms: int

    class Config:
        frozen = False


class TechnicalPerformanceContext(BaseModel):
    # Optional threshold override for 'fast_enough'
    latency_threshold_ms: int = Field(default=2000, ge=0, description="Threshold for fast_enough flag")

    class Config:
        frozen = False


class TechnicalPerformanceOutput(BaseModel):
    latency_ms: int
    fast_enough: bool

    def as_dict(self) -> Dict[str, Any]:
        return {"latency_ms": int(self.latency_ms), "fast_enough": bool(self.fast_enough)}


# -----------------------------
# Metric implementation
# -----------------------------
def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def evaluate(run: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compute technical performance values.

    Inputs:
      - run.duration_ms (required), run.status
      - context.latency_threshold_ms (optional, default 2000)

    Output:
      {
        "latency_ms": int,
        "fast_enough": bool
      }
    """
    try:
        if not isinstance(run, dict):
            return {"error": "invalid_run_type", "reason": "run must be a dict-like RunResult"}

        data = {
            "status": run.get("status", "success"),
            "duration_ms": run.get("duration_ms"),
        }

        # Validate input schema
        try:
            inp = TechnicalPerformanceInput(**data)  # type: ignore
        except ValidationError as ve:
            logger.error(f"technical_performance: invalid input: {ve}")
            # Still try to salvage latency if present
            latency = _safe_int(run.get("duration_ms", 0), 0)
            return {"latency_ms": latency, "fast_enough": False, "error": "validation_error", "reason": "invalid_input"}

        ctx = TechnicalPerformanceContext(**(context or {}))  # uses default if missing
        latency_ms = _safe_int(inp.duration_ms, 0)
        fast_enough = bool(latency_ms <= ctx.latency_threshold_ms)

        out = TechnicalPerformanceOutput(latency_ms=latency_ms, fast_enough=fast_enough)
        return out.as_dict()
    except Exception as e:  # Non-fatal by design
        logger.exception("technical_performance metric failed")
        latency = _safe_int((run or {}).get("duration_ms", 0), 0) if isinstance(run, dict) else 0
        return {"latency_ms": latency, "fast_enough": False, "error": "exception", "reason": str(e)}


# -----------------------------
# Registry auto-registration
# -----------------------------
register_metric("technical_performance", evaluate)