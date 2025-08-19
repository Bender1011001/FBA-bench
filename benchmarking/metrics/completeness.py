from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field, ValidationError
except Exception:  # pragma: no cover
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self, *_, **__):
            return self.__dict__
    def Field(*_, **__):  # type: ignore
        return None

from .registry import register_metric

logger = logging.getLogger(__name__)


class CompletenessInput(BaseModel):
    output: Any = Field(description="Scenario output; usually a dict")

    class Config:
        frozen = False


class CompletenessContext(BaseModel):
    required_fields: List[str] = Field(default_factory=list, description="List of required top-level fields")
    allow_nested: bool = Field(default=False, description="If True, use dot-paths in required_fields")

    class Config:
        frozen = False


def _has_path(d: Any, path: str) -> bool:
    if not isinstance(d, dict):
        return False
    cur = d
    for part in path.split("."):
        if not (isinstance(cur, dict) and part in cur):
            return False
        cur = cur[part]
    return True


def evaluate(run: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Completeness metric.

    Inputs:
      - run.output dict
      - context.required_fields: list[str]
      - context.allow_nested: if True interpret fields as dot-paths

    Output:
      {"present": int, "required": int, "completeness": float}
    """
    try:
        if not isinstance(run, dict):
            return {"present": 0, "required": 0, "completeness": 0.0, "error": "invalid_run_type"}
        try:
            inp = CompletenessInput(output=run.get("output"))  # type: ignore
        except ValidationError as ve:
            logger.error(f"completeness: invalid input {ve}")
            return {"present": 0, "required": 0, "completeness": 0.0, "error": "validation_error"}

        ctx = CompletenessContext(**(context or {}))  # type: ignore
        req = [f for f in (ctx.required_fields or []) if isinstance(f, str) and f.strip()]
        total = len(req)
        if total == 0:
            return {"present": 0, "required": 0, "completeness": 0.0, "reason": "no_required_fields"}

        output = inp.output if isinstance(inp.output, dict) else {}
        present = 0
        for field in req:
            if ctx.allow_nested and "." in field:
                if _has_path(output, field):
                    present += 1
            else:
                if field in output:
                    present += 1

        completeness = float(present) / float(total) if total > 0 else 0.0
        return {"present": int(present), "required": int(total), "completeness": completeness}
    except Exception as e:
        logger.exception("completeness metric failed")
        return {"present": 0, "required": 0, "completeness": 0.0, "error": "exception", "reason": str(e)}


register_metric("completeness", evaluate)