from __future__ import annotations

import logging
from typing import Any, Dict, Optional

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


class PolicyComplianceInput(BaseModel):
    output: Any = Field(description="Scenario output; may include 'policy_violations' as int or list")

    class Config:
        frozen = False


class PolicyComplianceOutput(BaseModel):
    policy_violations: int
    compliant: bool

    def as_dict(self) -> Dict[str, Any]:
        return {"policy_violations": int(self.policy_violations), "compliant": bool(self.compliant)}


def _extract_violation_count(output: Any) -> int:
    try:
        if isinstance(output, dict):
            v = output.get("policy_violations")
            if isinstance(v, int):
                return max(0, v)
            if isinstance(v, list):
                return len(v)
            if isinstance(v, dict):
                # e.g., {"count": 2}
                if "count" in v and isinstance(v["count"], int):
                    return max(0, int(v["count"]))
        # If scalar
        if isinstance(output, int):
            return max(0, output)
    except Exception:
        pass
    return 0


def evaluate(run: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Policy compliance metric.

    Input:
      - run.output["policy_violations"] as int or list or {"count": int}

    Output:
      {"policy_violations": int, "compliant": bool}
    """
    try:
        if not isinstance(run, dict):
            return {"policy_violations": 0, "compliant": False, "error": "invalid_run_type"}
        try:
            inp = PolicyComplianceInput(output=run.get("output"))  # type: ignore
        except ValidationError as ve:
            logger.error(f"policy_compliance: invalid input {ve}")
            return {"policy_violations": 0, "compliant": False, "error": "validation_error"}

        count = _extract_violation_count(inp.output)
        return PolicyComplianceOutput(policy_violations=count, compliant=(count == 0)).as_dict()
    except Exception as e:
        logger.exception("policy_compliance metric failed")
        return {"policy_violations": 0, "compliant": False, "error": "exception", "reason": str(e)}


register_metric("policy_compliance", evaluate)