from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List

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


class CostEfficiencyInput(BaseModel):
    # RunResult-like fields we inspect for usage info
    output: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None

    class Config:
        frozen = False


class CostEfficiencyContext(BaseModel):
    # score_like resolution
    score_value: Optional[float] = Field(default=None, description="Explicit score-like value for numerator")
    score_field_path: Optional[str] = Field(default=None, description="Dot-path in run.output to read score")
    # cost_like resolution
    min_cost_field: str = Field(default="cost", description="Preferred cost field key to read directly")
    usage_paths: List[str] = Field(
        default_factory=lambda: [
            "token_usage.total_tokens",
            "token_usage.input_tokens",
            "token_usage.output_tokens",
            "usage.total_tokens",
            "usage.tokens",
        ]
    )
    # token to cost conversion (default 1 token == 1 unit to avoid assumptions)
    token_to_cost_rate: float = Field(default=1.0, ge=0.0, description="Multiplier to convert tokens to cost units")

    class Config:
        frozen = False


def _get_nested(d: Any, path: str) -> Any:
    if not isinstance(d, dict) or not path:
        return None
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return float(x)
        if isinstance(x, str):
            s = x.strip().lower().replace("$", "").replace(",", "")
            return float(s)
        return None
    except Exception:
        return None


def _resolve_cost_like(inp: CostEfficiencyInput, ctx: CostEfficiencyContext) -> Optional[float]:
    # direct cost field from artifacts -> metrics -> output
    for src in (inp.artifacts, inp.metrics, inp.output):
        if isinstance(src, dict) and ctx.min_cost_field in src:
            v = _as_float(src.get(ctx.min_cost_field))
            if v is not None:
                return v

    # look for nested usage dicts and derive pseudo-cost from tokens
    for src in (inp.artifacts, inp.metrics, inp.output):
        if not isinstance(src, dict):
            continue
        for p in ctx.usage_paths:
            val = _get_nested(src, p)
            tokens = _as_float(val)
            if tokens is not None and tokens > 0:
                return tokens * float(ctx.token_to_cost_rate)

    # sometimes providers place usage under "usage": {"prompt_tokens":...,"completion_tokens":...}
    for src in (inp.artifacts, inp.metrics, inp.output):
        if not isinstance(src, dict):
            continue
        usage = src.get("usage") or src.get("token_usage")
        if isinstance(usage, dict):
            total = 0.0
            found = False
            for k in ("total_tokens", "prompt_tokens", "input_tokens", "completion_tokens", "output_tokens"):
                v = _as_float(usage.get(k))
                if v is not None:
                    total += v
                    found = True
            if found and total > 0:
                return total * float(ctx.token_to_cost_rate)

    return None


def _resolve_score_like(inp: CostEfficiencyInput, ctx: CostEfficiencyContext) -> float:
    # explicit context-provided score wins
    if ctx.score_value is not None:
        return float(ctx.score_value)

    # path in output
    if ctx.score_field_path:
        v = _get_nested(inp.output or {}, ctx.score_field_path)
        f = _as_float(v)
        if f is not None:
            return float(f)

    # default neutral score
    return 1.0


def evaluate(run: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Cost-efficiency metric.

    If token/usage or cost present, compute:
        efficiency = score_like / cost_like

    Resolution order:
    - cost_like: run.artifacts.cost or run.metrics.cost or run.output.cost
                 else tokens converted via token_to_cost_rate
    - score_like: context.score_value or run.output[score_field_path] else 1.0

    Output:
      {"supported": bool, "efficiency": float|None, "reason": str|None}
    """
    try:
        if not isinstance(run, dict):
            return {"supported": False, "efficiency": None, "reason": "invalid_run_type"}

        try:
            inp = CostEfficiencyInput(
                output=run.get("output"),
                metrics=run.get("metrics"),
                artifacts=run.get("artifacts"),
            )  # type: ignore
        except ValidationError as ve:
            logger.error(f"cost_efficiency: invalid input {ve}")
            return {"supported": False, "efficiency": None, "reason": "validation_error"}

        ctx = CostEfficiencyContext(**(context or {}))  # type: ignore

        cost_like = _resolve_cost_like(inp, ctx)
        if cost_like is None:
            return {"supported": False, "efficiency": None, "reason": "missing_usage"}

        if cost_like <= 0:
            return {"supported": False, "efficiency": None, "reason": "non_positive_cost"}

        score_like = _resolve_score_like(inp, ctx)
        efficiency = float(score_like) / float(cost_like)

        return {"supported": True, "efficiency": efficiency, "reason": None}
    except Exception as e:
        logger.exception("cost_efficiency metric failed")
        return {"supported": False, "efficiency": None, "reason": str(e)}


register_metric("cost_efficiency", evaluate)