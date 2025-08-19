from __future__ import annotations

import logging
import re
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

_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


class RobustnessInput(BaseModel):
    output: Any = Field(description="Scenario output; typically a string or structured content")

    class Config:
        frozen = False


class RobustnessContext(BaseModel):
    expected_signal: Any = Field(default=None, description="Reference signal to compare against")
    mode: str = Field(default="normalized_overlap", description="normalized_overlap|exact_casefold")
    case_insensitive: bool = Field(default=True)
    strip_whitespace: bool = Field(default=True)

    class Config:
        frozen = False


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (str, bytes)):
        return x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
    # favor 'text' or 'summary' keys if dict
    if isinstance(x, dict):
        for k in ("text", "summary", "value", "content"):
            if k in x:
                return _to_text(x[k])
    return str(x)


def _normalize(s: str, case_insensitive: bool, strip_ws: bool) -> str:
    if case_insensitive:
        s = s.casefold()
    if strip_ws:
        s = _WS_RE.sub(" ", s).strip()
    return s


def _jaccard_chars(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return float(inter / union) if union > 0 else 0.0


def evaluate(run: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Robustness metric.

    Compares run.output to an expected signal under an invariant normalization
    (whitespace squashing, case-folding). Returns a score in [0,1].

    Modes:
      - normalized_overlap (default): Jaccard overlap on normalized characters
      - exact_casefold: boolean exact equality under casefold + whitespace squash

    Output:
      {"robustness_score": float}
    """
    try:
        if not isinstance(run, dict):
            return {"robustness_score": 0.0, "error": "invalid_run_type"}
        try:
            inp = RobustnessInput(output=run.get("output"))  # type: ignore
        except ValidationError as ve:
            logger.error(f"robustness: invalid input {ve}")
            return {"robustness_score": 0.0, "error": "validation_error"}

        ctx = RobustnessContext(**(context or {}))  # type: ignore
        out_text = _to_text(inp.output)
        exp_text = _to_text(ctx.expected_signal)

        a = _normalize(out_text, ctx.case_insensitive, ctx.strip_whitespace)
        b = _normalize(exp_text, ctx.case_insensitive, ctx.strip_whitespace)

        mode = (ctx.mode or "normalized_overlap").lower().strip()
        if mode == "exact_casefold":
            score = 1.0 if a == b else 0.0
        else:
            score = _jaccard_chars(a, b)

        # clip to [0,1]
        score = max(0.0, min(1.0, float(score)))
        return {"robustness_score": score}
    except Exception as e:
        logger.exception("robustness metric failed")
        return {"robustness_score": 0.0, "error": "exception", "reason": str(e)}


register_metric("robustness", evaluate)