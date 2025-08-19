from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set

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

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


class KeywordCoverageInput(BaseModel):
    # run.output is expected to be dict or str; for summarization, we look at "summary" in dict
    output: Any = Field(description="Scenario output; dict with 'summary' or a raw string")

    class Config:
        frozen = False


class KeywordCoverageContext(BaseModel):
    keywords: List[str] = Field(default_factory=list, description="List of focus keywords to check")
    field_path: Optional[str] = Field(default="summary", description="If output is dict, the field to read")
    case_insensitive: bool = Field(default=True)
    strip_punctuation: bool = Field(default=True)
    unique_match: bool = Field(default=True, description="Count keyword present at least once (unique) or by frequency")

    class Config:
        frozen = False


class KeywordCoverageOutput(BaseModel):
    keyword_hits: int
    keyword_total: int
    coverage: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "keyword_hits": int(self.keyword_hits),
            "keyword_total": int(self.keyword_total),
            "coverage": float(self.coverage),
        }


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (str, bytes)):
        return x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
    return str(x)


def _normalize_text(s: str, case_insensitive: bool, strip_punctuation: bool) -> str:
    if strip_punctuation:
        s = _PUNCT_RE.sub(" ", s)
    if case_insensitive:
        s = s.lower()
    return s


def _extract_field(output: Any, field_path: Optional[str]) -> Any:
    if field_path is None or not isinstance(output, dict):
        return output
    cur = output
    for part in field_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def evaluate(run: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Keyword coverage for summarization scenarios.

    Inputs:
      - run.output (dict with "summary" or str)
      - context.keywords: list[str] (required to be meaningful)
      - context.field_path: default "summary"
      - case_insensitive / strip_punctuation flags
      - unique_match: if True counts presence, else frequency sum

    Output:
      {"keyword_hits": int, "keyword_total": int, "coverage": float}
    """
    try:
        if not isinstance(run, dict):
            return {"keyword_hits": 0, "keyword_total": 0, "coverage": 0.0, "error": "invalid_run_type"}

        try:
            inp = KeywordCoverageInput(output=run.get("output"))  # type: ignore
        except ValidationError as ve:
            logger.error(f"keyword_coverage: invalid input {ve}")
            return {"keyword_hits": 0, "keyword_total": 0, "coverage": 0.0, "error": "validation_error"}

        ctx = KeywordCoverageContext(**(context or {}))  # type: ignore
        kw_list = [k for k in (ctx.keywords or []) if isinstance(k, str) and k.strip()]
        total = len(kw_list)
        if total == 0:
            return {"keyword_hits": 0, "keyword_total": 0, "coverage": 0.0, "reason": "no_keywords"}

        field_val = _extract_field(inp.output, ctx.field_path)
        text = _to_text(field_val)
        text = _normalize_text(text, ctx.case_insensitive, ctx.strip_punctuation)

        hits = 0
        if ctx.unique_match:
            for k in kw_list:
                k_norm = _normalize_text(k, ctx.case_insensitive, ctx.strip_punctuation).strip()
                if not k_norm:
                    continue
                if k_norm in text:
                    hits += 1
        else:
            # frequency-based: count occurrences
            for k in kw_list:
                k_norm = _normalize_text(k, ctx.case_insensitive, ctx.strip_punctuation).strip()
                if not k_norm:
                    continue
                # naive frequency count
                hits += text.count(k_norm)

        # Normalize coverage to [0,1] by default with unique presence semantics.
        # If frequency-based, we still clip to [0,1] relative to total keywords.
        coverage = min(1.0, float(hits) / float(total)) if total > 0 else 0.0

        return KeywordCoverageOutput(keyword_hits=hits, keyword_total=total, coverage=coverage).as_dict()
    except Exception as e:
        logger.exception("keyword_coverage metric failed")
        return {"keyword_hits": 0, "keyword_total": 0, "coverage": 0.0, "error": "exception", "reason": str(e)}


register_metric("keyword_coverage", evaluate)