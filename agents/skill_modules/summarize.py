from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from pydantic import Field, ValidationError

from .base import Skill, SkillExecutionError, SkillInputModel, SkillOutputModel

try:
    from fba_bench.core.logging import setup_logging  # type: ignore
except Exception:  # pragma: no cover
    setup_logging = None  # type: ignore

if setup_logging:
    try:
        setup_logging()
    except Exception:
        pass

logger = logging.getLogger(__name__)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    # Normalize whitespace
    txt = re.sub(r"\s+", " ", text.strip())
    if not txt:
        return []
    # Simple deterministic sentence split
    parts = _SENTENCE_SPLIT_RE.split(txt)
    # Ensure sentences have trailing punctuation removed to avoid duplicates when joining
    return [p.strip() for p in parts if p.strip()]


def _token_count(s: str) -> int:
    # Deterministic heuristic: tokens approximated by whitespace-separated words
    if not s:
        return 0
    return len([w for w in s.split(" ") if w])


class SummarizeInput(SkillInputModel):
    text: str = Field(..., description="Input text to summarize", min_length=1)
    max_tokens: int = Field(
        128,
        description="Maximum token budget for the output (whitespace tokens heuristic)",
        ge=1,
        le=2048,
        json_schema_extra={"example": 128},
    )

    model_config = {"json_schema_extra": {"examples": [{"text": "Sentence one. Sentence two.", "max_tokens": 8}]}}


class SummarizeOutput(SkillOutputModel):
    summary: str = Field(..., description="Deterministic extractive summary")

    model_config = {"json_schema_extra": {"examples": [{"summary": "Sentence one. Sentence two."}]}}


class SummarizeSkill(Skill):
    """
    Deterministic extractive summarizer.

    Strategy:
    - Split text into sentences (., !, ?).
    - Take sentences in order until token budget is reached.
    - Join with a single space.
    """

    name = "summarize"
    description = "Extractive summarization: select earliest sentences within token budget."
    input_model = SummarizeInput
    output_model = SummarizeOutput

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data = self.input_model.model_validate(params)
        except ValidationError as e:
            raise SkillExecutionError(f"Invalid input: {e}") from e

        sentences = _split_sentences(data.text)
        if not sentences:
            out = self.output_model(summary="")
            return out.model_dump()

        budget = int(data.max_tokens)
        selected: List[str] = []
        used = 0
        for s in sentences:
            tc = _token_count(s)
            if tc == 0:
                continue
            if used + tc > budget:
                break
            selected.append(s)
            used += tc

        # If nothing could be selected within budget (likely very small budget), fallback to first sentence truncated
        if not selected:
            first = sentences[0]
            words = first.split(" ")
            truncated = " ".join(words[:max(1, budget)])
            selected = [truncated]

        summary = " ".join(selected).strip()
        out = self.output_model(summary=summary)
        return out.model_dump()