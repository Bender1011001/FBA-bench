from __future__ import annotations

"""
Research Summarization Scenario

Multi-document input requiring targeted extraction and summary under token constraints.

Public interface exposed by this module:
- [python.class ResearchSummarizationConfig(BaseModel)](benchmarking/scenarios/research_summarization.py:1)
- [python.def generate_input(seed: int|None, params: dict|None) -> dict](benchmarking/scenarios/research_summarization.py:1)
- [python.async def run(input_payload: dict, runner_callable: Callable[[dict], Awaitable[dict]], timeout_seconds: int|None=None) -> dict](benchmarking/scenarios/research_summarization.py:1)

Registration:
- Registers itself under key "research_summarization" via the global ScenarioRegistry.
"""

import random
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from .registry import scenario_registry


def _rnd(seed: Optional[int]) -> random.Random:
    r = random.Random()
    if seed is not None:
        r.seed(seed)
    return r


def _estimate_tokens(text: str) -> int:
    # Simple deterministic token estimate via whitespace split; avoids external deps
    return len(text.split())


def _inject_noise(r: random.Random, text: str, noise_probability: float) -> str:
    # Deterministically inject simple noise sentences with a fixed template
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    augmented: List[str] = []
    for s in sentences:
        augmented.append(s)
        if s and r.random() < noise_probability:
            augmented.append(
                f"Note: Auxiliary observation {r.randint(100,999)} indicates limited applicability in edge contexts."
            )
    return " ".join(augmented)


class ResearchSummarizationConfig(BaseModel):
    """
    Configuration schema (Pydantic v2) for Research Summarization Scenario.
    """

    num_docs: int = Field(5, ge=1, le=50, description="Number of abstracts to synthesize")
    max_tokens: int = Field(200, ge=10, le=5000, description="Max tokens allowed for the summary")
    focus_keywords: Optional[List[str]] = Field(
        default=None,
        description="List of keywords the summary must address; case-insensitive matching",
    )
    noise_probability: float = Field(
        0.0,
        ge=0.0,
        le=0.5,
        description="Probability to inject a noise sentence after each sentence in each document",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "num_docs": 5,
                    "max_tokens": 180,
                    "focus_keywords": ["Q3", "revenue", "growth"],
                    "noise_probability": 0.1,
                }
            ]
        }
    }

    @field_validator("noise_probability")
    @classmethod
    def _check_noise(cls, v: float) -> float:
        if not (0.0 <= v <= 0.5):
            raise ValueError("noise_probability must be within [0.0, 0.5]")
        return v


_CORPUS_THEMES = [
    # theme, keyword variants
    ("financial performance", ["revenue", "profit", "margin", "Q3", "guidance"]),
    ("customer analytics", ["cohort", "retention", "churn", "LTV"]),
    ("product metrics", ["adoption", "activation", "MAU", "DAU"]),
    ("operations", ["throughput", "latency", "SLA", "uptime"]),
    ("research impact", ["novelty", "citation", "benchmark", "dataset"]),
]


def _make_doc(r: random.Random, idx: int, target_keywords: List[str]) -> Dict[str, Any]:
    theme, variants = _CORPUS_THEMES[idx % len(_CORPUS_THEMES)]
    # Ensure at least one overlap with target keywords with probability; otherwise include theme variants
    overlap_kw = None
    if target_keywords and r.random() < 0.7:
        overlap_kw = r.choice(target_keywords)
    else:
        overlap_kw = r.choice(variants)

    other_kw = r.sample(variants, k=min(2, len(variants)))
    sentences = [
        f"This study analyzes {theme} signals with emphasis on {overlap_kw}.",
        f"Our method improves {other_kw[0]} and maintains robust {other_kw[-1]} under stress.",
        f"Experimental evaluation spans Q{r.randint(1,4)} across two fiscal years.",
        f"Key finding: {overlap_kw} increased by {r.randint(5, 30)}% relative to baseline.",
        f"We discuss limitations and implications for future {theme} work.",
    ]
    return {
        "id": f"D{idx+1:03d}",
        "title": f"Research on {theme.title()} {idx+1}",
        "abstract": " ".join(sentences),
        "theme": theme,
        "keywords": list(set([overlap_kw] + other_kw)),
    }


def generate_input(seed: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Deterministically build a set of doc abstracts with overlapping keywords; optionally inject noise.
    """
    params = params or {}
    try:
        cfg = ResearchSummarizationConfig(**params)
    except ValidationError as e:
        raise ValueError(str(e)) from e

    r = _rnd(seed)
    focus_keywords = [k.lower() for k in (cfg.focus_keywords or [])]
    docs: List[Dict[str, Any]] = []
    for i in range(cfg.num_docs):
        d = _make_doc(r, i, focus_keywords)
        if cfg.noise_probability > 0.0:
            d["abstract"] = _inject_noise(r, d["abstract"], cfg.noise_probability)
        docs.append(d)

    prompt = (
        "Produce a focused summary consolidating the key findings across the provided abstracts. "
        "Limit output to max_tokens, and explicitly address the focus_keywords where relevant. "
        "Avoid copying noise; prefer concrete numbers and outcome statements."
    )

    payload = {
        "config": cfg.model_dump(),
        "seed": seed,
        "documents": docs,
        "focus_keywords": focus_keywords,
        "max_tokens": cfg.max_tokens,
        "task": "Targeted research summarization with constrained length and keyword coverage.",
        "prompt": prompt,
    }
    return payload


async def run(
    input_payload: Dict[str, Any],
    runner_callable: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute the scenario by calling the runner and evaluating its response.

    Runner contract:
    - Input: {
        "documents": list[{"id","title","abstract"}],
        "focus_keywords": list[str],
        "max_tokens": int,
        "prompt": str,
        "seed": int|None
      }
    - Output expected fields:
      {"summary": str}
    """
    runner_input = {
        "documents": input_payload.get("documents", []),
        "focus_keywords": input_payload.get("focus_keywords", []),
        "max_tokens": input_payload.get("max_tokens", 200),
        "prompt": input_payload.get("prompt"),
        "seed": input_payload.get("seed"),
        "instructions": (
            "Read all abstracts, extract salient quantitative outcomes, and compose a concise, "
            "non-redundant summary. Address focus keywords explicitly. Stay within max_tokens."
        ),
    }

    raw = await runner_callable(runner_input)

    summary: str = str(raw.get("summary", "")).strip()
    max_tokens: int = int(runner_input["max_tokens"])
    tokens = _estimate_tokens(summary)
    length_ok = tokens <= max_tokens

    # Compute keyword hits, case-insensitive
    fk: List[str] = list(runner_input.get("focus_keywords", []))
    lowered = summary.lower()
    hits = 0
    for kw in fk:
        # hit if keyword or its simple plural/singular appears
        if kw in lowered or (kw.endswith("s") and kw[:-1] in lowered) or (kw + "s") in lowered:
            hits += 1

    # Coverage score combines hits normalized and a brevity bonus (within constraint)
    coverage_score = 0.0
    if fk:
        coverage_score = hits / len(fk)
    brevity_bonus = 1.0 if length_ok else max(0.0, 1.0 - (tokens - max_tokens) / max(max_tokens, 1))
    final_score = round(0.7 * coverage_score + 0.3 * brevity_bonus, 4)

    return {
        "summary": summary,
        "coverage_score": float(final_score),
        "length_ok": bool(length_ok),
        "keyword_hits": int(hits),
    }


# Register with the scenario registry under the key "research_summarization".
@dataclass
class ResearchSummarizationScenarioAdapter:
    Config = ResearchSummarizationConfig

    @staticmethod
    def generate_input(seed: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return generate_input(seed=seed, params=params)

    @staticmethod
    async def run(
        input_payload: Dict[str, Any],
        runner_callable: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        return await globals()["run"](input_payload, runner_callable, timeout_seconds)


scenario_registry.register("research_summarization", ResearchSummarizationScenarioAdapter)