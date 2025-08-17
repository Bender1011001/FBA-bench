from __future__ import annotations

"""
Structured LLM output contracts for FBA-Bench agent runners.

These Pydantic models define the validated, strongly-typed results that LLM-backed
agents must return. Runners should instruct models to emit JSON that conforms to
FbaDecision.model_json_schema(), then parse/validate the response using
FbaDecision.model_validate_json().

Usage example (inside a runner):

    from fba_bench.core.llm_outputs import FbaDecision
    from pydantic import ValidationError

    schema = FbaDecision.model_json_schema()
    # Embed `schema` in the prompt and instruct JSON-only output

    llm_text = run_llm_and_get_text()
    try:
        decision = FbaDecision.model_validate_json(llm_text)
        return decision.model_dump()
    except ValidationError as e:
        logger.error("LLM output validation failed: %s", e)
        raise AgentRunnerDecisionError("Invalid LLM output") from e
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class PriceDecision(BaseModel):
    asin: str = Field(..., description="The product ASIN.")
    new_price: float = Field(..., gt=0.0, description="The recommended new price in USD.")
    reasoning: str = Field(..., min_length=1, description="The justification for this price.")

    @field_validator("asin")
    @classmethod
    def asin_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("asin must be a non-empty string")
        return v.strip()


class FbaDecision(BaseModel):
    pricing_decisions: List[PriceDecision] = Field(
        ...,
        description="A list of pricing decisions to execute."
    )
    meta: Optional[dict] = Field(
        default=None,
        description="Optional metadata about the decision, such as constraints applied or model confidence."
    )

    def to_dict(self) -> dict:
        """
        Convenience method to produce a plain dict suitable for downstream usage.
        """
        return self.model_dump()

    def to_json(self) -> str:
        """
        Convenience method to produce a JSON string of the decision.
        """
        return self.model_dump_json()