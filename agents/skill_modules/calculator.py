from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError

from .base import Skill, SkillExecutionError, SkillInputModel, SkillOutputModel, safe_arithmetic_eval

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


class CalculatorInput(SkillInputModel):
    expression: str = Field(
        ...,
        description="Arithmetic expression using +, -, *, /, //, %, ** and parentheses",
        min_length=1,
        json_schema_extra={"example": {"expression": "12*(3+4)"}},
    )


class CalculatorOutput(SkillOutputModel):
    result: float = Field(..., description="Numeric result as float", json_schema_extra={"example": 84.0})
    steps: List[str] = Field(
        default_factory=list,
        description="Deterministic evaluation steps for traceability",
        json_schema_extra={"example": ["parsed AST", "evaluated nodes safely"]},
    )


class CalculatorSkill(Skill):
    """
    Deterministic, safe arithmetic calculator.

    - Rejects any non-arithmetic constructs.
    - Guards exponent size and magnitude.
    """

    name = "calculator"
    description = "Deterministic arithmetic evaluator supporting +, -, *, /, //, %, ** and parentheses."
    input_model = CalculatorInput
    output_model = CalculatorOutput

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data = self.input_model.model_validate(params)
        except ValidationError as e:
            raise SkillExecutionError(f"Invalid input: {e}") from e

        expr = data.expression.strip()
        # Basic character filter to give fast-fail before AST parsing
        allowed_chars = set("0123456789.+-*/()% ")
        # include power and floor-div tokens
        # '**' and '//' are composed of allowed chars above
        if any(ch not in allowed_chars for ch in expr):
            raise SkillExecutionError("Expression contains disallowed characters")

        try:
            result = safe_arithmetic_eval(expr)
        except SkillExecutionError:
            # bubble up deterministic errors as-is
            raise
        except Exception as e:  # pragma: no cover
            raise SkillExecutionError(f"Unexpected error during evaluation: {e}") from e

        # Provide minimal deterministic steps (no introspection of AST to keep simple and safe)
        steps = ["parsed AST", "evaluated nodes safely"]

        out = self.output_model(result=float(result), steps=steps)
        return out.model_dump()

    async def arun(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Use default async-to-sync wrapper from base if desired; here compute is CPU-light
        return await super().arun(params)