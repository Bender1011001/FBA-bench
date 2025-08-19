from __future__ import annotations

import logging
from typing import Any, Dict, Optional

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


class LookupInput(SkillInputModel):
    key: str = Field(..., description="Key to look up", min_length=1)
    table: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary mapping keys to values. Case-sensitive keys.",
        json_schema_extra={"example": {"SKU-42": "Widget XL"}},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"key": "SKU-42", "table": {"SKU-42": "Widget XL"}}]
        }
    }


class LookupOutput(SkillOutputModel):
    value: Optional[str] = Field(default=None, description="Lookup result or None if not found")
    found: bool = Field(default=False, description="True if key was found exactly")


class LookupSkill(Skill):
    """
    Deterministic dictionary lookup.

    - Case-sensitive by default (documented).
    - Returns {'value': None, 'found': False} if missing.
    """

    name = "lookup"
    description = "Exact, deterministic dictionary lookup (case-sensitive keys)."
    input_model = LookupInput
    output_model = LookupOutput

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data = self.input_model.model_validate(params)
        except ValidationError as e:
            raise SkillExecutionError(f"Invalid input: {e}") from e

        val = data.table.get(data.key)
        out = self.output_model(value=val, found=(val is not None))
        return out.model_dump()