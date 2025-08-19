from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

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


_FIELD_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_\-\s]+)\s*:\s*(.+?)\s*$")


def _normalize_field_name(name: str) -> str:
    return name.strip().lower()


def _parse_key_value_lines(text: str) -> Dict[str, str]:
    """Parse simple 'Field: value' lines into a dict (case-insensitive keys)."""
    result: Dict[str, str] = {}
    for line in text.splitlines():
        m = _FIELD_LINE_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if key and value is not None:
            result[_normalize_field_name(key)] = value.strip()
    return result


def _parse_json_like_blocks(text: str) -> Dict[str, str]:
    """
    Attempt to find JSON-like content and parse it safely.
    We:
    - Extract substrings that look like JSON objects {...}
    - Try json.loads; if it fails, skip
    - Only collect primitive str/number/bool values, and stringify them
    """
    extracted: Dict[str, str] = {}
    # Greedy but safe approach: find braces-balanced blocks roughly
    # For simplicity and determinism, scan for lines that appear to be JSON objects
    candidates: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        if depth > 0:
            buf.append(ch)
        if ch == "}":
            depth -= 1
            if depth == 0 and buf:
                candidates.append("".join(buf))
                buf = []
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(k, str):
                        key = _normalize_field_name(k)
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            extracted[key] = "" if v is None else str(v)
        except Exception:
            continue
    return extracted


class ExtractFieldsInput(SkillInputModel):
    text: str = Field(..., description="Source text to extract fields from", min_length=1)
    fields: List[str] = Field(
        ...,
        description="List of field names to extract. Matching is case-insensitive.",
        min_items=1,
        json_schema_extra={"example": ["name", "date"]},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "Name: Jane Doe\nDate: 2025-07-04", "fields": ["name", "date"]},
            ]
        }
    }


class ExtractFieldsOutput(SkillOutputModel):
    extracted: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary of extracted field values. Only includes found fields.",
        json_schema_extra={"example": {"name": "Jane Doe", "date": "2025-07-04"}},
    )


class ExtractFieldsSkill(Skill):
    """
    Deterministic field extractor.

    Strategy:
    - Parse 'Field: value' lines (case-insensitive field names)
    - Additionally parse JSON-like blocks and collect primitive fields
    - For requested fields, return those found (case-insensitive match)
    - Missing fields are omitted from the output
    """

    name = "extract_fields"
    description = "Extract key fields from plain text using simple patterns and JSON-like hints."
    input_model = ExtractFieldsInput
    output_model = ExtractFieldsOutput

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data = self.input_model.model_validate(params)
        except ValidationError as e:
            raise SkillExecutionError(f"Invalid input: {e}") from e

        text = data.text
        target_fields = [_normalize_field_name(f) for f in data.fields if isinstance(f, str) and f.strip()]
        if not target_fields:
            raise SkillExecutionError("No valid fields provided")

        # Collect candidates from two strategies
        kv_a = _parse_key_value_lines(text)
        kv_b = _parse_json_like_blocks(text)

        merged: Dict[str, str] = {}
        merged.update(kv_a)
        # Do not override values from A with B if A already provided a value
        for k, v in kv_b.items():
            merged.setdefault(k, v)

        out_map: Dict[str, str] = {}
        for tf in target_fields:
            if tf in merged:
                out_map[tf] = merged[tf]

        out = self.output_model(extracted=out_map)
        return out.model_dump()