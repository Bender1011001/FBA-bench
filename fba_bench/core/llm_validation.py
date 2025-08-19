from __future__ import annotations

"""
Validation utilities for LLM outputs defined in fba_bench.core.llm_outputs.

Features:
- Pydantic v2-based strict and non-strict validation for defined contracts
- JSON parsing and safe, optional loose type coercions in non-strict mode
- JSON Schema export via Pydantic's model_json_schema()
- Optional jsonschema-based validation path for strict external schema checks
- Centralized logging of validation outcomes using the project's logging utilities
- Convenience mapping from contract names to concrete models

Public API:
- get_schema(model)
- validate_output(model, payload, strict=True)
- validate_with_jsonschema(schema, payload)
- validate_by_name(contract, payload, strict=True)
- CONTRACT_REGISTRY
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from fba_bench.core.logging import setup_logging  # Ensure consistent formatting/handlers
from fba_bench.core.llm_outputs import (
    AgentResponse,
    FbaDecision,
    TaskPlan,
    ToolCall,
)

# Initialize logging (idempotent)
setup_logging()
logger = logging.getLogger(__name__)


# ---- Contract Registry -------------------------------------------------------

CONTRACT_REGISTRY: Dict[str, Type[BaseModel]] = {
    "fba_decision": FbaDecision,
    "task_plan": TaskPlan,
    "tool_call": ToolCall,
    "agent_response": AgentResponse,
}


# ---- Helpers ----------------------------------------------------------------


def _truncate_payload_for_log(payload: Union[str, bytes, Dict[str, Any]], limit: int = 600) -> str:
    try:
        if isinstance(payload, (str, bytes)):
            s = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
            return (s[:limit] + "...") if len(s) > limit else s
        # dict-like
        s = json.dumps(payload, ensure_ascii=False)
        return (s[:limit] + "...") if len(s) > limit else s
    except Exception:
        return "<unserializable payload>"


def _parse_payload(payload: Union[str, bytes, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, bytes):
        try:
            return json.loads(payload.decode("utf-8", errors="replace"))
        except Exception as e:
            raise ValueError(f"Failed to parse bytes payload to JSON: {e}") from e
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception as e:
            raise ValueError(f"Failed to parse string payload to JSON: {e}") from e
    raise TypeError(f"Unsupported payload type: {type(payload).__name__}")


def coerce_loose_types(obj: Any) -> Any:
    """
    Best-effort, safe coercions for non-strict mode:
    - "123" -> 123 when used as numeric
    - "123.45" -> 123.45
    - "true"/"false" -> True/False (case-insensitive) if unambiguous
    - Trim whitespace around strings
    This is applied generically; Pydantic will still perform type validation.
    """
    if isinstance(obj, dict):
        return {k: coerce_loose_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce_loose_types(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        # boolean
        low = s.lower()
        if low in ("true", "false"):
            return low == "true"
        # int
        try:
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return int(s)
        except Exception:
            pass
        # float
        try:
            # Reject if it contains non-numeric except one dot / leading sign
            f = float(s)
            return f
        except Exception:
            return s
    return obj


def _build_model_variant(base: Type[BaseModel], *, strict: bool) -> Type[BaseModel]:
    """
    Build a model subclass with adjusted model_config:
    - strict=True: no coercions, forbid extra
    - strict=False: allow coercions, ignore extra (strip unknowns)
    """
    cfg = ConfigDict(strict=strict, extra="forbid" if strict else "ignore")
    # Create a dynamic subclass to adjust config without mutating original model.
    # Pydantic v2: prefer passing model_config instead of __config__.
    return create_model(  # type: ignore[call-arg]
        base.__name__ + ("Strict" if strict else "Lax"),
        __base__=base,
        model_config=cfg,
    )


def _normalize_pydantic_errors(e: ValidationError) -> List[Dict[str, Any]]:
    errs: List[Dict[str, Any]] = []
    for err in e.errors():
        loc = "/".join(str(p) for p in err.get("loc", []))
        errs.append(
            {
                "loc": loc,
                "msg": err.get("msg", ""),
                "type": err.get("type", ""),
                "input": err.get("input", None),
                "ctx": err.get("ctx", None),
            }
        )
    return errs


# ---- Public API --------------------------------------------------------------


def get_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Return the JSON Schema for the provided Pydantic model.
    """
    try:
        schema = model.model_json_schema()
        return schema
    except Exception as e:
        logger.exception("Failed generating schema for model=%s: %s", model.__name__, e)
        raise


def validate_output(
    model: Type[BaseModel],
    payload: Union[str, bytes, Dict[str, Any]],
    strict: bool = True,
) -> Tuple[bool, Optional[BaseModel], List[Dict[str, Any]]]:
    """
    Validate the given payload (JSON string/bytes/dict) against the provided model.
    Behavior depends on 'strict':
    - strict=True: extra fields forbidden; type coercions disabled; exact type matching required
    - strict=False: extra fields ignored/stripped; safe coercions attempted

    Returns: (ok, instance_or_none, errors)
    """
    truncated = _truncate_payload_for_log(payload)
    try:
        data = _parse_payload(payload)
    except Exception as e:
        error = {"loc": "root", "msg": str(e), "type": "json_parse_error"}
        logger.error(
            "LLM output parse failure for model=%s | errors=1 | payload=%s",
            model.__name__,
            truncated,
        )
        return False, None, [error]

    if not strict:
        data = coerce_loose_types(data)

    variant = _build_model_variant(model, strict=strict)
    try:
        instance = variant.model_validate(data)
        # Success logging at debug level to avoid noisy logs at scale
        logger.debug(
            "LLM output validated successfully for model=%s | strict=%s",
            model.__name__,
            strict,
        )
        return True, instance, []
    except ValidationError as e:
        errors = _normalize_pydantic_errors(e)
        logger.warning(
            "LLM output validation failed for model=%s | strict=%s | error_count=%d | payload=%s",
            model.__name__,
            strict,
            len(errors),
            truncated,
        )
        return False, None, errors
    except Exception as e:
        # Unexpected error
        err = {"loc": "root", "msg": str(e), "type": "internal_error"}
        logger.exception(
            "LLM output validation crashed for model=%s | strict=%s | payload=%s",
            model.__name__,
            strict,
            truncated,
        )
        return False, None, [err]


def validate_with_jsonschema(schema: Dict[str, Any], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate payload using jsonschema. Returns a list of normalized errors:
    [{"path": "...", "message": "...", "validator": "..."}]. Empty list if valid.
    Guarded by try/except import to keep dependency optional.
    """
    try:
        from jsonschema import ValidationError as JSValidationError  # type: ignore
        from jsonschema import validate as js_validate  # type: ignore
    except Exception:
        # jsonschema isn't available; return a sentinel error letting caller decide
        return [
            {
                "path": "root",
                "message": "jsonschema library not available; install 'jsonschema' to enable strict validation",
                "validator": "import",
            }
        ]

    try:
        js_validate(instance=payload, schema=schema)
        return []
    except JSValidationError as e:  # pragma: no cover - depends on input
        path = "/".join(str(p) for p in list(e.path))
        return [
            {
                "path": path if path else "root",
                "message": e.message,
                "validator": getattr(e, "validator", "unknown"),
            }
        ]
    except Exception as e:  # pragma: no cover - unexpected
        return [{"path": "root", "message": str(e), "validator": "internal"}]


def validate_by_name(
    contract: str, payload: Union[str, bytes, Dict[str, Any]], strict: bool = True
) -> Tuple[bool, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convenience wrapper to validate by contract registry key.
    On success returns (True, model_dump(), []).
    On failure returns (False, None, errors).
    """
    model = CONTRACT_REGISTRY.get(contract)
    if model is None:
        err = {
            "loc": "contract",
            "msg": f"Unknown contract '{contract}'",
            "type": "unknown_contract",
        }
        logger.error("Unknown LLM contract requested: %s", contract)
        return False, None, [err]

    ok, instance, errors = validate_output(model, payload, strict=strict)
    if not ok or instance is None:
        return False, None, errors
    # Return sanitized dict
    return True, instance.model_dump(), []