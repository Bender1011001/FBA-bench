from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Type

from pydantic import BaseModel

from .base import Skill, SkillExecutionError

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

# Global registry mapping unique skill key to Skill class
RUNNING_REGISTRY: Dict[str, Type[Skill]] = {}


def register(name: str, cls: Type[Skill]) -> None:
    """
    Register a Skill class by unique key.

    Raises:
        ValueError if name already registered or cls invalid.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Skill name must be a non-empty string")
    if not issubclass(cls, Skill):
        raise ValueError("cls must be a subclass of Skill")
    if name in RUNNING_REGISTRY:
        raise ValueError(f"Skill '{name}' is already registered")
    RUNNING_REGISTRY[name] = cls
    logger.debug(f"Registered skill '{name}' -> {cls.__name__}")


def create(name: str, config: Dict[str, Any] | None = None) -> Skill:
    """
    Resolve and instantiate a registered skill by name.

    Raises:
        ValueError if name unknown.
    """
    cls = RUNNING_REGISTRY.get(name)
    if not cls:
        supported = ", ".join(sorted(RUNNING_REGISTRY.keys()))
        raise ValueError(f"Unknown skill '{name}'. Supported: [{supported}]")
    return cls(config=config)


def list_skills() -> List[Dict[str, Any]]:
    """
    List registered skills with minimal metadata.
    """
    items: List[Dict[str, Any]] = []
    for name, cls in sorted(RUNNING_REGISTRY.items(), key=lambda kv: kv[0]):
        # Attempt to access schemas; fallback to None
        input_schema: Dict[str, Any] | None = None
        output_schema: Dict[str, Any] | None = None
        try:
            if hasattr(cls, "input_model") and issubclass(cls.input_model, BaseModel):  # type: ignore
                input_schema = cls.input_model.model_json_schema()  # type: ignore
            if hasattr(cls, "output_model") and issubclass(cls.output_model, BaseModel):  # type: ignore
                output_schema = cls.output_model.model_json_schema()  # type: ignore
        except Exception:
            pass
        items.append(
            {
                "name": name,
                "class": cls.__name__,
                "description": getattr(cls, "description", ""),
                "input_schema": input_schema,
                "output_schema": output_schema,
            }
        )
    return items


# Auto-register built-in skills on import
def _auto_register_builtins() -> None:
    """
    Import and register builtin skills. Import-time side effects populate RUNNING_REGISTRY.
    """
    from .calculator import CalculatorSkill  # noqa: F401
    from .summarize import SummarizeSkill  # noqa: F401
    from .extract_fields import ExtractFieldsSkill  # noqa: F401
    from .lookup import LookupSkill  # noqa: F401
    from .transform_text import TransformTextSkill  # noqa: F401

    # Explicitly ensure they are registered (idempotent if module-level register is in class file)
    for cls in (CalculatorSkill, SummarizeSkill, ExtractFieldsSkill, LookupSkill, TransformTextSkill):
        try:
            register(cls.name, cls)
        except ValueError:
            # Already registered, ignore
            pass


# Eagerly load builtins
try:
    _auto_register_builtins()
except Exception as e:  # pragma: no cover - this should not fail; log and continue
    logger.error(f"Failed auto-registering built-in skills: {e}")