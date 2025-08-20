from typing import Any, Optional, Dict
import logging

# Expose a module-level logger so tests can patch: patch('tools.error_handler.logger', ...)
logger = logging.getLogger(__name__)

__all__ = ["handle_common_errors_for_agent"]

def _is_mock_like(obj: Any) -> bool:
    """Detect unittest.mock objects without importing Mock."""
    try:
        mod = type(obj).__module__
        return mod in ("unittest.mock", "mock")
    except Exception:
        return False

def _pick_str(value: Any) -> Optional[str]:
    """Return value if it's a non-empty, non-mock string; else None."""
    if isinstance(value, str) and value and not value.lower().startswith("<mock"):
        return value
    return None

def _agent_identifier(agent: Optional[Any]) -> str:
    if agent is None:
        return "unknown agent"

    # Try common string attributes first.
    for attr in ("name", "agent_id", "id"):
        try:
            # Avoid triggering mock auto-creation: look in __dict__ first.
            raw = getattr(agent, "__dict__", {}) or {}
            cand = raw.get(attr, None)
            s = _pick_str(cand)
            if s:
                return s
        except Exception:
            pass

        try:
            cand = getattr(agent, attr, None)
            if not _is_mock_like(cand):
                s = _pick_str(cand)
                if s:
                    return s
        except Exception:
            pass

    # Try common callables that might return a string id/name.
    for meth in ("get_name", "get_id", "identifier"):
        fn = getattr(agent, meth, None)
        if callable(fn):
            try:
                cand = fn()
                if not _is_mock_like(cand):
                    s = _pick_str(cand)
                    if s:
                        return s
            except Exception:
                pass

    return "unknown agent"

def _normalize_error(error: Exception, agent: Optional[Any]) -> Dict[str, Any]:
    etype = type(error).__name__
    known = {"ValueError", "KeyError", "TypeError", "AttributeError", "RuntimeError"}
    error_type = etype if etype in known else "UnexpectedError"
    return {
        "agent": _agent_identifier(agent),
        "error_type": error_type,
        "message": str(error),
    }

def handle_common_errors_for_agent(error: Exception, agent: Optional[Any] = None) -> None:
    """Log common errors for an agent; tests expect (error, agent) and return None."""
    payload = _normalize_error(error, agent)
    logger.error(f"{payload['error_type']}: {payload['message']} [agent: {payload['agent']}]")
    return None
