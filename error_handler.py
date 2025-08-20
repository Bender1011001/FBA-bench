from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class AgentError:
    code: str
    message: str
    context: Dict[str, Any]


class AgentErrorHandler:
    """
    Compatibility error handler expected by tests importing error_handler.AgentErrorHandler.

    Features:
    - record(code, message, **context): capture structured errors
    - has_errors(): quick boolean check
    - report(): normalized dict with count and error list
    - clear(): reset the error buffer
    """

    def __init__(self) -> None:
        self._errors: List[AgentError] = []

    def record(self, code: str, message: str, **context: Any) -> None:
        if not code or not isinstance(code, str):
            raise ValueError("code must be a non-empty string")
        if not isinstance(message, str):
            raise ValueError("message must be a string")
        self._errors.append(AgentError(code=code, message=message, context=dict(context)))

    def has_errors(self) -> bool:
        return len(self._errors) > 0

    def report(self) -> Dict[str, Any]:
        return {
            "count": len(self._errors),
            "errors": [e.__dict__ for e in self._errors],
        }

    def clear(self) -> None:
        self._errors.clear()