from __future__ import annotations

from typing import Any, Dict, Optional


class AppError(Exception):
    """
    Base application error with structured fields for central handling.
    """

    def __init__(self, *, code: str, message: str, status_code: int, detail: Optional[Any] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.detail = detail


class SimulationNotFoundError(AppError):
    """
    Raised when a simulation cannot be found.
    """

    def __init__(self, simulation_id: str, detail: Optional[Dict[str, Any]] = None) -> None:
        data: Dict[str, Any] = {"simulation_id": simulation_id}
        if isinstance(detail, dict):
            data.update(detail)
        super().__init__(
            code="simulation_not_found",
            message=f"Simulation '{simulation_id}' not found",
            status_code=404,
            detail=data,
        )


class SimulationStateError(AppError):
    """
    Raised when a simulation is not in an expected state for an operation.
    """

    def __init__(self, simulation_id: str, expected: str, actual: str, detail: Optional[Dict[str, Any]] = None) -> None:
        data: Dict[str, Any] = {"simulation_id": simulation_id, "expected": expected, "actual": actual}
        if isinstance(detail, dict):
            data.update(detail)
        super().__init__(
            code="invalid_simulation_state",
            message=f"Invalid simulation state for '{simulation_id}'",
            status_code=400,
            detail=data,
        )


__all__ = [
    "AppError",
    "SimulationNotFoundError",
    "SimulationStateError",
]