from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class RolloutCriteria:
    """
    Success criteria for a rollout phase.

    Fields used by tests:
      - min_success_rate: float (0..1)
      - max_error_rate: float (0..1)
      - max_response_time_ms: int
    """
    min_success_rate: float
    max_error_rate: float
    max_response_time_ms: int


@dataclass
class RolloutPhase:
    """
    A rollout phase definition.

    Fields used by tests:
      - phase_name: str
      - traffic_percentage: float (0..1)
      - duration_hours: int
      - success_criteria: RolloutCriteria
    """
    phase_name: str
    traffic_percentage: float
    duration_hours: int
    success_criteria: RolloutCriteria


class GradualRolloutManager:
    """
    Minimal deterministic rollout manager used by tests.

    Provides:
      - await initialize_phase(phase: RolloutPhase) -> bool
      - await configure_traffic_routing(percentage: float) -> bool
      - await validate_phase_criteria(phase: RolloutPhase, monitoring_data: dict) -> bool
    """

    def __init__(self) -> None:
        self._current_phase: str | None = None
        self._current_traffic: float = 0.0

    async def initialize_phase(self, phase: RolloutPhase) -> bool:
        if not phase.phase_name or phase.traffic_percentage < 0 or phase.traffic_percentage > 1:
            return False
        self._current_phase = phase.phase_name
        logger.info("Initialized rollout phase: %s", phase.phase_name)
        return True

    async def configure_traffic_routing(self, percentage: float) -> bool:
        if percentage < 0 or percentage > 1:
            return False
        self._current_traffic = float(percentage)
        logger.info("Configured traffic routing to %.2f%%", self._current_traffic * 100.0)
        return True

    async def validate_phase_criteria(self, phase: RolloutPhase, monitoring_data: Dict[str, Any]) -> bool:
        """
        Validate provided monitoring metrics against phase.success_criteria.
        Tests supply keys:
          - success_rate: float
          - error_rate: float
          - response_time_ms: int
        """
        crit = phase.success_criteria
        try:
            success_rate = float(monitoring_data.get("success_rate", 0.0))
            error_rate = float(monitoring_data.get("error_rate", 1.0))
            response_time_ms = int(monitoring_data.get("response_time_ms", 10_000))
        except Exception:
            return False

        ok = (
            success_rate >= crit.min_success_rate and
            error_rate <= crit.max_error_rate and
            response_time_ms <= crit.max_response_time_ms
        )
        logger.info(
            "Phase '%s' criteria validation: success_rate=%.3f (>= %.3f), error_rate=%.3f (<= %.3f), response=%dms (<= %dms) -> %s",
            phase.phase_name,
            success_rate, crit.min_success_rate,
            error_rate, crit.max_error_rate,
            response_time_ms, crit.max_response_time_ms,
            "PASS" if ok else "FAIL",
        )
        return ok


__all__ = [
    "GradualRolloutManager",
    "RolloutPhase",
    "RolloutCriteria",
]