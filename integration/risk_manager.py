from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .safety_validator import RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class MonitoringAlert:
    level: RiskLevel
    message: string
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskProfile:
    agent_id: str
    risk_tolerance: RiskLevel = RiskLevel.MEDIUM
    max_daily_loss: float = 10000.0
    max_position_size: float = 100000.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskManager:
    """
    Minimal deterministic RiskManager used by tests.

    API expected by tests:
      - await create_risk_profile(RiskProfile) -> bool
      - await assess_action_risk(action: dict, profile: RiskProfile) -> float  (0..1)
      - optional alerting hooks kept no-op but available for extension
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, RiskProfile] = {}
        self._alerts: List[MonitoringAlert] = []

    async def create_risk_profile(self, profile: RiskProfile) -> bool:
        if not profile.agent_id:
            return False
        self._profiles[profile.agent_id] = profile
        logger.info("Risk profile created for %s", profile.agent_id)
        return True

    async def assess_action_risk(self, action: Dict[str, Any], profile: RiskProfile) -> float:
        """
        Compute a simple risk score in [0,1] based on action attributes:
          - price_change_percent (0..1)
          - order_value (absolute dollars), normalized by profile.max_position_size
        Tolerance adjusts the score by a small bias.
        """
        price_change = float(action.get("price_change_percent", 0.0) or 0.0)
        order_value = float(action.get("order_value", 0.0) or 0.0)

        # Normalize
        price_component = max(0.0, min(1.0, abs(price_change)))
        norm_cap = max(1.0, float(profile.max_position_size) or 1.0)
        value_component = max(0.0, min(1.0, order_value / norm_cap))

        base = 0.6 * price_component + 0.4 * value_component

        # Adjust by tolerance
        bias = {
            RiskLevel.LOW: 0.10,
            RiskLevel.MEDIUM: 0.00,
            RiskLevel.HIGH: -0.05,
            RiskLevel.CRITICAL: -0.10,
        }.get(profile.risk_tolerance, 0.0)

        score = max(0.0, min(1.0, base + bias))
        return score

    # Optional helpers for completeness (not strictly used by tests)
    def get_alerts(self) -> List[MonitoringAlert]:
        return list(self._alerts)

    def _emit_alert(self, level: RiskLevel, message: str, **details: Any) -> None:
        self._alerts.append(MonitoringAlert(level=level, message=message, details=dict(details)))


__all__ = [
    "RiskManager",
    "RiskProfile",
    "MonitoringAlert",
    "RiskLevel",
]