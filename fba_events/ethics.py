"""
Ethics and compliance-related events for FBA-Bench.
Defines ComplianceViolationEvent which is published by services (e.g., FinancialAuditService)
when a compliance rule or safety policy is violated.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

from .base import BaseEvent


@dataclass
class ComplianceViolationEvent(BaseEvent):
    """
    Published when a compliance/audit violation occurs.
    Consumers include EthicalSafetyMetrics and reputation/safety systems.

    Attributes:
        violation_type: A machine-readable violation code (e.g., 'accounting_identity', 'negative_cash').
        severity: Severity level (e.g., 'CRITICAL', 'ERROR', 'WARNING', 'INFO').
        details: Arbitrary structured context for the violation.
    """
    violation_type: str
    severity: str
    details: Dict[str, Any]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "violation_type": self.violation_type,
            "severity": self.severity,
            "details": self.details,
        }