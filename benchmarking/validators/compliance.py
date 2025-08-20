from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union


@dataclass(frozen=True)
class Rule:
    """A single compliance rule."""
    name: str
    description: str = ""
    # Path to value within the payload, dot-separated (e.g., "order.item.price")
    path: Optional[str] = None
    # Expected type for the value (if provided)
    expected_type: Optional[type] = None
    # Whether the field must be present and not None
    required: bool = False
    # Numeric range constraints (inclusive bounds)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    # String length constraints (inclusive bounds)
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    # Regex pattern the value must match (for str)
    regex: Optional[str] = None
    # Set of allowed values (for exact membership checks)
    allowed_values: Optional[Iterable[Any]] = None
    # Custom callable for advanced validation: fn(value) -> tuple[bool, str|None]
    custom_validator: Optional[Any] = None
    # If True, a failure on this rule marks the overall result as non-compliant
    critical: bool = False


@dataclass
class RuleEvaluation:
    rule: Rule
    passed: bool
    message: str = ""
    value: Any = None


@dataclass
class ComplianceReport:
    compliant: bool
    score: float
    total_rules: int
    passed_rules: int
    failed_rules: int
    critical_failures: int
    evaluations: List[RuleEvaluation] = field(default_factory=list)


class ComplianceValidator:
    """Validates arbitrary payloads against a set of compliance rules.

    Design goals:
    - Deterministic and side-effect free
    - Robust for heterogeneous payloads (dicts, nested dicts/lists)
    - Human-readable report with scoring
    """

    def __init__(self, strict: bool = False) -> None:
        """
        :param strict: When True, any rule failure makes report.compliant False even if non-critical.
        """
        self.strict = strict

    def validate(
        self,
        payload: Mapping[str, Any] | Any,
        rules: Iterable[Rule],
    ) -> ComplianceReport:
        evaluations: List[RuleEvaluation] = []
        total = 0
        passed = 0
        critical_failures = 0

        for rule in rules:
            total += 1
            ok, msg, value = self._evaluate_rule(payload, rule)
            if ok:
                passed += 1
                evaluations.append(RuleEvaluation(rule=rule, passed=True, message="OK", value=value))
            else:
                if rule.critical:
                    critical_failures += 1
                evaluations.append(RuleEvaluation(rule=rule, passed=False, message=msg or "Failed", value=value))

        failed = total - passed
        # Score: percent of rules passed (0..1)
        score = (passed / total) if total else 1.0

        if self.strict:
            compliant = failed == 0
        else:
            # Non-strict: compliant if no critical failures and score >= 0.8 by default
            compliant = (critical_failures == 0) and (score >= 0.8)

        return ComplianceReport(
            compliant=compliant,
            score=score,
            total_rules=total,
            passed_rules=passed,
            failed_rules=failed,
            critical_failures=critical_failures,
            evaluations=evaluations,
        )

    def _evaluate_rule(
        self,
        payload: Mapping[str, Any] | Any,
        rule: Rule,
    ) -> Tuple[bool, Optional[str], Any]:
        # Resolve value by path if provided, else operate on root payload
        value = self._resolve_path(payload, rule.path) if rule.path else payload

        # Required presence
        if rule.required and value is None:
            return False, f"Required value missing at path '{rule.path or '<root>'}'", value

        # If not required and missing, count as pass (no constraint to check)
        if value is None:
            return True, None, value

        # Type check
        if rule.expected_type is not None and not isinstance(value, rule.expected_type):
            return False, f"Type mismatch: expected {rule.expected_type.__name__}, got {type(value).__name__}", value

        # Numeric range
        if isinstance(value, (int, float)):
            if rule.min_value is not None and value < rule.min_value:
                return False, f"Value {value} < min {rule.min_value}", value
            if rule.max_value is not None and value > rule.max_value:
                return False, f"Value {value} > max {rule.max_value}", value

        # String constraints
        if isinstance(value, str):
            if rule.min_length is not None and len(value) < rule.min_length:
                return False, f"Length {len(value)} < min_length {rule.min_length}", value
            if rule.max_length is not None and len(value) > rule.max_length:
                return False, f"Length {len(value)} > max_length {rule.max_length}", value
            if rule.regex is not None:
                if not re.fullmatch(rule.regex, value):
                    return False, f"Regex mismatch: pattern '{rule.regex}'", value

        # Membership constraint
        if rule.allowed_values is not None:
            allowed_set = set(rule.allowed_values)
            if value not in allowed_set:
                return False, f"Value '{value}' not in allowed set {sorted(allowed_set)}", value

        # Custom validator
        if rule.custom_validator is not None:
            try:
                res = rule.custom_validator(value)
            except Exception as ex:
                return False, f"Custom validator error: {ex}", value
            ok, msg = self._normalize_custom_result(res)
            if not ok:
                return False, msg or "Custom validator failed", value

        return True, None, value

    def _resolve_path(self, payload: Any, path: Optional[str]) -> Any:
        if not path:
            return payload
        current = payload
        for part in path.split("."):
            if current is None:
                return None
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            elif isinstance(current, (list, tuple)):
                # Support list indices in path
                try:
                    idx = int(part)
                except ValueError:
                    return None
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            else:
                # Fallback to attribute access
                try:
                    current = getattr(current, part)
                except Exception:
                    return None
        return current

    @staticmethod
    def _normalize_custom_result(res: Any) -> Tuple[bool, Optional[str]]:
        """Normalize custom validator result into (ok, message)."""
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], bool):
            return bool(res[0]), str(res[1]) if res[1] is not None else None
        if isinstance(res, bool):
            return res, None
        # Any truthy value treated as pass without message
        return bool(res), None