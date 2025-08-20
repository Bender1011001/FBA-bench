from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TypedDict, Union

Number = Union[int, float, Decimal]


class ConsistencyMetrics(TypedDict, total=False):
    # Aggregate metrics commonly referenced by tests
    average_consistency: float
    consistency_variance: float
    tests_meeting_threshold: int
    total_tests: int
    # State and API-specific metrics
    state_synchronization: float
    api_response_consistency: float


@dataclass
class ValidationResult:
    """Normalized validation result used by integration tests."""
    passed: bool
    score: float
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class ConsistencyChecker:
    """
    Provides deterministic, conservative consistency checks between:
    - Simulation vs Sandbox aggregated results
    - Exported environment states
    - API responses across environments

    Design goals:
    - Return a score in [0.0, 1.0]
    - Be stable and order-independent
    - Ignore extraneous keys and focus on comparable structure and numeric values
    - Treat missing keys as partial mismatch rather than hard failure
    """

    def __init__(self, numeric_tolerance: float = 1e-6) -> None:
        # numeric_tolerance: absolute tolerance when comparing numbers
        self.numeric_tolerance = float(numeric_tolerance)

    # Public async APIs expected by tests

    async def calculate_consistency(self, sim_result: Mapping[str, Any], sandbox_result: Mapping[str, Any]) -> float:
        """
        Calculate overall consistency between two structured results (dict-like).
        Compares overlapping fields with a weighted score that favors numeric closeness,
        while accounting for partial structural mismatches.
        """
        return await asyncio.to_thread(self._structural_similarity, sim_result, sandbox_result)

    async def verify_state_consistency(self, simulation_env: Any, sandbox_env: Any) -> float:
        """
        Export state from both environments and compare deterministically.
        Uses the same structural similarity metric specialized for state payloads.
        """
        # Prefer exporting fresh state from both to validate synchronization
        try:
            sim_state = await simulation_env.export_state()
        except TypeError:
            # Allow sync export_state fallback
            sim_state = simulation_env.export_state()

        try:
            sandbox_state = await sandbox_env.export_state()
        except TypeError:
            sandbox_state = sandbox_env.export_state()

        return await asyncio.to_thread(self._structural_similarity, sim_state, sandbox_state)

    async def compare_api_responses(self, sim_response: Any, sandbox_response: Any) -> float:
        """
        Compare API responses across environments. For dicts, compares shared keys and numeric fields.
        For lists, compares element-by-element using dict/primitive comparison where applicable.
        Falls back to normalized string equality when types are not directly comparable.
        """
        return await asyncio.to_thread(self._api_response_similarity, sim_response, sandbox_response)

    # Internal similarity utilities

    def _api_response_similarity(self, a: Any, b: Any) -> float:
        # Dispatch based on types
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            return self._mapping_similarity(a, b)
        if isinstance(a, list) and isinstance(b, list):
            return self._list_similarity(a, b)
        if self._is_number(a) and self._is_number(b):
            return self._numeric_similarity(float(a), float(b))
        # String/primitive fallback
        return 1.0 if self._normalize_primitive(a) == self._normalize_primitive(b) else 0.0

    def _structural_similarity(self, a: Any, b: Any) -> float:
        """
        Recursively compare structured content. Returns a score in [0,1].
        Weights:
          - numeric closeness emphasized
          - structural key overlap
          - graceful handling of missing/extra keys
        """
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            return self._mapping_similarity(a, b)
        if isinstance(a, list) and isinstance(b, list):
            return self._list_similarity(a, b)
        if self._is_number(a) and self._is_number(b):
            return self._numeric_similarity(float(a), float(b))
        return 1.0 if self._normalize_primitive(a) == self._normalize_primitive(b) else 0.0

    def _mapping_similarity(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
        if not a and not b:
            return 1.0
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        shared = a_keys & b_keys
        if not shared:
            # No overlapping keys: structural mismatch
            return 0.0 if (a_keys or b_keys) else 1.0

        # Key overlap score (Jaccard)
        key_overlap = len(shared) / len(a_keys | b_keys)

        # Value similarity across shared keys
        value_scores: List[float] = []
        for k in shared:
            value_scores.append(self._structural_similarity(a[k], b[k]))
        value_score = sum(value_scores) / max(len(value_scores), 1)

        # Balance structural and value similarity
        # Favor values but keep structure relevant
        return max(0.0, min(1.0, 0.3 * key_overlap + 0.7 * value_score))

    def _list_similarity(self, a: List[Any], b: List[Any]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        # Compare up to min length pairwise, then penalize for length difference
        n = min(len(a), len(b))
        pair_scores = [self._structural_similarity(a[i], b[i]) for i in range(n)]
        base = sum(pair_scores) / max(n, 1)

        # Length penalty (soft)
        length_penalty = 1.0 - (abs(len(a) - len(b)) / max(len(a), len(b)))
        length_penalty = max(0.0, min(1.0, length_penalty))

        return max(0.0, min(1.0, 0.8 * base + 0.2 * length_penalty))

    def _numeric_similarity(self, a: float, b: float) -> float:
        # Absolute tolerance wins near zero; relative tolerance for larger magnitudes
        if math.isclose(a, b, rel_tol=1e-9, abs_tol=self.numeric_tolerance):
            return 1.0
        # Relative similarity: 1 - relative error, clamped to [0,1]
        denom = max(abs(a), abs(b), self.numeric_tolerance)
        rel_err = abs(a - b) / denom
        return max(0.0, min(1.0, 1.0 - rel_err))

    @staticmethod
    def _is_number(x: Any) -> bool:
        return isinstance(x, (int, float, Decimal)) and not isinstance(x, bool)

    @staticmethod
    def _normalize_primitive(x: Any) -> str:
        # Normalize timestamps or datetimes if present
        if isinstance(x, datetime):
            return x.isoformat()
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="ignore")
        return str(x)


# Convenience helpers exposed for potential external use

def summarize_consistency_scores(scores: Iterable[float]) -> ConsistencyMetrics:
    s = list(scores)
    if not s:
        return ConsistencyMetrics(average_consistency=0.0, consistency_variance=0.0, tests_meeting_threshold=0, total_tests=0)
    avg = sum(s) / len(s)
    var = 0.0
    if len(s) > 1:
        mean = avg
        var = sum((x - mean) ** 2 for x in s) / (len(s) - 1)
    return ConsistencyMetrics(
        average_consistency=max(0.0, min(1.0, avg)),
        consistency_variance=max(0.0, var),
        tests_meeting_threshold=sum(1 for x in s if x >= 0.85),
        total_tests=len(s),
    )