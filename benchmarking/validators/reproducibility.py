from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ReproducibilityReport:
    overall_score: float
    issues: List[Dict[str, Any]]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "issues": list(self.issues),
            "summary": dict(self.summary),
        }


class ReproducibilityValidator:
    """
    Minimal, production-safe reproducibility validator expected by tests:
    from benchmarking.validators.reproducibility import ReproducibilityValidator

    API:
      validate_reproducibility(run_id: str, reference_results: dict, current_results: dict) -> ReproducibilityReport

    Behavior:
      - Compares two series 'total_profit_series' if present
      - Computes a simple normalized score in [0,1] based on element-wise absolute difference
      - Reports issues if lengths mismatch or if series missing
      - Never raises; always returns a structured report suitable for JSON serialization
    """

    def __init__(self, tolerance: float = 1e-6) -> None:
        self.tolerance = float(tolerance)

    def validate_reproducibility(
        self,
        run_id: str,
        reference_results: Dict[str, Any],
        current_results: Dict[str, Any],
    ) -> ReproducibilityReport:
        issues: List[Dict[str, Any]] = []
        summary: Dict[str, Any] = {"run_id": str(run_id)}

        ref_series = []
        cur_series = []

        try:
            ref_series = list(reference_results.get("total_profit_series") or [])
            cur_series = list(current_results.get("total_profit_series") or [])
        except Exception:
            issues.append({
                "severity": "error",
                "message": "invalid_input_structure",
            })
            return ReproducibilityReport(
                overall_score=0.0,
                issues=issues,
                summary=summary,
            )

        if not ref_series or not cur_series:
            issues.append({
                "severity": "warning",
                "message": "missing_series",
                "reference_len": len(ref_series),
                "current_len": len(cur_series),
            })
            summary.update({
                "reference_len": len(ref_series),
                "current_len": len(cur_series),
            })
            return ReproducibilityReport(
                overall_score=0.0,
                issues=issues,
                summary=summary,
            )

        if len(ref_series) != len(cur_series):
            issues.append({
                "severity": "warning",
                "message": "length_mismatch",
                "reference_len": len(ref_series),
                "current_len": len(cur_series),
            })

        n = min(len(ref_series), len(cur_series))
        if n == 0:
            return ReproducibilityReport(
                overall_score=0.0,
                issues=issues,
                summary=summary,
            )

        # Compute normalized error (L1 relative to reference magnitude + epsilon)
        eps = 1e-12
        diffs = []
        for i in range(n):
            r = float(ref_series[i])
            c = float(cur_series[i])
            denom = max(abs(r), eps)
            diffs.append(abs(c - r) / denom)

        avg_rel_error = sum(diffs) / n if n > 0 else 1.0
        # Map error to score: 1.0 when identical, decays with error
        score = max(0.0, 1.0 - avg_rel_error)

        # Record minor deviations above tolerance
        if any(d > self.tolerance for d in diffs):
            issues.append({
                "severity": "info",
                "message": "deviations_detected",
                "avg_relative_error": avg_rel_error,
                "max_relative_error": max(diffs),
            })

        summary.update({
            "count": n,
            "avg_relative_error": avg_rel_error,
            "max_relative_error": max(diffs) if diffs else 0.0,
        })

        return ReproducibilityReport(
            overall_score=score,
            issues=issues,
            summary=summary,
        )