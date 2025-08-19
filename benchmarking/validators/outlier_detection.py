from __future__ import annotations

"""
Outlier detection validator.

Key: "outlier_detection"

Flags runs whose duration_ms are outliers using Median Absolute Deviation (MAD):
- An outlier satisfies |x - median| > k * MAD, with default k=5.
- MAD is scaled by 1.4826 to be consistent with stddev for normal distributions.

Context:
- {"k": float} optional, default 5.0

Output:
- {"issues":[...], "summary":{"median":..., "mad":..., "k":..., "outliers":[indices...]}}
"""

from typing import Any, Dict, List, Optional, Tuple
from .registry import register_validator
from .types import Issue, ValidationOutput, normalize_output


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    n = len(v)
    mid = n // 2
    if n % 2 == 1:
        return float(v[mid])
    return float((v[mid - 1] + v[mid]) / 2)


def _mad(values: List[float], med: float) -> float:
    if not values:
        return 0.0
    abs_dev = [abs(x - med) for x in values]
    mad = _median(abs_dev)
    # scale to be comparable to stddev under normality
    return float(1.4826 * mad)


def outlier_detection_validate(report: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = context or {}
    k: float = float(ctx.get("k", 5.0))
    out = ValidationOutput()

    runs = report.get("runs") or []
    durations: List[Tuple[int, float]] = []
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            continue
        d = run.get("duration_ms")
        if isinstance(d, int) or isinstance(d, float):
            durations.append((idx, float(d)))

    vals = [d for _, d in durations]
    med = _median(vals)
    mad = _mad(vals, med)
    outliers: List[int] = []

    if mad == 0.0:
        # If all durations are equal, no outliers. Record info.
        out.summary.details.update({"median": med, "mad": 0.0, "k": k, "outliers": []})
        return normalize_output(out)

    for idx, d in durations:
        if abs(d - med) > k * mad:
            outliers.append(idx)
            out.add_issue(
                Issue(
                    id="duration_outlier",
                    severity="warning",
                    message=f"Run[{idx}] duration_ms={d} is an outlier (median={med}, MAD={mad}, k={k})",
                    path=["runs", str(idx), "duration_ms"],
                )
            )

    out.summary.details.update({"median": med, "mad": mad, "k": k, "outliers": outliers})
    return normalize_output(out)


register_validator("outlier_detection", outlier_detection_validate)