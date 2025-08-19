from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Aggregation utilities for function-style metrics
# - Input runs are dicts compatible with benchmarking.core.engine.RunResult.model_dump()
# - Metric values are expected under run["metrics"][metric_key]
# - Handle numeric, boolean, dict-of-numerics, and mixed missing values safely


def _flatten_numeric_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(_flatten_numeric_dict(v, key, sep))
    return out


def _numeric_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "stddev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _boolean_stats(values: List[bool]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "true_count": 0, "false_count": 0, "success_rate": 0.0}
    true_count = sum(1 for v in values if bool(v))
    count = len(values)
    return {
        "count": count,
        "true_count": true_count,
        "false_count": count - true_count,
        "success_rate": float(true_count / count),
    }


def _collect_metric_values(runs: List[Dict[str, Any]], metric_key: str) -> Tuple[List[float], List[bool], List[Dict[str, float]]]:
    nums: List[float] = []
    bools: List[bool] = []
    dicts: List[Dict[str, float]] = []

    for r in runs:
        metrics = (r or {}).get("metrics") or {}
        if metric_key not in metrics:
            continue
        val = metrics.get(metric_key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
            nums.append(float(val))
        elif isinstance(val, bool):
            bools.append(bool(val))
        elif isinstance(val, dict):
            flat = _flatten_numeric_dict(val)
            if flat:
                dicts.append(flat)
        # else: ignore non-supported types for aggregation
    return nums, bools, dicts


def aggregate_metric_values(runs: List[Dict[str, Any]], metric_key: str) -> Dict[str, Any]:
    """
    Compute aggregate statistics for a specific metric across runs.

    Returns a dict including available sections:
    - "numeric": {count, mean, median, stddev, min, max}
    - "boolean": {count, true_count, false_count, success_rate}
    - "by_field": {subfield: numeric_stats} for dict-valued metrics (flattened)
    - "missing": number of runs without the metric_key present
    """
    total_runs = len(runs or [])
    nums, bools, dicts = _collect_metric_values(runs or [], metric_key)

    out: Dict[str, Any] = {}
    missing = total_runs - (len(nums) + len(bools) + len(dicts))
    out["missing"] = int(max(0, missing))

    if nums:
        out["numeric"] = _numeric_stats(nums)
    if bools:
        out["boolean"] = _boolean_stats(bools)
    if dicts:
        fields: Dict[str, List[float]] = {}
        for d in dicts:
            for k, v in d.items():
                fields.setdefault(k, []).append(v)
        out["by_field"] = {k: _numeric_stats(vs) for k, vs in fields.items()}

    return out


def aggregate_all(runs: List[Dict[str, Any]], metric_keys: List[str]) -> Dict[str, Any]:
    """
    Aggregate multiple metrics across runs.

    Returns:
      {
        metric_key: aggregate_metric_values(...),
        ...
      }
    """
    result: Dict[str, Any] = {}
    for key in (metric_keys or []):
        try:
            result[key] = aggregate_metric_values(runs or [], key)
        except Exception as e:
            result[key] = {"error": str(e)}
    return result