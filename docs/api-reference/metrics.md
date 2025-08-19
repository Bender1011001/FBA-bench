# Metrics: Registry, Interfaces, and Built-ins

This section documents the metrics system, including the function-style registry and the built-in metrics. Metrics are discoverable by key, accept a RunResult-like dict input, and return normalized dict outputs. All built-ins are deterministic and use Pydantic v2 schemas for input/context.

Registry API
- Register: [`python.def register_metric(key: str, fn: Callable[[dict, dict|None], dict]) -> None`](benchmarking/metrics/registry.py:1)
- Lookup: [`python.def get_metric(key: str) -> Callable`](benchmarking/metrics/registry.py:1)
- List: [`python.def list_metrics() -> list[str]`](benchmarking/metrics/registry.py:1)
- Convenience in package: [`python.def metric_keys() -> List[str]`](benchmarking/metrics/__init__.py:1)

Engine integration
- The Engine prefers function-style metrics by key. When computing metrics for a run, it calls each metric function with:
  - run: RunResult-like dict (see [`python.class RunResult`](benchmarking/core/engine.py:793))
  - context: dict containing scenario metadata (e.g., {"scenario_key": ..., "params": {...}})
- Implementation detail: the engine passes partial metric results from earlier metrics in config order to later ones via run["metrics"] (see [`python.def _apply_metrics`](benchmarking/core/engine.py:1)).

Aggregation helpers
- Single metric aggregation: [`python.def aggregate_metric_values(runs: list[dict], metric_key: str) -> dict`](benchmarking/metrics/aggregate.py:1)
- Multiple: [`python.def aggregate_all(runs: list[dict], metric_keys: list[str]) -> dict`](benchmarking/metrics/aggregate.py:1)

Built-in metrics

1) technical_performance
- Module: [`benchmarking/metrics/technical_performance_v2.py`](benchmarking/metrics/technical_performance_v2.py:1)
- Key: "technical_performance"
- Input: run.duration_ms, run.status; Context: {"latency_threshold_ms": 2000} default
- Output example:
  {"latency_ms": 1234, "fast_enough": true}

2) accuracy_score
- Module: [`benchmarking/metrics/accuracy_score.py`](benchmarking/metrics/accuracy_score.py:1)
- Key: "accuracy_score"
- Modes:
  - exact: deep equality for structured outputs, normalized string equality otherwise
  - overlap: token-set Jaccard overlap ratio
- Context example:
  {"expected_output": "hello world", "mode":"exact", "field_path": "data.summary", "case_insensitive": true}
- Output example:
  {"accuracy": 1.0, "mode": "exact"}

3) keyword_coverage
- Module: [`benchmarking/metrics/keyword_coverage.py`](benchmarking/metrics/keyword_coverage.py:1)
- Key: "keyword_coverage"
- Input: run.output["summary"] (or context.field_path)
- Context: {"keywords": ["Q3","revenue"], "unique_match": true}
- Output example:
  {"keyword_hits": 2, "keyword_total": 2, "coverage": 1.0}

4) policy_compliance
- Module: [`benchmarking/metrics/policy_compliance.py`](benchmarking/metrics/policy_compliance.py:1)
- Key: "policy_compliance"
- Input: run.output["policy_violations"] as int|list|{"count": int}
- Output example:
  {"policy_violations": 0, "compliant": true}

5) robustness
- Module: [`benchmarking/metrics/robustness.py`](benchmarking/metrics/robustness.py:1)
- Key: "robustness"
- Context: {"expected_signal": "...", "mode":"normalized_overlap"|"exact_casefold"}
- Output example:
  {"robustness_score": 0.92}

6) cost_efficiency
- Module: [`benchmarking/metrics/cost_efficiency.py`](benchmarking/metrics/cost_efficiency.py:1)
- Key: "cost_efficiency"
- Behavior: compute score/cost ratio if cost or token usage present; else not supported with reason.
- Context: {"score_value": 4.0} or {"score_field_path":"metrics.score"}, token conversion rate via {"token_to_cost_rate": 0.5}
- Output example (supported):
  {"supported": true, "efficiency": 2.0, "reason": null}
- Output example (not supported):
  {"supported": false, "efficiency": null, "reason": "missing_usage"}

7) completeness
- Module: [`benchmarking/metrics/completeness.py`](benchmarking/metrics/completeness.py:1)
- Key: "completeness"
- Context: {"required_fields": ["accepted","revenue","fulfilled_rate"], "allow_nested": false}
- Output example:
  {"present": 2, "required": 3, "completeness": 0.6667}

8) custom_scriptable
- Module: [`benchmarking/metrics/custom_scriptable.py`](benchmarking/metrics/custom_scriptable.py:1)
- Key: "custom_scriptable"
- Behavior: evaluate a minimal safe boolean expression, referencing run fields and prior metric values (engine passes partials incrementally).
- Context: {"expression": "duration_ms < 1500 and accuracy_score >= 0.8"}
- Output example:
  {"result": true, "expression": "duration_ms < 1500 and accuracy_score >= 0.8"}

Safety and determinism
- Exceptions in metric functions are caught by the engine; metrics return partials with error reason and do not fail the pipeline.
- If randomness is needed, derive it from provided seeds in the run payload to ensure determinism.

Examples (Engine)
```python
from benchmarking.core.engine import EngineConfig, RunnerSpec, ScenarioSpec, run_benchmark

config = EngineConfig(
    scenarios=[ScenarioSpec(key="research_summarization", params={"focus_keywords":["Q3","revenue"]}, repetitions=1, seeds=[101])],
    runners=[RunnerSpec(key="diy", config={"agent_id": "baseline-1"})],
    metrics=["technical_performance","keyword_coverage","accuracy_score"],
)
report = run_benchmark(config)