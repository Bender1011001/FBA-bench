# Validators: Registry, Interfaces, and Built-ins

This section documents the validators system, including the function-style registry and the built-in validators. Validators are discoverable by key, accept a ScenarioReport-like dict input (or EngineReport context when noted), and return normalized dict outputs. All built-ins are deterministic and use safe defaults.

Registry API
- Register: [`python.def register_validator(key: str, fn: Callable[[dict, dict|None], dict]) -> None`](benchmarking/validators/registry.py:1)
- Lookup: [`python.def get_validator(key: str) -> Callable`](benchmarking/validators/registry.py:1)
- List: [`python.def list_validators() -> list[str]`](benchmarking/validators/registry.py:1)

Interface for each validator function
- [`python.def validate(report: dict, context: dict|None=None) -> dict`](benchmarking/validators/types.py:1)
  - report is a dict representation of [`python.class ScenarioReport`](benchmarking/core/engine.py:823).model_dump()
  - context may include scenario_key, params, engine totals, config digest, expected_seeds, and validator-specific options
- Normalized return structure:
  - {"issues": list[dict], "summary": dict}
  - issue: {"id": str, "severity": "info"|"warning"|"error", "message": str, "path": list[str]|None}

Engine integration
- The Engine applies configured validators by key after runs and scenario aggregation, collecting results under aggregates["validations"] for each scenario.

Built-in validators

1) structural_consistency
- Module: [`benchmarking/validators/structural_consistency.py`](benchmarking/validators/structural_consistency.py:1)
- Key: "structural_consistency"
- Input: ScenarioReport-like dict with "runs" containing RunResult-like dicts (see [`python.class RunResult`](benchmarking/core/engine.py:793))
- Behavior: checks presence and types of required fields (scenario_key, runner_key, status, duration_ms, metrics; output: dict when success) and returns issues for missing/invalid fields. Summarizes counts by status.

2) determinism_check
- Module: [`benchmarking/validators/determinism_check.py`](benchmarking/validators/determinism_check.py:1)
- Key: "determinism_check"
- Behavior: for runs that share (runner_key, seed), compares output values either for exact equality or numeric near-equality within tolerance.
- Context:
  - {"tolerance": float, "fields": list[str] | None}
  - fields are dot paths within run.output; if omitted, compares common top-level fields.

3) reproducibility_metadata
- Module: [`benchmarking/validators/reproducibility_metadata.py`](benchmarking/validators/reproducibility_metadata.py:1)
- Key: "reproducibility_metadata"
- Behavior: verifies that runs include seeds and that they align with EngineConfig; warns on missing seeds; errors on unexpected seeds; optionally checks per-run config_digest against EngineReport digest if present in context.
- Context: {"expected_seeds": list[int] | None, "config_digest": str | None}

4) schema_adherence
- Module: [`benchmarking/validators/schema_adherence.py`](benchmarking/validators/schema_adherence.py:1)
- Key: "schema_adherence"
- Behavior: validates run.output against a simple contract (required nested fields and their types).
- Context:
  - {"contract": {"required": {"field": "type_name" | ["type_name", ...] | {"path": "...", "type": "type_name"}}}}
  - Types: "str","int","float","bool","list","dict","number"

5) outlier_detection
- Module: [`benchmarking/validators/outlier_detection.py`](benchmarking/validators/outlier_detection.py:1)
- Key: "outlier_detection"
- Behavior: flags runs whose duration_ms are outliers using robust stats (median and MAD): |x - median| > k * MAD; default k=5. Returns warning issues and summary stats.
- Context: {"k": float}

6) fairness_balance
- Module: [`benchmarking/validators/fairness_balance.py`](benchmarking/validators/fairness_balance.py:1)
- Key: "fairness_balance"
- Behavior: groups runs by a categorical (runner_key, seed, or nested field) and checks that a metric (from metrics or output) does not differ more than a threshold across group means. Reports imbalance as error otherwise info.
- Context:
  - {"group": "runner_key"|"...", "metric_path": "metrics.accuracy"|"...", "threshold": float, "min_group_size": int}

Usage examples

EngineConfig example (attach validators by key):
- See [`python.class EngineConfig(BaseModel)`](benchmarking/core/engine.py:760)

Simple call:
```python
from benchmarking.core.engine import EngineConfig, RunnerSpec, ScenarioSpec, run_benchmark

config = EngineConfig(
    scenarios=[ScenarioSpec(key="multiturn_tool_use", params={"steps": 1}, repetitions=1, seeds=[7])],
    runners=[RunnerSpec(key="diy", config={"agent_id": "baseline-1"})],
    metrics=[],
    validators=["structural_consistency","reproducibility_metadata","outlier_detection"],
)
report = run_benchmark(config)
# validation results per scenario:
for s in report.scenario_reports:
    print(s.aggregates.get("validations"))
```

Schema notes
- Validators are deterministic and do not raise; they return safe issue lists with severities and structured summaries. If randomness is necessary, it must derive from provided seeds and be documented.
