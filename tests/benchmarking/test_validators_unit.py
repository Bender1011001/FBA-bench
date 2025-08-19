import math
import pytest

from benchmarking.validators.registry import register_validator, get_validator, list_validators
from benchmarking.validators import structural_consistency  # noqa: F401 (auto-registers)
from benchmarking.validators import determinism_check  # noqa: F401
from benchmarking.validators import reproducibility_metadata  # noqa: F401
from benchmarking.validators import schema_adherence  # noqa: F401
from benchmarking.validators import outlier_detection  # noqa: F401
from benchmarking.validators import fairness_balance  # noqa: F401


def test_registry_register_and_get_and_list():
    # sanity existing built-ins
    keys = list_validators()
    assert "structural_consistency" in keys
    assert "determinism_check" in keys

    # add a temporary validator
    def dummy(report: dict, context: dict | None = None) -> dict:
        return {"issues": [], "summary": {"count": 0}}

    register_validator("unit_dummy", dummy)
    assert "unit_dummy" in list_validators()
    fn = get_validator("unit_dummy")
    out = fn({}, {})
    assert isinstance(out, dict)
    assert "issues" in out and "summary" in out

    with pytest.raises(KeyError):
        get_validator("unknown_key_xyz")


def test_structural_consistency_malformed_runs():
    fn = get_validator("structural_consistency")
    report = {
        "scenario_key": "example",
        "runs": [
            {"runner_key": "a", "status": "success", "duration_ms": 10, "metrics": {}, "output": {"x": 1}},  # missing scenario_key
            {"scenario_key": "example", "runner_key": "b", "status": "error", "duration_ms": -5, "metrics": {}, "output": {"y": 1}},
            "not_a_dict",
        ],
        "aggregates": {},
    }
    out = fn(report, {})
    issues = out["issues"]
    # expect missing_field, negative_duration, unexpected_output_on_failure, invalid_run_type
    assert any(i["id"] == "missing_field" for i in issues)
    assert any(i["id"] == "negative_duration" for i in issues)
    assert any(i["id"] == "unexpected_output_on_failure" for i in issues)
    assert any(i["id"] == "invalid_run_type" for i in issues)


def test_determinism_check_detects_inconsistency_and_tolerance():
    fn = get_validator("determinism_check")
    # two runs same runner+seed, inconsistent numeric field "value"
    report = {
        "scenario_key": "s",
        "runs": [
            {"scenario_key": "s", "runner_key": "r", "seed": 7, "status": "success", "duration_ms": 1, "metrics": {}, "output": {"value": 1.0}},
            {"scenario_key": "s", "runner_key": "r", "seed": 7, "status": "success", "duration_ms": 2, "metrics": {}, "output": {"value": 3.0}},
        ],
        "aggregates": {},
    }
    out = fn(report, {"tolerance": 0.0, "fields": ["value"]})
    assert any(i["id"] == "determinism_mismatch" for i in out["issues"])

    # with tolerance large enough, mismatch should not be flagged
    out2 = fn(report, {"tolerance": 5.0, "fields": ["value"]})
    assert not any(i["id"] == "determinism_mismatch" for i in out2["issues"])
    # should include an info determinism_ok
    assert any(i["id"] == "determinism_ok" for i in out2["issues"])


def test_schema_adherence_minimal_contract_missing_and_type():
    fn = get_validator("schema_adherence")
    report = {
        "scenario_key": "s",
        "runs": [
            {"scenario_key": "s", "runner_key": "r", "seed": 1, "status": "success", "duration_ms": 1, "metrics": {}, "output": {"a": 1, "b": "x"}},
            {"scenario_key": "s", "runner_key": "r", "seed": 2, "status": "success", "duration_ms": 1, "metrics": {}, "output": {"a": "not_int"}},
        ],
        "aggregates": {},
    }
    contract = {"required": {"a": "int", "b": "str"}}
    out = fn(report, {"contract": contract})
    issues = out["issues"]
    # run[0] ok; run[1] missing 'b' and a wrong type
    assert any(i["id"] == "schema_missing_field" for i in issues)
    assert any(i["id"] == "schema_type_mismatch" for i in issues)


def test_outlier_detection_mad_flags_outlier_configurable_k():
    fn = get_validator("outlier_detection")
    # durations: mostly around 100, one at 1000
    runs = []
    for d in [98, 101, 100, 99, 102, 97, 1000]:
        runs.append({"scenario_key": "s", "runner_key": "r", "seed": d, "status": "success", "duration_ms": d, "metrics": {}, "output": {"ok": True}})
    report = {"scenario_key": "s", "runs": runs, "aggregates": {}}
    # k small to ensure detection
    out = fn(report, {"k": 3.0})
    assert any(i["id"] == "duration_outlier" for i in out["issues"])
    # median should be around 100 and outliers should include index of 1000 (last idx=6)
    assert 6 in out["summary"]["details"]["outliers"]


def test_fairness_balance_detects_imbalance():
    fn = get_validator("fairness_balance")
    report = {
        "scenario_key": "s",
        "runs": [
            {"scenario_key": "s", "runner_key": "A", "seed": 1, "status": "success", "duration_ms": 1, "metrics": {"accuracy": 0.95}, "output": {}},
            {"scenario_key": "s", "runner_key": "A", "seed": 2, "status": "success", "duration_ms": 1, "metrics": {"accuracy": 0.96}, "output": {}},
            {"scenario_key": "s", "runner_key": "B", "seed": 3, "status": "success", "duration_ms": 1, "metrics": {"accuracy": 0.70}, "output": {}},
            {"scenario_key": "s", "runner_key": "B", "seed": 4, "status": "success", "duration_ms": 1, "metrics": {"accuracy": 0.71}, "output": {}},
        ],
        "aggregates": {},
    }
    # large difference between groups
    out = fn(report, {"group": "runner_key", "metric_path": "metrics.accuracy", "threshold": 0.1, "min_group_size": 1})
    assert any(i["id"] == "fairness_imbalance" for i in out["issues"])

    # increase threshold to avoid error
    out2 = fn(report, {"group": "runner_key", "metric_path": "metrics.accuracy", "threshold": 0.5})
    assert any(i["id"] == "fairness_within_threshold" for i in out2["issues"])


def test_reproducibility_metadata_warning_and_error_paths():
    fn = get_validator("reproducibility_metadata")
    report = {
        "scenario_key": "s",
        "runs": [
            {"scenario_key": "s", "runner_key": "r", "seed": None, "status": "success", "duration_ms": 1, "metrics": {}, "output": {}},
            {"scenario_key": "s", "runner_key": "r", "seed": 101, "status": "success", "duration_ms": 1, "metrics": {}, "output": {}},
            {"scenario_key": "s", "runner_key": "r", "seed": 202, "status": "success", "duration_ms": 1, "metrics": {}, "output": {}},
        ],
        "aggregates": {},
    }
    # expected seeds do not include 202
    out = fn(report, {"expected_seeds": [101, 303], "config_digest": "abc"})
    issues = out["issues"]
    assert any(i["id"] == "missing_seed" for i in issues)
    assert any(i["id"] == "unexpected_seed" for i in issues)
    # informational about per-run digest not present
    assert any(i["id"] == "per_run_digest_missing" for i in issues)