import math
import pytest

from benchmarking.metrics.registry import register_metric, get_metric, list_metrics
from benchmarking.metrics import registry as regmod
from benchmarking.metrics.aggregate import aggregate_metric_values, aggregate_all

# Import built-in function metrics to ensure auto-registration took place on import
import benchmarking.metrics.technical_performance_v2  # noqa: F401
import benchmarking.metrics.accuracy_score  # noqa: F401
import benchmarking.metrics.keyword_coverage  # noqa: F401
import benchmarking.metrics.policy_compliance  # noqa: F401
import benchmarking.metrics.robustness  # noqa: F401
import benchmarking.metrics.cost_efficiency  # noqa: F401
import benchmarking.metrics.completeness  # noqa: F401
import benchmarking.metrics.custom_scriptable  # noqa: F401


def test_registry_contains_expected_keys():
    keys = set(list_metrics())
    expected = {
        "technical_performance",
        "accuracy_score",
        "keyword_coverage",
        "policy_compliance",
        "robustness",
        "cost_efficiency",
        "completeness",
        "custom_scriptable",
    }
    assert expected.issubset(keys)


def test_get_metric_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_metric("definitely_unknown_metric_key")


def test_technical_performance_basic():
    fn = get_metric("technical_performance")
    out = fn({"status": "success", "duration_ms": 1200}, {"latency_threshold_ms": 1500})
    assert out["latency_ms"] == 1200
    assert out["fast_enough"] is True

    out2 = fn({"status": "success", "duration_ms": 2200}, None)
    assert out2["latency_ms"] == 2200
    assert out2["fast_enough"] is False


def test_accuracy_exact_and_overlap():
    fn = get_metric("accuracy_score")
    # exact equality
    out = fn({"output": "Hello World"}, {"expected_output": "hello world", "mode": "exact", "case_insensitive": True})
    assert out["mode"] == "exact"
    assert out["accuracy"] == 1.0

    # overlap (Jaccard)
    out2 = fn({"output": "apple banana cherry"}, {"expected_output": "banana cherry date", "mode": "overlap"})
    # tokens: a={apple,banana,cherry}, b={banana,cherry,date} => inter=2, union=4 -> 0.5
    assert abs(out2["accuracy"] - 0.5) < 1e-9
    assert out2["mode"] == "overlap"


def test_keyword_coverage_unique_and_freq():
    fn = get_metric("keyword_coverage")
    run = {"output": {"summary": "Q3 revenue increased while Q2 was flat. Revenue guidance up."}}
    ctx = {"keywords": ["Q3", "revenue", "profit"], "field_path": "summary"}
    out = fn(run, ctx)
    assert out["keyword_total"] == 3
    assert out["keyword_hits"] >= 2
    assert 0 <= out["coverage"] <= 1

    # frequency mode
    out2 = fn(run, {**ctx, "unique_match": False})
    assert out2["keyword_hits"] >= out["keyword_hits"]
    assert 0 <= out2["coverage"] <= 1


def test_policy_compliance_counts_variants():
    fn = get_metric("policy_compliance")
    assert fn({"output": {"policy_violations": 0}}, None) == {"policy_violations": 0, "compliant": True}
    assert fn({"output": {"policy_violations": [1, 2]}}, None) == {"policy_violations": 2, "compliant": False}
    assert fn({"output": {"policy_violations": {"count": 3}}}, None) == {"policy_violations": 3, "compliant": False}


def test_robustness_modes():
    fn = get_metric("robustness")
    a = fn({"output": "Hello World"}, {"expected_signal": "  hello    world ", "mode": "exact_casefold"})
    assert a["robustness_score"] == 1.0
    b = fn({"output": "abc"}, {"expected_signal": "xyz", "mode": "normalized_overlap"})
    assert 0.0 <= b["robustness_score"] <= 1.0


def test_cost_efficiency_supported_and_na():
    fn = get_metric("cost_efficiency")
    # supported via cost field
    run = {"output": {"cost": 2.0}, "metrics": {}, "artifacts": {}}
    out = fn(run, {"score_value": 4.0})
    assert out["supported"] is True
    assert abs(out["efficiency"] - 2.0) < 1e-9

    # supported via tokens usage
    run2 = {"output": {"token_usage": {"total_tokens": 100}}}
    out2 = fn(run2, {"token_to_cost_rate": 0.5, "score_value": 50.0})
    assert out2["supported"] is True
    assert abs(out2["efficiency"] - (50.0 / (100 * 0.5))) < 1e-9

    # not supported
    out3 = fn({"output": {}}, None)
    assert out3["supported"] is False
    assert out3["reason"] == "missing_usage"


def test_completeness_top_level_and_nested():
    fn = get_metric("completeness")
    out = fn({"output": {"a": 1, "b": 2}}, {"required_fields": ["a", "c"]})
    assert out["required"] == 2
    assert out["present"] == 1
    assert abs(out["completeness"] - 0.5) < 1e-9

    out2 = fn({"output": {"a": {"x": 1}, "b": 2}}, {"required_fields": ["a.x", "a.y"], "allow_nested": True})
    assert out2["required"] == 2
    assert out2["present"] == 1


def test_custom_scriptable_safe_eval_and_blocking():
    fn = get_metric("custom_scriptable")
    # Provide run with a dependent metric value to test env mapping
    run = {"duration_ms": 1200, "status": "success", "metrics": {"accuracy_score": 0.85, "technical_performance": {"fast_enough": True}}}
    expr = "duration_ms < 1500 and (accuracy_score >= 0.8 or technical_performance__fast_enough)"
    out = fn(run, {"expression": expr})
    assert out["result"] is True

    # Block disallowed names
    out2 = fn(run, {"expression": "__import__('os').system('echo boom')"})
    assert out2["result"] is False
    assert "error" in out2


def test_aggregation_numeric_boolean_and_by_field():
    runs = [
        {"metrics": {
            "accuracy_score": 0.8,
            "technical_performance": {"latency_ms": 1000, "fast_enough": True}
        }},
        {"metrics": {
            "accuracy_score": 1.0,
            "technical_performance": {"latency_ms": 1500, "fast_enough": True}
        }},
        {"metrics": {
            "accuracy_score": 0.4,
            "technical_performance": {"latency_ms": 2500, "fast_enough": False}
        }},
    ]
    acc_agg = aggregate_metric_values(runs, "accuracy_score")
    assert "numeric" in acc_agg and abs(acc_agg["numeric"]["mean"] - (0.8 + 1.0 + 0.4) / 3.0) < 1e-9

    tech_agg = aggregate_metric_values(runs, "technical_performance")
    assert "by_field" in tech_agg and "latency_ms" in tech_agg["by_field"]
    assert "boolean" in tech_agg and "success_rate" in tech_agg["boolean"]

    all_agg = aggregate_all(runs, ["accuracy_score", "technical_performance", "nonexistent"])
    assert "nonexistent" in all_agg and isinstance(all_agg["nonexistent"].get("missing", 0), int)