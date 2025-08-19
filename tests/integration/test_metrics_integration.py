import asyncio
import pytest

from benchmarking.core.engine import Engine, EngineConfig, ScenarioSpec, RunnerSpec, summarize_scenario
from benchmarking.scenarios.registry import scenario_registry

# Ensure function-style metrics are registered on import
import benchmarking.metrics.technical_performance_v2  # noqa: F401
import benchmarking.metrics.accuracy_score  # noqa: F401
import benchmarking.metrics.keyword_coverage  # noqa: F401


@pytest.mark.integration
@pytest.mark.asyncio
async def test_engine_with_function_metrics_and_aggregation(monkeypatch):
    # Use built-in scenario key to exercise registry path; we will patch execution to be deterministic.
    sc_key = "research_summarization"

    # Sanity: scenario should be registered by its module
    # If not, import will register (module itself does scenario_registry.register on import)
    try:
        scenario_registry.get(sc_key)
    except Exception:
        pytest.importorskip("benchmarking.scenarios.research_summarization")

    # Patch the internal execution helper to return a deterministic output
    async def fake_execute_scenario(_scenario_target, _runner, _payload):
        # Provide a summary string that matches expected keywords
        return {"summary": "Q3 revenue increased significantly this quarter."}

    import benchmarking.core.engine as engmod
    monkeypatch.setattr(engmod, "_execute_scenario", fake_execute_scenario)

    # Also patch runner creation to avoid external dependencies
    async def fake_create_runner(key, config):
        # The scenario adapter expects a callable "runner"; engine passes it into scenario.run
        async def runner_callable(request: dict):
            # echo back request for completeness
            return {"ok": True, "request": request}
        return runner_callable

    monkeypatch.setattr(engmod, "create_runner", fake_create_runner)

    # Engine configuration: include metric keys and pass metric context via ScenarioSpec.params
    # - accuracy_score: provide expected_output (exact mode)
    # - keyword_coverage: provide keywords
    # - technical_performance: threshold default is fine
    params = {
        "expected_output": "Q3 revenue increased significantly this quarter.",
        "mode": "exact",
        "keywords": ["Q3", "revenue"],
    }
    cfg = EngineConfig(
        scenarios=[ScenarioSpec(key=sc_key, params=params, repetitions=1, seeds=[123], timeout_seconds=5)],
        runners=[RunnerSpec(key="diy", config={"agent_id": "test-agent"})],
        metrics=["technical_performance", "keyword_coverage", "accuracy_score"],
        validators=[],
        parallelism=1,
        retries=0,
    )

    engine = Engine(cfg)
    report = await engine.run()

    # Verify scenario report exists
    assert report.scenario_reports and len(report.scenario_reports) == 1
    sr = report.scenario_reports[0]
    assert sr.scenario_key == sc_key
    assert sr.runs and len(sr.runs) == 1

    run = sr.runs[0]
    # Metrics should include our three keys with dict outputs for the function metrics
    assert "technical_performance" in run.metrics
    assert "keyword_coverage" in run.metrics
    assert "accuracy_score" in run.metrics

    tech = run.metrics["technical_performance"]
    cov = run.metrics["keyword_coverage"]
    acc = run.metrics["accuracy_score"]

    assert isinstance(tech, dict) and "latency_ms" in tech and "fast_enough" in tech
    assert isinstance(cov, dict) and cov["keyword_total"] == 2 and 0.0 <= cov["coverage"] <= 1.0
    assert isinstance(acc, dict) and acc["mode"] in ("exact", "overlap") and 0.0 <= acc["accuracy"] <= 1.0

    # Aggregation helper should compute means for numeric metrics and counts for categorical
    aggs = summarize_scenario(sr)
    assert "metrics" in aggs and "mean" in aggs["metrics"]
    # No numeric scalar metric in our three (accuracy is numeric) verify it is present
    means = aggs["metrics"]["mean"]
    if "accuracy_score" in means:
        assert 0.0 <= means["accuracy_score"] <= 1.0