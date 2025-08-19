import asyncio
import pytest

from benchmarking.core.engine import (
    Engine,
    EngineConfig,
    ScenarioSpec,
    RunnerSpec,
    EngineReport,
)
from benchmarking.scenarios.registry import scenario_registry


@pytest.fixture(autouse=True)
def clear_scenarios_registry():
    scenario_registry.clear()
    yield
    scenario_registry.clear()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_engine_integration_end_to_end(monkeypatch):
    # Minimal deterministic scenario: returns the seed for verification
    async def simple_scenario(runner, payload):
        # pretend to use runner.decide etc.; just echo deterministic output
        return {"value": payload.get("seed")}

    scen_key = "integration.simple_suite"
    scenario_registry.register(scen_key, simple_scenario)

    # Provide a trivial runner to avoid framework deps
    from agent_runners import registry as runner_registry

    class TestRunner:
        agent_id = "test"

    monkeypatch.setattr(runner_registry, "create_runner", lambda k, c: TestRunner())

    # Dummy metric returns a constant numeric value (enables aggregation assertions)
    class TestMetric:
        def calculate(self, data):
            # data is dict; return a float suitable for aggregation
            return 1.0

    from benchmarking.core.engine import MetricRegistry

    def create_metric(self, name, config=None):
        if name == "dummy_metric":
            return TestMetric()
        return None

    monkeypatch.setattr(MetricRegistry, "create_metric", create_metric, raising=True)

    # Dummy validator records a simple issue
    class DummyValidationResult:
        def __init__(self):
            self.is_valid = True
            self.errors = []
            self.metadata = {"note": "ok"}

        def to_dict(self):
            return {"is_valid": self.is_valid, "errors": self.errors, "metadata": self.metadata}

    class TestValidator:
        def __init__(self, cfg=None):
            pass

        def validate(self, data, **kwargs):
            # Return a result object convertible to dict
            return DummyValidationResult()

    from benchmarking.core.engine import ValidatorRegistry

    def create_validator(self, name, config=None):
        if name == "dummy_validator":
            return TestValidator()
        return None

    monkeypatch.setattr(ValidatorRegistry, "create_validator", create_validator, raising=True)

    cfg = EngineConfig(
        scenarios=[
            ScenarioSpec(key=scen_key, repetitions=2, seeds=[11, 22], timeout_seconds=3),
        ],
        runners=[RunnerSpec(key="diy", config={"agent_id": "test"})],
        metrics=["dummy_metric"],
        validators=["dummy_validator"],
        parallelism=2,
        retries=0,
    )

    eng = Engine(cfg)
    report = await eng.run()

    # Structure assertions
    assert isinstance(report, EngineReport)
    assert report.started_at > 0
    assert report.finished_at >= report.started_at
    assert isinstance(report.config_digest, str) and len(report.config_digest) > 8
    assert len(report.scenario_reports) == 1

    sr = report.scenario_reports[0]
    assert sr.scenario_key == scen_key
    assert len(sr.runs) == 2 * 1  # seeds x runners
    statuses = {r.status for r in sr.runs}
    assert statuses == {"success"}

    # Per-run output and metrics
    seeds_seen = sorted([r.seed for r in sr.runs])
    assert seeds_seen == [11, 22]
    for r in sr.runs:
        assert r.output is not None
        assert r.metrics.get("dummy_metric") == 1.0

    # Aggregates present
    ag = sr.aggregates
    assert ag["runs"] == 2
    assert ag["pass_count"] == 2
    assert ag["fail_count"] == 0
    assert ag["metrics"]["mean"]["dummy_metric"] == 1.0
    # Validator attached
    assert "validations" in ag
    assert ag["validations"][0]["name"] == "dummy_validator"

    # Totals
    totals = report.totals
    assert totals["runs"] == 2
    assert totals["success"] == 2
    assert totals["failed"] == 0
    assert totals["metrics"]["mean"]["dummy_metric"] == 1.0