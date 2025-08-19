import asyncio
import types
import pytest

from pydantic import ValidationError

# Import new engine API and models
from benchmarking.core.engine import (
    Engine,
    EngineConfig,
    ScenarioSpec,
    RunnerSpec,
    RunResult,
    ScenarioReport,
    EngineReport,
    summarize_scenario,
    compute_totals,
)

from benchmarking.scenarios.registry import scenario_registry


@pytest.fixture(autouse=True)
def clear_scenarios_registry():
    # Ensure a clean registry for each test
    scenario_registry.clear()
    yield
    scenario_registry.clear()


def _register_scenario_fn(key: str, fn):
    scenario_registry.register(key, fn)  # the registry stores callables/classes


@pytest.mark.unit
def test_config_validation_parallelism_and_empty_scenarios():
    # Bad parallelism
    with pytest.raises(ValidationError):
        EngineConfig(
            scenarios=[ScenarioSpec(key="x")],
            runners=[RunnerSpec(key="diy", config={"agent_id": "a"})],
            parallelism=0,
        )
    # Empty scenarios fails
    with pytest.raises(ValidationError):
        EngineConfig(
            scenarios=[],
            runners=[RunnerSpec(key="diy", config={"agent_id": "a"})],
        )
    # Empty runners fails
    with pytest.raises(ValidationError):
        EngineConfig(
            scenarios=[ScenarioSpec(key="x")],
            runners=[],
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_timeout_handling(monkeypatch):
    # Slow scenario that sleeps beyond timeout
    async def slow_scenario(runner, payload):
        await asyncio.sleep(0.5)
        return {"ok": True}

    key = "unit_slow"
    _register_scenario_fn(key, slow_scenario)

    cfg = EngineConfig(
        scenarios=[ScenarioSpec(key=key, timeout_seconds=1)],  # set at scenario level; we will tighten via config below
        runners=[RunnerSpec(key="diy", config={"agent_id": "dummy"})],
        metrics=[],
        validators=[],
        parallelism=1,
    )

    # Force timeout at 0.05s by overriding scenario spec in place
    cfg.scenarios[0].timeout_seconds = 0.05

    # Avoid instantiating a real runner by monkeypatching create_runner to a dummy object
    from agent_runners import registry as runner_registry

    class DummyRunner:
        agent_id = "dummy"

    monkeypatch.setattr(runner_registry, "create_runner", lambda k, c: DummyRunner())

    eng = Engine(cfg)
    report = await eng.run()
    assert isinstance(report, EngineReport)
    assert report.scenario_reports and report.scenario_reports[0].runs
    statuses = [r.status for r in report.scenario_reports[0].runs]
    assert statuses == ["timeout"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retries_flaky_scenario(monkeypatch):
    # Flaky scenario: fails once then succeeds
    async def flaky_scenario(runner, payload):
        cnt = getattr(flaky_scenario, "_count", 0)
        setattr(flaky_scenario, "_count", cnt + 1)
        if cnt == 0:
            raise RuntimeError("transient")
        return {"ok": True, "value": payload.get("seed")}

    key = "unit_flaky"
    _register_scenario_fn(key, flaky_scenario)

    cfg = EngineConfig(
        scenarios=[ScenarioSpec(key=key, repetitions=1, seeds=[7], timeout_seconds=2)],
        runners=[RunnerSpec(key="diy", config={"agent_id": "dummy"})],
        metrics=[],
        validators=[],
        parallelism=1,
        retries=1,  # allow one retry
    )

    from agent_runners import registry as runner_registry

    class DummyRunner:
        agent_id = "dummy"

    monkeypatch.setattr(runner_registry, "create_runner", lambda k, c: DummyRunner())

    eng = Engine(cfg)
    report = await eng.run()
    run = report.scenario_reports[0].runs[0]
    assert run.status == "success"
    assert run.output and run.output.get("ok") is True
    assert run.seed == 7


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metrics_application_mean_aggregates(monkeypatch):
    # Scenario returns deterministic output
    async def scenario_ok(runner, payload):
        return {"data": "x"}

    key = "unit_metrics"
    _register_scenario_fn(key, scenario_ok)

    # Dummy metric instance that always returns 1.0
    class DummyMetric:
        def calculate(self, data):
            return 1.0

    # Patch MetricRegistry.create_metric to return our dummy for 'dummy_metric'
    from benchmarking.core.engine import MetricRegistry

    def create_metric(self, name, config=None):
        if name == "dummy_metric":
            return DummyMetric()
        return None

    monkeypatch.setattr(MetricRegistry, "create_metric", create_metric, raising=True)

    from agent_runners import registry as runner_registry

    class DummyRunner:
        agent_id = "dummy"

    monkeypatch.setattr(runner_registry, "create_runner", lambda k, c: DummyRunner())

    cfg = EngineConfig(
        scenarios=[ScenarioSpec(key=key, repetitions=2, seeds=[1, 2])],
        runners=[RunnerSpec(key="diy", config={"agent_id": "dummy"})],
        metrics=["dummy_metric"],
        validators=[],
        parallelism=2,
    )

    eng = Engine(cfg)
    report = await eng.run()
    sr = report.scenario_reports[0]
    # All runs should carry dummy_metric=1.0
    assert all(r.metrics.get("dummy_metric") == 1.0 for r in sr.runs)
    # Aggregated mean equals 1.0
    assert pytest.approx(sr.aggregates["metrics"]["mean"]["dummy_metric"], rel=0, abs=1e-9) == 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validator_injection(monkeypatch):
    # Simple scenario returns output
    async def scenario_ok(runner, payload):
        return {"ok": True}

    key = "unit_validator"
    _register_scenario_fn(key, scenario_ok)

    # Dummy validator returning a structure with to_dict
    class DummyValidationResult:
        def __init__(self):
            self.is_valid = False
            self.errors = [{"code": "E1", "message": "issue"}]

        def to_dict(self):
            return {"is_valid": self.is_valid, "errors": self.errors}

    class DummyValidator:
        def __init__(self, cfg=None):
            pass

        def validate(self, data, **kwargs):
            return DummyValidationResult()

    # Patch ValidatorRegistry.create_validator
    from benchmarking.core.engine import ValidatorRegistry

    def create_validator(self, name, config=None):
        if name == "dummy_validator":
            return DummyValidator()
        return None

    monkeypatch.setattr(ValidatorRegistry, "create_validator", create_validator, raising=True)

    from agent_runners import registry as runner_registry

    class DummyRunner:
        agent_id = "dummy"

    monkeypatch.setattr(runner_registry, "create_runner", lambda k, c: DummyRunner())

    cfg = EngineConfig(
        scenarios=[ScenarioSpec(key=key)],
        runners=[RunnerSpec(key="diy", config={"agent_id": "dummy"})],
        metrics=[],
        validators=["dummy_validator"],
        parallelism=1,
    )

    eng = Engine(cfg)
    report = await eng.run()
    sr = report.scenario_reports[0]
    assert "validations" in sr.aggregates
    assert isinstance(sr.aggregates["validations"], list)
    assert sr.aggregates["validations"][0]["name"] == "dummy_validator"