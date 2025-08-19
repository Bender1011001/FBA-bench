import asyncio
import pytest

from benchmarking.core.engine import Engine, EngineConfig, ScenarioSpec, RunnerSpec
import benchmarking.core.engine as engmod
from agent_runners import registry as runner_registry


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validators_integration_with_engine(monkeypatch):
    # Patch runner creation to a dummy object (scenario won't use it due to patched executor)
    class DummyRunner:
        agent_id = "dummy"

    monkeypatch.setattr(runner_registry, "create_runner", lambda key, cfg: DummyRunner())

    # Patch scenario execution to deterministic outputs and inject one slow run for outlier detection
    async def fake_execute_scenario(target, runner, payload):
        # Deterministic output
        seed = payload.get("seed")
        if seed == 999:
            await asyncio.sleep(0.2)  # induce a slow run => clear duration outlier
        return {"results": [{"seed": seed, "ok": True}]}

    monkeypatch.setattr(engmod, "_execute_scenario", fake_execute_scenario)

    # Use an existing scenario key so _resolve_scenario succeeds ("multiturn_tool_use" is documented/built-in)
    sc_key = "multiturn_tool_use"
    cfg = EngineConfig(
        scenarios=[ScenarioSpec(key=sc_key, params={"steps": 1}, repetitions=1, seeds=[101, 202, 999], timeout_seconds=5)],
        runners=[RunnerSpec(key="diy", config={"agent_id": "test-agent"})],
        metrics=[],
        validators=["structural_consistency", "reproducibility_metadata", "outlier_detection"],
        parallelism=1,
        retries=0,
    )

    eng = Engine(cfg)
    report = await eng.run()

    assert report is not None
    assert len(report.scenario_reports) == 1
    srep = report.scenario_reports[0]
    assert srep.scenario_key == sc_key

    # Aggregates should contain validations
    validations = srep.aggregates.get("validations")
    assert isinstance(validations, list) and len(validations) == 3

    # Map by validator key
    vmap = {v.get("validator") or v.get("name"): v for v in validations}

    # structural_consistency present
    assert "structural_consistency" in vmap
    assert "summary" in vmap["structural_consistency"]

    # reproducibility_metadata present, should mention observed_seeds
    assert "reproducibility_metadata" in vmap
    repro = vmap["reproducibility_metadata"]
    assert "summary" in repro and "details" in repro["summary"]
    observed = repro["summary"]["details"].get("observed_seeds")
    assert observed is not None and all(isinstance(s, int) for s in observed)

    # outlier_detection should flag the intentionally slow run
    assert "outlier_detection" in vmap
    outdet = vmap["outlier_detection"]
    # there should be at least one issue with id "duration_outlier" or summary contains non-empty outliers
    ids = [i.get("id") for i in outdet.get("issues", [])]
    summary_outliers = (outdet.get("summary") or {}).get("details", {}).get("outliers", [])
    assert ("duration_outlier" in ids) or (isinstance(summary_outliers, list) and len(summary_outliers) >= 1)