import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from fba_bench_api.main import create_app as _create_app


@pytest.mark.asyncio
async def test_experiment_stop_and_results_flow():
    app = _create_app()
    transport = ASGITransport(app=app, lifespan="on")
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1) Create draft experiment
        create_payload = {
            "name": "IntegrationFlow",
            "description": "Flow test",
            "agent_id": str(uuid.uuid4()),
            "scenario_id": "scenario-abc",
            "params": {"k": 1},
        }
        r_create = await client.post("/api/v1/experiments", json=create_payload)
        assert r_create.status_code == 201
        created = r_create.json()
        exp_id = created["id"]
        assert created["status"] == "draft"

        # 2) Transition to running
        r_patch = await client.patch(f"/api/v1/experiments/{exp_id}", json={"status": "running"})
        assert r_patch.status_code == 200
        running = r_patch.json()
        assert running["status"] == "running"

        # 3) Stop -> completed
        r_stop = await client.post(f"/api/v1/experiments/{exp_id}/stop")
        assert r_stop.status_code == 200
        stopped = r_stop.json()
        assert stopped["status"] == "completed"

        # 4) Results shape for completed experiment
        r_results = await client.get(f"/api/v1/experiments/{exp_id}/results")
        assert r_results.status_code == 200
        results = r_results.json()
        assert results["experiment_id"] == exp_id
        assert isinstance(results.get("results", []), list)
        assert isinstance(results.get("summary", {}), dict)
        assert results["summary"].get("status") == "completed"


@pytest.mark.asyncio
async def test_stop_invalid_when_not_running_returns_400():
    app = _create_app()
    transport = ASGITransport(app=app, lifespan="on")
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Create draft
        payload = {
            "name": "StopInvalid",
            "agent_id": str(uuid.uuid4()),
            "scenario_id": "scenario-xyz",
            "params": {},
        }
        r_create = await client.post("/api/v1/experiments", json=payload)
        assert r_create.status_code == 201
        exp_id = r_create.json()["id"]
        assert r_create.json()["status"] == "draft"

        # Attempt to stop while not running -> 400
        r_stop = await client.post(f"/api/v1/experiments/{exp_id}/stop")
        assert r_stop.status_code == 400