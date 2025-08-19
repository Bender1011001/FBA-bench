from __future__ import annotations

import os
from fastapi.testclient import TestClient
import pytest

# Use isolated SQLite DB for this test module
TEST_DB_URL = "sqlite:///./test_api_simulations.db"
os.environ["DATABASE_URL"] = TEST_DB_URL

from fba_bench_api.main import create_app  # noqa: E402
from fba_bench_api.models.base import Base  # noqa: E402
from fba_bench_api.core import database as db  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_db():
    Base.metadata.drop_all(bind=db.engine)
    Base.metadata.create_all(bind=db.engine)
    yield
    Base.metadata.drop_all(bind=db.engine)


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def _create_agent(client: TestClient) -> str:
    r = client.post("/api/v1/agents", json={"name": "Runner", "framework": "baseline", "config": {"t": 1}})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _create_experiment(client: TestClient, agent_id: str) -> str:
    payload = {
        "name": "ExpSim",
        "description": "for sim",
        "agent_id": agent_id,
        "scenario_id": "scenario-x",
        "params": {"seed": 123},
    }
    r = client.post("/api/v1/experiments", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_simulation_create_start_stop_persist(client: TestClient):
    # prepare experiment (optional link)
    agent_id = _create_agent(client)
    exp_id = _create_experiment(client, agent_id)

    # create
    r = client.post("/api/v1/simulation", json={"experiment_id": exp_id, "metadata": {"note": "adhoc"}})
    assert r.status_code == 201, r.text
    sim = r.json()
    sim_id = sim["id"]
    assert sim["status"] == "pending"
    assert sim["websocket_topic"] == f"simulation-progress:{sim_id}"
    assert sim["metadata"] == {"note": "adhoc"}

    # fresh session — start
    app2 = create_app()
    with TestClient(app2) as c2:
        r2 = c2.post(f"/api/v1/simulation/{sim_id}/start")
        assert r2.status_code == 200, r2.text
        started = r2.json()
        assert started["status"] == "running"
        assert started["websocket_topic"] == f"simulation-progress:{sim_id}"

    # fresh session — stop
    app3 = create_app()
    with TestClient(app3) as c3:
        r3 = c3.post(f"/api/v1/simulation/{sim_id}/stop")
        assert r3.status_code == 200, r3.text
        stopped = r3.json()
        assert stopped["status"] == "stopped"

        # get reflects persisted state
        r4 = c3.get(f"/api/v1/simulation/{sim_id}")
        assert r4.status_code == 200
        got = r4.json()
        assert got["status"] == "stopped"
        assert got["websocket_topic"] == f"simulation-progress:{sim_id}"