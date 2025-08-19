from __future__ import annotations

import os
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

# Use isolated SQLite DB for this test module
TEST_DB_URL = "sqlite:///./test_api_experiments.db"
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
    r = client.post("/api/v1/agents", json={"name": "Agent X", "framework": "baseline", "config": {"t": 1}})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_experiment_crud_and_status_transitions(client: TestClient):
    agent_id = _create_agent(client)

    # Create experiment (draft)
    payload = {
        "name": "Exp1",
        "description": "desc",
        "agent_id": agent_id,
        "scenario_id": "scn-1",
        "params": {"k": 1},
    }
    r = client.post("/api/v1/experiments", json=payload)
    assert r.status_code == 201, r.text
    exp = r.json()
    exp_id = exp["id"]
    assert exp["status"] == "draft"
    assert exp["created_at"] and exp["updated_at"]

    # Fetch with fresh client to verify persistence
    app2 = create_app()
    with TestClient(app2) as c2:
        r2 = c2.get(f"/api/v1/experiments/{exp_id}")
        assert r2.status_code == 200
        got = r2.json()
        assert got["id"] == exp_id
        assert got["name"] == "Exp1"
        assert got["params"] == {"k": 1}

        # List
        r3 = c2.get("/api/v1/experiments")
        assert r3.status_code == 200
        items = r3.json()
        assert any(i["id"] == exp_id for i in items)

        # Invalid transition: draft -> completed should fail
        r_bad = c2.patch(f"/api/v1/experiments/{exp_id}", json={"status": "completed"})
        assert r_bad.status_code == 400

        # Valid transition: draft -> running
        r4 = c2.patch(f"/api/v1/experiments/{exp_id}", json={"status": "running", "params": {"k": 2}})
        assert r4.status_code == 200
        updated = r4.json()
        assert updated["status"] == "running"
        assert updated["params"] == {"k": 2}

    # Next transition: running -> completed persists
    app3 = create_app()
    with TestClient(app3) as c3:
        r5 = c3.patch(f"/api/v1/experiments/{exp_id}", json={"status": "completed"})
        assert r5.status_code == 200
        done = r5.json()
        assert done["status"] == "completed"

        # Delete experiment
        r6 = c3.delete(f"/api/v1/experiments/{exp_id}")
        assert r6.status_code == 204

    # Verify 404 after delete
    app4 = create_app()
    with TestClient(app4) as c4:
        r7 = c4.get(f"/api/v1/experiments/{exp_id}")
        assert r7.status_code == 404