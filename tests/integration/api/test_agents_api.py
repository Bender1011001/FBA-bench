from __future__ import annotations

import os
import uuid
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

# Ensure DATABASE_URL points to a temp SQLite file BEFORE importing the app/DB
TEST_DB_URL = "sqlite:///./test_api_agents.db"
os.environ["DATABASE_URL"] = TEST_DB_URL

from fba_bench_api.main import create_app  # noqa: E402
from fba_bench_api.models.base import Base  # noqa: E402
from fba_bench_api.core import database as db  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_db():
    # Recreate schema cleanly for each test module
    Base.metadata.drop_all(bind=db.engine)
    Base.metadata.create_all(bind=db.engine)
    yield
    Base.metadata.drop_all(bind=db.engine)


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_agent_crud_persists_across_requests(client: TestClient):
    # Create
    payload = {"name": "Agent One", "framework": "baseline", "config": {"temp": 0.2}}
    r = client.post("/api/v1/agents", json=payload)
    assert r.status_code == 201, r.text
    created = r.json()
    assert "id" in created and created["framework"] == "baseline"
    agent_id = created["id"]
    assert created["config"] == {"temp": 0.2}
    assert created["websocket_topic" if False else "id"]  # no-op keep mypy happy

    # New client (new session) — verify persistence
    app2 = create_app()
    with TestClient(app2) as c2:
        # Get
        r2 = c2.get(f"/api/v1/agents/{agent_id}")
        assert r2.status_code == 200, r2.text
        fetched = r2.json()
        assert fetched["id"] == agent_id
        assert fetched["name"] == "Agent One"

        # List
        r3 = c2.get("/api/v1/agents")
        assert r3.status_code == 200
        items = r3.json()
        assert any(a["id"] == agent_id for a in items)

        # Update
        upd = {"name": "Agent Uno", "config": {"temp": 0.3}}
        r4 = c2.patch(f"/api/v1/agents/{agent_id}", json=upd)
        assert r4.status_code == 200, r4.text
        updated = r4.json()
        assert updated["name"] == "Agent Uno"
        assert updated["config"] == {"temp": 0.3}
        assert updated["updated_at"] is not None

    # Fresh client — verify update persisted
    app3 = create_app()
    with TestClient(app3) as c3:
        r5 = c3.get(f"/api/v1/agents/{agent_id}")
        assert r5.status_code == 200
        persisted = r5.json()
        assert persisted["name"] == "Agent Uno"
        # Delete
        r6 = c3.delete(f"/api/v1/agents/{agent_id}")
        assert r6.status_code == 204

    # Ensure 404 after deletion
    app4 = create_app()
    with TestClient(app4) as c4:
        r7 = c4.get(f"/api/v1/agents/{agent_id}")
        assert r7.status_code == 404