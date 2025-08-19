import os
from typing import Any, Dict

from fastapi.testclient import TestClient

from fba_bench import __version__
from fba_bench_api.main import app


def test_version_exposed():
    assert __version__ == "3.0.0"


def _assert_health_payload(payload: Dict[str, Any]):
    assert payload.get("status") == "healthy"
    assert payload.get("service") == "FBA-Bench Research Toolkit API"
    assert payload.get("version") == "3.0.0"
    assert isinstance(payload.get("timestamp"), str)
    assert "websocket_connections" in payload
    assert "uptime_s" in payload
    # Build metadata presence (may be "unknown")
    assert "git_sha" in payload
    assert "build_time" in payload
    assert "environment" in payload


def test_health_basic_ok():
    with TestClient(app) as client:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        _assert_health_payload(r.json())


def test_health_resilient_without_optional_deps(monkeypatch):
    # Ensure flags are off and URLs missing
    monkeypatch.delenv("CHECK_REDIS", raising=False)
    monkeypatch.delenv("CHECK_DB", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("FBA_BENCH_REDIS_URL", raising=False)
    monkeypatch.delenv("FBA_REDIS_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("FBA_BENCH_DB_URL", raising=False)

    with TestClient(app) as client:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        _assert_health_payload(r.json())


def test_health_resilient_with_checks_enabled_but_missing_urls(monkeypatch):
    # Enable checks but omit URLs — endpoint must not raise
    monkeypatch.setenv("CHECK_REDIS", "1")
    monkeypatch.setenv("CHECK_DB", "1")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("FBA_BENCH_REDIS_URL", raising=False)
    monkeypatch.delenv("FBA_REDIS_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("FBA_BENCH_DB_URL", raising=False)

    with TestClient(app) as client:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        _assert_health_payload(data)
        # When URLs are missing, redis_ok/db_ok keys might be absent; if present, must be bool
        if "redis_ok" in data:
            assert isinstance(data["redis_ok"], bool)
        if "db_ok" in data:
            assert isinstance(data["db_ok"], bool)