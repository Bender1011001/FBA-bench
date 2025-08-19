import os
import pytest
from fastapi.testclient import TestClient

from fba_bench_api.main import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_config_get_merged_env_and_overrides(client: TestClient, monkeypatch):
    # Ensure env-derived defaults
    monkeypatch.setenv("ENABLE_OBSERVABILITY", "true")
    monkeypatch.setenv("PROFILE", "testing")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://env-endpoint:4318")

    resp = client.get("/api/v1/config")
    assert resp.status_code == 200
    data = resp.json()
    # env-derived present
    assert data["enable_observability"] is True
    assert data["profile"] == "testing"
    assert data["telemetry_endpoint"] == "http://env-endpoint:4318"


def test_config_patch_overrides_and_merge(client: TestClient, monkeypatch):
    # Set env; then patch should override merged view
    monkeypatch.setenv("ENABLE_OBSERVABILITY", "false")
    monkeypatch.setenv("PROFILE", "development")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://env-endpoint:4318")

    # Patch runtime overrides
    payload = {"enable_observability": True, "profile": "runtime", "telemetry_endpoint": "http://rt:4318"}
    resp = client.patch("/api/v1/config", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["enable_observability"] is True
    assert data["profile"] == "runtime"
    assert data["telemetry_endpoint"] == "http://rt:4318"

    # Subsequent GET reflects overrides merged over env
    resp = client.get("/api/v1/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enable_observability"] is True
    assert data["profile"] == "runtime"
    assert data["telemetry_endpoint"] == "http://rt:4318"