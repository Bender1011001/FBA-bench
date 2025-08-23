import pytest
from httpx import AsyncClient, ASGITransport
from fba_bench_api.main import app, create_app as _create_app


@pytest.mark.asyncio
async def test_settings_get_and_post_update_reflected():
    # Use isolated app instance to avoid cross-test state
    isolated_app = _create_app()
    transport = ASGITransport(app=isolated_app, lifespan="on")
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Initial GET should return defaults with required sections
        r_get = await client.get("/api/v1/settings")
        assert r_get.status_code == 200
        data = r_get.json()
        assert "settings" in data and isinstance(data["settings"], dict)
        settings = data["settings"]

        # Validate presence of top-level groups
        assert "apiKeys" in settings and isinstance(settings["apiKeys"], dict)
        assert "defaults" in settings and isinstance(settings["defaults"], dict)
        assert "ui" in settings and isinstance(settings["ui"], dict)

        # Validate representative keys within each group (shape only)
        api_keys = settings["apiKeys"]
        for k in ["openai", "anthropic", "google", "cohere", "openrouter"]:
            assert k in api_keys

        defaults = settings["defaults"]
        for k in ["defaultLLM", "defaultScenario", "defaultAgent", "defaultMetrics", "autoSave", "notifications"]:
            assert k in defaults

        ui = settings["ui"]
        for k in ["theme", "language", "timezone"]:
            assert k in ui

        # POST a small update (raw settings payload allowed)
        updated = {
            **settings,
            "defaults": {**defaults, "defaultLLM": "gpt-4o"},
            "ui": {**ui, "theme": "dark"},
        }
        r_post = await client.post("/api/v1/settings", json=updated)
        assert r_post.status_code == 204

        # Subsequent GET reflects changes
        r_get2 = await client.get("/api/v1/settings")
        assert r_get2.status_code == 200
        s2 = r_get2.json()["settings"]
        assert s2["defaults"]["defaultLLM"] == "gpt-4o"
        assert s2["ui"]["theme"] == "dark"


@pytest.mark.asyncio
async def test_metrics_endpoints_shapes():
    # Use shared app instance; metrics endpoints are resilient with stable shapes
    transport = ASGITransport(app=app, lifespan="on")
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # /api/metrics/audit
        ra = await client.get("/api/metrics/audit")
        assert ra.status_code == 200
        audit = ra.json()
        for k in ["processed_transactions", "total_violations", "current_position", "tolerance_cents"]:
            assert k in audit
        assert isinstance(audit["current_position"], dict)

        # /api/metrics/ledger
        rl = await client.get("/api/metrics/ledger")
        assert rl.status_code == 200
        ledger = rl.json()
        for k in ["total_assets", "total_liabilities", "accounting_identity_valid", "identity_difference"]:
            assert k in ledger

        # /api/metrics/bsr
        rb = await client.get("/api/metrics/bsr")
        assert rb.status_code == 200
        bsr = rb.json()
        assert "products" in bsr and isinstance(bsr["products"], list)
        # competitor_count may be absent; if present should be an int
        if "competitor_count" in bsr and bsr["competitor_count"] is not None:
            assert isinstance(bsr["competitor_count"], int)

        # /api/metrics/fees
        rf = await client.get("/api/metrics/fees")
        assert rf.status_code == 200
        fees = rf.json()
        assert isinstance(fees, dict)