import pytest
from httpx import AsyncClient, ASGITransport
from fba_bench_api.main import app, create_app as _create_app


@pytest.mark.asyncio
async def test_health_alias_parity_and_security_headers():
    transport = ASGITransport(app=app, lifespan="on")
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        r1 = await client.get("/health")
        r2 = await client.get("/api/v1/health")

        assert r1.status_code == r2.status_code
        d1 = r1.json()
        d2 = r2.json()

        # Compare stable subset keys and their values
        keys = {"status", "redis", "event_bus", "db"}
        assert keys.issubset(d1.keys())
        assert keys.issubset(d2.keys())
        assert {k: d1.get(k) for k in keys} == {k: d2.get(k) for k in keys}

        # Security headers present on /health
        headers = r1.headers
        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("X-Frame-Options") == "DENY"
        assert headers.get("Referrer-Policy") == "no-referrer"
        assert headers.get("Cross-Origin-Resource-Policy") == "same-site"


@pytest.mark.asyncio
async def test_rate_limiting_settings_but_not_health():
    # Use a fresh app instance to avoid cross-test rate-limit interference
    isolated_app = _create_app()
    transport = ASGITransport(app=isolated_app, lifespan="on")
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Exceed default API_RATE_LIMIT (100/minute) on a rate-limited endpoint
        codes = []
        for _ in range(120):
            resp = await client.get("/api/v1/settings")
            codes.append(resp.status_code)
        assert any(code == 429 for code in codes), "Expected at least one 429 for settings endpoint"

        # Health endpoints are exempt; should never return 429s even under load
        health_codes = []
        for _ in range(200):
            resp = await client.get("/health")
            health_codes.append(resp.status_code)
        assert all(code != 429 for code in health_codes), "Health endpoint must not be rate limited"