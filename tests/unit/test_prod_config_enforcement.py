import pytest
from httpx import AsyncClient, ASGITransport

# Import inside tests to ensure create_app reads current env via os.getenv at call time
from fba_bench_api.main import create_app as _create_app


def _dummy_pem() -> str:
    return "-----BEGIN PUBLIC KEY-----\nX\n-----END PUBLIC KEY-----"


def _clear_env(monkeypatch):
    # Auth-related
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.delenv("FBA_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("AUTH_TEST_BYPASS", raising=False)
    monkeypatch.delenv("AUTH_JWT_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("AUTH_JWT_ALG", raising=False)
    monkeypatch.delenv("AUTH_JWT_ISSUER", raising=False)
    monkeypatch.delenv("AUTH_JWT_AUDIENCE", raising=False)
    # CORS-related
    monkeypatch.delenv("FBA_CORS_ALLOW_ORIGINS", raising=False)
    # Environment detection
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("ENV", raising=False)


def test_prod_requires_public_key_when_auth_enabled(monkeypatch):
    _clear_env(monkeypatch)
    # Protected env
    monkeypatch.setenv("APP_ENV", "production")
    # Ensure CORS is valid to isolate JWT failure
    monkeypatch.setenv("FBA_CORS_ALLOW_ORIGINS", "https://app.example.com")
    # Ensure AUTH flags are unset so defaults apply (enabled in protected)
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.delenv("AUTH_JWT_PUBLIC_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc:
        _create_app()
    msg = str(exc.value)
    assert "AUTH_ENABLED=true" in msg
    assert "AUTH_JWT_PUBLIC_KEY" in msg


def test_staging_rejects_wildcard_cors(monkeypatch):
    _clear_env(monkeypatch)
    # Protected env
    monkeypatch.setenv("APP_ENV", "staging")
    # Provide a valid key so CORS error is surfaced
    monkeypatch.setenv("AUTH_JWT_PUBLIC_KEY", _dummy_pem())
    # Defaults should enable auth in protected env
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    # Invalid wildcard in protected env
    monkeypatch.setenv("FBA_CORS_ALLOW_ORIGINS", "*")

    with pytest.raises(RuntimeError) as exc:
        _create_app()
    msg = str(exc.value)
    assert "FBA_CORS_ALLOW_ORIGINS" in msg
    assert "not '*'" in msg


@pytest.mark.asyncio
async def test_dev_defaults_allow_and_app_starts_without_key(monkeypatch):
    _clear_env(monkeypatch)
    # Dev env (non-protected): unset all env markers
    # Defaults should disable auth and allow wildcard CORS
    monkeypatch.setenv("FBA_CORS_ALLOW_ORIGINS", "*")

    app = _create_app()
    transport = ASGITransport(app=app, lifespan="on")
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        r = await client.get("/api/v1/health")
        assert r.status_code in (200, 503)


def test_staging_with_valid_cors_and_key_starts(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("AUTH_JWT_PUBLIC_KEY", _dummy_pem())
    monkeypatch.setenv(
        "FBA_CORS_ALLOW_ORIGINS",
        "https://app.staging.example.com,https://admin.staging.example.com",
    )

    # Should not raise
    _create_app()