from __future__ import annotations
import logging
import os
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from fba_bench_api.core.lifespan import lifespan
from fba_bench.core.logging import setup_logging, RequestIdMiddleware
from fba_bench_api.api.exception_handlers import add_exception_handlers
from fba_bench import __version__
from fba_bench_api.core.container import AppContainer

from fba_bench_api.api.routes import root as root_routes
from fba_bench_api.api.routes import config as config_routes
from fba_bench_api.api.routes import simulation as sim_routes
from fba_bench_api.api.routes import realtime as realtime_routes
from fba_bench_api.api.routes import agents as agents_routes
from fba_bench_api.api.routes import experiments as exp_routes
from fba_bench_api.api.routes import settings as settings_routes
from fba_bench_api.api.routes import metrics as metrics_routes

# Centralized, idempotent logging initialization
setup_logging()
logger = logging.getLogger("fba_bench_api")

# JWT verification (RS256) middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
import time
import jwt  # PyJWT
from jwt import PyJWKClient

# Rate limiting (slowapi)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

def _is_protected_env() -> bool:
    """
    Treat environment as 'protected' if any of ENVIRONMENT, APP_ENV, or ENV is one of:
    ['production', 'prod', 'staging']. Defaults to non-protected otherwise.
    """
    for key in ("ENVIRONMENT", "APP_ENV", "ENV"):
        val = os.getenv(key)
        if val and val.strip().lower() in ("production", "prod", "staging"):
            return True
    return False


def env_bool(name: str, default: bool) -> bool:
    """
    Parse a boolean environment variable with robust truthy/falsy handling.
    Accepts: 1/0, true/false, yes/no, on/off (case-insensitive).
    Falls back to default if unset or unparsable.
    Also checks a backward-compatible 'FBA_<NAME>' alias if primary is unset.
    """
    raw = os.getenv(name)
    if raw is None:
        raw = os.getenv(f"FBA_{name}")  # maintain compatibility with prior FBA_AUTH_ENABLED, etc.
    if raw is None:
        return bool(default)
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return bool(default)


AUTH_JWT_ALG = os.getenv("AUTH_JWT_ALG", "RS256")
AUTH_JWT_PUBLIC_KEY = os.getenv("AUTH_JWT_PUBLIC_KEY")
AUTH_JWT_ISSUER = os.getenv("AUTH_JWT_ISSUER")
AUTH_JWT_AUDIENCE = os.getenv("AUTH_JWT_AUDIENCE")
AUTH_JWT_CLOCK_SKEW = int(os.getenv("AUTH_JWT_CLOCK_SKEW", "60") or "60")
# Enable/disable auth via env. Defaults to disabled for tests/dev unless explicitly enabled.
AUTH_ENABLED = (os.getenv("AUTH_ENABLED") or os.getenv("FBA_AUTH_ENABLED") or "false").strip().lower() in ("1", "true", "yes", "on")
# Explicit test bypass (default true to keep integration tests unauthenticated unless opted-in)
AUTH_TEST_BYPASS = (os.getenv("AUTH_TEST_BYPASS", "true").strip().lower() in ("1", "true", "yes", "on"))
# Gate docs in production behind auth if requested
AUTH_PROTECT_DOCS = (os.getenv("AUTH_PROTECT_DOCS", "false").strip().lower() in ("1", "true", "yes", "on"))
# Default API rate limit (configurable)
API_RATE_LIMIT = (os.getenv("API_RATE_LIMIT", "100/minute").strip() or "100/minute")

UNPROTECTED_PATHS = {
    "/health",
    "/api/v1/health",
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
}

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Allow health and docs unauthenticated; protect the rest
        if path in UNPROTECTED_PATHS or path.startswith("/ws"):  # ws authenticated separately if needed
            return await call_next(request)
        # Global bypass for tests/dev unless explicitly disabled
        if not AUTH_ENABLED or AUTH_TEST_BYPASS:
            return await call_next(request)

        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        if not auth or not auth.lower().startswith("bearer "):
            return JSONResponse({"detail": "Missing bearer token"}, status_code=401)

        token = auth.split(" ", 1)[1].strip()
        try:
            options = {
                "require": ["exp", "iat"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": AUTH_JWT_AUDIENCE is not None,
                "verify_iss": AUTH_JWT_ISSUER is not None,
            }
            # Prefer static PEM public key via env for simplicity and security
            if AUTH_JWT_PUBLIC_KEY:
                payload = jwt.decode(
                    token,
                    AUTH_JWT_PUBLIC_KEY,
                    algorithms=[AUTH_JWT_ALG],
                    audience=AUTH_JWT_AUDIENCE,
                    issuer=AUTH_JWT_ISSUER,
                    leeway=AUTH_JWT_CLOCK_SKEW,
                    options=options,
                )
            else:
                # If a JWKS URL were provided (not in current spec), add support here.
                return JSONResponse({"detail": "JWT public key not configured"}, status_code=500)

            # Attach identity to request.state for handlers
            request.state.user = {
                "sub": payload.get("sub"),
                "scope": payload.get("scope"),
                "roles": payload.get("roles"),
            }
        except Exception as e:
            logger.warning("JWT verification failed: %s", e)
            return JSONResponse({"detail": "Invalid token"}, status_code=401)

        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Basic hardening headers
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
        return response


def _get_cors_allowed_origins() -> List[str]:
    # Locked-down production origins as confirmed
    defaults = ["https://app.fba.example.com", "https://console.fba.example.com"]
    raw = os.getenv("FBA_CORS_ALLOW_ORIGINS")
    if not raw:
        return defaults
    # Comma-separated list in env
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or defaults


def create_app() -> FastAPI:
    # Resolve environment and security defaults
    protected = _is_protected_env()

    # Recompute auth flags from env with protected-aware defaults
    global AUTH_ENABLED, AUTH_TEST_BYPASS, AUTH_JWT_PUBLIC_KEY
    AUTH_ENABLED = env_bool("AUTH_ENABLED", default=protected)
    AUTH_TEST_BYPASS = env_bool("AUTH_TEST_BYPASS", default=(not protected))
    AUTH_JWT_PUBLIC_KEY = (os.getenv("AUTH_JWT_PUBLIC_KEY") or "").strip() or None

    # Fail fast on insecure/misconfigured setups
    if AUTH_ENABLED and not AUTH_JWT_PUBLIC_KEY:
        raise RuntimeError("AUTH_ENABLED=true but AUTH_JWT_PUBLIC_KEY is not set. Provide an RSA public key (PEM).")

    # CORS must be explicit in protected environments
    raw_cors = os.getenv("FBA_CORS_ALLOW_ORIGINS")
    if protected:
        if not raw_cors or raw_cors.strip() in ("", "*"):
            raise RuntimeError("FBA_CORS_ALLOW_ORIGINS must be a comma-separated allow-list (not '*') in staging/production.")

    # Compute docs gating after resolving AUTH flags
    protect_docs = AUTH_PROTECT_DOCS and AUTH_ENABLED
    default_docs = "/docs"
    default_redoc = "/redoc"
    default_openapi = "/openapi.json"

    docs_url = None if protect_docs else default_docs
    redoc_url = None if protect_docs else default_redoc
    openapi_url = None if protect_docs else default_openapi

    # Construct FastAPI with gated docs
    app = FastAPI(
        title="FBA-Bench Research Toolkit API",
        description="Real-time simulation data API for research and analysis, with control.",
        version=__version__,
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )

    # Adjust UNPROTECTED_PATHS based on docs availability (always keep health aliases)
    if docs_url is None:
        UNPROTECTED_PATHS.discard(default_docs)
    else:
        UNPROTECTED_PATHS.add(docs_url)
    if redoc_url is None:
        UNPROTECTED_PATHS.discard(default_redoc)
    else:
        UNPROTECTED_PATHS.add(redoc_url)
    if openapi_url is None:
        UNPROTECTED_PATHS.discard(default_openapi)
    else:
        UNPROTECTED_PATHS.add(openapi_url)

    # Dependency Injection container
    app.state.container = AppContainer()

    # Correlation id middleware (adds X-Request-ID and injects into logs)
    app.add_middleware(RequestIdMiddleware)

    # Security headers middleware (basic hardening)
    app.add_middleware(SecurityHeadersMiddleware)

    # Rate Limiting (global default with health exemptions)
    limiter = Limiter(key_func=get_remote_address, default_limits=[API_RATE_LIMIT])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # JWT middleware (protects all but health/docs) - only if explicitly enabled and key provided
    if AUTH_ENABLED and AUTH_JWT_PUBLIC_KEY:
        app.add_middleware(JWTAuthMiddleware)
        logger.info("JWTAuthMiddleware enabled")
    else:
        logger.info("JWTAuthMiddleware disabled (AUTH_ENABLED=%s, PUBLIC_KEY=%s)", AUTH_ENABLED, bool(AUTH_JWT_PUBLIC_KEY))

    allow_origins = _get_cors_allowed_origins()
    # Hardened CORS: explicit origins, no credentials, limited methods/headers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
        expose_headers=["Content-Type"],
        max_age=600,
    )

    # Exception handlers
    add_exception_handlers(app)

    # Routers
    app.include_router(root_routes.router)
    app.include_router(config_routes.router)
    app.include_router(sim_routes.router)
    app.include_router(realtime_routes.router)
    app.include_router(agents_routes.router)
    app.include_router(exp_routes.router)
    app.include_router(settings_routes.router)
    app.include_router(metrics_routes.router)

    @app.options("/{path:path}")
    async def options_handler(path: str):
        return {"message": "OK"}

    # Health endpoint (unauthenticated) with Redis/DB/EventBus checks (exempt from rate limiting)
    @app.get("/health")
    @limiter.exempt
    async def health(request: Request):
        from starlette.responses import JSONResponse as _JSON
        status: dict = {"status": "ok", "redis": "unknown", "event_bus": "unknown", "db": "unknown"}

        # Redis
        try:
            from fba_bench_api.core.redis_client import get_redis
            r = await get_redis()
            pong = await r.ping()
            status["redis"] = "ok" if pong else "down"
        except Exception as e:
            status["redis"] = f"down:{type(e).__name__}"

        # Event bus via DI container
        try:
            container = request.app.state.container  # type: ignore[attr-defined]
            bus = container.event_bus() if container else None
            status["event_bus"] = "ok" if bus is not None else "down"
        except Exception as e:
            status["event_bus"] = f"down:{type(e).__name__}"

        # Database
        try:
            from sqlalchemy import create_engine, text
            db_url = os.getenv("DATABASE_URL", "sqlite:///./fba_bench.db")
            eng = create_engine(db_url, future=True)
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            status["db"] = "ok"
        except Exception as e:
            status["db"] = f"down:{type(e).__name__}"

        http_status = 200 if all(v == "ok" for k, v in status.items() if k != "status") else 503
        status["status"] = "ok" if http_status == 200 else "degraded"
        return _JSON(status, status_code=http_status)

    # Alias route matching frontend path; identical behavior/auth as /health (also exempt)
    @app.get("/api/v1/health")
    @limiter.exempt
    async def health_v1(request: Request):
        # Delegate to the primary health handler to ensure identical payload and status
        return await health(request)

    return app


app = create_app()