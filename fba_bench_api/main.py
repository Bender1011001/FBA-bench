from __future__ import annotations
import logging
import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fba_bench_api.core.lifespan import lifespan
from fba_bench.core.logging import setup_logging, RequestIdMiddleware
from fba_bench_api.api.exception_handlers import add_exception_handlers
from fba_bench import __version__

from fba_bench_api.api.routes import root as root_routes
from fba_bench_api.api.routes import config as config_routes
from fba_bench_api.api.routes import simulation as sim_routes
from fba_bench_api.api.routes import realtime as realtime_routes
from fba_bench_api.api.routes import agents as agents_routes
from fba_bench_api.api.routes import experiments as exp_routes

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

AUTH_JWT_ALG = os.getenv("AUTH_JWT_ALG", "RS256")
AUTH_JWT_PUBLIC_KEY = os.getenv("AUTH_JWT_PUBLIC_KEY")
AUTH_JWT_ISSUER = os.getenv("AUTH_JWT_ISSUER")
AUTH_JWT_AUDIENCE = os.getenv("AUTH_JWT_AUDIENCE")
AUTH_JWT_CLOCK_SKEW = int(os.getenv("AUTH_JWT_CLOCK_SKEW", "60") or "60")

UNPROTECTED_PATHS = {
    "/health",
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
    app = FastAPI(
        title="FBA-Bench Research Toolkit API",
        description="Real-time simulation data API for research and analysis, with control.",
        version=__version__,
        lifespan=lifespan,
    )
    # Correlation id middleware (adds X-Request-ID and injects into logs)
    app.add_middleware(RequestIdMiddleware)
    # JWT middleware (protects all but health/docs)
    app.add_middleware(JWTAuthMiddleware)

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

    @app.options("/{path:path}")
    async def options_handler(path: str):
        return {"message": "OK"}

    # Health endpoint (unauthenticated) with Redis/DB/EventBus checks
    @app.get("/health")
    async def health():
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

        # Event bus
        try:
            from fba_bench_api.core.state import active_event_bus
            bus = active_event_bus()
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

    return app


app = create_app()