from __future__ import annotations
import logging
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

def create_app() -> FastAPI:
    app = FastAPI(
        title="FBA-Bench Research Toolkit API",
        description="Real-time simulation data API for research and analysis, with control.",
        version=__version__,
        lifespan=lifespan,
    )
    # Correlation id middleware (adds X-Request-ID and injects into logs)
    app.add_middleware(RequestIdMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
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

    return app

app = create_app()