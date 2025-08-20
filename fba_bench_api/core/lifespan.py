from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from .state import dashboard_service, active_event_bus
from .state import dashboard_service as _dash_ref, active_event_bus as _bus_ref
from .persistence import config_persistence_manager
from services.dashboard_api_service import DashboardAPIService
from fba_events.bus import InMemoryEventBus as EventBus
from fba_bench_api.api.dependencies import connection_manager  # fixed import
from fba_bench_api.core.redis_client import close_redis  # Graceful Redis shutdown

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FBA-Bench API…")
    # init persistence layer cache
    config_persistence_manager.initialize_from_storage()

    # Event bus + dashboard service
    bus = EventBus()
    dash = DashboardAPIService(bus)
    # expose to global state
    globals()["_dash_ref"] = dash
    globals()["_bus_ref"] = bus

    await bus.start()
    await connection_manager.start()
    logger.info("Event bus + connection manager started")

    try:
        yield
    finally:
        try:
            await dash.stop()
        except Exception:
            pass
        try:
            await bus.stop()
        except Exception:
            pass
        try:
            await connection_manager.stop()
        except Exception:
            pass
        try:
            await close_redis()
        except Exception:
            pass
        logger.info("FBA-Bench API stopped")