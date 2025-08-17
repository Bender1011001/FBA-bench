from __future__ import annotations
from typing import Dict, Any, Optional
import logging

# External, already in your repo
from services.dashboard_api_service import DashboardAPIService
from event_bus import EventBus

logger = logging.getLogger(__name__)

# In-memory stores (persisted to disk via persistence layer)
experiment_configs_db: Dict[str, Dict[str, Any]] = {}
simulation_configs_db: Dict[str, Dict[str, Any]] = {}
templates_db: Dict[str, Dict[str, Any]] = {}

# Long-lived singletons initialized in lifespan
dashboard_service: Optional[DashboardAPIService] = None
active_event_bus: Optional[EventBus] = None