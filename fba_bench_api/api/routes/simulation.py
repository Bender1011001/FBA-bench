from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional, Literal

from fastapi import APIRouter, Depends, status
from fba_bench_api.api.di import get_event_bus, get_simulation_orchestrator
from fba_bench_api.api.errors import SimulationNotFoundError, SimulationStateError
from pydantic import BaseModel, Field
from fba_bench_api.core.redis_client import get_redis
from sqlalchemy.orm import Session
from fba_bench_api.core.database import get_db_session
from fba_bench_api.core.persistence import PersistenceManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/simulation", tags=["Simulation"])

# Existing orchestrator-based endpoints remain above replaced block are now removed.
# We will keep backward-compatible start/stop/pause/resume only if wired elsewhere.
# New minimal simulation run management (in-memory) per acceptance criteria.

SimStatus = Literal["pending", "running", "stopped", "completed", "failed"]

class SimulationCreate(BaseModel):
    experiment_id: Optional[str] = Field(None, description="Optional experiment id to associate")
    metadata: Optional[dict] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {"experiment_id": None, "metadata": {"note": "ad-hoc run"}}
        }

class Simulation(BaseModel):
    id: str
    experiment_id: Optional[str] = None
    status: SimStatus
    websocket_topic: str
    created_at: datetime
    updated_at: datetime
    metadata: Optional[dict] = None

def get_pm(db: Session = Depends(get_db_session)) -> PersistenceManager:
    return PersistenceManager(db)

# Utilities
import uuid as _uuid
def _uuid4() -> str:
    return str(_uuid.uuid4())

def _now() -> datetime:
    return datetime.utcnow()

def _topic(sim_id: str) -> str:
    return f"simulation-progress:{sim_id}"

async def _publish_status(sim_id: str, status_value: str) -> None:
    # Publish to Redis if available; swallow errors to avoid test flakiness
    try:
        redis = await get_redis()
        await redis.publish(_topic(sim_id), status_value)
    except Exception as exc:
        logger.info("Redis publish skipped/unavailable: %s", exc)

# Routes

@router.post("", response_model=Simulation, status_code=status.HTTP_201_CREATED, description="Create a simulation record (pending). Returns websocket topic to subscribe for progress.")
async def create_simulation(payload: SimulationCreate, pm: PersistenceManager = Depends(get_pm)):
    sim_id = _uuid4()
    item = {
        "id": sim_id,
        "experiment_id": payload.experiment_id,
        "status": "pending",
        "websocket_topic": _topic(sim_id),
        "created_at": _now(),
        "updated_at": _now(),
        "metadata": payload.metadata or {},
    }
    created = pm.simulations().create(item)
    return Simulation(**created)

@router.post("/{simulation_id}/start", response_model=Simulation, description="Start a pending simulation")
async def start_simulation(simulation_id: str, pm: PersistenceManager = Depends(get_pm)):
    current = pm.simulations().get(simulation_id)
    if not current:
        raise SimulationNotFoundError(simulation_id)
    if current["status"] != "pending":
        raise SimulationStateError(simulation_id, expected="pending", actual=current.get("status", "unknown"))
    updates = {"status": "running", "updated_at": _now()}
    updated = pm.simulations().update(simulation_id, updates)
    await _publish_status(simulation_id, "running")
    return Simulation(**updated)  # type: ignore[arg-type]

@router.post("/{simulation_id}/stop", response_model=Simulation, description="Stop a running simulation")
async def stop_simulation(simulation_id: str, pm: PersistenceManager = Depends(get_pm)):
    current = pm.simulations().get(simulation_id)
    if not current:
        raise SimulationNotFoundError(simulation_id)
    if current["status"] != "running":
        raise SimulationStateError(simulation_id, expected="running", actual=current.get("status", "unknown"))
    updates = {"status": "stopped", "updated_at": _now()}
    updated = pm.simulations().update(simulation_id, updates)
    await _publish_status(simulation_id, "stopped")
    return Simulation(**updated)  # type: ignore[arg-type]

@router.get("/{simulation_id}", response_model=Simulation, description="Get simulation status")
async def get_simulation(simulation_id: str, pm: PersistenceManager = Depends(get_pm)):
    current = pm.simulations().get(simulation_id)
    if not current:
        raise SimulationNotFoundError(simulation_id)
    return Simulation(**current)

# ---------------------------------------------------------------------------
# Back-compat orchestrator controls (DI-backed), used by frontend UI
# These manage the SimulationOrchestrator lifecycle directly and return
# a minimal OK envelope. Database records are managed separately via CRUD.
# ---------------------------------------------------------------------------

@router.post("/start", description="Start orchestrator (compat)")
async def compat_start(
    bus = Depends(get_event_bus),
    orch = Depends(get_simulation_orchestrator),
):
    await orch.start(bus)
    return {"ok": True, "status": "starting"}

@router.post("/stop", description="Stop orchestrator (compat)")
async def compat_stop(
    orch = Depends(get_simulation_orchestrator),
):
    await orch.stop()
    return {"ok": True, "status": "stopped"}

@router.post("/pause", description="Pause orchestrator (compat)")
async def compat_pause(
    orch = Depends(get_simulation_orchestrator),
):
    await orch.pause()
    return {"ok": True, "status": "paused"}

@router.post("/resume", description="Resume orchestrator (compat)")
async def compat_resume(
    orch = Depends(get_simulation_orchestrator),
):
    await orch.resume()
    return {"ok": True, "status": "running"}

@router.post("/emergency-stop", description="Emergency stop orchestrator (compat)")
async def compat_emergency_stop(
    orch = Depends(get_simulation_orchestrator),
):
    await orch.stop()
    return {"ok": True, "status": "stopped"}