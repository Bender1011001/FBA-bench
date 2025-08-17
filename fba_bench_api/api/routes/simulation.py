from __future__ import annotations
import uuid
from fastapi import APIRouter, HTTPException
from fba_bench_api.core.state import active_event_bus
from fba_bench_api.core.state import simulation_configs_db
from fba_bench_api.models.simulation import (
    SimulationStartRequest, SimulationControlResponse, SimulationStatusResponse
)
from simulation_orchestrator import SimulationOrchestrator, SimulationConfig
from api.dependencies import simulation_manager

router = APIRouter(prefix="/api/v1/simulation", tags=["Simulation"])

@router.post("/start", response_model=SimulationControlResponse)
async def start_simulation(start_request: SimulationStartRequest):
    config_id = start_request.config_id
    sim_id = start_request.simulation_id or str(uuid.uuid4())
    if sim_id in simulation_manager.get_all_simulation_ids():
        raise HTTPException(409, f"Simulation '{sim_id}' already running")
    config_data = simulation_configs_db.get(config_id)
    if not config_data:
        raise HTTPException(404, f"Configuration '{config_id}' not found")

    sim_config = SimulationConfig(
        tick_interval_seconds=config_data.get("tick_interval_seconds", 1.0),
        max_ticks=config_data.get("max_ticks"),
        start_time=config_data.get("start_time"),
        time_acceleration=config_data.get("time_acceleration", 1.0),
        seed=config_data.get("seed"),
    )
    orch = SimulationOrchestrator(sim_config)
    await simulation_manager.add_orchestrator(sim_id, orch)
    try:
        await orch.start(active_event_bus)
        return SimulationControlResponse(
            success=True,
            message=f"Simulation '{sim_id}' started with config '{config_id}'",
            simulation_id=sim_id,
            status=orch.get_status(),
        )
    except Exception as e:
        await simulation_manager.remove_orchestrator(sim_id)
        raise HTTPException(500, f"Failed to start simulation: {e}")

@router.post("/stop/{simulation_id}", response_model=SimulationControlResponse)
async def stop_simulation(simulation_id: str):
    orch = await simulation_manager.get_orchestrator(simulation_id)
    if not orch.is_running:
        raise HTTPException(400, f"Simulation '{simulation_id}' is not running")
    await orch.stop()
    await simulation_manager.remove_orchestrator(simulation_id)
    return SimulationControlResponse(success=True, message=f"Simulation '{simulation_id}' stopped.", simulation_id=simulation_id)

@router.post("/pause/{simulation_id}", response_model=SimulationControlResponse)
async def pause_simulation(simulation_id: str):
    orch = await simulation_manager.get_orchestrator(simulation_id)
    if not orch.is_running:
        raise HTTPException(400, f"Simulation '{simulation_id}' is not running")
    if orch.is_paused:
        raise HTTPException(400, f"Simulation '{simulation_id}' is already paused")
    await orch.pause()
    return SimulationControlResponse(success=True, message=f"Simulation '{simulation_id}' paused.", simulation_id=simulation_id, status=orch.get_status())

@router.post("/resume/{simulation_id}", response_model=SimulationControlResponse)
async def resume_simulation(simulation_id: str):
    orch = await simulation_manager.get_orchestrator(simulation_id)
    if not orch.is_running:
        raise HTTPException(400, f"Simulation '{simulation_id}' is not running")
    if not orch.is_paused:
        raise HTTPException(400, f"Simulation '{simulation_id}' is not paused")
    await orch.resume()
    return SimulationControlResponse(success=True, message=f"Simulation '{simulation_id}' resumed.", simulation_id=simulation_id, status=orch.get_status())

@router.get("/status/{simulation_id}", response_model=SimulationStatusResponse)
async def get_simulation_status(simulation_id: str):
    orch = await simulation_manager.get_orchestrator(simulation_id)
    return SimulationStatusResponse(simulation_id=simulation_id, **orch.get_status())