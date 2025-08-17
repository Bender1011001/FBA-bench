from __future__ import annotations
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from fba_bench_api.core.persistence import config_persistence_manager
from fba_bench_api.models.simulation import (
    SimulationConfigCreate, SimulationConfigUpdate, SimulationConfigResponse
)
from fba_bench_api.models.experiments import (
    ConfigTemplateSave, ConfigTemplateResponse,
    BenchmarkConfigRequest, BenchmarkConfigResponse,
)

router = APIRouter(prefix="/api/v1/config", tags=["Configuration"])

@router.post("/simulation", response_model=SimulationConfigResponse, status_code=status.HTTP_201_CREATED)
async def create_simulation_configuration(config_data: SimulationConfigCreate):
    config_id = str(uuid.uuid4())
    now = datetime.now()
    try:
        entry = config_data.model_dump(exclude_unset=True)
        entry.update({
            "config_id": config_id,
            "created_at": now,
            "updated_at": now,
            "base_parameters": entry.get("base_parameters", {}) or {}
        })
        if not config_persistence_manager.save_simulation_config(config_id, entry):
            raise HTTPException(500, "Failed to persist configuration")
        return SimulationConfigResponse(**entry)
    except ValidationError as e:
        raise HTTPException(422, detail=e.errors())

@router.get("/simulation/{config_id}", response_model=SimulationConfigResponse)
async def get_simulation_configuration(config_id: str):
    config = config_persistence_manager.load_simulation_config(config_id)
    if not config:
        raise HTTPException(404, "Configuration not found")
    return SimulationConfigResponse(**config)

@router.put("/simulation/{config_id}", response_model=SimulationConfigResponse)
async def update_simulation_configuration(config_id: str, config_data: SimulationConfigUpdate):
    current = config_persistence_manager.load_simulation_config(config_id)
    if not current:
        raise HTTPException(404, "Configuration not found")
    update = config_data.model_dump(exclude_unset=True)
    if "base_parameters" in update and isinstance(update["base_parameters"], dict):
        current.setdefault("base_parameters", {}).update(update.pop("base_parameters"))
    current.update(update)
    current["updated_at"] = datetime.now()
    if not config_persistence_manager.save_simulation_config(config_id, current):
        raise HTTPException(500, "Failed to save updated config")
    return SimulationConfigResponse(**current)

@router.delete("/simulation/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_simulation_configuration(config_id: str):
    if not config_persistence_manager.load_simulation_config(config_id):
        raise HTTPException(404, "Configuration not found")
    if not config_persistence_manager.delete_simulation_config(config_id):
        raise HTTPException(500, "Failed to delete configuration")
    return {"message": "deleted"}  # 204 handled by FastAPI

@router.get("/templates", response_model=list[ConfigTemplateResponse])
async def list_configuration_templates():
    items: list[ConfigTemplateResponse] = []
    for name in config_persistence_manager.list_templates():
        data = config_persistence_manager.load_template(name)
        if data:
            items.append(ConfigTemplateResponse(
                template_name=name,
                description=data.get("description"),
                config_data=data.get("config_data", {}),
                created_at=data.get("created_at"),
            ))
    return items

@router.post("/templates", response_model=ConfigTemplateResponse, status_code=status.HTTP_201_CREATED)
async def save_configuration_as_template(template_data: ConfigTemplateSave):
    config = config_persistence_manager.load_simulation_config(template_data.config_id)
    if not config:
        raise HTTPException(404, f"Configuration '{template_data.config_id}' not found")
    if config_persistence_manager.load_template(template_data.template_name):
        raise HTTPException(409, f"Template '{template_data.template_name}' already exists")
    payload = {
        "template_name": template_data.template_name,
        "description": template_data.description,
        "config_data": {**config, "template_description": template_data.description},
    }
    if not config_persistence_manager.save_template(template_data.template_name, payload):
        raise HTTPException(500, "Failed to save template")
    return ConfigTemplateResponse(
        template_name=template_data.template_name,
        description=template_data.description,
        config_data=payload["config_data"],
        created_at=datetime.now()
    )

@router.post("/benchmark/config", response_model=BenchmarkConfigResponse)
async def create_benchmark_configuration(config_data: BenchmarkConfigRequest):
    config_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    simulation_config = {
        "config_id": config_id,
        "name": config_data.simulationSettings.get("simulationName", f"Config-{config_id[:8]}"),
        "description": config_data.simulationSettings.get("description", ""),
        "created_at": now,
        "updated_at": now,
        "tick_interval_seconds": config_data.simulationSettings.get("tickInterval", 1),
        "max_ticks": config_data.simulationSettings.get("duration", 1000),
        "seed": config_data.simulationSettings.get("randomSeed", 42),
        "initial_price": config_data.simulationSettings.get("initialPrice", 10.0),
        "initial_inventory": config_data.simulationSettings.get("inventory", 100),
        "agent_configs": config_data.agentConfigs,
        "llm_settings": config_data.llmSettings,
        "constraints": config_data.constraints,
        "experiment_settings": config_data.experimentSettings,
        "original_frontend_config": config_data.model_dump(),
    }
    if not config_persistence_manager.save_simulation_config(config_id, simulation_config):
        raise HTTPException(500, "Failed to persist configuration")
    return BenchmarkConfigResponse(
        success=True,
        message="Benchmark configuration created",
        config_id=config_id,
        created_at=simulation_config["created_at"]
    )