from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Literal

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from fba_bench_api.core.database_async import get_async_db_session
from fba_bench_api.core.persistence_async import AsyncPersistenceManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/experiments", tags=["Experiments"])

# Status enum and validation
ExperimentStatus = Literal["draft", "running", "completed", "failed"]

# Pydantic Schemas
class ExperimentBase(BaseModel):
    name: str = Field(..., min_length=1, description="Experiment name")
    description: Optional[str] = Field(None, description="Optional description")
    agent_id: str = Field(..., min_length=1, description="Associated agent id (UUID4 string)")
    scenario_id: Optional[str] = Field(None, description="Scenario identifier")
    params: dict = Field(default_factory=dict, description="Arbitrary parameters for the run")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must be non-empty")
        return v.strip()

class ExperimentCreate(ExperimentBase):
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Exp1",
                "description": "Benchmark agent on scenario-abc",
                "agent_id": "7f3a3a2f-6f2b-4bfb-8b9b-2b7b0f5f8e12",
                "scenario_id": "scenario-abc",
                "params": {"k": 1, "seed": 42}
            }
        }

class ExperimentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1)
    description: Optional[str] = None
    params: Optional[dict] = None
    status: Optional[ExperimentStatus] = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("name must be non-empty when provided")
        return v.strip() if v else v

    class Config:
        json_schema_extra = {
            "example": {"name": "Exp1-updated", "status": "running", "params": {"k": 2}}
        }

class Experiment(ExperimentBase):
    id: str
    status: ExperimentStatus = "draft"
    created_at: datetime
    updated_at: datetime

def get_pm(db: AsyncSession = Depends(get_async_db_session)) -> AsyncPersistenceManager:
    return AsyncPersistenceManager(db)

# Utilities
import uuid as _uuid
def _uuid4() -> str:
    return str(_uuid.uuid4())

def _now() -> datetime:
    return datetime.utcnow()

# Transition validation
def _validate_transition(current: ExperimentStatus, desired: ExperimentStatus) -> None:
    allowed = {
        "draft": {"running"},
        "running": {"completed", "failed"},
        "completed": set(),
        "failed": set(),
    }
    if desired not in allowed[current]:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Invalid status transition: {current} -> {desired}",
        )

# Routes

@router.get("", response_model=list[Experiment], description="List all experiments")
async def list_experiments(pm: AsyncPersistenceManager = Depends(get_pm)):
    items = await pm.experiments().list()
    return [Experiment(**i) for i in items]

@router.post("", response_model=Experiment, status_code=status.HTTP_201_CREATED, description="Create a new experiment (starts as draft)")
async def create_experiment(payload: ExperimentCreate, pm: AsyncPersistenceManager = Depends(get_pm)):
    data = payload.model_dump()
    item = {
        "id": _uuid4(),
        "name": data["name"],
        "description": data.get("description"),
        "agent_id": data["agent_id"],
        "scenario_id": data.get("scenario_id"),
        "params": data.get("params") or {},
        "status": "draft",
        "created_at": _now(),
        "updated_at": _now(),
    }
    created = await pm.experiments().create(item)
    return Experiment(**created)

@router.get("/{experiment_id}", response_model=Experiment, description="Retrieve experiment by id")
async def get_experiment(experiment_id: str, pm: AsyncPersistenceManager = Depends(get_pm)):
    item = await pm.experiments().get(experiment_id)
    if not item:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Experiment '{experiment_id}' not found")
    return Experiment(**item)

@router.patch("/{experiment_id}", response_model=Experiment, description="Update experiment metadata or transition status")
async def update_experiment(experiment_id: str, payload: ExperimentUpdate, pm: AsyncPersistenceManager = Depends(get_pm)):
    current = await pm.experiments().get(experiment_id)
    if not current:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Experiment '{experiment_id}' not found")
    update_data = payload.model_dump(exclude_unset=True)
    # Handle status transitions
    if "status" in update_data:
        _validate_transition(current["status"], update_data["status"])  # type: ignore[arg-type]
    update_data["updated_at"] = _now()
    updated = await pm.experiments().update(experiment_id, update_data)
    return Experiment(**updated)  # type: ignore[arg-type]

@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT, description="Delete an experiment")
async def delete_experiment(experiment_id: str, pm: AsyncPersistenceManager = Depends(get_pm)):
    ok = await pm.experiments().delete(experiment_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Experiment '{experiment_id}' not found")
    return None


@router.post("/{experiment_id}/stop", response_model=Experiment, description="Gracefully stop a running experiment (transitions to 'completed')")
async def stop_experiment(experiment_id: str, pm: AsyncPersistenceManager = Depends(get_pm)):
    current = await pm.experiments().get(experiment_id)
    if not current:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Experiment '{experiment_id}' not found")
    if current.get("status") != "running":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Experiment not running")
    update = {"status": "completed", "updated_at": _now()}
    updated = await pm.experiments().update(experiment_id, update)
    return Experiment(**updated)


@router.get("/{experiment_id}/results", description="Retrieve experiment results (placeholder payload if not implemented)")
async def get_experiment_results(experiment_id: str, pm: AsyncPersistenceManager = Depends(get_pm)):
    current = await pm.experiments().get(experiment_id)
    if not current:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Experiment '{experiment_id}' not found")
    current_status = current.get("status", "unknown")
    return {
        "experiment_id": experiment_id,
        "results": [],
        "summary": {"status": current_status},
    }