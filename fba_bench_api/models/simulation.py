from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class SimulationSnapshot(BaseModel):
    current_tick: Optional[int] = None
    simulation_time: Optional[str] = None
    last_update: Optional[str] = None
    uptime_seconds: Optional[int] = None
    products: Optional[Dict[str, Any]] = None
    competitors: Optional[Dict[str, Any]] = None
    market_summary: Optional[Dict[str, Any]] = None
    financial_summary: Optional[Dict[str, Any]] = None
    agents: Optional[Dict[str, Any]] = None
    command_stats: Optional[Dict[str, Any]] = None
    event_stats: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class EventFilter(BaseModel):
    event_type: Optional[str] = None
    limit: int = 20
    since_tick: Optional[int] = None

class SimulationConfigCreate(BaseModel):
    name: str
    description: Optional[str] = None
    tick_interval_seconds: float = Field(1.0)
    max_ticks: Optional[int] = None
    start_time: Optional[datetime] = None
    time_acceleration: float = Field(1.0)
    seed: Optional[int] = None
    base_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)

class SimulationConfigUpdate(SimulationConfigCreate):
    name: Optional[str] = None

class SimulationConfigResponse(BaseModel):
    config_id: str
    name: str
    description: Optional[str]
    tick_interval_seconds: float
    max_ticks: Optional[int]
    start_time: Optional[datetime]
    time_acceleration: float
    seed: Optional[int]
    base_parameters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class SimulationStartRequest(BaseModel):
    config_id: str
    simulation_id: Optional[str] = None

class SimulationControlResponse(BaseModel):
    success: bool
    message: str
    simulation_id: Optional[str] = None
    status: Optional[Dict[str, Any]] = None

class SimulationStatusResponse(BaseModel):
    is_running: bool
    is_paused: bool
    current_tick: int
    real_time: str
    simulation_time: str
    config: Dict[str, Any]
    statistics: Dict[str, Any]
    simulation_id: Optional[str] = None
    message: Optional[str] = None