from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal, Dict, Any

from pydantic import BaseModel, Field


# ---- Realtime / Simulation Snapshot ----

SimStatus = Literal["idle", "running", "stopped"]


class AgentSnapshot(BaseModel):
    slug: str = Field(..., description="Agent identifier")
    display_name: str = Field(..., description="Human friendly name")
    state: str = Field(..., description="Agent state string")


class KpiSnapshot(BaseModel):
    revenue: float = Field(..., ge=0.0, description="Total revenue (float)")
    profit: float = Field(..., description="Total profit (float)")
    units_sold: int = Field(..., ge=0, description="Total units sold")


class SimulationSnapshot(BaseModel):
    status: SimStatus = Field(..., description="Simulation status")
    tick: int = Field(..., ge=0)
    kpis: KpiSnapshot
    agents: List[AgentSnapshot] = Field(default_factory=list)
    timestamp: datetime = Field(..., description="UTC ISO timestamp")


# ---- Realtime / Recent Events ----

class RecentEventsResponse(BaseModel):
    events: List[Dict[str, Any]] = Field(default_factory=list)
    event_type: Optional[str] = Field(None, description="Filter applied (sales|commands|...)")
    limit: int = Field(..., ge=1, le=100)
    total_returned: int = Field(..., ge=0)
    timestamp: datetime
    filtered: bool
    since_tick: Optional[int] = Field(None, ge=0)