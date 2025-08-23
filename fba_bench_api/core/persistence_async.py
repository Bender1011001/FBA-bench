from __future__ import annotations

"""
Async persistence layer using SQLAlchemy AsyncSession.

Provides:
- AsyncAgentRepository
- AsyncExperimentRepository
- AsyncSimulationRepository
- AsyncPersistenceManager

These mirror the sync repositories in fba_bench_api/core/persistence.py but use
awaitable DB operations for full non-blocking I/O.
"""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from fba_bench_api.models.base import utcnow
from fba_bench_api.models.agent import AgentORM
from fba_bench_api.models.experiment import ExperimentORM, ExperimentStatusEnum
from fba_bench_api.models.simulation import SimulationORM


class AsyncAgentRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def list(self) -> list[dict]:
        result = await self.db.execute(select(AgentORM))
        return [a.to_dict() for a in result.scalars().all()]

    async def create(self, data: dict) -> dict:
        obj = AgentORM(
            id=data["id"],
            name=data["name"],
            framework=data["framework"],
            config=data.get("config") or {},
        )
        self.db.add(obj)
        await self.db.flush()
        return obj.to_dict()

    async def get(self, agent_id: str) -> Optional[dict]:
        obj = await self.db.get(AgentORM, agent_id)
        return obj.to_dict() if obj else None

    async def update(self, agent_id: str, data: dict) -> Optional[dict]:
        obj = await self.db.get(AgentORM, agent_id)
        if not obj:
            return None
        if "name" in data and data["name"] is not None:
            obj.name = data["name"]
        if "framework" in data and data["framework"] is not None:
            obj.framework = data["framework"]
        if "config" in data and data["config"] is not None:
            obj.config = data["config"]
        obj.updated_at = utcnow()
        await self.db.flush()
        return obj.to_dict()

    async def delete(self, agent_id: str) -> bool:
        obj = await self.db.get(AgentORM, agent_id)
        if not obj:
            return False
        # delete() is synchronous API on AsyncSession but schedules the deletion; flush persists it.
        self.db.delete(obj)
        await self.db.flush()
        return True


class AsyncExperimentRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def list(self) -> list[dict]:
        result = await self.db.execute(select(ExperimentORM))
        return [e.to_dict() for e in result.scalars().all()]

    async def create(self, data: dict) -> dict:
        obj = ExperimentORM(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            agent_id=data["agent_id"],
            scenario_id=data.get("scenario_id"),
            params=data.get("params") or {},
            status=ExperimentStatusEnum.draft,
        )
        self.db.add(obj)
        await self.db.flush()
        return obj.to_dict()

    async def get(self, exp_id: str) -> Optional[dict]:
        obj = await self.db.get(ExperimentORM, exp_id)
        return obj.to_dict() if obj else None

    async def update(self, exp_id: str, data: dict) -> Optional[dict]:
        obj = await self.db.get(ExperimentORM, exp_id)
        if not obj:
            return None
        if "name" in data and data["name"] is not None:
            obj.name = data["name"]
        if "description" in data:
            obj.description = data["description"]
        if "params" in data and data["params"] is not None:
            obj.params = data["params"]
        if "status" in data and data["status"] is not None:
            obj.status = ExperimentStatusEnum(data["status"])
        obj.updated_at = utcnow()
        await self.db.flush()
        return obj.to_dict()

    async def delete(self, exp_id: str) -> bool:
        obj = await self.db.get(ExperimentORM, exp_id)
        if not obj:
            return False
        self.db.delete(obj)
        await self.db.flush()
        return True


class AsyncSimulationRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, data: dict) -> dict:
        obj = SimulationORM(
            id=data["id"],
            experiment_id=data.get("experiment_id"),
            # status defaults to pending
            sim_metadata=data.get("metadata") or {},
        )
        self.db.add(obj)
        await self.db.flush()
        return obj.to_dict_with_topic()

    async def get(self, sim_id: str) -> Optional[dict]:
        obj = await self.db.get(SimulationORM, sim_id)
        return obj.to_dict_with_topic() if obj else None

    async def update(self, sim_id: str, data: dict) -> Optional[dict]:
        obj = await self.db.get(SimulationORM, sim_id)
        if not obj:
            return None
        # Only allow status and metadata updates here; routes validate transitions
        if "status" in data and data["status"] is not None:
            obj.status = data["status"]
        if "metadata" in data and data["metadata"] is not None:
            obj.sim_metadata = data["metadata"]
        obj.updated_at = utcnow()
        await self.db.flush()
        return obj.to_dict_with_topic()


class AsyncPersistenceManager:
    """
    Provides typed async repositories bound to an AsyncSession.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    def agents(self) -> AsyncAgentRepository:
        return AsyncAgentRepository(self.db)

    def experiments(self) -> AsyncExperimentRepository:
        return AsyncExperimentRepository(self.db)

    def simulations(self) -> AsyncSimulationRepository:
        return AsyncSimulationRepository(self.db)