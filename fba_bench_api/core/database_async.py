from __future__ import annotations

"""
Async SQLAlchemy session/engine for non-blocking DB I/O.

- Uses async drivers (aiosqlite for SQLite, asyncpg for Postgres)
- Provides FastAPI dependency get_async_db_session() (one session per-request)
- create_db_tables_async() helper to initialize schema at startup if needed
"""

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from fba_bench_api.models.base import Base


def _resolve_async_url() -> str:
    """
    Convert synchronous SQLAlchemy URL to async-driver URL when possible.
    - sqlite:///path -> sqlite+aiosqlite:///path
    - postgresql://... -> postgresql+asyncpg://...
    Otherwise, return as-is (user must ensure correct async dialect).
    """
    raw = os.getenv("DATABASE_URL", "sqlite:///./fba_bench.db")
    if raw.startswith("sqlite:///"):
        return f"sqlite+aiosqlite:///{raw.split('sqlite:///', 1)[1]}"
    if raw.startswith("postgresql://"):
        return f"postgresql+asyncpg://{raw.split('postgresql://', 1)[1]}"
    return raw


# Create async engine + sessionmaker
ASYNC_DATABASE_URL: str = _resolve_async_url()
async_engine: AsyncEngine = create_async_engine(ASYNC_DATABASE_URL, future=True, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, expire_on_commit=False, autoflush=False, autocommit=False, class_=AsyncSession)


async def create_db_tables_async() -> None:
    """
    Initialize DB schema using async engine.

    Safe to call at startup; no-ops if tables already exist.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI async dependency.
    Yields an AsyncSession per request; commits on success and rolls back on errors.
    """
    session: AsyncSession = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()