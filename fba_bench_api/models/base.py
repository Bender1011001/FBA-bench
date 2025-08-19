from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import DateTime, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator, TEXT


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class JSONEncoded(TypeDecorator):
    """
    SQLite-safe JSON TypeDecorator.
    Uses TEXT storage with JSON serialization for cross-DB compatibility.
    """
    impl = TEXT

    cache_ok = True

    def process_bind_param(self, value: Any, dialect) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value: Optional[str], dialect) -> Any:
        if value is None:
            return None
        return json.loads(value)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# Ensure updated_at auto-updates on UPDATE
@event.listens_for(Base, "before_update", propagate=True)
def _timestamp_before_update(mapper, connection, target):
    if hasattr(target, "updated_at"):
        setattr(target, "updated_at", utcnow())