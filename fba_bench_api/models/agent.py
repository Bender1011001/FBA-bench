from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from sqlalchemy import String, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, JSONEncoded, TimestampMixin, utcnow


class FrameworkEnum(str):
    baseline = "baseline"
    langchain = "langchain"
    crewai = "crewai"
    custom = "custom"


class AgentORM(TimestampMixin, Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID4 string
    name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    framework: Mapped[str] = mapped_column(
        SAEnum(
            FrameworkEnum.baseline,
            FrameworkEnum.langchain,
            FrameworkEnum.crewai,
            FrameworkEnum.custom,
            name="framework_enum",
            native_enum=False,
            validate_strings=True,
        ),
        nullable=False,
    )
    config: Mapped[dict] = mapped_column(JSONEncoded, nullable=False, default=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "framework": self.framework,
            "config": self.config or {},
            "created_at": self.created_at if isinstance(self.created_at, datetime) else utcnow(),
            "updated_at": self.updated_at if isinstance(self.updated_at, datetime) else utcnow(),
        }