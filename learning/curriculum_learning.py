from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class CurriculumLevel:
    level_id: str
    created_at: str
    updated_at: str
    data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    difficulty: Optional[str] = None
    version: int = 1


class CurriculumLearning:
    """
    Production-ready curriculum manager.

    Features:
    - Add/get curriculum levels
    - Track per-student progress
    - Optional persistence to storage_dir as JSON files
    """

    def __init__(self, storage_dir: Optional[str] = None) -> None:
        self._curriculum: Dict[str, CurriculumLevel] = {}
        self._student_progress: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._storage_dir = storage_dir
        if self._storage_dir:
            os.makedirs(self._storage_dir, exist_ok=True)
            self._load_from_disk()

    def add_curriculum_level(self, level_data: Dict[str, Any]) -> str:
        """Add a new level and return its level_id."""
        now = datetime.now(timezone.utc).isoformat()
        level_id = str(level_data.get("level_id") or f"lvl-{uuid.uuid4().hex[:10]}")
        level = CurriculumLevel(
            level_id=level_id,
            created_at=now,
            updated_at=now,
            data={k: v for k, v in level_data.items() if k != "level_id"},
            tags=list(level_data.get("tags", [])),
            difficulty=level_data.get("difficulty"),
            version=int(level_data.get("version", 1)),
        )
        self._curriculum[level_id] = level
        self._persist_level(level)
        logger.info("Curriculum level added: %s", level_id)
        return level_id

    def get_curriculum_level(self, level_id: str) -> Optional[Dict[str, Any]]:
        """Return level dict or None."""
        lvl = self._curriculum.get(level_id)
        if not lvl:
            return None
        return self._level_to_dict(lvl)

    def update_student_progress(self, student_id: str, level_id: str, progress: Dict[str, Any]) -> None:
        """
        Merge student progress for a level. Raises if level unknown.
        Common fields: status, score, attempts, last_completed_at.
        """
        if level_id not in self._curriculum:
            raise KeyError(f"Unknown level_id: {level_id}")
        student = self._student_progress.setdefault(student_id, {})
        lvl_progress = student.setdefault(level_id, {})
        lvl_progress.update(progress or {})
        lvl_progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        logger.debug("Updated progress student=%s level=%s", student_id, level_id)

    def list_levels(self) -> List[Dict[str, Any]]:
        """List all levels as dicts."""
        return [self._level_to_dict(v) for v in self._curriculum.values()]

    def get_student_progress(self, student_id: str, level_id: Optional[str] = None) -> Dict[str, Any]:
        """Get progress for student; optionally a single level."""
        data = self._student_progress.get(student_id, {})
        return data if level_id is None else dict(data.get(level_id, {}))

    def _persist_level(self, level: CurriculumLevel) -> None:
        if not self._storage_dir:
            return
        path = os.path.join(self._storage_dir, f"{level.level_id}.json")
        payload = self._level_to_dict(level)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning("Failed to persist curriculum level %s: %s", level.level_id, e)

    def _load_from_disk(self) -> None:
        try:
            for name in os.listdir(self._storage_dir or ""):
                if not name.endswith(".json"):
                   continue
                path = os.path.join(self._storage_dir or "", name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    lvl = CurriculumLevel(
                        level_id=str(data["level_id"]),
                        created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
                        updated_at=str(data.get("updated_at") or datetime.now(timezone.utc).isoformat()),
                        data=dict(data.get("data", {})),
                        tags=list(data.get("tags", [])),
                        difficulty=data.get("difficulty"),
                        version=int(data.get("version", 1)),
                    )
                    self._curriculum[lvl.level_id] = lvl
                except Exception as e:
                    logger.warning("Failed to load curriculum file %s: %s", path, e)
        except Exception as e:
            logger.warning("Failed to scan curriculum storage: %s", e)

    @staticmethod
    def _level_to_dict(level: CurriculumLevel) -> Dict[str, Any]:
        return {
            "level_id": level.level_id,
            "created_at": level.created_at,
            "updated_at": level.updated_at,
            "data": dict(level.data),
            "tags": list(level.tags),
            "difficulty": level.difficulty,
            "version": level.version,
        }