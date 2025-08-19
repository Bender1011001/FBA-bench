"""
Simple Alembic migration runner for FBA-Bench.

Usage:
  - Upgrade to head:
      poetry run python scripts/db_migrate.py upgrade head
  - Downgrade to base:
      poetry run python scripts/db_migrate.py downgrade base

DATABASE_URL is read from environment (defaults to sqlite:///./fba_bench.db).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config


def make_config() -> Config:
    repo_root = Path(__file__).resolve().parents[1]
    ini_path = repo_root / "alembic.ini"
    if not ini_path.exists():
        raise FileNotFoundError("alembic.ini not found at repo root")

    cfg = Config(str(ini_path))
    cfg.set_main_option("script_location", str((repo_root / "alembic").as_posix()))
    # Allow env.py to resolve DATABASE_URL; set if present to help some tools
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: scripts/db_migrate.py [upgrade|downgrade|current|heads] [target]")
        return 2

    action = argv[1]
    target = argv[2] if len(argv) >= 3 else "head"

    cfg = make_config()

    if action == "upgrade":
        command.upgrade(cfg, target)
    elif action == "downgrade":
        command.downgrade(cfg, target)
    elif action == "current":
        command.current(cfg)
    elif action == "heads":
        command.heads(cfg)
    else:
        print(f"Unknown action: {action}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))