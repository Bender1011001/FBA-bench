import os
import shutil
from pathlib import Path

import pytest

alembic = pytest.importorskip("alembic", reason="Alembic not installed")
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402


@pytest.mark.integration
def test_alembic_upgrade_and_downgrade_cycle(tmp_path: Path, monkeypatch):
    """
    Validates that Alembic migrations can be applied and reversed programmatically.

    Steps:
    - Prepare a temp SQLite database file URL.
    - Configure Alembic to use local script_location and override sqlalchemy.url via env.
    - Upgrade to head.
    - Downgrade to base.
    - Upgrade to head again.
    """
    # Workspace root (repo root where alembic.ini and alembic/ live)
    repo_root = Path(__file__).resolve().parents[2]

    # Ensure a clean temp directory for the DB file
    workdir = tmp_path / "alembic_tmp"
    workdir.mkdir(parents=True, exist_ok=True)

    # SQLite file for this test (prefer file-based to ensure consistent behavior)
    db_path = workdir / "alembic_test.db"
    db_url = f"sqlite:///{db_path.as_posix()}"

    # Set DATABASE_URL so alembic/env.py reads it (preferred over sqlalchemy.url here)
    monkeypatch.setenv("DATABASE_URL", db_url)

    # Prepare Alembic Config
    alembic_ini = repo_root / "alembic.ini"
    assert alembic_ini.exists(), "alembic.ini must exist at repo root"

    cfg = Config(str(alembic_ini))
    # Script location should be the 'alembic' dir in repo
    cfg.set_main_option("script_location", str((repo_root / "alembic").as_posix()))
    # Avoid reading sqlalchemy.url from file; env.py reads DATABASE_URL instead
    if cfg.get_main_option("sqlalchemy.url"):
        cfg.set_main_option("sqlalchemy.url", db_url)

    # Execute upgrade/downgrade cycle
    command.upgrade(cfg, "head")
    # Reaching here means upgrade succeeded; database was created and migrations applied.

    command.downgrade(cfg, "base")
    # Reaching here means downgrade succeeded.

    command.upgrade(cfg, "head")
    # Reaching here means a full cycle works.

    # Sanity checks
    assert db_path.exists(), "SQLite DB file should exist after migrations"
    # Leave the SQLite file for inspection on failure; clean up on success
    try:
        shutil.rmtree(workdir)
    except Exception:
        # Best-effort cleanup; not fatal for CI
        pass