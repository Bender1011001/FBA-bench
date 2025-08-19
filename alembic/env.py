from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import create_engine, pool
from sqlalchemy.engine import Connection
from sqlalchemy import engine_from_config
from alembic import context

# Configure Python logging (optional; integrates with alembic.ini)
if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)

# Target metadata: import the Base metadata from backend models without importing the whole app
# This avoids coupling to runtime-only modules.
from fba_bench_api.models.base import Base  # noqa: E402

target_metadata = Base.metadata

# Read database URL from environment with sensible default for development
DEFAULT_SQLITE_URL = "sqlite:///./fba_bench.db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)


def get_url() -> str:
    # Alembic prefers explicit URL resolution; avoid alembic.ini sqlalchemy.url
    return DATABASE_URL


def include_object(object, name, type_, reflected, compare_to):
    """
    Hook to filter objects from autogenerate if needed.
    Currently, include everything; keep deterministic migrations.
    """
    return True


def process_revision_directives(context, revision, directives):
    """
    Ensure deterministic message formatting or other directive tweaks if desired.
    Currently a no-op.
    """
    # Example:
    # for script in directives:
    #     script.message = script.message.strip().lower().replace(" ", "_")
    return


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.
    This configures the context with just a URL and not an Engine.
    By skipping Engine creation, we don't need DBAPI access.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        process_revision_directives=process_revision_directives,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.
    In this scenario we need to create an Engine and associate a connection with the context.
    """
    url = get_url()

    connect_args = {}
    if url.startswith("sqlite"):
        # Required for SQLite file DBs when used in some runtimes; safe for Alembic too.
        connect_args["check_same_thread"] = False

    engine = create_engine(
        url,
        poolclass=pool.NullPool,
        connect_args=connect_args,
        future=True,
    )

    with engine.connect() as connection:  # type: Connection
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
            process_revision_directives=process_revision_directives,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()