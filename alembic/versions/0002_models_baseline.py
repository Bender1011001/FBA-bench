"""models baseline

Revision ID: 0002_models_baseline
Revises: 0001_initial_baseline
Create Date: 2025-08-18 16:40:00.000000

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0002_models_baseline"
down_revision = "0001_initial_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # agents
    op.create_table(
        "agents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False, index=True),
        sa.Column(
            "framework",
            sa.Enum("baseline", "langchain", "crewai", "custom", name="framework_enum", native_enum=False, validate_strings=True),
            nullable=False,
        ),
        sa.Column("config", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_agents_name", "agents", ["name"])

    # experiments
    op.create_table(
        "experiments",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False, index=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("agent_id", sa.String(length=36), sa.ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False, index=True),
        sa.Column("scenario_id", sa.String(length=255), nullable=True),
        sa.Column("params", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("draft", "running", "completed", "failed", name="experiment_status_enum", native_enum=False, validate_strings=True),
            nullable=False,
            server_default=sa.text("'draft'"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_experiments_name", "experiments", ["name"])
    op.create_index("ix_experiments_agent_id", "experiments", ["agent_id"])

    # simulations
    op.create_table(
        "simulations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("experiment_id", sa.String(length=36), sa.ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column(
            "status",
            sa.Enum("pending", "running", "stopped", "completed", "failed", name="simulation_status_enum", native_enum=False, validate_strings=True),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("metadata", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_simulations_experiment_id", "simulations", ["experiment_id"])


def downgrade() -> None:
    # drop simulations
    op.drop_index("ix_simulations_experiment_id", table_name="simulations")
    op.drop_table("simulations")
    op.execute("DROP TYPE IF EXISTS simulation_status_enum")

    # drop experiments
    op.drop_index("ix_experiments_agent_id", table_name="experiments")
    op.drop_index("ix_experiments_name", table_name="experiments")
    op.drop_table("experiments")
    op.execute("DROP TYPE IF EXISTS experiment_status_enum")

    # drop agents
    op.drop_index("ix_agents_name", table_name="agents")
    op.drop_table("agents")
    op.execute("DROP TYPE IF EXISTS framework_enum")