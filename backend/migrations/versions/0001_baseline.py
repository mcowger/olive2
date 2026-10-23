"""Baseline schema matching Olive SQLite database.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-10-13 00:00:00.000000
"""

from __future__ import annotations

from alembic import op

from olive.db.migrations import (
    ADDITIVE_COLUMNS,
    CREATE_INDEX_STATEMENTS,
    CREATE_TABLE_STATEMENTS,
)

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    for stmt in CREATE_TABLE_STATEMENTS:
        if "alembic_version" not in stmt:
            op.execute(stmt)
    conn = op.get_bind()
    for table, column, definition in ADDITIVE_COLUMNS:
        try:
            conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except Exception as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
    for stmt in CREATE_INDEX_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for table in [
        "chat_messages",
        "logs",
        "templates",
        "sync_state",
        "plaud_deleted_files",
        "plaud_ingest_state",
        "stage_checkpoints",
        "stage_runs",
        "meeting_speakers",
        "speakers",
        "artifacts",
        "recordings",
        "meetings",
    ]:
        op.execute(f"DROP TABLE IF EXISTS {table}")
