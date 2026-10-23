"""Database migrations, built-in template seeding, and legacy repair (port of migrations.ts)."""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from sqlalchemy.engine import Connection, Engine

CREATE_TABLE_STATEMENTS: list[str] = [
    """CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    start_time INTEGER NOT NULL,
    end_time INTEGER NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    tags TEXT NOT NULL DEFAULT '[]',
    primary_transcript_artifact_id TEXT REFERENCES artifacts(id),
    primary_summary_artifact_id TEXT REFERENCES artifacts(id),
    last_error TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
  )""",
    """CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id),
    path TEXT NOT NULL,
    mime TEXT NOT NULL,
    duration_ms INTEGER,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    provider_recording_id TEXT,
    created_at INTEGER NOT NULL
  )""",
    """CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id),
    recording_id TEXT REFERENCES recordings(id),
    kind TEXT NOT NULL,
    provider TEXT NOT NULL,
    format TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at INTEGER NOT NULL
  )""",
    """CREATE TABLE IF NOT EXISTS speakers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider_ids TEXT NOT NULL DEFAULT '{}',
    enrolled_at INTEGER,
    enrollment_clip_paths TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL
  )""",
    """CREATE TABLE IF NOT EXISTS meeting_speakers (
    meeting_id TEXT NOT NULL REFERENCES meetings(id),
    speaker_id TEXT NOT NULL REFERENCES speakers(id),
    evidence_artifact_id TEXT REFERENCES artifacts(id),
    PRIMARY KEY (meeting_id, speaker_id)
  )""",
    """CREATE TABLE IF NOT EXISTS stage_runs (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id),
    stage TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    provider_job_id TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    started_at INTEGER,
    finished_at INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
  )""",
    """CREATE TABLE IF NOT EXISTS stage_checkpoints (
    meeting_id TEXT NOT NULL REFERENCES meetings(id),
    stage TEXT NOT NULL,
    run_id TEXT NOT NULL,
    doc TEXT NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (meeting_id, stage)
  )""",
    """CREATE TABLE IF NOT EXISTS plaud_ingest_state (
    meeting_id TEXT PRIMARY KEY REFERENCES meetings(id),
    plaud_file_id TEXT NOT NULL UNIQUE,
    first_seen_at INTEGER NOT NULL,
    pcs_deadline_at INTEGER NOT NULL,
    pcs_resolved INTEGER NOT NULL DEFAULT 0
  )""",
    """CREATE TABLE IF NOT EXISTS plaud_deleted_files (\n    plaud_file_id TEXT PRIMARY KEY,\n    deleted_at INTEGER NOT NULL\n  )""",
    """CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
  )""",
    """CREATE TABLE IF NOT EXISTS templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    system_prompt TEXT NOT NULL DEFAULT '',
    user_prompt TEXT NOT NULL,
    is_default INTEGER NOT NULL DEFAULT 0,
    is_builtin INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
  )""",
    """CREATE TABLE IF NOT EXISTS logs (
    id TEXT PRIMARY KEY,
    level TEXT NOT NULL,
    category TEXT NOT NULL,
    message TEXT NOT NULL,
    meeting_id TEXT REFERENCES meetings(id),
    details TEXT,
    created_at INTEGER NOT NULL
  )""",
    """CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    usage TEXT,
    created_at INTEGER NOT NULL
  )""",
    """CREATE TABLE IF NOT EXISTS usage_events (
    id TEXT PRIMARY KEY,
    meeting_id TEXT REFERENCES meetings(id),
    kind TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    audio_seconds REAL,
    cost_usd REAL,
    source TEXT NOT NULL UNIQUE,
    detail TEXT,
    created_at INTEGER NOT NULL
  )""",
    """CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
  )""",
]

CREATE_INDEX_STATEMENTS: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_meetings_start_time ON meetings(start_time DESC)",
    "CREATE INDEX IF NOT EXISTS idx_recordings_meeting ON recordings(meeting_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_recordings_provider_id ON recordings(provider, provider_recording_id) WHERE provider_recording_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_meeting ON artifacts(meeting_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_stage_runs_meeting_stage ON stage_runs(meeting_id, stage)",
    "CREATE INDEX IF NOT EXISTS idx_templates_name ON templates(name)",
    "CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)",
    "CREATE INDEX IF NOT EXISTS idx_logs_category ON logs(category)",
    "CREATE INDEX IF NOT EXISTS idx_logs_meeting_id ON logs(meeting_id)",
    "CREATE INDEX IF NOT EXISTS idx_chat_messages_meeting_created ON chat_messages(meeting_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_usage_events_created ON usage_events(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_usage_events_meeting ON usage_events(meeting_id)",
]

ADDITIVE_COLUMNS: list[tuple[str, str, str]] = [
    ("meetings", "last_error", "TEXT"),
    ("recordings", "provider_recording_id", "TEXT"),
    ("artifacts", "recording_id", "TEXT REFERENCES recordings(id)"),
    ("speakers", "provider_ids", "TEXT NOT NULL DEFAULT '{}'"),
    ("speakers", "enrolled_at", "INTEGER"),
    ("speakers", "enrollment_clip_paths", "TEXT NOT NULL DEFAULT '[]'"),
    ("stage_runs", "provider_job_id", "TEXT"),
    ("stage_runs", "attempts", "INTEGER NOT NULL DEFAULT 0"),
    ("stage_runs", "last_error", "TEXT"),
    ("stage_runs", "started_at", "INTEGER"),
    ("stage_runs", "finished_at", "INTEGER"),
    ("plaud_ingest_state", "pcs_resolved", "INTEGER NOT NULL DEFAULT 0"),
    ("plaud_ingest_state", "duplicate_of_meeting_id", "TEXT REFERENCES meetings(id)"),
]

DEFAULT_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "Executive Summary",
        "description": "Structured meeting overview with context, key discussion points, decisions made, and assigned action items.",
        "system_prompt": "You are an expert executive assistant and meeting analyst. Produce clear, structured, professional meeting summaries in Markdown format with precise speaker attribution and actionable next steps. Do not invent, guess, or hallucinate participant names or speaker identities that are not explicitly stated in the transcript or participant list. Only attribute statements to the actual speaker labels provided in the transcript.",
        "user_prompt": """# Meeting Summary: {{title}}
**Date:** {{date}}
**Participants:** {{speakers}}

## Executive Overview
Summarize the main purpose, high-level context, and outcome of the meeting in 2-3 concise paragraphs.

## Key Discussion Points
- Detail each key topic discussed, noting differing perspectives, relevant metrics, and context.

## Decisions Made
- Clear, bulleted list of all explicit decisions agreed upon during the conversation.

## Action Items & Next Steps
- [ ] **[Owner]**: Specific task description (include deadlines/timeline if mentioned)

---
### Meeting Transcript:
{{transcript}}""",
        "is_default": 1,
        "is_builtin": 1,
    },
    {
        "id": "00000000-0000-0000-0000-000000000002",
        "name": "1-on-1 Catchup",
        "description": "Focused on individual priorities, project updates, blockers, feedback, and mutual commitments.",
        "system_prompt": "You are an executive coach and engineering manager summarizing a 1-on-1 meeting. Emphasize personal updates, blockers, feedback, and commitments. Do not invent, guess, or hallucinate participant names or speaker identities that are not explicitly stated in the transcript or participant list. Only attribute statements to the actual speaker labels provided in the transcript.",
        "user_prompt": """# 1-on-1 Summary: {{title}}
**Date:** {{date}}
**Participants:** {{speakers}}

## Progress & Wins
- Recent achievements, project updates, and positive milestones discussed.

## Blockers & Concerns
- Any obstacles, risks, team dependencies, or challenges raised that need resolution.

## Feedback & Career Growth
- Feedback exchanged, development opportunities, coaching notes, or long-term goals.

## Commitments & Follow-ups
- [ ] **[Owner]**: Agreed action item or follow-up

---
### Meeting Transcript:
{{transcript}}""",
        "is_default": 0,
        "is_builtin": 1,
    },
    {
        "id": "00000000-0000-0000-0000-000000000003",
        "name": "Technical Architecture & Design Review",
        "description": "Detailed engineering review covering architecture, technical trade-offs, scalability, and open questions.",
        "system_prompt": "You are a principal software engineer and technical architect reviewing meeting notes. Focus on architectural decisions, trade-offs, constraints, and engineering action items. Do not invent, guess, or hallucinate participant names or speaker identities that are not explicitly stated in the transcript or participant list. Only attribute statements to the actual speaker labels provided in the transcript.",
        "user_prompt": """# Technical Architecture Review: {{title}}
**Date:** {{date}}
**Participants:** {{speakers}}

## Problem Statement & Context
- Technical goals, motivation, and scope of the proposed system or change.

## Architecture & Design Decisions
- Architectural approach, component responsibilities, data flow, and key decisions.

## Trade-offs & Alternatives Considered
- Alternatives discussed, pros/cons, and explicit trade-offs.

## Non-Functional Requirements & Risks
- Security, performance, scale, reliability, failure modes, and monitoring needs.

## Open Questions & Action Items
- [ ] **[Owner]**: Follow-up task, benchmark, prototype, or design doc update

---
### Meeting Transcript:
{{transcript}}""",
        "is_default": 0,
        "is_builtin": 1,
    },
    {
        "id": "00000000-0000-0000-0000-000000000004",
        "name": "Action Items & Decisions",
        "description": "Concise, zero-fluff extraction of decisions made and task assignments with owners.",
        "system_prompt": "You are an agile project manager extracting strictly decisions and actionable next steps. Omit general conversational filler. Do not invent, guess, or hallucinate participant names or speaker identities that are not explicitly stated in the transcript or participant list.",
        "user_prompt": """# Decisions & Action Items: {{title}}
**Date:** {{date}}
**Participants:** {{speakers}}

## Decisions Log
- **Decision**: Context and agreed outcome.

## Action Items Matrix
- [ ] **[Owner]**: Task description (Due date / milestone if mentioned)

---
### Meeting Transcript:
{{transcript}}""",
        "is_default": 0,
        "is_builtin": 1,
    },
]


def _extract_sqlite_conn(target: sqlite3.Connection | Connection | Engine) -> sqlite3.Connection:
    if isinstance(target, sqlite3.Connection):
        return target
    if isinstance(target, Engine):
        raw = target.raw_connection()
        return raw.driver_connection  # type: ignore[return-value]
    if isinstance(target, Connection):
        return target.connection.driver_connection  # type: ignore[return-value]
    raise TypeError(f"Unsupported database target: {type(target)}")


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def _seed_builtin_templates(conn: sqlite3.Connection) -> None:
    now = int(time.time() * 1000)
    for tpl in DEFAULT_TEMPLATES:
        conn.execute(
            """
            INSERT OR IGNORE INTO templates
              (id, name, description, system_prompt, user_prompt, is_default, is_builtin, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tpl["id"],
                tpl["name"],
                tpl["description"],
                tpl["system_prompt"],
                tpl["user_prompt"],
                tpl["is_default"],
                tpl["is_builtin"],
                now,
                now,
            ),
        )
        conn.execute(
            """
            UPDATE templates
            SET system_prompt = ?
            WHERE id = ? AND is_builtin = 1
            """,
            (tpl["system_prompt"], tpl["id"]),
        )


def _repair_corrupted_plaud_durations(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE meetings
        SET end_time = start_time + ((end_time - start_time) / 1000)
        WHERE source = 'plaud'
          AND (end_time - start_time) > 86400000
          AND ((end_time - start_time) % 1000) = 0
        """
    )
    conn.execute(
        """
        UPDATE recordings
        SET duration_ms = duration_ms / 1000
        WHERE provider = 'plaud'
          AND duration_ms IS NOT NULL
          AND duration_ms > 86400000
          AND (duration_ms % 1000) = 0
        """
    )


def _stamp_alembic_baseline(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT version_num FROM alembic_version LIMIT 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('0001_baseline')")


def run_migrations(target: sqlite3.Connection | Connection | Engine) -> None:
    if isinstance(target, Engine):
        raw_conn = target.raw_connection()
        try:
            sqlite_conn: sqlite3.Connection = raw_conn.driver_connection  # type: ignore[assignment]
            _run_migrations_sqlite(sqlite_conn)
            raw_conn.commit()
        finally:
            raw_conn.close()
        return

    sqlite_conn = _extract_sqlite_conn(target)
    _run_migrations_sqlite(sqlite_conn)
    sqlite_conn.commit()


def _run_migrations_sqlite(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    for statement in CREATE_TABLE_STATEMENTS:
        conn.execute(statement)

    for table, column, definition in ADDITIVE_COLUMNS:
        _add_column_if_missing(conn, table, column, definition)

    for statement in CREATE_INDEX_STATEMENTS:
        conn.execute(statement)

    _seed_builtin_templates(conn)
    _repair_corrupted_plaud_durations(conn)
    _stamp_alembic_baseline(conn)
