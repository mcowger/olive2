"""cancel_dbos_workflows_for_meeting must work even when DBOS is not launched yet.

Regression: DBOS launches lazily on first enqueue, so after a server restart
the cancel path used to return 0 immediately and leave queued workflows
running. The helper must launch DBOS on demand and then cancel.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import text

# Import order matters: services first avoids the workflows<->ml cycle.
import olive.services.transcription  # noqa: F401
from olive.db import create_db
from olive.workflows import transcription_workflow as wf


async def test_cancel_launches_dbos_when_not_launched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = create_db(":memory:")
    meeting_id = "m-cancel-dbos"
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :mid, 'Cancel DBOS', 1, 2, 'upload', 'ready', '[]',
                  NULL, NULL, NULL, 1, 1
                )
                """
            ),
            {"mid": meeting_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO stage_runs (
                  id, meeting_id, stage, status, provider_job_id, attempts,
                  last_error, started_at, finished_at, created_at, updated_at
                ) VALUES (
                  'sr-1', :mid, 'local_transcribe', 'cancelled', 'local-abcd1234', 1,
                  NULL, 1, NULL, 1, 1
                )
                """
            ),
            {"mid": meeting_id},
        )

    dbos_path = tmp_path / "dbos.sqlite"
    with sqlite3.connect(str(dbos_path)) as sconn:
        sconn.execute(
            "CREATE TABLE workflow_status (workflow_uuid TEXT, parent_workflow_id TEXT, status TEXT)"
        )
        sconn.executemany(
            "INSERT INTO workflow_status VALUES (?, ?, ?)",
            [
                ("local-abcd1234", None, "ENQUEUED"),
                ("local-abcd1234-6", "local-abcd1234", "ENQUEUED"),
                ("local-other999", None, "ENQUEUED"),
            ],
        )

    state = {"launched": False, "ensure_calls": 0}
    cancelled: list[str] = []

    async def fake_ensure(**_kwargs: Any) -> str:
        state["ensure_calls"] += 1
        state["launched"] = True
        return "sqlite:///tmp/dbos.sqlite"

    async def fake_cancel(workflow_id: str) -> None:
        cancelled.append(workflow_id)

    monkeypatch.setattr(wf, "is_dbos_launched", lambda: state["launched"])
    monkeypatch.setattr(wf, "ensure_dbos_launched_async", fake_ensure)
    monkeypatch.setattr(wf, "get_existing_dbos_sqlite_path", lambda **_k: dbos_path)
    monkeypatch.setattr(wf, "DBOS", SimpleNamespace(cancel_workflow_async=fake_cancel))

    count = await wf.cancel_dbos_workflows_for_meeting(
        meeting_id,
        db=handle.engine,
        config_dir=str(tmp_path),
        meetings_dir=str(tmp_path),
    )

    assert state["ensure_calls"] == 1
    assert count == 2
    assert sorted(cancelled) == ["local-abcd1234", "local-abcd1234-6"]
    handle.engine.dispose()


async def test_cancel_matches_insights_job_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``insights:{meeting_id}:{hex8}`` jobs (the ``insights`` queue) are cancelled with the meeting."""
    handle = create_db(":memory:")
    dbos_path = tmp_path / "dbos.sqlite"
    with sqlite3.connect(str(dbos_path)) as sconn:
        sconn.execute(
            "CREATE TABLE workflow_status (workflow_uuid TEXT, parent_workflow_id TEXT, status TEXT)"
        )
        sconn.executemany(
            "INSERT INTO workflow_status VALUES (?, ?, ?)",
            [
                ("insights:m-ins:abcd1234", None, "ENQUEUED"),
                ("insights:m-other:abcd1234", None, "ENQUEUED"),
                ("insights:m-ins:done5678", None, "SUCCESS"),
            ],
        )

    cancelled: list[str] = []

    async def fake_cancel(workflow_id: str) -> None:
        cancelled.append(workflow_id)

    monkeypatch.setattr(wf, "is_dbos_launched", lambda: True)
    monkeypatch.setattr(wf, "get_existing_dbos_sqlite_path", lambda **_k: dbos_path)
    monkeypatch.setattr(wf, "DBOS", SimpleNamespace(cancel_workflow_async=fake_cancel))

    count = await wf.cancel_dbos_workflows_for_meeting(
        "m-ins", db=handle.engine, config_dir=str(tmp_path), meetings_dir=str(tmp_path)
    )

    assert count == 1
    assert cancelled == ["insights:m-ins:abcd1234"]
    handle.engine.dispose()
