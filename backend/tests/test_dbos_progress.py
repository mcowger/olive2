"""Active DBOS transcription runs must map to their meeting regardless of how they were started."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from dbos._serialization import DefaultSerializer

import olive.app  # noqa: F401  (import order avoids a workflows <-> ml circular import)
from olive.workflows import transcription_workflow as tw

MEETING_ID = "11111111-2222-3333-4444-555555555555"


def _make_dbos_db(path: Path, rows: list[tuple[str, str, str, str | None, dict[str, Any]]]) -> None:
    """rows: (workflow_uuid, status, name, queue_name, workflow_input)."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE workflow_status (
            workflow_uuid TEXT PRIMARY KEY, status TEXT, name TEXT, queue_name TEXT,
            parent_workflow_id TEXT, created_at INTEGER, priority INTEGER DEFAULT 0
        );
        CREATE TABLE workflow_input (workflow_uuid TEXT PRIMARY KEY, inputs TEXT);
        """
    )
    ser = DefaultSerializer()
    for i, (uuid, status, name, queue, inp) in enumerate(rows):
        conn.execute(
            "INSERT INTO workflow_status VALUES (?,?,?,?,NULL,?,0)",
            (uuid, status, name, queue, 1_000 + i),
        )
        conn.execute(
            "INSERT INTO workflow_input VALUES (?,?)",
            (uuid, ser.serialize({"args": [inp], "kwargs": {}})),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def dbos_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "dbos.sqlite"
    monkeypatch.setattr(tw, "is_dbos_launched", lambda: True)
    monkeypatch.setattr(tw, "get_existing_dbos_sqlite_path", lambda **_: path)
    tw._DBOS_PROGRESS_CACHE = None
    tw._WF_MEETING_CACHE.clear()
    return path


def test_active_run_without_stage_link_maps_to_meeting(dbos_path: Path) -> None:
    # A recovered run: its id matches no stage_runs.provider_job_id and no task context.
    _make_dbos_db(
        dbos_path,
        [
            (
                "dbos-485fabef",
                "PENDING",
                "<temp>.step_transcribe_asr",
                "asr",
                {"meetingId": MEETING_ID, "workDir": "/tmp/x"},
            )
        ],
    )

    progress = tw.get_dbos_active_progress_map(None)

    assert MEETING_ID in progress
    assert progress[MEETING_ID]["stage"]


def test_finished_run_is_not_reported(dbos_path: Path) -> None:
    _make_dbos_db(
        dbos_path,
        [
            (
                "dbos-done",
                "SUCCESS",
                "<temp>.step_transcribe_asr",
                "asr",
                {"meetingId": MEETING_ID},
            )
        ],
    )

    assert MEETING_ID not in tw.get_dbos_active_progress_map(None)


def test_stage_workflow_maps_to_meeting_by_id_and_args(dbos_path: Path) -> None:
    """Flat stages pass (stage, payload); progress must still find the meeting."""
    _make_dbos_db(
        dbos_path,
        [
            (
                "stage:" + MEETING_ID + ":abcd1234:asr",
                "PENDING",
                "<temp>.run_local_stage",
                "asr",
                None,
            )
        ],
    )
    # Store args in the stage-workflow shape: (stage_name, payload).
    import sqlite3

    ser = DefaultSerializer()
    with sqlite3.connect(dbos_path) as conn:
        conn.execute(
            "UPDATE workflow_input SET inputs = ? WHERE workflow_uuid = ?",
            (
                ser.serialize(
                    {
                        "args": ["asr", {"taskId": "t1", "meetingId": MEETING_ID}],
                        "kwargs": {},
                    }
                ),
                "stage:" + MEETING_ID + ":abcd1234:asr",
            ),
        )
        conn.commit()

    progress = tw.get_dbos_active_progress_map(None)

    assert MEETING_ID in progress
    assert progress[MEETING_ID]["stage"] == "transcribing"
