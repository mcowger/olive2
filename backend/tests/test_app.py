"""HTTP API health and meetings list tests (port of app.test.ts)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import text

from olive.app import create_app
from olive.db import create_db


def test_serves_health_and_lists_meetings() -> None:
    handle = create_db(":memory:")
    app = create_app({"db": handle.engine})
    client = TestClient(app)

    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    assert health_res.json() == {"status": "ok"}

    empty_res = client.get("/api/meetings")
    assert empty_res.status_code == 200
    assert empty_res.json() == {
        "meetings": [],
        "pagination": {"limit": 50, "offset": 0, "total": 0, "hasMore": False, "nextOffset": None},
    }

    now = int(time.time() * 1000)
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, :title, :start_time, :end_time, :source, :status, :tags,
                  NULL, NULL, NULL, :created_at, :updated_at
                )
                """
            ),
            {
                "id": "meeting-api-1",
                "title": "API fixture",
                "start_time": now,
                "end_time": now + 30_000,
                "source": "upload",
                "status": "ready",
                "tags": json.dumps(["test", "api"]),
                "created_at": now,
                "updated_at": now,
            },
        )
        conn.execute(
            text(
                """
                INSERT INTO artifacts (
                  id, meeting_id, recording_id, kind, provider, format, path, created_at
                ) VALUES (
                  :id, :meeting_id, NULL, 'summary', 'fixture', 'md', 'summaries/fixture.md', :created_at
                )
                """
            ),
            {"id": "artifact-api-1", "meeting_id": "meeting-api-1", "created_at": now},
        )
        conn.execute(
            text(
                "UPDATE meetings SET primary_summary_artifact_id = 'artifact-api-1' WHERE id = 'meeting-api-1'"
            )
        )

    pop_res = client.get("/api/meetings")
    assert pop_res.status_code == 200
    assert pop_res.json() == {
        "meetings": [
            {
                "id": "meeting-api-1",
                "title": "API fixture",
                "startTime": now,
                "endTime": now + 30_000,
                "source": "upload",
                "status": "ready",
                "tags": ["test", "api"],
                "primaryTranscriptArtifactId": None,
                "primarySummaryArtifactId": "artifact-api-1",
                "lastError": None,
                "createdAt": now,
                "updatedAt": now,
                "audioStatus": "missing",
                "canRetryAudio": False,
            }
        ],
        "pagination": {"limit": 50, "offset": 0, "total": 1, "hasMore": False, "nextOffset": None},
    }

    search_match = client.get("/api/meetings?search=fixture")
    assert search_match.status_code == 200
    assert len(search_match.json()["meetings"]) == 1

    search_no_match = client.get("/api/meetings?search=nonexistenttermxyz")
    assert search_no_match.status_code == 200
    assert len(search_no_match.json()["meetings"]) == 0

    handle.engine.dispose()


def test_update_meeting_start_time_shifts_duration_and_moves_folder(tmp_path: Path) -> None:
    from olive.layout import ensure_meeting_folder, meeting_folder_name, meeting_paths

    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    # 2026-10-14 ~02:00 UTC; a -5h shift crosses the UTC date boundary.
    start = 1_791_952_800_000
    duration = 60_000
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, :title, :start_time, :end_time, :source, :status, :tags,
                  NULL, NULL, NULL, :created_at, :updated_at
                )
                """
            ),
            {
                "id": "meeting-time-1",
                "title": "Night Meeting",
                "start_time": start,
                "end_time": start + duration,
                "source": "upload",
                "status": "ready",
                "tags": json.dumps([]),
                "created_at": start,
                "updated_at": start,
            },
        )
        paths = ensure_meeting_folder(meeting_paths(meetings_dir, start, "Night Meeting", "meeting-time-1"))
        (Path(paths.audio_dir) / "clip.m4a").write_bytes(b"audio-bytes")

    old_folder = Path(meetings_dir) / meeting_folder_name(start, "Night Meeting", "meeting-time-1")
    assert (old_folder / "audio" / "clip.m4a").read_bytes() == b"audio-bytes"

    app = create_app({"db": handle.engine, "meetingsDir": meetings_dir})
    client = TestClient(app)

    new_start = start - 5 * 3_600_000
    res = client.put("/api/meetings/meeting-time-1/start-time", json={"startTime": new_start})
    assert res.status_code == 200
    body = res.json()["meeting"]
    assert body["startTime"] == new_start
    assert body["endTime"] == new_start + duration

    new_folder = Path(meetings_dir) / meeting_folder_name(new_start, "Night Meeting", "meeting-time-1")
    assert not old_folder.exists()
    assert (new_folder / "audio" / "clip.m4a").read_bytes() == b"audio-bytes"

    # No-op rewrite of the same timestamp keeps the folder in place.
    repeat = client.put("/api/meetings/meeting-time-1/start-time", json={"startTime": new_start})
    assert repeat.status_code == 200
    assert new_folder.exists()

    assert (
        client.put("/api/meetings/meeting-time-1/start-time", json={"startTime": "nope"}).status_code == 400
    )
    assert client.put("/api/meetings/meeting-time-1/start-time", json={}).status_code == 400
    assert client.put("/api/meetings/missing/start-time", json={"startTime": new_start}).status_code == 404

    handle.engine.dispose()
