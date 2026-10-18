"""Bulk re-transcribe API tests (POST /api/meetings/bulk-transcribe)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from olive.app import create_app
from olive.db import create_db
from olive.layout import meeting_paths
from olive.providers.speechmatics import SpeechmaticsClient
from olive.services.transcription import TranscriptionService
from olive.workflows.queues import QUEUE_ASSEMBLYAI, QUEUE_DRIVER, QUEUE_MUSE, QUEUE_SPEECHMATICS

AUDIO_BYTES = b"fake audio bytes"


def _create_mock_speechmatics_client():
    class _MockBatch:
        async def createTranscriptionJob(self, *_args: Any) -> dict[str, str]:
            return {"id": "sm-job-bulk"}

        async def getJob(self, jid: str) -> dict[str, Any]:
            return {
                "job": {
                    "id": jid,
                    "status": "done",
                    "created_at": "2026-08-16T12:00:00Z",
                    "data_name": "test.m4a",
                    "duration": 5.0,
                }
            }

        async def getJobResult(self, _jid: str, fmt: str = "json-v2") -> Any:
            if fmt == "json-v2":
                return {
                    "format": "2.0",
                    "job": {"id": "sm-job-bulk"},
                    "results": [
                        {
                            "type": "word",
                            "start_time": 0.1,
                            "end_time": 0.5,
                            "alternatives": [{"content": "Hello", "confidence": 0.9}],
                        }
                    ],
                }
            return "Hello."

        async def deleteJob(self, jid: str) -> dict[str, Any]:
            return {"job": {"id": jid, "status": "deleted", "created_at": "", "data_name": ""}}

    return SpeechmaticsClient({"apiKey": "test-key", "batchClient": _MockBatch()})


def _seed_meeting(handle: Any, meetings_dir: str, meeting_id: str, title: str, start: int) -> None:
    paths = meeting_paths(meetings_dir, start, title, meeting_id)
    Path(paths.audio_dir).mkdir(parents=True, exist_ok=True)
    (Path(paths.folder) / "audio/test.m4a").write_bytes(AUDIO_BYTES)
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, :title, :now, :end, 'upload', 'ready', '[]',
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {"id": meeting_id, "title": title, "now": start, "end": start + 5000},
        )
        conn.execute(
            text(
                """
                INSERT INTO recordings (
                  id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at
                ) VALUES (
                  :rid, :mid, 'audio/test.m4a', 'audio/mp4', 5000, :sz,
                  :sha, 'upload', NULL, :now
                )
                """
            ),
            {
                "rid": f"rec-{meeting_id}",
                "mid": meeting_id,
                "sz": len(AUDIO_BYTES),
                "sha": f"sha-{meeting_id}",
                "now": start,
            },
        )


def _patch_transcription_enqueue_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace real DBOS enqueue with a no-op so API tests don't leak background work.

    Without this, ``start_transcription`` enqueues a durable workflow that keeps
    running after the test disposes its ``:memory:`` engine; the orphaned job
    then fails with ``no such table: meetings`` once the in-memory DB is gone.
    """
    import olive.services.transcription as tx_mod

    async def _fake_ensure(**_kwargs: Any) -> str:
        return "sqlite:///fake-dbos.sqlite"

    async def _fake_enqueue(queue_name: str, fn: Any, **kwargs: Any) -> str:
        return kwargs.get("workflow_id") or "handle-fake"

    monkeypatch.setattr(tx_mod, "ensure_dbos_launched_async", _fake_ensure)
    monkeypatch.setattr(tx_mod, "enqueue_driver_job", _fake_enqueue)


def _make_client(tmp_path: Path):
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    now = 1_700_000_000_000
    _seed_meeting(handle, meetings_dir, "m-bulk-1", "Bulk Meeting One", now)
    _seed_meeting(handle, meetings_dir, "m-bulk-2", "Bulk Meeting Two", now + 86_400_000)
    service = TranscriptionService(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "speechmaticsClient": _create_mock_speechmatics_client(),
        }
    )
    app = create_app(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "transcriptionService": service,
        }
    )
    return TestClient(app), handle


def test_bulk_transcribe_explicit_ids_queues_background_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_transcription_enqueue_noop(monkeypatch)
    client, handle = _make_client(tmp_path)
    try:
        res = client.post(
            "/api/meetings/bulk-transcribe",
            json={"meetingIds": ["m-bulk-1", "m-bulk-2"], "provider": "speechmatics", "force": True},
        )
        assert res.status_code == 202, res.text
        data = res.json()
        assert data["provider"] == "speechmatics"
        assert data["total"] == 2
        assert data["queued"] == 2
        assert data["skipped"] == 0
        assert {r["meetingId"] for r in data["results"]} == {"m-bulk-1", "m-bulk-2"}
    finally:
        handle.engine.dispose()


def test_bulk_transcribe_filters_by_date_and_rejects_bad_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_transcription_enqueue_noop(monkeypatch)
    client, handle = _make_client(tmp_path)
    try:
        bad = client.post("/api/meetings/bulk-transcribe", json={"provider": "google-stt"})
        assert bad.status_code == 400

        # Date filter matching only the second meeting
        res = client.post(
            "/api/meetings/bulk-transcribe",
            json={
                "provider": "speechmatics",
                "filters": {"startTime": 1_700_000_000_001},
            },
        )
        assert res.status_code == 202, res.text
        data = res.json()
        assert data["total"] == 1
        assert data["results"][0]["meetingId"] == "m-bulk-2"

        # Unknown meeting id is reported as skipped, not a 500
        res2 = client.post(
            "/api/meetings/bulk-transcribe",
            json={"meetingIds": ["m-bulk-1", "m-missing"], "provider": "speechmatics"},
        )
        assert res2.status_code == 202, res2.text
        data2 = res2.json()
        assert data2["total"] == 2
        assert data2["skipped"] == 1
    finally:
        handle.engine.dispose()


def test_bulk_transcribe_requires_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transcription_enqueue_noop(monkeypatch)
    client, handle = _make_client(tmp_path)
    try:
        res = client.post(
            "/api/meetings/bulk-transcribe",
            json={"provider": "speechmatics", "filters": {"search": "no-such-meeting-xyz"}},
        )
        assert res.status_code == 400
        assert "No meetings matched" in res.json()["error"]
    finally:
        handle.engine.dispose()


@pytest.mark.parametrize(
    ("provider", "expected_queue"),
    [
        ("local", QUEUE_DRIVER),
        ("speechmatics", QUEUE_SPEECHMATICS),
        ("assemblyai", QUEUE_ASSEMBLYAI),
        ("muse", QUEUE_MUSE),
        ("local:cohere", QUEUE_DRIVER),
    ],
)
async def test_start_transcription_enqueues_on_provider_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str, expected_queue: str
) -> None:
    """The DBOS job queue is the SOLE path for background transcription work.

    start_transcription must enqueue onto the provider's named queue with a
    per-meeting deduplication id — never fire a bare asyncio task. Drivers
    go through enqueue_driver_job (workflow context) because DBOS forbids
    child-workflow enqueue from inside a step.
    """
    import olive.services.transcription as tx_mod

    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    _seed_meeting(handle, meetings_dir, "m-q-1", "Queue Meeting", 1_700_000_000_000)
    service = TranscriptionService({"db": handle.engine, "meetingsDir": meetings_dir})

    calls: list[dict[str, Any]] = []

    async def fake_ensure(**_kwargs: Any) -> str:
        return "sqlite:///fake-dbos.sqlite"

    async def fake_enqueue(queue_name: str, fn: Any, **kwargs: Any) -> str:
        calls.append({"queue": queue_name, **kwargs})
        return "handle-1"

    monkeypatch.setattr(tx_mod, "ensure_dbos_launched_async", fake_ensure)
    monkeypatch.setattr(tx_mod, "enqueue_driver_job", fake_enqueue)
    try:
        res = await service.start_transcription("m-q-1", {"provider": provider})
        assert res["status"] == "running"
        assert res["jobId"].startswith("transcribe:m-q-1:")
        assert len(calls) == 1
        assert calls[0]["queue"] == expected_queue
        assert calls[0]["deduplication_id"] == "transcribe:m-q-1"
        assert calls[0]["meeting_id"] == "m-q-1"
        assert calls[0]["workflow_id"] == res["jobId"]
        # Queued progress is emitted synchronously at enqueue time.
        progress = service.get_transcription_progress("m-q-1")
        assert progress is not None and progress.stage == "queued"
    finally:
        handle.engine.dispose()


async def test_bulk_transcribe_registers_durable_workflows(tmp_path: Path) -> None:
    """End-to-end: bulk enqueue leaves durable, meeting-tagged workflows."""
    from dbos import DBOS

    from olive.workflows import cancel_dbos_workflows_for_meeting

    # File-backed DB: background drivers hold this engine past the test body,
    # and a fresh :memory: connection after dispose would have no tables.
    handle = create_db(str(tmp_path / "olive.sqlite"))
    meetings_dir = str(tmp_path / "meetings")
    _seed_meeting(handle, meetings_dir, "m-bulk-1", "Bulk Meeting One", 1_700_000_000_000)
    _seed_meeting(handle, meetings_dir, "m-bulk-2", "Bulk Meeting Two", 1_700_000_000_001)
    service = TranscriptionService(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "speechmaticsClient": _create_mock_speechmatics_client(),
        }
    )
    try:
        res = await service.start_bulk_transcription(
            {"meetingIds": ["m-bulk-1", "m-bulk-2"], "provider": "speechmatics"}
        )
        assert res["total"] == 2
        assert res["queued"] == 2
        workflows = await DBOS.list_workflows_async(limit=50, load_input=True)
        ids = {str(w.workflow_id) for w in workflows}
        assert any(i.startswith("transcribe:m-bulk-1:") for i in ids)
        assert any(i.startswith("transcribe:m-bulk-2:") for i in ids)
    finally:
        for mid in ("m-bulk-1", "m-bulk-2"):
            try:
                await cancel_dbos_workflows_for_meeting(mid, db=handle.engine, meetings_dir=meetings_dir)
            except Exception:
                pass
        handle.engine.dispose()
