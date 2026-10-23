"""End-to-end DBOS tests for the flat local transcription stage pipeline.

Local transcription runs as five fire-and-forget stage workflows
(``ffmpeg_decode -> diarize -> {speaker_embed || asr} -> align``), one per queue.
Nothing awaits the pipeline, so these tests observe ``stage_runs`` and meeting
status instead of a returned result.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from sqlalchemy import text

from local_pipeline_helpers import wait_for_meeting_status
from olive.app import create_app
from olive.db import create_db
from olive.layout import meeting_paths
from olive.ml import encode_wav
from olive.services.transcription import TranscriptionService


def _create_dummy_wav(freq_hz: float = 220.0, duration_sec: float = 3.5, sample_rate: int = 16000) -> bytes:
    num_samples = int(sample_rate * duration_sec)
    t = np.arange(num_samples, dtype=np.float32) / float(sample_rate)
    samples = 0.4 * np.sin(2.0 * np.pi * freq_hz * t)
    return encode_wav(samples, sample_rate)


def _seed_meeting(
    handle: Any,
    meetings_dir: str,
    meeting_id: str,
    recording_id: str,
    title: str,
    start_time: int,
    audio_bytes: bytes,
    *,
    sha: str,
) -> Any:
    paths = meeting_paths(meetings_dir, start_time, title, meeting_id)
    Path(paths.audio_dir).mkdir(parents=True, exist_ok=True)
    (Path(paths.folder) / "audio/sample.wav").write_bytes(audio_bytes)

    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, :title, :now, :end, 'upload', 'processing', '[]',
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {"id": meeting_id, "title": title, "now": start_time, "end": start_time + 4000},
        )
        conn.execute(
            text(
                """
                INSERT INTO recordings (
                  id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at
                ) VALUES (
                  :rid, :mid, 'audio/sample.wav', 'audio/wav', 4000, :sz,
                  :sha, 'upload', NULL, :now
                )
                """
            ),
            {"rid": recording_id, "mid": meeting_id, "sz": len(audio_bytes), "sha": sha, "now": start_time},
        )
    return paths


def _service(handle: Any, meetings_dir: str, config_dir: str) -> TranscriptionService:
    return TranscriptionService(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "configDir": config_dir,
            "pipelineOptions": {"asrConfig": {"modelId": "mock"}},
        }
    )


async def test_http_server_stays_responsive_while_transcription_runs_in_background(
    tmp_path: Path,
) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    config_dir = str(tmp_path / "config")
    transcription_service = _service(handle, meetings_dir, config_dir)
    app = create_app(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "configDir": config_dir,
            "transcriptionService": transcription_service,
        }
    )

    meeting_id = "11111111-2222-3333-4444-555555555555"
    recording_id = "22222222-3333-4444-5555-666666666666"
    start_time = int(time.time() * 1000)
    audio_bytes = _create_dummy_wav(330.0, 4.0)
    _seed_meeting(
        handle,
        meetings_dir,
        meeting_id,
        recording_id,
        "Non-Blocking Test Meeting",
        start_time,
        audio_bytes,
        sha="test-hash",
    )

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            transcribe_task = asyncio.create_task(
                client.post(
                    f"/api/meetings/{meeting_id}/transcribe",
                    json={"provider": "local", "modelId": "mock"},
                )
            )
            await asyncio.sleep(0.01)
            t0 = time.perf_counter()
            meetings_res = await client.get("/api/meetings")
            active_res = await client.get("/api/transcriptions/active")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            assert meetings_res.status_code == 200
            assert active_res.status_code == 200
            # Responsive while the pipeline runs. The bound is generous because
            # the suite runs in parallel; the pipeline itself takes far longer.
            assert elapsed_ms < 750.0

            transcribe_res = await transcribe_task
            assert transcribe_res.status_code == 200

            await wait_for_meeting_status(handle.engine, meeting_id, "ready")
    finally:
        handle.engine.dispose()


async def test_flat_pipeline_persists_per_stage_runs_and_checkpoints(
    tmp_path: Path,
) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    config_dir = str(tmp_path / "config")
    transcription_service = _service(handle, meetings_dir, config_dir)

    meeting_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    recording_id = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
    start_time = int(time.time() * 1000)
    audio_bytes = _create_dummy_wav(280.0, 3.0)
    paths = _seed_meeting(
        handle,
        meetings_dir,
        meeting_id,
        recording_id,
        "DBOS Staged Pipeline Meeting",
        start_time,
        audio_bytes,
        sha="dbos-stage-hash",
    )

    try:
        res = await transcription_service.start_transcription(
            meeting_id, {"provider": "local", "modelId": "mock"}
        )
        assert res["status"] == "running"
        assert res["jobId"].startswith(f"stage:{meeting_id}:")

        await wait_for_meeting_status(handle.engine, meeting_id, "ready")

        with handle.engine.connect() as conn:
            stage_rows = (
                conn.execute(
                    text("SELECT stage, status FROM stage_runs WHERE meeting_id = :mid"),
                    {"mid": meeting_id},
                )
                .mappings()
                .all()
            )
        stages_by_name = {r["stage"]: r["status"] for r in stage_rows}
        for expected_stage in (
            "local_transcribe",
            "ffmpeg_decode",
            "diarize",
            "speaker_embed",
            "asr",
            "align",
        ):
            assert stages_by_name.get(expected_stage) == "done"

        # Stage hand-off state lives in the database and is cleaned up when the
        # run ends; nothing intermediate is left in the meeting folder.
        with handle.engine.connect() as conn:
            leftover = conn.execute(
                text("SELECT COUNT(*) AS n FROM stage_checkpoints WHERE meeting_id = :mid"),
                {"mid": meeting_id},
            ).mappings().one()
        assert leftover["n"] == 0
        assert not (Path(paths.transcripts_dir) / ".work").exists()
        assert (Path(config_dir) / "dbos.sqlite").exists()

        # No stage was routed through a shared "pipeline" queue.
        import sqlite3

        with sqlite3.connect(str(Path(config_dir) / "dbos.sqlite"), timeout=1.0) as sconn:
            queue_names = {
                str(row[0])
                for row in sconn.execute(
                    "SELECT DISTINCT queue_name FROM workflow_status WHERE workflow_uuid LIKE ?",
                    (f"stage:{meeting_id}:%",),
                ).fetchall()
            }
        assert queue_names == {"ffmpeg", "diarize", "speaker_embed", "asr", "align"}
    finally:
        handle.engine.dispose()


async def test_post_transcribe_background_returns_202_and_streams_progress(
    tmp_path: Path,
) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    config_dir = str(tmp_path / "config")
    transcription_service = _service(handle, meetings_dir, config_dir)
    app = create_app(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "configDir": config_dir,
            "transcriptionService": transcription_service,
        }
    )

    meeting_id = "cccccccc-dddd-eeee-ffff-000000000000"
    recording_id = "dddddddd-eeee-ffff-0000-111111111111"
    start_time = int(time.time() * 1000)
    audio_bytes = _create_dummy_wav(300.0, 3.0)
    _seed_meeting(
        handle,
        meetings_dir,
        meeting_id,
        recording_id,
        "Background 202 Meeting",
        start_time,
        audio_bytes,
        sha="bg-202-hash",
    )

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            res = await client.post(
                f"/api/meetings/{meeting_id}/transcribe",
                json={"background": True, "provider": "local", "modelId": "mock"},
            )
            assert res.status_code == 202
            body = res.json()
            assert body["status"] == "running"
            assert body["jobId"].startswith(f"stage:{meeting_id}:")
            assert body["alreadyRunning"] is False

            await wait_for_meeting_status(handle.engine, meeting_id, "ready")

            debug_res = await client.get("/api/debug/queues")
            assert debug_res.status_code == 200
            debug_data = debug_res.json()
            assert debug_data["summary"]["totalQueues"] == 10
            queue_names = [q["name"] for q in debug_data["queues"]]
            assert queue_names == [
                "plaud_sync",
                "ffmpeg",
                "diarize",
                "speaker_embed",
                "asr",
                "align",
                "speechmatics",
                "assemblyai",
                "muse",
                "summarize",
            ]
            ffmpeg_q = next(q for q in debug_data["queues"] if q["name"] == "ffmpeg")
            assert len(ffmpeg_q["recentJobs"]) >= 1
            assert ffmpeg_q["recentJobs"][0]["name"] == "stage:ffmpeg_decode"
            assert ffmpeg_q["recentJobs"][0]["meetingTitle"] == "Background 202 Meeting"
    finally:
        handle.engine.dispose()


async def test_second_request_reports_already_running_for_a_live_run(
    tmp_path: Path,
) -> None:
    """A duplicate start while stages are still queued must not restart the run."""
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    config_dir = str(tmp_path / "config")
    transcription_service = _service(handle, meetings_dir, config_dir)

    meeting_id = "99999999-8888-7777-6666-555555555555"
    start_time = int(time.time() * 1000)
    audio_bytes = _create_dummy_wav(200.0, 3.0)
    _seed_meeting(
        handle,
        meetings_dir,
        meeting_id,
        "rec-already-running",
        "Already Running Meeting",
        start_time,
        audio_bytes,
        sha="already-running-hash",
    )

    try:
        first = await transcription_service.start_transcription(
            meeting_id, {"provider": "local", "modelId": "mock"}
        )
        assert first["alreadyRunning"] is False
        second = await transcription_service.start_transcription(
            meeting_id, {"provider": "local", "modelId": "mock"}
        )
        assert second["alreadyRunning"] is True
        assert second["jobId"] == first["jobId"]

        await wait_for_meeting_status(handle.engine, meeting_id, "ready")
    finally:
        handle.engine.dispose()


@pytest.mark.parametrize("_case", [0])
def test_module_imports_cleanly(_case: int) -> None:
    """Guard against accidental reinstatement of the removed nested orchestration."""
    from olive.workflows import transcription_workflow as tw

    assert not hasattr(tw, "run_staged_local_transcription")
    assert not hasattr(tw, "local_transcription_workflow")
    assert not hasattr(tw, "run_transcription_driver")


async def test_finalize_is_idempotent(tmp_path: Path) -> None:
    """The align stage's finalize may run twice (crash/retry); artifacts stay single."""
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    config_dir = str(tmp_path / "config")
    transcription_service = _service(handle, meetings_dir, config_dir)

    meeting_id = "12121212-3434-5656-7878-909090909090"
    start_time = int(time.time() * 1000)
    audio_bytes = _create_dummy_wav(150.0, 2.0)
    _seed_meeting(
        handle,
        meetings_dir,
        meeting_id,
        "rec-finalize-idempotent",
        "Finalize Idempotent",
        start_time,
        audio_bytes,
        sha="finalize-hash",
    )
    result = {
        "transcript": {
            "segments": [
                {"startMs": 0, "endMs": 1000, "speaker": "Speaker 1", "text": "Hello there."}
            ],
            "language": "en",
            "durationMs": 2000,
        },
        "discoveredSpeakers": [],
    }

    try:
        await transcription_service.complete_local_transcript(meeting_id, result, "wf-1")
        await transcription_service.complete_local_transcript(meeting_id, result, "wf-1")

        with handle.engine.connect() as conn:
            artifacts = (
                conn.execute(
                    text(
                        "SELECT kind, format, provider FROM artifacts WHERE meeting_id = :mid ORDER BY format"
                    ),
                    {"mid": meeting_id},
                )
                .mappings()
                .all()
            )
        assert [(a["kind"], a["format"], a["provider"]) for a in artifacts] == [
            ("transcript", "json", "local"),
            ("transcript", "txt", "local"),
        ]
        with handle.engine.connect() as conn:
            meeting = (
                conn.execute(
                    text("SELECT status, primary_transcript_artifact_id FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .one()
            )
            stage = (
                conn.execute(
                    text("SELECT status FROM stage_runs WHERE meeting_id = :mid AND stage = :stage"),
                    {"mid": meeting_id, "stage": "local_transcribe"},
                )
                .mappings()
                .one()
            )
        assert meeting["status"] == "ready"
        assert stage["status"] == "done"
    finally:
        handle.engine.dispose()


async def test_streaming_transcribe_follows_the_durable_run_to_completion(
    tmp_path: Path,
) -> None:
    """POST ?stream=true must not report "done" the instant the run is enqueued."""
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    config_dir = str(tmp_path / "config")
    transcription_service = _service(handle, meetings_dir, config_dir)
    app = create_app(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "configDir": config_dir,
            "transcriptionService": transcription_service,
        }
    )

    meeting_id = "abcdefab-cdef-abcd-efab-cdefabcdefab"
    start_time = int(time.time() * 1000)
    _seed_meeting(
        handle,
        meetings_dir,
        meeting_id,
        "rec-streaming",
        "Streaming Meeting",
        start_time,
        _create_dummy_wav(240.0, 3.0),
        sha="streaming-hash",
    )

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            events: list[str] = []
            async with client.stream(
                "POST",
                f"/api/meetings/{meeting_id}/transcribe",
                json={"stream": True, "provider": "local", "modelId": "mock"},
            ) as resp:
                assert resp.status_code == 200
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        events.append(line.split(":", 1)[1].strip())
            assert "result" in events, events
            assert events[-1] == "result", events
        await wait_for_meeting_status(handle.engine, meeting_id, "ready")
    finally:
        handle.engine.dispose()
