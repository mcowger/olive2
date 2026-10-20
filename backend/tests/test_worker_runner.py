"""LocalTranscriptionWorkerRunner and background concurrency tests (port of worker-runner.test.ts)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from sqlalchemy import text

from olive.app import create_app
from olive.db import create_db
from olive.layout import meeting_paths
from olive.ml import AbortController, LocalTranscriptionWorkerRunner, encode_wav
from olive.services.transcription import TranscriptionService


def _create_dummy_wav(freq_hz: float = 220.0, duration_sec: float = 3.5, sample_rate: int = 16000) -> bytes:
    num_samples = int(sample_rate * duration_sec)
    t = np.arange(num_samples, dtype=np.float32) / float(sample_rate)
    samples = 0.4 * np.sin(2.0 * np.pi * freq_hz * t)
    return encode_wav(samples, sample_rate)


async def test_worker_runner_transcribes_audio_in_background_and_reports_progress(
    tmp_path: Path,
) -> None:
    runner = LocalTranscriptionWorkerRunner({"asrConfig": {"modelId": "mock"}})
    try:
        audio_bytes = _create_dummy_wav(260.0, 3.5)
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(audio_bytes)

        progress_updates: list[Any] = []
        result = await runner.transcribe(
            {
                "audioPath": str(audio_path),
                "modelId": "mock",
                "onProgress": lambda u: progress_updates.append(u),
            }
        )
        assert result is not None
        assert len(result.transcript.segments) > 0
        assert len(progress_updates) > 0
    finally:
        runner.stop()


async def test_worker_runner_handles_cancellation_via_abort_signal(tmp_path: Path) -> None:
    runner = LocalTranscriptionWorkerRunner({"asrConfig": {"modelId": "mock"}})
    try:
        audio_bytes = _create_dummy_wav(260.0, 4.0)
        audio_path = tmp_path / "cancel.wav"
        audio_path.write_bytes(audio_bytes)

        abort_controller = AbortController()
        task = asyncio.create_task(
            runner.transcribe(
                {
                    "audioPath": str(audio_path),
                    "modelId": "mock",
                    "signal": abort_controller.signal,
                }
            )
        )
        await asyncio.sleep(0.02)
        abort_controller.abort()

        with pytest.raises(RuntimeError, match="cancelled"):
            await task
    finally:
        runner.stop()


async def test_http_server_stays_responsive_while_transcription_runs_in_background(
    tmp_path: Path,
) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    runner = LocalTranscriptionWorkerRunner({"asrConfig": {"modelId": "mock"}})
    transcription_service = TranscriptionService(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "localPipeline": runner,
        }
    )
    app = create_app(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "transcriptionService": transcription_service,
        }
    )

    meeting_id = "11111111-2222-3333-4444-555555555555"
    recording_id = "22222222-3333-4444-5555-666666666666"
    start_time = int(time.time() * 1000)
    paths = meeting_paths(meetings_dir, start_time, "Non-Blocking Test Meeting", meeting_id)
    Path(paths.audio_dir).mkdir(parents=True, exist_ok=True)
    audio_bytes = _create_dummy_wav(330.0, 4.0)
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
                  :id, 'Non-Blocking Test Meeting', :now, :end, 'upload', 'processing', '[]',
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {"id": meeting_id, "now": start_time, "end": start_time + 4000},
        )
        conn.execute(
            text(
                """
                INSERT INTO recordings (
                  id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at
                ) VALUES (
                  :rid, :mid, 'audio/sample.wav', 'audio/wav', 4000, :sz,
                  'test-hash', 'upload', NULL, :now
                )
                """
            ),
            {"rid": recording_id, "mid": meeting_id, "sz": len(audio_bytes), "now": start_time},
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
            assert elapsed_ms < 200.0

            transcribe_res = await transcribe_task
            assert transcribe_res.status_code == 200
    finally:
        runner.stop()
        handle.engine.dispose()


async def test_dbos_staged_pipeline_persists_per_stage_runs_and_checkpoints(
    tmp_path: Path,
) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    config_dir = str(tmp_path / "config")
    runner = LocalTranscriptionWorkerRunner(
        {"asrConfig": {"modelId": "mock"}},
        config_dir=config_dir,
        meetings_dir=meetings_dir,
    )
    transcription_service = TranscriptionService(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "configDir": config_dir,
            "localPipeline": runner,
        }
    )

    meeting_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    recording_id = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
    start_time = int(time.time() * 1000)
    paths = meeting_paths(meetings_dir, start_time, "DBOS Staged Pipeline Meeting", meeting_id)
    Path(paths.audio_dir).mkdir(parents=True, exist_ok=True)
    audio_bytes = _create_dummy_wav(280.0, 3.0)
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
                  :id, 'DBOS Staged Pipeline Meeting', :now, :end, 'upload', 'processing', '[]',
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {"id": meeting_id, "now": start_time, "end": start_time + 3000},
        )
        conn.execute(
            text(
                """
                INSERT INTO recordings (
                  id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at
                ) VALUES (
                  :rid, :mid, 'audio/sample.wav', 'audio/wav', 3000, :sz,
                  'dbos-stage-hash', 'upload', NULL, :now
                )
                """
            ),
            {"rid": recording_id, "mid": meeting_id, "sz": len(audio_bytes), "now": start_time},
        )

    try:
        res = await transcription_service.transcribe_meeting(
            meeting_id,
            {"provider": "local", "modelId": "mock"},
        )
        assert res["status"] == "done"

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

        work_root = Path(paths.transcripts_dir) / ".work"
        work_dirs = list(work_root.iterdir())
        assert len(work_dirs) == 1
        checkpoint_dir = work_dirs[0]
        assert (checkpoint_dir / "diarization.json").exists()
        assert (checkpoint_dir / "speaker_clusters.json").exists()
        assert (checkpoint_dir / "raw_asr.json").exists()
        assert not (checkpoint_dir / "16k_mono.wav").exists()
        assert (Path(config_dir) / "dbos.sqlite").exists()
    finally:
        runner.stop()
        handle.engine.dispose()


async def test_post_transcribe_background_returns_202_and_streams_progress(
    tmp_path: Path,
) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    config_dir = str(tmp_path / "config")
    runner = LocalTranscriptionWorkerRunner(
        {"asrConfig": {"modelId": "mock"}},
        config_dir=config_dir,
        meetings_dir=meetings_dir,
    )
    transcription_service = TranscriptionService(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "configDir": config_dir,
            "localPipeline": runner,
        }
    )
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
    paths = meeting_paths(meetings_dir, start_time, "Background 202 Meeting", meeting_id)
    Path(paths.audio_dir).mkdir(parents=True, exist_ok=True)
    audio_bytes = _create_dummy_wav(300.0, 3.0)
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
                  :id, 'Background 202 Meeting', :now, :end, 'upload', 'processing', '[]',
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {"id": meeting_id, "now": start_time, "end": start_time + 3000},
        )
        conn.execute(
            text(
                """
                INSERT INTO recordings (
                  id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at
                ) VALUES (
                  :rid, :mid, 'audio/sample.wav', 'audio/wav', 3000, :sz,
                  'bg-202-hash', 'upload', NULL, :now
                )
                """
            ),
            {"rid": recording_id, "mid": meeting_id, "sz": len(audio_bytes), "now": start_time},
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
            assert body["jobId"].startswith(f"transcribe:{meeting_id}:")
            assert body["alreadyRunning"] is False

            for _ in range(100):
                detail_res = await client.get(f"/api/meetings/{meeting_id}")
                if detail_res.json()["meeting"]["status"] == "ready":
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("Background transcription did not reach ready status")

            debug_res = await client.get("/api/debug/queues")
            assert debug_res.status_code == 200
            debug_data = debug_res.json()
            assert debug_data["summary"]["totalQueues"] == 12
            queue_names = [q["name"] for q in debug_data["queues"]]
            assert queue_names == [
                "pipeline",
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
                "insights",
            ]
            ffmpeg_q = next(q for q in debug_data["queues"] if q["name"] == "ffmpeg")
            assert len(ffmpeg_q["recentJobs"]) >= 1
            assert ffmpeg_q["recentJobs"][0]["name"] == "step_decode_audio"
            assert ffmpeg_q["recentJobs"][0]["meetingTitle"] == "Background 202 Meeting"
    finally:
        runner.stop()
        handle.engine.dispose()
