"""Meeting deletion tests (port of meetings-delete.test.ts)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from sqlalchemy import text

from olive.db import create_db
from olive.layout import meeting_paths
from olive.services.meetings import delete_meeting, get_meeting


def _seed_full_meeting(engine, meetings_dir: str) -> tuple[str, Path]:
    now = int(time.time() * 1000)
    meeting_id = "meeting-delete-1"
    title = "Meeting To Delete"

    paths = meeting_paths(meetings_dir, now, title, meeting_id)
    Path(paths.transcripts_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.audio_dir).mkdir(parents=True, exist_ok=True)

    transcript_rel = "transcripts/transcript.json"
    (Path(paths.folder) / transcript_rel).write_text(json.dumps({"segments": []}), encoding="utf-8")
    audio_rel = "audio/recording.wav"
    (Path(paths.folder) / audio_rel).write_text("fake-audio", encoding="utf-8")

    with engine.begin() as conn:
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
            {"id": meeting_id, "title": title, "now": now, "end": now + 600_000},
        )
        recording_id = "recording-delete-1"
        conn.execute(
            text(
                """
                INSERT INTO recordings (
                  id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at
                ) VALUES (
                  :id, :mid, :path, 'audio/wav', 600000, 10, 'deadbeef',
                  'upload', NULL, :now
                )
                """
            ),
            {"id": recording_id, "mid": meeting_id, "path": audio_rel, "now": now},
        )
        transcript_artifact_id = "artifact-transcript-delete-1"
        conn.execute(
            text(
                """
                INSERT INTO artifacts (
                  id, meeting_id, recording_id, kind, provider, format, path, created_at
                ) VALUES (
                  :id, :mid, :rid, 'transcript', 'speechmatics', 'json', :path, :now
                )
                """
            ),
            {
                "id": transcript_artifact_id,
                "mid": meeting_id,
                "rid": recording_id,
                "path": transcript_rel,
                "now": now,
            },
        )
        conn.execute(
            text("UPDATE meetings SET primary_transcript_artifact_id = :aid WHERE id = :mid"),
            {"aid": transcript_artifact_id, "mid": meeting_id},
        )
        speaker_id = "speaker-delete-1"
        conn.execute(
            text(
                """
                INSERT INTO speakers (
                  id, name, provider_ids, enrolled_at, enrollment_clip_paths, created_at
                ) VALUES (:id, 'Alice', '{}', NULL, '[]', :now)
                """
            ),
            {"id": speaker_id, "now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO meeting_speakers (meeting_id, speaker_id, evidence_artifact_id)
                VALUES (:mid, :sid, :aid)
                """
            ),
            {"mid": meeting_id, "sid": speaker_id, "aid": transcript_artifact_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO stage_runs (
                  id, meeting_id, stage, status, provider_job_id, attempts,
                  last_error, started_at, finished_at, created_at, updated_at
                ) VALUES (
                  'stage-run-delete-1', :mid, 'transcribe', 'done', NULL, 1,
                  NULL, :now, :now, :now, :now
                )
                """
            ),
            {"mid": meeting_id, "now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO stage_checkpoints (meeting_id, stage, run_id, doc, updated_at)
                VALUES (:mid, 'diarize', 'run-delete', '{}', :now)
                """
            ),
            {"mid": meeting_id, "now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO chat_messages (
                  id, meeting_id, role, content, provider, model, usage, created_at
                ) VALUES ('chat-delete-1', :mid, 'user', 'hello', NULL, NULL, NULL, :now)
                """
            ),
            {"mid": meeting_id, "now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO logs (
                  id, level, category, message, meeting_id, details, created_at
                ) VALUES ('log-delete-1', 'info', 'test', 'seeded log', :mid, NULL, :now)
                """
            ),
            {"mid": meeting_id, "now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO usage_events (
                  id, meeting_id, kind, provider, model, input_tokens,
                  output_tokens, total_tokens, audio_seconds, cost_usd,
                  source, detail, created_at
                ) VALUES (
                  'usage-delete-1', :mid, 'chat', 'test-provider', 'test-model',
                  10, 5, 15, NULL, 0.000001,
                  'usage-delete-source-1', NULL, :now
                )
                """
            ),
            {"mid": meeting_id, "now": now},
        )

    return meeting_id, Path(paths.folder)


def test_delete_meeting_removes_meeting_related_rows_and_files(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    meeting_id, folder = _seed_full_meeting(handle.engine, meetings_dir)

    assert folder.exists()
    assert delete_meeting(handle.engine, meeting_id, meetings_dir) is True
    assert get_meeting(handle.engine, meeting_id, meetings_dir) is None
    assert not folder.exists()

    with handle.engine.connect() as conn:
        for tbl in [
            "recordings",
            "artifacts",
            "meeting_speakers",
            "stage_runs",
            "stage_checkpoints",
            "chat_messages",
            "logs",
        ]:
            rem = conn.execute(
                text(f"SELECT * FROM {tbl} WHERE meeting_id = :mid"),
                {"mid": meeting_id},
            ).fetchall()
            assert len(rem) == 0

        # Usage history is detached (meeting_id nulled) so global spend
        # totals survive, but per-meeting attribution is gone.
        usage_rows = conn.execute(
            text("SELECT * FROM usage_events WHERE id = 'usage-delete-1'"),
        ).fetchall()
        assert len(usage_rows) == 1
        assert usage_rows[0]._mapping["meeting_id"] is None

        speaker = conn.execute(text("SELECT * FROM speakers WHERE id = 'speaker-delete-1'")).first()
        assert speaker is not None

    handle.engine.dispose()


def test_delete_meeting_returns_false_when_not_found(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    assert delete_meeting(handle.engine, "does-not-exist", str(tmp_path)) is False
    handle.engine.dispose()
