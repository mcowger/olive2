"""Tests for database migrations, layout, and settings (port of migrations.test.ts)."""

from __future__ import annotations

import json
import time

from sqlalchemy import text

from olive.db import create_db, run_migrations
from olive.layout import date_stamp, meeting_folder_name, meeting_paths, sanitize_filename
from olive.settings import (
    DEFAULT_LOCAL_ASR_MODEL,
    AppConfig,
    clear_app_config_cache,
    load_app_config,
    mask_app_config,
    resolve_paths,
    save_app_config,
)


def test_migrations_apply_idempotently_and_round_trip_every_table() -> None:
    handle = create_db(":memory:")
    now = int(time.time() * 1000)

    # Apply a second time to verify idempotence
    run_migrations(handle.engine)

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
                  :primary_transcript_artifact_id, :primary_summary_artifact_id,
                  :last_error, :created_at, :updated_at
                )
                """
            ),
            {
                "id": "meeting-1",
                "title": "Migration fixture",
                "start_time": now,
                "end_time": now + 60_000,
                "source": "upload",
                "status": "pending",
                "tags": json.dumps(["fixture"]),
                "primary_transcript_artifact_id": None,
                "primary_summary_artifact_id": None,
                "last_error": None,
                "created_at": now,
                "updated_at": now,
            },
        )

        conn.execute(
            text(
                """
                INSERT INTO speakers (
                  id, name, provider_ids, enrolled_at, enrollment_clip_paths, created_at
                ) VALUES (
                  :id, :name, :provider_ids, :enrolled_at, :enrollment_clip_paths, :created_at
                )
                """
            ),
            {
                "id": "speaker-1",
                "name": "Fixture Speaker",
                "provider_ids": json.dumps({"speechmatics": ["voice-1"]}),
                "enrolled_at": now,
                "enrollment_clip_paths": json.dumps(["clips/fixture.wav"]),
                "created_at": now,
            },
        )

        conn.execute(
            text(
                """
                INSERT INTO recordings (
                  id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at
                ) VALUES (
                  :id, :meeting_id, :path, :mime, :duration_ms, :size_bytes, :sha256,
                  :provider, :provider_recording_id, :created_at
                )
                """
            ),
            {
                "id": "recording-1",
                "meeting_id": "meeting-1",
                "path": "audio/recording-1.m4a",
                "mime": "audio/mp4",
                "duration_ms": 60_000,
                "size_bytes": 1234,
                "sha256": "sha256-fixture",
                "provider": "upload",
                "provider_recording_id": "provider-recording-1",
                "created_at": now,
            },
        )

        conn.execute(
            text(
                """
                INSERT INTO artifacts (
                  id, meeting_id, recording_id, kind, provider, format, path, created_at
                ) VALUES (
                  :id, :meeting_id, :recording_id, :kind, :provider, :format, :path, :created_at
                )
                """
            ),
            {
                "id": "artifact-1",
                "meeting_id": "meeting-1",
                "recording_id": "recording-1",
                "kind": "transcript",
                "provider": "upload",
                "format": "json",
                "path": "transcripts/upload.json",
                "created_at": now,
            },
        )

        conn.execute(
            text(
                """
                INSERT INTO meeting_speakers (meeting_id, speaker_id, evidence_artifact_id)
                VALUES (:meeting_id, :speaker_id, :evidence_artifact_id)
                """
            ),
            {
                "meeting_id": "meeting-1",
                "speaker_id": "speaker-1",
                "evidence_artifact_id": "artifact-1",
            },
        )

        conn.execute(
            text(
                """
                INSERT INTO stage_runs (
                  id, meeting_id, stage, status, provider_job_id, attempts,
                  last_error, started_at, finished_at, created_at, updated_at
                ) VALUES (
                  :id, :meeting_id, :stage, :status, :provider_job_id, :attempts,
                  :last_error, :started_at, :finished_at, :created_at, :updated_at
                )
                """
            ),
            {
                "id": "stage-run-1",
                "meeting_id": "meeting-1",
                "stage": "speechmatics_transcribe",
                "status": "pending",
                "provider_job_id": None,
                "attempts": 0,
                "last_error": None,
                "started_at": None,
                "finished_at": None,
                "created_at": now,
                "updated_at": now,
            },
        )

        checkpoint_doc = json.dumps({"turns": [{"startMs": 0, "endMs": 1000}]})
        conn.execute(
            text(
                """
                INSERT INTO stage_checkpoints (meeting_id, stage, run_id, doc, updated_at)
                VALUES ('meeting-1', 'diarize', 'run-1', :doc, :now)
                """
            ),
            {"doc": checkpoint_doc, "now": now},
        )
        checkpoint = conn.execute(text("SELECT * FROM stage_checkpoints")).mappings().one()
        assert dict(checkpoint) == {
            "meeting_id": "meeting-1",
            "stage": "diarize",
            "run_id": "run-1",
            "doc": checkpoint_doc,
            "updated_at": now,
        }

        conn.execute(
            text(
                """
                INSERT INTO plaud_ingest_state (
                  meeting_id, plaud_file_id, first_seen_at, pcs_deadline_at, pcs_resolved
                ) VALUES (
                  :meeting_id, :plaud_file_id, :first_seen_at, :pcs_deadline_at, :pcs_resolved
                )
                """
            ),
            {
                "meeting_id": "meeting-1",
                "plaud_file_id": "plaud-file-1",
                "first_seen_at": now,
                "pcs_deadline_at": now + 86_400_000,
                "pcs_resolved": 0,
            },
        )

        conn.execute(
            text("INSERT INTO sync_state (key, value) VALUES (:key, :value)"),
            {"key": "last-sync", "value": str(now)},
        )

        meeting = conn.execute(text("SELECT * FROM meetings WHERE id = 'meeting-1'")).mappings().first()
        assert dict(meeting or {}) == {
            "id": "meeting-1",
            "title": "Migration fixture",
            "start_time": now,
            "end_time": now + 60_000,
            "source": "upload",
            "status": "pending",
            "tags": '["fixture"]',
            "primary_transcript_artifact_id": None,
            "primary_summary_artifact_id": None,
            "last_error": None,
            "created_at": now,
            "updated_at": now,
        }

        recording = conn.execute(text("SELECT * FROM recordings WHERE id = 'recording-1'")).mappings().first()
        assert recording is not None
        assert recording["meeting_id"] == "meeting-1"
        assert recording["sha256"] == "sha256-fixture"

        artifact = conn.execute(text("SELECT * FROM artifacts WHERE id = 'artifact-1'")).mappings().first()
        assert artifact is not None
        assert artifact["recording_id"] == "recording-1"

        speaker = conn.execute(text("SELECT * FROM speakers WHERE id = 'speaker-1'")).mappings().first()
        assert speaker is not None
        assert speaker["name"] == "Fixture Speaker"

        templates = conn.execute(text("SELECT * FROM templates")).mappings().all()
        assert len(templates) >= 4
        default_tpl = next((t for t in templates if t["is_default"] == 1), None)
        assert default_tpl is not None
        assert default_tpl["name"] == "Executive Summary"

        alembic_ver = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        assert alembic_ver == "0001_baseline"

    handle.engine.dispose()


def test_layout_matches_typescript_exactly() -> None:
    assert sanitize_filename("..//Q1: Design / Review?*\n\t..") == "Q1_Design_Review"
    assert date_stamp(1700000000000) == "2023-11-14"
    assert (
        meeting_folder_name(1700000000000, "Weekly Sync / Planning", "abcdef12-3456-7890")
        == "2023-11-14_Weekly_Sync_Planning__abcdef12"
    )
    paths = meeting_paths("/tmp/meetings", 1700000000000, "Test", "12345678-90ab")
    assert paths.folder.endswith("2023-11-14_Test__12345678")


def test_settings_coerces_qwen_and_masks_hf_token(tmp_path) -> None:
    clear_app_config_cache()
    paths = resolve_paths({"OLIVE_CONFIG_DIR": str(tmp_path)})
    cfg = AppConfig.model_validate(
        {
            "localAsrModel": "ggml-org/Qwen3-ASR-1.7B-GGUF",
            "hfToken": "hf_secret_123456",
        }
    )
    assert cfg.localAsrModel == DEFAULT_LOCAL_ASR_MODEL
    save_app_config(cfg, paths)
    loaded = load_app_config(paths)
    assert loaded.localAsrModel == DEFAULT_LOCAL_ASR_MODEL
    assert loaded.hfToken == "hf_secret_123456"
    masked = mask_app_config(loaded)
    assert masked["hfToken"] == "••••••••3456"
    assert masked["hasHfToken"] is True
