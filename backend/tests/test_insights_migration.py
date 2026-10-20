"""Tests for the Olive Insights schema (Phase 0): tables, cascades, and backup coverage."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from olive.db import create_db, run_migrations, set_db_for_tests
from olive.services import split as split_module
from olive.services.backup import BackupService
from olive.services.meetings import delete_meeting
from olive.services.speakers import SpeakerService
from olive.settings import AppConfig

NOW = 1_700_000_000_000
INSIGHTS_TABLES = ("meeting_speaker_signals", "meeting_moments", "transcript_imports", "speaker_aliases")


def _insert_meeting(conn, meeting_id: str, *, title: str = "Insights fixture", start: int = NOW) -> None:
    conn.execute(
        text(
            """
            INSERT INTO meetings (
              id, title, start_time, end_time, source, status, tags, created_at, updated_at
            ) VALUES (:id, :title, :start, :end, 'upload', 'ready', '[]', :now, :now)
            """
        ),
        {"id": meeting_id, "title": title, "start": start, "end": start + 600_000, "now": NOW},
    )


def _insert_speaker(conn, speaker_id: str, name: str) -> None:
    conn.execute(
        text("INSERT INTO speakers (id, name, created_at) VALUES (:id, :name, :now)"),
        {"id": speaker_id, "name": name, "now": NOW},
    )


def _insert_signal(conn, signal_id: str, meeting_id: str, speaker_id: str | None, label: str = "S1") -> None:
    conn.execute(
        text(
            """
            INSERT INTO meeting_speaker_signals (
              id, meeting_id, speaker_id, speaker_label, identity_source, identity_confidence,
              dimensions, talk_ms, turn_count, interrupt_count, stance, transcript_hash,
              prompt_version, model, provider, transcript_artifact_id, created_at
            ) VALUES (
              :id, :mid, :sid, :label, 'voiceprint', 0.9,
              '{"warmth": 0.4}', 1000, 3, 1, 'supportive', 'hash', 'v1', 'm', 'p', 'art-1', :now
            )
            """
        ),
        {"id": signal_id, "mid": meeting_id, "sid": speaker_id, "label": label, "now": NOW},
    )


def _insert_moment(
    conn,
    moment_id: str,
    meeting_id: str,
    speaker_id: str | None,
    toward_speaker_id: str | None = None,
) -> None:
    conn.execute(
        text(
            """
            INSERT INTO meeting_moments (
              id, meeting_id, speaker_id, speaker_label, toward_speaker_id, toward_label,
              start_ms, end_ms, turn_start_idx, turn_end_idx, event_type, valence, framework,
              dimension, severity, quote, rationale, guidance, verified, identity_source,
              identity_confidence, transcript_hash, prompt_version, created_at
            ) VALUES (
              :id, :mid, :sid, 'S1', :tid, 'S2',
              1000, 2000, 1, 2, 'interruption', 'negative', 'fw',
              'respect', 0.5, 'quote', 'why', 'try this', 1, 'voiceprint',
              0.8, 'hash', 'v1', :now
            )
            """
        ),
        {"id": moment_id, "mid": meeting_id, "sid": speaker_id, "tid": toward_speaker_id, "now": NOW},
    )


def _insert_alias(conn, alias_id: str, alias_norm: str, speaker_id: str | None) -> None:
    conn.execute(
        text(
            "INSERT INTO speaker_aliases (id, alias_norm, speaker_id, kind, confirmed, created_at) "
            "VALUES (:id, :alias, :sid, 'nickname', 1, :now)"
        ),
        {"id": alias_id, "alias": alias_norm, "sid": speaker_id, "now": NOW},
    )


def _count(conn, table: str) -> int:
    return int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())


def test_insights_tables_columns_and_indexes_exist_and_are_idempotent() -> None:
    handle = create_db(":memory:")
    run_migrations(handle.engine)  # second application must be a no-op

    with handle.engine.connect() as conn:
        tables = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        indexes = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='index'"))}
        ms_columns = {r[1] for r in conn.execute(text("PRAGMA table_info(meeting_speakers)"))}
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalars().all()

    assert set(INSIGHTS_TABLES) <= tables
    assert {"identity_source", "identity_confidence"} <= ms_columns
    assert {
        "idx_signals_meeting",
        "idx_signals_speaker",
        "idx_moments_meeting",
        "idx_moments_speaker",
        "idx_moments_event",
        "idx_moments_start",
        "idx_speaker_aliases",
    } <= indexes
    # No FTS or per-turn exchange tables belong to this phase.
    assert not any("fts" in t.lower() or t == "exchanges" for t in tables)
    # Schema is built from the in-code statement lists; no new alembic revision.
    assert version == ["0001_baseline"]


def test_additive_columns_are_added_to_legacy_meeting_speakers() -> None:
    from olive.db.migrations import ADDITIVE_COLUMNS

    assert ("meeting_speakers", "identity_source", "TEXT") in ADDITIVE_COLUMNS
    assert ("meeting_speakers", "identity_confidence", "REAL") in ADDITIVE_COLUMNS

    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1")
        _insert_speaker(conn, "sp1", "Alice")
        conn.execute(
            text(
                "INSERT INTO meeting_speakers (meeting_id, speaker_id, identity_source, identity_confidence) "
                "VALUES ('m1', 'sp1', 'voiceprint', 0.75)"
            )
        )
        row = conn.execute(text("SELECT identity_source, identity_confidence FROM meeting_speakers")).one()
    assert tuple(row) == ("voiceprint", 0.75)


def test_every_insights_table_round_trips_and_applies_defaults() -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1")
        _insert_speaker(conn, "sp1", "Alice")
        _insert_signal(conn, "sig1", "m1", "sp1")
        _insert_moment(conn, "mom1", "m1", "sp1", "sp1")
        _insert_alias(conn, "al1", "ali", "sp1")
        conn.execute(
            text(
                "INSERT INTO transcript_imports (id, content_sha256, meeting_id, source_format, subtype, created_at) "
                "VALUES ('imp1', 'sha-1', 'm1', 'vtt', 'zoom', :now)"
            ),
            {"now": NOW},
        )
        # Defaults: only required columns supplied.
        conn.execute(
            text(
                "INSERT INTO meeting_speaker_signals (id, meeting_id, speaker_label, created_at) "
                "VALUES ('sig2', 'm1', 'S2', :now)"
            ),
            {"now": NOW},
        )
        conn.execute(
            text(
                "INSERT INTO meeting_moments (id, meeting_id, start_ms, event_type, valence, quote, created_at) "
                "VALUES ('mom2', 'm1', 5, 'praise', 'positive', 'nice', :now)"
            ),
            {"now": NOW},
        )
        conn.execute(
            text("INSERT INTO speaker_aliases (id, alias_norm, created_at) VALUES ('al2', 'bob', :now)"),
            {"now": NOW},
        )

    with handle.engine.connect() as conn:
        sig = conn.execute(text("SELECT * FROM meeting_speaker_signals WHERE id='sig1'")).mappings().one()
        assert sig["dimensions"] == '{"warmth": 0.4}'
        assert sig["talk_ms"] == 1000 and sig["identity_confidence"] == 0.9
        sig2 = conn.execute(text("SELECT * FROM meeting_speaker_signals WHERE id='sig2'")).mappings().one()
        assert (sig2["dimensions"], sig2["talk_ms"], sig2["turn_count"], sig2["interrupt_count"]) == (
            "{}",
            0,
            0,
            0,
        )
        assert sig2["speaker_id"] is None
        mom = conn.execute(text("SELECT * FROM meeting_moments WHERE id='mom1'")).mappings().one()
        assert mom["verified"] == 1 and mom["toward_speaker_id"] == "sp1" and mom["quote"] == "quote"
        mom2 = conn.execute(text("SELECT * FROM meeting_moments WHERE id='mom2'")).mappings().one()
        assert mom2["verified"] == 0 and mom2["speaker_id"] is None
        alias2 = conn.execute(text("SELECT * FROM speaker_aliases WHERE id='al2'")).mappings().one()
        assert alias2["confirmed"] == 0 and alias2["speaker_id"] is None
        for table in INSIGHTS_TABLES:
            assert _count(conn, table) >= 1


def test_orm_models_are_exported_and_match_tables() -> None:
    from olive.db import (
        MeetingMomentModel,
        MeetingSpeakerSignalModel,
        SpeakerAliasModel,
        TranscriptImportModel,
    )

    assert MeetingSpeakerSignalModel.__tablename__ == "meeting_speaker_signals"
    assert MeetingMomentModel.__tablename__ == "meeting_moments"
    assert TranscriptImportModel.__tablename__ == "transcript_imports"
    assert SpeakerAliasModel.__tablename__ == "speaker_aliases"

    handle = create_db(":memory:")
    with handle.engine.connect() as conn:
        for model in (
            MeetingSpeakerSignalModel,
            MeetingMomentModel,
            TranscriptImportModel,
            SpeakerAliasModel,
        ):
            live = {r[1] for r in conn.execute(text(f"PRAGMA table_info({model.__tablename__})"))}
            assert {c.name for c in model.__table__.columns} == live


def test_foreign_keys_are_enforced_on_insights_tables() -> None:
    handle = create_db(":memory:")
    with pytest.raises(IntegrityError), handle.engine.begin() as conn:
        _insert_signal(conn, "sig1", "missing-meeting", None)


def test_transcript_imports_content_sha256_is_unique() -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1")
        _insert_meeting(conn, "m2")
        conn.execute(
            text(
                "INSERT INTO transcript_imports (id, content_sha256, meeting_id, created_at) "
                "VALUES ('i1', 'same', 'm1', :now)"
            ),
            {"now": NOW},
        )
    with pytest.raises(IntegrityError), handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO transcript_imports (id, content_sha256, meeting_id, created_at) "
                "VALUES ('i2', 'same', 'm2', :now)"
            ),
            {"now": NOW},
        )


def test_speaker_aliases_unique_per_alias_and_speaker() -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _insert_speaker(conn, "sp1", "Alice")
        _insert_speaker(conn, "sp2", "Alicia")
        _insert_alias(conn, "a1", "ali", "sp1")
        _insert_alias(conn, "a2", "ali", "sp2")  # same alias, different speaker: allowed
    with pytest.raises(IntegrityError), handle.engine.begin() as conn:
        _insert_alias(conn, "a3", "ali", "sp1")


def test_delete_meeting_removes_insights_rows_without_integrity_error(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1")
        _insert_meeting(conn, "m2")
        _insert_speaker(conn, "sp1", "Alice")
        _insert_signal(conn, "sig1", "m1", "sp1")
        _insert_signal(conn, "sig2", "m2", "sp1")
        _insert_moment(conn, "mom1", "m1", "sp1")
        _insert_moment(conn, "mom2", "m2", "sp1")
        for mid, iid in (("m1", "i1"), ("m2", "i2")):
            conn.execute(
                text(
                    "INSERT INTO transcript_imports (id, content_sha256, meeting_id, created_at) "
                    "VALUES (:i, :s, :m, :now)"
                ),
                {"i": iid, "s": f"sha-{iid}", "m": mid, "now": NOW},
            )

    assert delete_meeting(handle.engine, "m1", str(tmp_path)) is True

    with handle.engine.connect() as conn:
        for table in ("meeting_speaker_signals", "meeting_moments", "transcript_imports"):
            ids = conn.execute(text(f"SELECT meeting_id FROM {table}")).scalars().all()
            assert ids == ["m2"], table
        assert _count(conn, "speakers") == 1


async def test_merge_speakers_repoints_insights_to_survivor() -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1")
        _insert_speaker(conn, "src", "Al")
        _insert_speaker(conn, "dst", "Alice")
        _insert_speaker(conn, "other", "Bob")
        _insert_signal(conn, "sig1", "m1", "src")
        _insert_moment(conn, "mom1", "m1", "src", "other")
        _insert_moment(conn, "mom2", "m1", "other", "src")
        _insert_alias(conn, "al1", "al", "src")
        _insert_alias(conn, "al2", "dup", "src")
        _insert_alias(conn, "al3", "dup", "dst")  # would collide on (alias_norm, speaker_id)

    service = SpeakerService(db=handle, config_dir="", embedding_extractor=object())  # type: ignore[arg-type]
    await service.merge_speakers("src", "dst")

    with handle.engine.connect() as conn:
        assert conn.execute(text("SELECT speaker_id FROM meeting_speaker_signals")).scalars().all() == ["dst"]
        moments = {
            r["id"]: (r["speaker_id"], r["toward_speaker_id"])
            for r in conn.execute(text("SELECT * FROM meeting_moments")).mappings()
        }
        assert moments == {"mom1": ("dst", "other"), "mom2": ("other", "dst")}
        aliases = sorted(
            (r["alias_norm"], r["speaker_id"])
            for r in conn.execute(text("SELECT * FROM speaker_aliases")).mappings()
        )
        assert aliases == [("al", "dst"), ("dup", "dst")]
        assert conn.execute(text("SELECT id FROM speakers WHERE id='src'")).first() is None


async def test_delete_speaker_nulls_insights_references() -> None:
    handle = create_db(":memory:")
    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1")
        _insert_speaker(conn, "gone", "Al")
        _insert_speaker(conn, "other", "Bob")
        _insert_signal(conn, "sig1", "m1", "gone")
        _insert_moment(conn, "mom1", "m1", "gone", "other")
        _insert_moment(conn, "mom2", "m1", "other", "gone")
        _insert_alias(conn, "al1", "al", "gone")

    service = SpeakerService(db=handle, config_dir="", embedding_extractor=object())  # type: ignore[arg-type]
    await service.delete_speaker("gone")

    with handle.engine.connect() as conn:
        assert conn.execute(text("SELECT speaker_id FROM meeting_speaker_signals")).scalars().all() == [None]
        moments = {
            r["id"]: (r["speaker_id"], r["toward_speaker_id"])
            for r in conn.execute(text("SELECT * FROM meeting_moments")).mappings()
        }
        assert moments == {"mom1": (None, "other"), "mom2": ("other", None)}
        assert conn.execute(text("SELECT speaker_id FROM speaker_aliases")).scalars().all() == [None]
        # Rows survive: the signal keeps its label so it can be re-attributed later.
        assert _count(conn, "meeting_speaker_signals") == 1


def test_split_meeting_deletes_stale_signals_and_moments(tmp_path: Path, monkeypatch) -> None:
    from olive.layout import meeting_paths

    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    paths = meeting_paths(meetings_dir, NOW, "Split me", "m1")
    audio = Path(paths.folder) / "audio" / "rec.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"RIFFfakeaudio")

    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1", title="Split me")
        _insert_meeting(conn, "other", title="Untouched")
        _insert_speaker(conn, "sp1", "Alice")
        conn.execute(
            text(
                "INSERT INTO recordings (id, meeting_id, path, mime, duration_ms, size_bytes, sha256, "
                "provider, created_at) VALUES ('r1', 'm1', 'audio/rec.wav', 'audio/wav', 600000, 13, "
                "'sha', 'upload', :now)"
            ),
            {"now": NOW},
        )
        _insert_signal(conn, "sig1", "m1", "sp1")
        _insert_moment(conn, "mom1", "m1", "sp1")
        _insert_signal(conn, "sig-other", "other", "sp1")
        _insert_moment(conn, "mom-other", "other", "sp1")

    def fake_cut(src: Path, dst: Path, start_ms: int, end_ms: int | None) -> None:
        dst.write_bytes(b"cut-" + str(start_ms).encode())

    monkeypatch.setattr(split_module, "_ffmpeg_cut", fake_cut)
    monkeypatch.setattr(split_module, "_probe_duration_ms", lambda _p: 600_000)

    result = split_module.split_meeting(handle.engine, meetings_dir, "m1", 300_000)
    assert result["newMeeting"]["id"]

    with handle.engine.connect() as conn:
        assert conn.execute(text("SELECT id FROM meeting_speaker_signals")).scalars().all() == ["sig-other"]
        assert conn.execute(text("SELECT id FROM meeting_moments")).scalars().all() == ["mom-other"]


def test_insights_enabled_setting_defaults_off_and_round_trips() -> None:
    assert AppConfig().insightsEnabled is False
    assert AppConfig.model_validate({"insightsEnabled": True}).insightsEnabled is True


def _backup_service(handle, tmp_path: Path) -> BackupService:
    config_dir = tmp_path / "config"
    return BackupService(
        {
            "db": handle.engine,
            "configDir": str(config_dir),
            "meetingsDir": str(tmp_path / "meetings"),
            "backupsDir": str(tmp_path / "backups"),
            "databasePath": str(config_dir / "olive.sqlite"),
            "settingsPath": str(config_dir / "settings.json"),
        }
    )


def _file_backed_handle(tmp_path: Path):
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "meetings").mkdir(parents=True, exist_ok=True)
    (tmp_path / "backups").mkdir(parents=True, exist_ok=True)
    handle = create_db(str(tmp_path / "config" / "olive.sqlite"))
    set_db_for_tests(handle)
    return handle


async def test_backup_restore_round_trip_includes_insights_tables(tmp_path: Path) -> None:
    handle = _file_backed_handle(tmp_path)
    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1")
        _insert_speaker(conn, "sp1", "Alice")
        _insert_signal(conn, "sig1", "m1", "sp1")
        _insert_moment(conn, "mom1", "m1", "sp1")
        _insert_alias(conn, "al1", "ali", "sp1")
        conn.execute(
            text(
                "INSERT INTO transcript_imports (id, content_sha256, meeting_id, created_at) "
                "VALUES ('i1', 'sha', 'm1', :now)"
            ),
            {"now": NOW},
        )

    service = _backup_service(handle, tmp_path)
    backup = await service.create_backup({"filename": "insights.tar.gz"})

    with handle.engine.begin() as conn:
        for table in ("meeting_speaker_signals", "meeting_moments", "transcript_imports", "speaker_aliases"):
            conn.execute(text(f"DELETE FROM {table}"))

    result = await service.restore_backup(backup["path"])
    assert result["success"] is True

    from olive.db import get_db

    with get_db().connect() as conn:
        for table in INSIGHTS_TABLES:
            assert _count(conn, table) == 1, table
        assert conn.execute(text("SELECT quote FROM meeting_moments")).scalar_one() == "quote"
    set_db_for_tests(None)


async def test_legacy_backup_without_insights_tables_still_restores(tmp_path: Path) -> None:
    import sqlite3

    handle = _file_backed_handle(tmp_path)
    with handle.engine.begin() as conn:
        _insert_meeting(conn, "m1")
    service = _backup_service(handle, tmp_path)
    backup = await service.create_backup({"filename": "base.tar.gz"})

    # Build a pre-insights database: strip the new tables and columns' tables.
    legacy_staging = tmp_path / "legacy-staging"
    legacy_staging.mkdir()
    with tarfile.open(backup["path"], "r:gz") as tar:
        tar.extractall(legacy_staging, filter="data")
    legacy_db = legacy_staging / "olive.sqlite"
    raw = sqlite3.connect(str(legacy_db))
    try:
        for table in INSIGHTS_TABLES:
            raw.execute(f"DROP TABLE {table}")
        raw.commit()
    finally:
        raw.close()
    legacy_tar = tmp_path / "legacy.tar.gz"
    with tarfile.open(legacy_tar, "w:gz") as tar:
        for item in legacy_staging.iterdir():
            tar.add(item, arcname=item.name)

    result = await service.restore_backup(str(legacy_tar))
    assert result["success"] is True
    assert result["stats"]["meetings"] == 1

    from olive.db import get_db

    with get_db().connect() as conn:
        tables = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        assert set(INSIGHTS_TABLES) <= tables  # re-created by migrations on reopen
    set_db_for_tests(None)


def test_schema_literals_include_insights_values() -> None:
    from typing import get_args

    from olive.schemas import ArtifactKind, MeetingSource

    assert "signals" in get_args(ArtifactKind)
    assert "import" in get_args(MeetingSource)
    assert json.dumps(get_args(MeetingSource))  # still plain string literals
