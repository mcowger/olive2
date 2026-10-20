"""Tests for the read-only Olive Insights corpus census (services/census.py)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from olive.db import create_db
from olive.layout import meeting_paths
from olive.services import census as census_mod
from olive.services.census import compute_census, format_census_report, open_readonly_engine

MINUTE_MS = 60_000


def _ms(year: int, month: int, day: int, hour: int = 0) -> int:
    return int(datetime(year, month, day, hour, tzinfo=UTC).timestamp() * 1000)


def _seed(
    db: Any,
    meetings_dir: Path,
    *,
    meeting_id: str,
    title: str,
    start: int,
    end: int,
    speakers: tuple[str, ...] = (),
    transcript: Any = None,
    link_primary: bool = True,
    write_file: bool = True,
    recording_duration_ms: int | None = None,
) -> None:
    """Insert a meeting, its speakers, an optional recording, and an optional transcript artifact."""
    folder = Path(meeting_paths(str(meetings_dir), start, title, meeting_id).folder)
    rel_path = "transcripts/transcript.json"
    with db.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, created_at, updated_at) "
                "VALUES (:id, :title, :start, :end, 'upload', 'ready', '[]', 0, 0)"
            ),
            {"id": meeting_id, "title": title, "start": start, "end": end},
        )
        for name in speakers:
            speaker_id = f"speaker-{name.lower().replace(' ', '-')}"
            conn.execute(
                text("INSERT OR IGNORE INTO speakers (id, name, created_at) VALUES (:id, :name, 0)"),
                {"id": speaker_id, "name": name},
            )
            conn.execute(
                text("INSERT INTO meeting_speakers (meeting_id, speaker_id) VALUES (:mid, :sid)"),
                {"mid": meeting_id, "sid": speaker_id},
            )
        if recording_duration_ms is not None:
            conn.execute(
                text(
                    "INSERT INTO recordings (id, meeting_id, path, mime, duration_ms, size_bytes, sha256, "
                    "provider, created_at) VALUES (:id, :mid, 'audio/rec.opus', 'audio/ogg', :dur, 1, :sha, "
                    "'plaud', 0)"
                ),
                {
                    "id": f"rec-{meeting_id}",
                    "mid": meeting_id,
                    "dur": recording_duration_ms,
                    "sha": f"sha-{meeting_id}",
                },
            )
        if transcript is not None:
            artifact_id = f"art-{meeting_id}"
            conn.execute(
                text(
                    "INSERT INTO artifacts (id, meeting_id, kind, provider, format, path, created_at) "
                    "VALUES (:id, :mid, 'transcript', 'local', 'json', :path, 0)"
                ),
                {"id": artifact_id, "mid": meeting_id, "path": rel_path},
            )
            if link_primary:
                conn.execute(
                    text("UPDATE meetings SET primary_transcript_artifact_id = :aid WHERE id = :mid"),
                    {"aid": artifact_id, "mid": meeting_id},
                )
            if write_file:
                target = folder / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(transcript), encoding="utf-8")


@pytest.fixture
def db() -> Any:
    handle = create_db(":memory:")
    yield handle
    handle.engine.dispose()


@pytest.fixture
def corpus(db: Any, tmp_path: Path) -> Path:
    """Five meetings covering every transcript shape, bucket, and speaker case.

    | meeting | start      | length | transcript            | speakers           |
    |---------|------------|--------|-----------------------|--------------------|
    | M1      | 2026-01-05 | 20 min | segments object (3)   | Alice Chen linked  |
    | M2      | 2026-01-20 | 2 h    | bare list (2)         | none               |
    | M3      | 2026-02-02 | 3 min  | raw plaud list (2)    | none               |
    | M4      | 2026-02-10 | 4 h    | none                  | none               |
    | M5      | 2026-02-15 | 45 min | primary, file missing | none               |
    """
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    _seed(
        db,
        meetings_dir,
        meeting_id="meeting-0001",
        title="Standup",
        start=_ms(2026, 1, 5, 10),
        end=_ms(2026, 1, 5, 10) + 20 * MINUTE_MS,
        speakers=("Alice Chen",),
        transcript={
            "segments": [
                {"startMs": 0, "endMs": 5000, "speaker": "Speaker 1", "text": "hello there team"},
                {"startMs": 5000, "endMs": 9000, "speaker": "Speaker 1", "text": "morning everyone"},
                {"startMs": 9000, "endMs": 15000, "speaker": "Alice Chen", "text": "let us begin now"},
            ]
        },
    )
    _seed(
        db,
        meetings_dir,
        meeting_id="meeting-0002",
        title="Planning",
        start=_ms(2026, 1, 20, 9),
        end=_ms(2026, 1, 20, 9) + 120 * MINUTE_MS,
        transcript=[
            {"startMs": 0, "endMs": 4000, "speaker": "Speaker 2", "text": "one two three"},
            {"startMs": 4000, "endMs": 8000, "speaker": "Bob", "text": "four five"},
        ],
    )
    _seed(
        db,
        meetings_dir,
        meeting_id="meeting-0003",
        title="Quick sync",
        start=_ms(2026, 2, 2, 14),
        end=_ms(2026, 2, 2, 14) + 3 * MINUTE_MS,
        recording_duration_ms=3 * MINUTE_MS,
        transcript=[
            {"start_time": 0, "end_time": 60000, "speaker": "Speaker 1", "content": "ok so yes"},
            {"start_time": 60000, "end_time": 120000, "speaker": "Speaker 1", "content": "agreed"},
        ],
    )
    _seed(
        db,
        meetings_dir,
        meeting_id="meeting-0004",
        title="All hands",
        start=_ms(2026, 2, 10, 8),
        end=_ms(2026, 2, 10, 8) + 240 * MINUTE_MS,
    )
    _seed(
        db,
        meetings_dir,
        meeting_id="meeting-0005",
        title="Retro",
        start=_ms(2026, 2, 15, 16),
        end=_ms(2026, 2, 15, 16) + 45 * MINUTE_MS,
        transcript={"segments": []},
        write_file=False,
    )
    return meetings_dir


def test_compute_census_totals_and_transcript_coverage(db: Any, corpus: Path) -> None:
    stats = compute_census(db, corpus)

    assert stats["meetings"] == {
        "total": 5,
        "with_primary_transcript": 4,
        "without_primary_transcript": 1,
        "fallback_transcript_only": 0,
        "no_transcript": 1,
    }
    assert stats["transcripts"]["loaded"] == 3
    assert stats["transcripts"]["load_failures"] == {"missing_file": 1}
    assert stats["transcripts"]["shapes"] == {"segments_object": 1, "bare_list": 1, "raw_plaud": 1}


def test_compute_census_segment_and_word_stats(db: Any, corpus: Path) -> None:
    transcripts = compute_census(db, corpus)["transcripts"]

    assert transcripts["segments"]["total"] == 7
    assert transcripts["segments"]["per_meeting"] == {"min": 2, "median": 2, "max": 3}
    assert transcripts["words"]["total"] == 18
    assert transcripts["words"]["per_meeting"] == {"min": 4, "median": 5, "max": 9}
    assert transcripts["words"]["estimated_tokens"] == 23


def test_compute_census_duration_histogram(db: Any, corpus: Path) -> None:
    assert compute_census(db, corpus)["duration_histogram"] == {
        "<5m": 1,
        "5-30m": 1,
        "30-90m": 1,
        "90m-3h": 1,
        ">3h": 1,
        "unknown": 0,
    }


def test_compute_census_generic_speaker_share(db: Any, corpus: Path) -> None:
    speakers = compute_census(db, corpus)["speakers"]

    assert speakers["generic_segments"] == 5
    assert speakers["named_segments"] == 2
    assert speakers["generic_share"] == pytest.approx(5 / 7)
    assert speakers["distinct_labels_per_meeting"] == {"min": 1, "median": 2, "max": 2}
    assert speakers["meetings_with_no_identified_speakers"] == 4


def test_compute_census_month_coverage(db: Any, corpus: Path) -> None:
    assert compute_census(db, corpus)["months"] == {"2026-01": 2, "2026-02": 3}


def test_compute_census_empty_database(db: Any, tmp_path: Path) -> None:
    stats = compute_census(db, tmp_path)

    assert stats["meetings"]["total"] == 0
    assert stats["transcripts"]["loaded"] == 0
    assert stats["transcripts"]["segments"]["per_meeting"] is None
    assert stats["speakers"]["generic_share"] is None
    assert stats["months"] == {}


def test_compute_census_duration_falls_back_to_transcript_span(db: Any, tmp_path: Path) -> None:
    start = _ms(2026, 3, 1, 9)
    _seed(db, tmp_path, meeting_id="meeting-0010", title="Span", start=start, end=start)
    _seed(
        db,
        tmp_path,
        meeting_id="meeting-0011",
        title="Span two",
        start=start,
        end=start,
        transcript={"segments": [{"startMs": 0, "endMs": 240000, "speaker": "Speaker 1", "text": "hi"}]},
    )

    assert compute_census(db, tmp_path)["duration_histogram"]["<5m"] == 1
    assert compute_census(db, tmp_path)["duration_histogram"]["unknown"] == 1


def test_compute_census_counts_fallback_transcript(db: Any, tmp_path: Path) -> None:
    start = _ms(2026, 3, 2, 9)
    _seed(
        db,
        tmp_path,
        meeting_id="meeting-0020",
        title="Unlinked",
        start=start,
        end=start + 10 * MINUTE_MS,
        transcript={"segments": [{"startMs": 0, "endMs": 1000, "speaker": "UU", "text": "one"}]},
        link_primary=False,
    )

    stats = compute_census(db, tmp_path)

    assert stats["meetings"]["with_primary_transcript"] == 0
    assert stats["meetings"]["fallback_transcript_only"] == 1
    assert stats["meetings"]["no_transcript"] == 0
    assert stats["transcripts"]["loaded"] == 1
    assert stats["speakers"]["generic_share"] == 1.0


def test_readonly_engine_reads_and_rejects_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "olive.sqlite"
    handle = create_db(str(db_path))
    _seed(
        handle,
        tmp_path,
        meeting_id="meeting-0030",
        title="Readonly",
        start=_ms(2026, 4, 1, 9),
        end=_ms(2026, 4, 1, 9) + MINUTE_MS,
    )
    handle.engine.dispose()

    engine = open_readonly_engine(str(db_path))
    try:
        stats = compute_census(engine, tmp_path)
        assert stats["meetings"]["total"] == 1
        with pytest.raises(OperationalError), engine.begin() as conn:
            conn.execute(text("DELETE FROM meetings"))
    finally:
        engine.dispose()


def test_format_census_report_includes_headline_figures(db: Any, corpus: Path) -> None:
    report = format_census_report(compute_census(db, corpus))

    assert "5 total" in report
    assert "2026-01" in report
    assert "<5m" in report
    assert "7 segments" in report


def test_transcript_read_helpers_are_public_with_private_aliases(tmp_path: Path) -> None:
    for name in ("load_transcript", "recording_duration", "resolve_transcript_row"):
        assert getattr(census_mod, name) is getattr(census_mod, f"_{name}")
        assert name in census_mod.__all__
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"segments": [{"startMs": 0, "endMs": 1000, "speaker": "A", "text": "hi"}]}))
    loaded = census_mod.load_transcript(path, 1000)
    assert not isinstance(loaded, str)
    assert isinstance(census_mod.load_transcript(tmp_path / "missing.json", None), str)
