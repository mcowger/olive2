"""Meeting merge tests: transcript stitching plus a DB-level merge run."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from sqlalchemy import text

from olive.db import create_db
from olive.layout import meeting_paths
from olive.services import merge as merge_module
from olive.services.merge import merge_meetings, merge_transcript_payloads, shift_segments


def test_shift_segments_offsets_words() -> None:
    segments = [
        {
            "startMs": 0,
            "endMs": 1_000,
            "speaker": "A",
            "text": "hello",
            "words": [{"startMs": 0, "endMs": 400, "word": "hello"}],
        }
    ]
    shifted = shift_segments(segments, 60_000)
    assert shifted[0]["startMs"] == 60_000
    assert shifted[0]["endMs"] == 61_000
    assert shifted[0]["start_time"] == 60_000
    assert shifted[0]["words"][0]["startMs"] == 60_000
    assert shifted[0]["words"][0]["endMs"] == 60_400
    # Input untouched.
    assert segments[0]["startMs"] == 0


def test_merge_transcript_payloads_orders_segments() -> None:
    first = {"segments": [{"startMs": 5_000, "endMs": 6_000, "text": "one"}], "language": "en"}
    second = {"segments": [{"startMs": 0, "endMs": 1_000, "text": "two"}]}
    merged = merge_transcript_payloads([(second, 10_000), (first, 0)])
    assert [s["text"] for s in merged["segments"]] == ["one", "two"]
    assert [s["startMs"] for s in merged["segments"]] == [5_000, 10_000]
    assert merged["durationMs"] == 11_000
    assert merged["language"] == "en"


def test_merge_transcript_payloads_bare_list() -> None:
    merged = merge_transcript_payloads(
        [
            ([{"startMs": 0, "endMs": 1_000, "text": "a"}], 0),
            ([{"startMs": 0, "endMs": 2_000, "text": "b"}], 30_000),
        ]
    )
    assert isinstance(merged, list)
    assert [(s["text"], s["startMs"]) for s in merged] == [("a", 0), ("b", 30_000)]


def test_merge_transcript_payloads_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError):
        merge_transcript_payloads([({"nope": True}, 0)])


def _seed_meeting(
    engine,
    meetings_dir: str,
    *,
    meeting_id: str,
    title: str,
    start_time: int,
    duration_ms: int,
    segments: list[dict],
    tags: list[str],
    speaker_id: str | None = None,
) -> Path:
    paths = meeting_paths(meetings_dir, start_time, title, meeting_id)
    folder = Path(paths.folder)
    (folder / "audio").mkdir(parents=True, exist_ok=True)
    (folder / "transcripts").mkdir(parents=True, exist_ok=True)
    audio_rel = "audio/rec.wav"
    (folder / audio_rel).write_bytes(f"audio-{meeting_id}".encode())
    transcript_rel = "transcripts/local.json"
    (folder / transcript_rel).write_text(json.dumps({"segments": segments}), encoding="utf-8")
    now = int(time.time() * 1000)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, :title, :start, :end, 'upload', 'ready', :tags,
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {
                "id": meeting_id,
                "title": title,
                "start": start_time,
                "end": start_time + duration_ms,
                "tags": json.dumps(tags),
                "now": now,
            },
        )
        recording_id = f"rec-{meeting_id}"
        conn.execute(
            text(
                """
                INSERT INTO recordings (
                  id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at
                ) VALUES (:id, :mid, :path, 'audio/wav', :dur, 10, :sha, 'upload', NULL, :now)
                """
            ),
            {
                "id": recording_id,
                "mid": meeting_id,
                "path": audio_rel,
                "dur": duration_ms,
                "sha": f"sha-{meeting_id}",
                "now": now,
            },
        )
        artifact_id = f"art-{meeting_id}"
        conn.execute(
            text(
                """
                INSERT INTO artifacts (
                  id, meeting_id, recording_id, kind, provider, format, path, created_at
                ) VALUES (:id, :mid, :rid, 'transcript', 'local', 'json', :path, :now)
                """
            ),
            {"id": artifact_id, "mid": meeting_id, "rid": recording_id, "path": transcript_rel, "now": now},
        )
        conn.execute(
            text("UPDATE meetings SET primary_transcript_artifact_id = :aid WHERE id = :mid"),
            {"aid": artifact_id, "mid": meeting_id},
        )
        if speaker_id:
            conn.execute(
                text(
                    "INSERT INTO speakers (id, name, provider_ids, enrolled_at, "
                    "enrollment_clip_paths, created_at) VALUES (:id, :name, '{}', NULL, '[]', :now)"
                ),
                {"id": speaker_id, "name": speaker_id, "now": now},
            )
            conn.execute(
                text(
                    "INSERT INTO meeting_speakers (meeting_id, speaker_id, evidence_artifact_id) "
                    "VALUES (:mid, :sid, :aid)"
                ),
                {"mid": meeting_id, "sid": speaker_id, "aid": artifact_id},
            )
    return folder


def test_merge_meetings_joins_audio_transcripts_and_deletes_sources(tmp_path: Path, monkeypatch) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")

    def fake_concat(sources, dst, ext):  # type: ignore[no-untyped-def]
        Path(dst).write_bytes(b"".join(Path(src).read_bytes() for src in sources))

    monkeypatch.setattr(merge_module, "_concat_audio", fake_concat)
    monkeypatch.setattr(merge_module, "_probe_duration_ms", lambda _path: None)

    # Second clip starts later but is passed first to prove chronological sort.
    first = _seed_meeting(
        handle.engine,
        meetings_dir,
        meeting_id="meeting-a",
        title="Clip A",
        start_time=1_700_000_000_000,
        duration_ms=60_000,
        segments=[{"startMs": 0, "endMs": 1_000, "speaker": "A", "text": "one"}],
        tags=["work"],
        speaker_id="speaker-a",
    )
    second = _seed_meeting(
        handle.engine,
        meetings_dir,
        meeting_id="meeting-b",
        title="Clip B",
        start_time=1_700_000_060_000,
        duration_ms=30_000,
        segments=[{"startMs": 0, "endMs": 500, "speaker": "B", "text": "two"}],
        tags=["work", "road"],
        speaker_id="speaker-b",
    )

    result = merge_meetings(handle.engine, meetings_dir, ["meeting-b", "meeting-a"])

    assert result["meeting"]["id"] == "meeting-a"
    assert result["meeting"]["endTime"] == 1_700_000_090_000
    assert sorted(result["meeting"]["tags"]) == ["road", "work"]
    assert result["deletedMeetingIds"] == ["meeting-b"]
    assert first.exists()
    assert not second.exists()

    with handle.engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM meetings")).scalar_one() == 1
        assert conn.execute(text("SELECT COUNT(*) FROM recordings")).scalar_one() == 1
        duration = conn.execute(text("SELECT duration_ms FROM recordings")).scalar_one()
        assert duration == 90_000
        speaker_ids = {row[0] for row in conn.execute(text("SELECT speaker_id FROM meeting_speakers")).all()}
        assert speaker_ids == {"speaker-a", "speaker-b"}

    payload = json.loads((first / "transcripts/local.json").read_text(encoding="utf-8"))
    assert [(s["text"], s["startMs"]) for s in payload["segments"]] == [("one", 0), ("two", 60_000)]
    assert payload["durationMs"] == 60_500
    assert (first / "audio/rec.wav").read_bytes() == b"audio-meeting-aaudio-meeting-b"
    assert (first / "transcripts/local.txt").read_text(encoding="utf-8").strip().endswith("B: two")


def test_merge_meetings_validation(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    _seed_meeting(
        handle.engine,
        meetings_dir,
        meeting_id="meeting-a",
        title="Clip A",
        start_time=1_700_000_000_000,
        duration_ms=10_000,
        segments=[],
        tags=[],
    )
    with pytest.raises(ValueError, match="at least two"):
        merge_meetings(handle.engine, meetings_dir, ["meeting-a"])
    with pytest.raises(ValueError, match="not found"):
        merge_meetings(handle.engine, meetings_dir, ["meeting-a", "missing"])
