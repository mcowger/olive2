"""Transcript import service: no-audio meetings from Zoom VTT / SRT / speaker-labeled text."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from olive.db import DbHandle, create_db
from olive.services.meetings import delete_meeting
from olive.services.transcript_import import MissingStartTimeError, TranscriptImportService

ZOOM_VTT = """WEBVTT

1
00:00:01.000 --> 00:00:04.500
Alice Smith: Hello everyone, thanks for joining.

2
00:00:06.500 --> 00:00:09.250
Bob Jones: Sounds good.

3
00:01:00.000 --> 00:01:05.000
Speaker 3: I have nothing to add.
"""
ZOOM_NAME = "GMT20240315-150000_Recording.transcript.vtt"
ZOOM_START_MS = int(datetime(2024, 3, 15, 15, 0, 0, tzinfo=UTC).timestamp() * 1000)

PLAIN_TEXT = "Alice Smith: Morning.\nBob Jones: Hi Alice.\n"


class FakeInsights:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def enqueue_analyze(self, meeting_id: str) -> bool:
        self.calls.append(meeting_id)
        return True


class Env:
    def __init__(self, tmp_path: Path) -> None:
        self.db: DbHandle = create_db(":memory:")
        self.meetings_dir = str(tmp_path / "meetings")
        self.insights = FakeInsights()
        self.service = TranscriptImportService(
            db=self.db,
            meetings_dir=self.meetings_dir,
            config_dir=str(tmp_path / "config"),
            insights_service=self.insights,
            now=lambda: 1_800_000_000_000,
        )

    def rows(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        with self.db.engine.connect() as conn:
            return [dict(r) for r in conn.execute(text(sql), params).mappings()]


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


async def _import_zoom(env: Env, **extra: Any) -> dict[str, Any]:
    return await env.service.import_transcript({"content": ZOOM_VTT, "filename": ZOOM_NAME, **extra})


async def test_creates_import_meeting_without_recording(env: Env) -> None:
    result = await _import_zoom(env)

    assert result["deduped"] is False
    assert result["format"] == "vtt"
    assert result["subtype"] == "zoom"
    assert result["segmentCount"] == 3
    assert result["startTime"] == ZOOM_START_MS
    assert result["timingSynthetic"] is False
    meeting = env.rows("SELECT * FROM meetings WHERE id = :id", id=result["meetingId"])[0]
    assert meeting["source"] == "import"
    assert meeting["status"] == "ready"
    assert json.loads(meeting["tags"]) == ["import", "zoom"]
    assert meeting["start_time"] == ZOOM_START_MS
    assert meeting["end_time"] == ZOOM_START_MS + 65_000
    assert env.rows("SELECT * FROM recordings") == []
    imports = env.rows("SELECT * FROM transcript_imports")
    assert [(r["meeting_id"], r["source_format"], r["subtype"]) for r in imports] == [
        (result["meetingId"], "vtt", "zoom")
    ]
    stage = env.rows("SELECT * FROM stage_runs WHERE meeting_id = :id", id=result["meetingId"])
    assert [(r["stage"], r["status"]) for r in stage] == [("import", "done")]


async def test_writes_json_and_txt_artifacts_and_sets_primary(env: Env) -> None:
    result = await _import_zoom(env)
    mid = result["meetingId"]

    arts = env.rows("SELECT * FROM artifacts WHERE meeting_id = :id ORDER BY format", id=mid)
    assert [(a["kind"], a["format"], a["provider"], a["recording_id"]) for a in arts] == [
        ("transcript", "json", "import:zoom", None),
        ("transcript", "txt", "import:zoom", None),
    ]
    meeting = env.rows("SELECT * FROM meetings WHERE id = :id", id=mid)[0]
    json_art = next(a for a in arts if a["format"] == "json")
    assert meeting["primary_transcript_artifact_id"] == json_art["id"]

    folder = Path(env.meetings_dir) / next(Path(env.meetings_dir).iterdir()).name
    data = json.loads((folder / json_art["path"]).read_text(encoding="utf-8"))
    assert [s["speaker"] for s in data["segments"]] == ["Alice Smith", "Bob Jones", "Speaker 3"]
    assert data["segments"][0]["startMs"] == 1000
    txt_art = next(a for a in arts if a["format"] == "txt")
    assert "Alice Smith: Hello everyone" in (folder / txt_art["path"]).read_text(encoding="utf-8")


async def test_dedupes_by_content_hash(env: Env) -> None:
    first = await _import_zoom(env)
    # Same text with a BOM and CRLF endings is the same content.
    second = await env.service.import_transcript(
        {"content": ("﻿" + ZOOM_VTT).replace("\n", "\r\n").encode("utf-8"), "filename": "other.vtt"}
    )

    assert second == {"deduped": True, "meetingId": first["meetingId"]}
    assert len(env.rows("SELECT id FROM meetings")) == 1
    assert env.insights.calls == [first["meetingId"]]


async def test_speakers_matched_by_name_get_name_identity(env: Env) -> None:
    with env.db.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO speakers (id, name, enrolled_at, created_at) VALUES ('sp-alice','Alice Smith',5,1)"
            )
        )
        conn.execute(text("INSERT INTO speakers (id, name, created_at) VALUES ('sp-bob','Robert Jones',1)"))
        conn.execute(
            text(
                "INSERT INTO speaker_aliases (id, alias_norm, speaker_id, kind, confirmed, created_at) "
                "VALUES ('al1','bob jones','sp-bob','name',1,1)"
            )
        )

    result = await _import_zoom(env)
    by_label = {s["label"]: s for s in result["speakers"]}

    assert by_label["Alice Smith"]["speakerId"] == "sp-alice"
    assert by_label["Bob Jones"]["speakerId"] == "sp-bob"
    assert by_label["Alice Smith"]["identitySource"] == "name"
    links = env.rows("SELECT * FROM meeting_speakers WHERE meeting_id = :id", id=result["meetingId"])
    assert {(r["speaker_id"], r["identity_source"]) for r in links} == {
        ("sp-alice", "name"),
        ("sp-bob", "name"),
    }
    # No duplicate speakers were created for the matched names.
    assert {r["name"] for r in env.rows("SELECT name FROM speakers")} == {"Alice Smith", "Robert Jones"}
    assert result["unmappedNames"] == []


async def test_new_names_create_unenrolled_speakers_with_aliases(env: Env) -> None:
    result = await _import_zoom(env)

    speakers = env.rows("SELECT * FROM speakers ORDER BY name")
    assert [(s["name"], s["enrolled_at"]) for s in speakers] == [("Alice Smith", None), ("Bob Jones", None)]
    aliases = env.rows("SELECT alias_norm FROM speaker_aliases ORDER BY alias_norm")
    assert [a["alias_norm"] for a in aliases] == ["alice smith", "bob jones"]

    # Importing the same names again (new content) reuses the speakers case-insensitively.
    again = await env.service.import_transcript(
        {"content": "ALICE SMITH: Again.\nbob jones: Yes.\n", "filename": "2024-04-01 sync.txt"}
    )
    assert len(env.rows("SELECT id FROM speakers")) == 2
    assert {s["speakerId"] for s in again["speakers"]} == {
        s["speakerId"] for s in result["speakers"] if s["speakerId"]
    }


async def test_generic_labels_are_not_linked(env: Env) -> None:
    result = await _import_zoom(env)

    generic = next(s for s in result["speakers"] if s["label"] == "Speaker 3")
    assert generic["speakerId"] is None
    assert generic["identitySource"] is None
    assert len(env.rows("SELECT * FROM meeting_speakers")) == 2
    assert "Speaker 3" not in {s["name"] for s in env.rows("SELECT name FROM speakers")}


async def test_me_alias_maps_to_the_me_speaker(env: Env) -> None:
    with env.db.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO speakers (id, name, enrolled_at, created_at) VALUES ('sp-me','Phil McNeely',9,1)"
            )
        )
    result = await env.service.import_transcript(
        {
            "content": "Phil: Let's begin.\nAlice Smith: Ok.\n",
            "filename": "2024-05-02-standup.txt",
            "meAliases": ["Phil McNeely", "Phil"],
        }
    )

    by_label = {s["label"]: s for s in result["speakers"]}
    assert by_label["Phil"]["speakerId"] == "sp-me"
    assert by_label["Phil"]["identitySource"] == "name"
    assert "phil" in {a["alias_norm"] for a in env.rows("SELECT alias_norm FROM speaker_aliases")}
    assert len(env.rows("SELECT id FROM speakers WHERE name LIKE 'Phil%'")) == 1


async def test_ambiguous_name_is_reported_not_guessed(env: Env) -> None:
    with env.db.engine.begin() as conn:
        conn.execute(text("INSERT INTO speakers (id, name, created_at) VALUES ('a1','Alice Smith',1)"))
        conn.execute(text("INSERT INTO speakers (id, name, created_at) VALUES ('a2','alice smith',2)"))
    result = await env.service.import_transcript(
        {"content": "Alice Smith: Hi.\nBob Jones: Hello.\n", "filename": "2024-05-02.txt"}
    )

    assert result["unmappedNames"] == ["Alice Smith"]
    alice = next(s for s in result["speakers"] if s["label"] == "Alice Smith")
    assert alice["speakerId"] is None
    assert len(env.rows("SELECT * FROM meeting_speakers")) == 1


async def test_user_start_time_wins_over_filename(env: Env) -> None:
    result = await _import_zoom(env, startTime=1_700_000_000_000)
    assert result["startTime"] == 1_700_000_000_000

    iso = await env.service.import_transcript(
        {"content": PLAIN_TEXT, "filename": "notes.txt", "startTime": "2024-06-01T10:30:00Z"}
    )
    assert iso["startTime"] == int(datetime(2024, 6, 1, 10, 30, tzinfo=UTC).timestamp() * 1000)


async def test_start_time_inferred_from_content_header(env: Env) -> None:
    result = await env.service.import_transcript(
        {"content": "Meeting on 2024-02-10 14:00\n" + PLAIN_TEXT, "filename": "notes.txt"}
    )
    assert result["startTime"] == int(datetime(2024, 2, 10, 14, 0, tzinfo=UTC).timestamp() * 1000)


async def test_rejects_when_no_start_time_derivable(env: Env) -> None:
    with pytest.raises(MissingStartTimeError):
        await env.service.import_transcript({"content": PLAIN_TEXT, "filename": "notes.txt"})

    assert env.rows("SELECT id FROM meetings") == []
    assert env.rows("SELECT id FROM transcript_imports") == []
    assert env.insights.calls == []
    assert not Path(env.meetings_dir).exists() or list(Path(env.meetings_dir).iterdir()) == []


async def test_invalid_start_time_and_empty_content_are_value_errors(env: Env) -> None:
    with pytest.raises(ValueError):
        await env.service.import_transcript({"content": PLAIN_TEXT, "filename": "n.txt", "startTime": "soon"})
    with pytest.raises(ValueError, match="No speech segments"):
        await env.service.import_transcript({"content": "   \n", "filename": "2024-01-01.txt"})


async def test_enqueues_analysis_exactly_once(env: Env) -> None:
    result = await _import_zoom(env)
    assert env.insights.calls == [result["meetingId"]]


async def test_enqueue_failure_keeps_import_and_warns(tmp_path: Path) -> None:
    class Boom:
        async def enqueue_analyze(self, meeting_id: str) -> bool:
            raise RuntimeError("queue down")

    env = Env(tmp_path)
    env.service.insights_service = Boom()
    result = await _import_zoom(env)

    assert len(env.rows("SELECT id FROM meetings")) == 1
    assert any("queue down" in w for w in result["warnings"])


async def test_works_without_insights_service(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.service.insights_service = None
    result = await _import_zoom(env)
    assert result["deduped"] is False


async def test_import_then_delete_meeting(env: Env) -> None:
    result = await _import_zoom(env)

    assert delete_meeting(env.db.engine, result["meetingId"], env.meetings_dir) is True

    for table in ("meetings", "artifacts", "transcript_imports", "meeting_speakers", "stage_runs"):
        assert env.rows(f"SELECT * FROM {table}") == [], table
    assert list(Path(env.meetings_dir).iterdir()) == []
    # Content can be re-imported after deletion.
    again = await _import_zoom(env)
    assert again["deduped"] is False


async def test_huge_digit_start_time_is_value_error(env: Env) -> None:
    with pytest.raises(ValueError):
        await env.service.import_transcript(
            {"content": PLAIN_TEXT, "filename": "n.txt", "startTime": "9" * 400}
        )
