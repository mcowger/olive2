"""MCP write tools tests (issue #8): speakers, tags, titles with plan/confirm guards."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import text

from olive.app import create_app
from olive.db import create_db
from olive.layout import meeting_paths

MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "Host": "127.0.0.1",
}


def _mcp_request(
    client: TestClient,
    method: str,
    params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    res = client.post(
        "/mcp",
        headers=MCP_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params or {},
        },
    )
    return res, res.json()


def _call_tool(client: TestClient, name: str, args: dict[str, Any]) -> dict[str, Any]:
    _, body = _mcp_request(client, "tools/call", {"name": name, "arguments": args})
    return body["result"]


def _parse_text(result: dict[str, Any]) -> Any:
    assert result.get("isError") is not True, result
    return json.loads(result["content"][0]["text"])


def _seed_meeting(engine, meetings_dir: str, meeting_id: str, title: str, tags: list[str]) -> Path:
    now = 1_700_000_000_000
    paths = meeting_paths(meetings_dir, now, title, meeting_id)
    Path(paths.folder).mkdir(parents=True, exist_ok=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, :title, :now, :end, 'upload', 'ready', :tags,
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {
                "id": meeting_id,
                "title": title,
                "now": now,
                "end": now + 60_000,
                "tags": json.dumps(tags),
            },
        )
    return Path(paths.folder)


def _seed_speaker(engine, speaker_id: str, name: str, meeting_id: str | None = None) -> None:
    now = 1_700_000_000_000
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO speakers (
                  id, name, provider_ids, enrolled_at, enrollment_clip_paths, created_at
                ) VALUES (:id, :name, '{}', NULL, '[]', :now)
                """
            ),
            {"id": speaker_id, "name": name, "now": now},
        )
        if meeting_id:
            conn.execute(
                text(
                    """
                    INSERT INTO meeting_speakers (meeting_id, speaker_id, evidence_artifact_id)
                    VALUES (:mid, :sid, NULL)
                    """
                ),
                {"mid": meeting_id, "sid": speaker_id},
            )


def test_write_tools_hidden_by_default(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    app = create_app({"db": handle.engine, "meetingsDir": str(tmp_path / "meetings")})
    client = TestClient(app)
    _, body = _mcp_request(client, "tools/list")
    names = [t["name"] for t in body["result"]["tools"]]
    assert "update_speaker" not in names
    assert "merge_speakers" not in names
    assert "delete_speaker" not in names
    assert "set_meeting_tags" not in names
    assert "update_meeting_title" not in names
    assert "summarize_meeting" not in names
    assert "delete_summary" not in names
    assert "set_primary_summary" not in names
    assert "import_speaker_library" not in names

    res = _call_tool(client, "update_speaker", {"speakerId": "x", "newName": "Y"})
    assert res.get("isError") is True
    assert "OLIVE_MCP_WRITE=1" in res["content"][0]["text"]
    handle.engine.dispose()


def test_write_tools_listed_when_enabled(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    app = create_app(
        {
            "db": handle.engine,
            "meetingsDir": str(tmp_path / "meetings"),
            "mcpWriteEnabled": True,
        }
    )
    client = TestClient(app)
    _, body = _mcp_request(client, "tools/list")
    names = [t["name"] for t in body["result"]["tools"]]
    for expected in [
        "update_speaker",
        "merge_speakers",
        "delete_speaker",
        "set_meeting_tags",
        "update_meeting_title",
        "import_speaker_library",
    ]:
        assert expected in names
    handle.engine.dispose()


def test_update_speaker_dry_run_guard_and_apply(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    _seed_speaker(handle.engine, "spk-1", "Matt")
    app = create_app({"db": handle.engine, "meetingsDir": meetings_dir, "mcpWriteEnabled": True})
    client = TestClient(app)

    plan = _parse_text(
        _call_tool(
            client,
            "update_speaker",
            {"speakerId": "spk-1", "newName": "Matthew Cowger", "dryRun": True},
        )
    )
    assert plan["dryRun"] is True
    assert plan["currentName"] == "Matt"
    assert plan["wouldUpdate"] is True
    # dryRun mutates nothing
    with handle.engine.connect() as conn:
        row = conn.execute(text("SELECT name FROM speakers WHERE id='spk-1'")).mappings().first()
        assert row and row["name"] == "Matt"

    # expectedName mismatch aborts
    err = _call_tool(
        client,
        "update_speaker",
        {"speakerId": "spk-1", "newName": "Matthew Cowger", "expectedName": "Wrong"},
    )
    assert err.get("isError") is True
    assert "Precondition failed" in err["content"][0]["text"]

    applied = _parse_text(
        _call_tool(
            client,
            "update_speaker",
            {"speakerId": "spk-1", "newName": "Matthew Cowger", "expectedName": "Matt"},
        )
    )
    assert applied["previousName"] == "Matt"
    assert applied["speaker"]["name"] == "Matthew Cowger"
    handle.engine.dispose()


def test_merge_and_delete_speaker_flow(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    mid = "meeting-merge-1"
    _seed_meeting(handle.engine, meetings_dir, mid, "Merge fixture", [])
    _seed_speaker(handle.engine, "spk-src", "Matt", mid)
    _seed_speaker(handle.engine, "spk-tgt", "Matthew Cowger", mid)
    app = create_app({"db": handle.engine, "meetingsDir": meetings_dir, "mcpWriteEnabled": True})
    client = TestClient(app)

    plan = _parse_text(
        _call_tool(
            client,
            "merge_speakers",
            {"sourceSpeakerId": "spk-src", "targetSpeakerId": "spk-tgt", "dryRun": True},
        )
    )
    assert plan["dryRun"] is True
    assert plan["source"]["name"] == "Matt"
    assert plan["target"]["name"] == "Matthew Cowger"
    assert plan["wouldDeleteSourceId"] == "spk-src"

    # guard mismatch aborts
    err = _call_tool(
        client,
        "merge_speakers",
        {
            "sourceSpeakerId": "spk-src",
            "targetSpeakerId": "spk-tgt",
            "expectedSourceName": "Nobody",
        },
    )
    assert err.get("isError") is True

    merged = _parse_text(
        _call_tool(
            client,
            "merge_speakers",
            {
                "sourceSpeakerId": "spk-src",
                "targetSpeakerId": "spk-tgt",
                "expectedSourceName": "Matt",
                "expectedTargetName": "Matthew Cowger",
            },
        )
    )
    assert merged["speaker"]["id"] == "spk-tgt"
    with handle.engine.connect() as conn:
        assert conn.execute(text("SELECT 1 FROM speakers WHERE id='spk-src'")).first() is None

    # delete with plan then apply
    del_plan = _parse_text(_call_tool(client, "delete_speaker", {"speakerId": "spk-tgt", "dryRun": True}))
    assert del_plan["wouldDelete"] is True
    assert del_plan["name"] == "Matthew Cowger"
    deleted = _parse_text(
        _call_tool(
            client,
            "delete_speaker",
            {"speakerId": "spk-tgt", "expectedName": "Matthew Cowger"},
        )
    )
    assert deleted["deleted"] is True
    handle.engine.dispose()


def test_set_meeting_tags_guard_and_apply(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    mid = "meeting-tags-1"
    _seed_meeting(handle.engine, meetings_dir, mid, "Tags fixture", ["acme"])
    app = create_app({"db": handle.engine, "meetingsDir": meetings_dir, "mcpWriteEnabled": True})
    client = TestClient(app)

    plan = _parse_text(
        _call_tool(
            client,
            "set_meeting_tags",
            {"meetingId": mid, "tags": ["acme", "q3"], "dryRun": True},
        )
    )
    assert plan["currentTags"] == ["acme"]
    assert plan["wouldUpdate"] is True

    err = _call_tool(
        client,
        "set_meeting_tags",
        {"meetingId": mid, "tags": ["acme", "q3"], "expectedTags": ["stale"]},
    )
    assert err.get("isError") is True
    assert "Precondition failed" in err["content"][0]["text"]

    applied = _parse_text(
        _call_tool(
            client,
            "set_meeting_tags",
            {"meetingId": mid, "tags": ["acme", "q3"], "expectedTags": ["acme"]},
        )
    )
    assert applied["previousTags"] == ["acme"]
    assert applied["meeting"]["tags"] == ["acme", "q3"]
    handle.engine.dispose()


def test_update_meeting_title_guard_apply_and_folder_rename(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    mid = "meeting-title-1"
    old_folder = _seed_meeting(handle.engine, meetings_dir, mid, "ASR guess Acme", [])
    assert old_folder.exists()
    app = create_app({"db": handle.engine, "meetingsDir": meetings_dir, "mcpWriteEnabled": True})
    client = TestClient(app)

    plan = _parse_text(
        _call_tool(
            client,
            "update_meeting_title",
            {"meetingId": mid, "newTitle": "Acme Q3 review", "dryRun": True},
        )
    )
    assert plan["currentTitle"] == "ASR guess Acme"
    assert plan["wouldUpdate"] is True
    assert old_folder.exists()

    err = _call_tool(
        client,
        "update_meeting_title",
        {"meetingId": mid, "newTitle": "Acme Q3 review", "expectedTitle": "Stale"},
    )
    assert err.get("isError") is True

    applied = _parse_text(
        _call_tool(
            client,
            "update_meeting_title",
            {
                "meetingId": mid,
                "newTitle": "Acme Q3 review",
                "expectedTitle": "ASR guess Acme",
            },
        )
    )
    assert applied["previousTitle"] == "ASR guess Acme"
    assert applied["meeting"]["title"] == "Acme Q3 review"
    assert not old_folder.exists()
    handle.engine.dispose()


def test_write_enabled_via_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLIVE_MCP_WRITE", "1")
    handle = create_db(":memory:")
    app = create_app({"db": handle.engine, "meetingsDir": str(tmp_path / "meetings")})
    client = TestClient(app)
    _, body = _mcp_request(client, "tools/list")
    names = [t["name"] for t in body["result"]["tools"]]
    assert "update_speaker" in names
    handle.engine.dispose()


def _seed_recording(engine, meeting_id: str) -> None:
    now = 1_700_000_000_000
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO recordings (
                  id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at
                ) VALUES (
                  :rid, :mid, 'audio/meeting.m4a', 'audio/mp4', 60000, 1234,
                  'sha-test', 'upload', NULL, :now
                )
                """
            ),
            {"rid": f"rec-{meeting_id}", "mid": meeting_id, "now": now},
        )


class _StubTranscription:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def start_transcription(
        self, meeting_id: str, opts: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.calls.append(("start", {"meetingId": meeting_id, **(opts or {})}))
        return {
            "status": "running",
            "jobId": f"job-{meeting_id[:8]}",
            "meetingId": meeting_id,
            "alreadyRunning": False,
        }

    async def start_bulk_transcription(self, opts: dict[str, Any] | None = None) -> dict[str, Any]:
        o = dict(opts or {})
        self.calls.append(("bulk", o))
        ids = list(o.get("meetingIds") or [])
        return {
            "bulkId": "bulk-1",
            "results": [{"meetingId": mid, "status": "running"} for mid in ids],
            "queued": len(ids),
        }

    async def cancel_transcription(self, meeting_id: str) -> dict[str, Any]:
        self.calls.append(("cancel", meeting_id))
        return {"success": True, "meetingId": meeting_id}

    async def request_speaker_reidentification(self, meeting_id: str) -> dict[str, Any]:
        self.calls.append(("reidentify", meeting_id))
        return {"status": "queued", "jobId": f"reid-{meeting_id[:8]}", "meetingId": meeting_id}


class _StubSummary:
    async def start_summary_generation(self, opts: dict[str, Any]) -> dict[str, Any]:
        return {"meetingId": opts.get("meetingId"), "stage": "running"}


class _StubTitle:
    async def generate_and_apply(
        self, meeting_id: str, provider: str | None = None, model: str | None = None
    ) -> dict[str, Any]:
        return {
            "meeting": {"id": meeting_id, "title": "Generated Title"},
            "title": "Generated Title",
            "provider": provider or "google",
            "model": model or "gemini-2.5-flash",
        }


class _StubPoller:
    async def retry_meeting_audio(self, meeting_id: str) -> dict[str, Any]:
        return {"status": "queued", "meetingId": meeting_id}


def _write_app(tmp_path: Path, engine, **services: Any):
    return create_app(
        {
            "db": engine,
            "meetingsDir": str(tmp_path / "meetings"),
            "mcpWriteEnabled": True,
            **services,
        }
    )


def test_write_tools_list_includes_second_wave(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    app = _write_app(tmp_path, handle.engine)
    client = TestClient(app)
    _, body = _mcp_request(client, "tools/list")
    names = [t["name"] for t in body["result"]["tools"]]
    for expected in [
        "transcribe_meeting",
        "bulk_transcribe",
        "cancel_transcription",
        "retry_meeting_audio",
        "reidentify_speakers",
        "delete_meeting",
        "split_meeting",
        "merge_meetings",
        "update_meeting_start_time",
        "auto_title_meeting",
        "summarize_meeting",
    ]:
        assert expected in names
    handle.engine.dispose()


def test_bulk_tagging_plan_and_apply(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    _seed_meeting(handle.engine, meetings_dir, "m-bulk-1", "Bulk one", ["acme"])
    _seed_meeting(handle.engine, meetings_dir, "m-bulk-2", "Bulk two", [])
    app = _write_app(tmp_path, handle.engine)
    client = TestClient(app)

    plan = _parse_text(
        _call_tool(
            client,
            "set_meeting_tags",
            {"meetingIds": ["m-bulk-1", "m-bulk-2", "missing"], "tags": ["acme", "q3"], "dryRun": True},
        )
    )
    assert plan["mode"] == "bulk"
    assert plan["summary"] == {"total": 3, "wouldUpdate": 2, "notFound": 1}

    applied = _parse_text(
        _call_tool(
            client,
            "set_meeting_tags",
            {"meetingIds": ["m-bulk-1", "m-bulk-2"], "tags": ["acme", "q3"]},
        )
    )
    assert applied["summary"] == {"total": 2, "updated": 2, "notFound": 0}

    err = _call_tool(
        client,
        "set_meeting_tags",
        {"meetingIds": ["m-bulk-1"], "tags": ["x"], "expectedTags": ["acme"]},
    )
    assert err.get("isError") is True
    handle.engine.dispose()


def test_transcribe_meeting_dry_run_and_queue(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    _seed_meeting(handle.engine, meetings_dir, "m-tr-1", "Transcribe me", [])
    _seed_recording(handle.engine, "m-tr-1")
    stub = _StubTranscription()
    app = _write_app(tmp_path, handle.engine, transcriptionService=stub)
    client = TestClient(app)

    plan = _parse_text(_call_tool(client, "transcribe_meeting", {"meetingId": "m-tr-1", "dryRun": True}))
    assert plan["wouldQueue"] is True
    assert stub.calls == []

    queued = _parse_text(
        _call_tool(client, "transcribe_meeting", {"meetingId": "m-tr-1", "provider": "local"})
    )
    assert queued["status"] == "running"
    assert queued["jobId"].startswith("job-")
    assert stub.calls[0][0] == "start"

    err = _call_tool(client, "transcribe_meeting", {"meetingId": "m-tr-1", "provider": "nope"})
    assert err.get("isError") is True
    handle.engine.dispose()


def test_bulk_transcribe_filters_dry_run_and_queue(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    _seed_meeting(handle.engine, meetings_dir, "m-bt-1", "Bulk T one", ["acme"])
    _seed_meeting(handle.engine, meetings_dir, "m-bt-2", "Bulk T two", ["other"])
    stub = _StubTranscription()
    app = _write_app(tmp_path, handle.engine, transcriptionService=stub)
    client = TestClient(app)

    plan = _parse_text(
        _call_tool(client, "bulk_transcribe", {"provider": "local", "tags": ["acme"], "dryRun": True})
    )
    assert plan["mode"] == "filters"
    assert plan["meetingIds"] == ["m-bt-1"]

    queued = _parse_text(_call_tool(client, "bulk_transcribe", {"provider": "local", "tags": ["acme"]}))
    assert queued["operation"] == "bulk_transcribe"
    handle.engine.dispose()


def test_cancel_reidentify_retry_flow(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    _seed_meeting(handle.engine, meetings_dir, "m-cr-1", "Cancel me", [])
    stub = _StubTranscription()
    poller = _StubPoller()
    app = _write_app(tmp_path, handle.engine, transcriptionService=stub, plaudPoller=poller)
    client = TestClient(app)

    cx_plan = _parse_text(_call_tool(client, "cancel_transcription", {"meetingId": "m-cr-1", "dryRun": True}))
    assert cx_plan["wouldCancel"] is True
    cx = _parse_text(_call_tool(client, "cancel_transcription", {"meetingId": "m-cr-1"}))
    assert cx["success"] is True

    ri = _parse_text(_call_tool(client, "reidentify_speakers", {"meetingId": "m-cr-1"}))
    assert ri["status"] == "queued"

    ra = _parse_text(_call_tool(client, "retry_meeting_audio", {"meetingId": "m-cr-1"}))
    assert ra["status"] == "queued"
    handle.engine.dispose()


def test_retry_without_poller_errors(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    _seed_meeting(handle.engine, meetings_dir, "m-nop-1", "No poller", [])
    app = _write_app(tmp_path, handle.engine, transcriptionService=_StubTranscription())
    client = TestClient(app)
    err = _call_tool(client, "retry_meeting_audio", {"meetingId": "m-nop-1"})
    assert err.get("isError") is True
    assert "Plaud" in err["content"][0]["text"]
    handle.engine.dispose()


def test_delete_meeting_guard_and_apply(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    folder = _seed_meeting(handle.engine, meetings_dir, "m-del-1", "Delete me", ["acme"])
    assert folder.exists()
    app = _write_app(tmp_path, handle.engine)
    client = TestClient(app)

    plan = _parse_text(_call_tool(client, "delete_meeting", {"meetingId": "m-del-1", "dryRun": True}))
    assert plan["wouldDelete"] is True
    assert folder.exists()

    err = _call_tool(client, "delete_meeting", {"meetingId": "m-del-1", "expectedTitle": "Stale"})
    assert err.get("isError") is True

    done = _parse_text(
        _call_tool(client, "delete_meeting", {"meetingId": "m-del-1", "expectedTitle": "Delete me"})
    )
    assert done["deleted"] is True
    assert not folder.exists()
    handle.engine.dispose()


def test_split_meeting_dry_run_and_guards(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    _seed_meeting(handle.engine, meetings_dir, "m-split-1", "Split me", [])
    _seed_recording(handle.engine, "m-split-1")
    app = _write_app(tmp_path, handle.engine)
    client = TestClient(app)

    plan = _parse_text(
        _call_tool(client, "split_meeting", {"meetingId": "m-split-1", "splitMs": 30_000, "dryRun": True})
    )
    assert plan["dryRun"] is True
    assert plan["wouldSplit"] is True
    assert plan["part1RangeMs"] == [0, 30_000]
    assert plan["part2RangeMs"] == [30_000, 60_000]
    assert plan["summariesDropped"] is True

    err = _call_tool(
        client, "split_meeting", {"meetingId": "m-split-1", "splitMs": 30_000, "expectedTitle": "Stale"}
    )
    assert err.get("isError") is True

    err = _call_tool(client, "split_meeting", {"meetingId": "m-split-1", "splitMs": 1000, "dryRun": True})
    assert err.get("isError") is True

    err = _call_tool(client, "split_meeting", {"meetingId": "m-split-1"})
    assert err.get("isError") is True

    err = _call_tool(client, "split_meeting", {"meetingId": "missing", "splitMs": 30_000})
    assert err.get("isError") is True
    handle.engine.dispose()


def test_merge_meetings_dry_run_and_apply(tmp_path: Path, monkeypatch) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    folder_a = _seed_meeting(handle.engine, meetings_dir, "m-merge-1", "Merge one", ["work"])
    folder_b = _seed_meeting(handle.engine, meetings_dir, "m-merge-2", "Merge two", ["road"])
    (folder_a / "audio").mkdir(parents=True, exist_ok=True)
    (folder_b / "audio").mkdir(parents=True, exist_ok=True)
    (folder_a / "audio/meeting.m4a").write_bytes(b"clip-a")
    (folder_b / "audio/meeting.m4a").write_bytes(b"clip-b")
    now = 1_700_000_000_000
    with handle.engine.begin() as conn:
        for mid in ("m-merge-1", "m-merge-2"):
            conn.execute(
                text(
                    """
                    INSERT INTO recordings (
                      id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                      provider, provider_recording_id, created_at
                    ) VALUES (:rid, :mid, 'audio/meeting.m4a', 'audio/mp4', 60000, 1234,
                              :sha, 'upload', NULL, :now)
                    """
                ),
                {"rid": f"rec-{mid}", "mid": mid, "sha": f"sha-{mid}", "now": now},
            )
        conn.execute(
            text("UPDATE meetings SET start_time = :start, end_time = :end WHERE id = 'm-merge-2'"),
            {"start": now + 60_000, "end": now + 120_000},
        )

    from olive.services import merge as merge_module

    def fake_concat(sources, dst, ext):  # type: ignore[no-untyped-def]
        Path(dst).write_bytes(b"".join(Path(src).read_bytes() for src in sources))

    monkeypatch.setattr(merge_module, "_concat_audio", fake_concat)
    monkeypatch.setattr(merge_module, "_probe_duration_ms", lambda _path: None)

    app = _write_app(tmp_path, handle.engine)
    client = TestClient(app)

    plan = _parse_text(
        _call_tool(
            client,
            "merge_meetings",
            {"meetingIds": ["m-merge-2", "m-merge-1"], "dryRun": True},
        )
    )
    assert plan["dryRun"] is True
    assert plan["wouldMerge"] is True
    assert plan["mergedMeetingId"] == "m-merge-1"
    assert plan["deletedMeetingIds"] == ["m-merge-2"]
    assert [m["title"] for m in plan["meetings"]] == ["Merge one", "Merge two"]

    err = _call_tool(client, "merge_meetings", {"meetingIds": ["m-merge-1"]})
    assert err.get("isError") is True
    err = _call_tool(client, "merge_meetings", {"meetingIds": ["m-merge-1", "missing"]})
    assert err.get("isError") is True

    merged = _parse_text(_call_tool(client, "merge_meetings", {"meetingIds": ["m-merge-2", "m-merge-1"]}))
    assert merged["operation"] == "merge_meetings"
    assert merged["meeting"]["id"] == "m-merge-1"
    assert merged["deletedMeetingIds"] == ["m-merge-2"]
    assert sorted(merged["meeting"]["tags"]) == ["road", "work"]
    assert not folder_b.exists()
    with handle.engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM meetings")).scalar_one() == 1
        assert conn.execute(text("SELECT duration_ms FROM recordings")).scalar_one() == 120_000
    handle.engine.dispose()


def test_update_start_time_auto_title_summarize(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    _seed_meeting(handle.engine, meetings_dir, "m-st-1", "Time me", [])
    app = _write_app(
        tmp_path,
        handle.engine,
        titleService=_StubTitle(),
        summaryService=_StubSummary(),
    )
    client = TestClient(app)

    with handle.engine.connect() as conn:
        cur = conn.execute(text("SELECT start_time FROM meetings WHERE id='m-st-1'")).mappings().first()
        assert cur is not None
        current_start = int(cur["start_time"])

    st_plan = _parse_text(
        _call_tool(
            client,
            "update_meeting_start_time",
            {"meetingId": "m-st-1", "newStartTime": current_start + 86400000, "dryRun": True},
        )
    )
    assert st_plan["wouldUpdate"] is True

    st = _parse_text(
        _call_tool(
            client,
            "update_meeting_start_time",
            {
                "meetingId": "m-st-1",
                "newStartTime": current_start + 86400000,
                "expectedStartTime": current_start,
            },
        )
    )
    assert st["previousStartTime"] == current_start

    at = _parse_text(_call_tool(client, "auto_title_meeting", {"meetingId": "m-st-1"}))
    assert at["title"] == "Generated Title"

    sm = _parse_text(_call_tool(client, "summarize_meeting", {"meetingId": "m-st-1"}))
    assert sm["started"] is True
    handle.engine.dispose()


def _seed_summaries(engine, meetings_dir: str, meeting_id: str) -> None:
    from olive.layout import meeting_paths as _meeting_paths

    now = 1_700_000_000_000
    with engine.connect() as conn:
        row = (
            conn.execute(text("SELECT title, start_time FROM meetings WHERE id = :id"), {"id": meeting_id})
            .mappings()
            .first()
        )
        assert row is not None
    paths = _meeting_paths(meetings_dir, int(row["start_time"]), str(row["title"]), meeting_id)
    summaries_dir = Path(paths.folder) / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    (summaries_dir / "s1.md").write_text("# First\n", encoding="utf-8")
    (summaries_dir / "s2.md").write_text("# Second\n", encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO artifacts (
                  id, meeting_id, recording_id, kind, provider, format, path, created_at
                ) VALUES
                  ('sum-1', :mid, NULL, 'summary', 'test', 'md', 'summaries/s1.md', :now),
                  ('sum-2', :mid, NULL, 'summary', 'test', 'md', 'summaries/s2.md', :now2)
                """
            ),
            {"mid": meeting_id, "now": now, "now2": now + 1000},
        )
        conn.execute(
            text("UPDATE meetings SET primary_summary_artifact_id = 'sum-1' WHERE id = :id"),
            {"id": meeting_id},
        )


def test_delete_and_set_primary_summary_flow(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    _seed_meeting(handle.engine, meetings_dir, "m-sum-1", "Summary fixture", [])
    _seed_summaries(handle.engine, meetings_dir, "m-sum-1")
    app = _write_app(tmp_path, handle.engine)
    client = TestClient(app)

    _, list_body = _mcp_request(client, "tools/list")
    names = [t["name"] for t in list_body["result"]["tools"]]
    assert "delete_summary" in names
    assert "set_primary_summary" in names
    assert "reassign_speaker" in names
    assert "plaud_sync_trigger" in names

    # set_primary dryRun + guard + apply
    sp_plan = _parse_text(
        _call_tool(
            client,
            "set_primary_summary",
            {"meetingId": "m-sum-1", "summaryId": "sum-2", "dryRun": True},
        )
    )
    assert sp_plan["dryRun"] is True
    assert sp_plan["currentPrimarySummaryId"] == "sum-1"
    assert sp_plan["wouldUpdate"] is True

    sp_err = _call_tool(
        client,
        "set_primary_summary",
        {
            "meetingId": "m-sum-1",
            "summaryId": "sum-2",
            "expectedPrimarySummaryId": "stale",
        },
    )
    assert sp_err.get("isError") is True
    assert "Precondition failed" in sp_err["content"][0]["text"]

    sp_applied = _parse_text(
        _call_tool(
            client,
            "set_primary_summary",
            {
                "meetingId": "m-sum-1",
                "summaryId": "sum-2",
                "expectedPrimarySummaryId": "sum-1",
            },
        )
    )
    assert sp_applied["updated"] is True
    assert sp_applied["previousPrimarySummaryId"] == "sum-1"
    assert sp_applied["primarySummaryId"] == "sum-2"

    sp_same = _parse_text(
        _call_tool(client, "set_primary_summary", {"meetingId": "m-sum-1", "summaryId": "sum-2"})
    )
    assert sp_same["updated"] is False

    # delete non-primary dryRun + guard + apply
    del_plan = _parse_text(
        _call_tool(
            client,
            "delete_summary",
            {"meetingId": "m-sum-1", "summaryId": "sum-1", "dryRun": True},
        )
    )
    assert del_plan["dryRun"] is True
    assert del_plan["isPrimary"] is False
    assert del_plan["remainingSummaryIds"] == ["sum-2"]
    assert del_plan["nextPrimarySummaryId"] == "sum-2"

    del_err = _call_tool(
        client,
        "delete_summary",
        {"meetingId": "m-sum-1", "summaryId": "sum-1", "expectedIsPrimary": True},
    )
    assert del_err.get("isError") is True
    assert "Precondition failed" in del_err["content"][0]["text"]

    deleted = _parse_text(
        _call_tool(
            client,
            "delete_summary",
            {"meetingId": "m-sum-1", "summaryId": "sum-1", "expectedIsPrimary": False},
        )
    )
    assert deleted["deleted"] is True
    assert deleted["wasPrimary"] is False
    assert deleted["primarySummaryId"] == "sum-2"
    assert deleted["remainingSummaryIds"] == ["sum-2"]

    # delete primary falls back to None when nothing remains
    del_primary_plan = _parse_text(
        _call_tool(
            client,
            "delete_summary",
            {"meetingId": "m-sum-1", "summaryId": "sum-2", "dryRun": True},
        )
    )
    assert del_primary_plan["isPrimary"] is True
    assert del_primary_plan["nextPrimarySummaryId"] is None

    del_primary = _parse_text(
        _call_tool(client, "delete_summary", {"meetingId": "m-sum-1", "summaryId": "sum-2"})
    )
    assert del_primary["wasPrimary"] is True
    assert del_primary["primarySummaryId"] is None
    assert del_primary["remainingSummaryIds"] == []

    missing = _call_tool(client, "delete_summary", {"meetingId": "m-sum-1", "summaryId": "sum-9"})
    assert missing.get("isError") is True
    assert "not found" in missing["content"][0]["text"].lower()

    missing_primary = _call_tool(
        client,
        "set_primary_summary",
        {"meetingId": "m-sum-1", "summaryId": "sum-9"},
    )
    assert missing_primary.get("isError") is True
    handle.engine.dispose()


def _seed_transcript_with_speakers(engine, meetings_dir: str, meeting_id: str) -> None:
    from olive.layout import meeting_paths as _meeting_paths

    with engine.connect() as conn:
        row = (
            conn.execute(text("SELECT title, start_time FROM meetings WHERE id = :id"), {"id": meeting_id})
            .mappings()
            .first()
        )
        assert row is not None
    paths = _meeting_paths(meetings_dir, int(row["start_time"]), str(row["title"]), meeting_id)
    (Path(paths.folder) / "transcripts").mkdir(parents=True, exist_ok=True)
    transcript = {
        "segments": [
            {"speaker": "Speaker 1", "text": "Hello", "startMs": 0, "endMs": 1000},
            {"speaker": "Speaker 2", "text": "Hi", "startMs": 1000, "endMs": 2000},
            {"speaker": "Speaker 1", "text": "How are you?", "startMs": 2000, "endMs": 3000},
        ]
    }
    (Path(paths.folder) / "transcripts/local.json").write_text(json.dumps(transcript), encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO artifacts (id, meeting_id, recording_id, kind, provider, format, path, created_at) "
                "VALUES ('art-tr-1', :mid, NULL, 'transcript', 'local', 'json', 'transcripts/local.json', :now)"
            ),
            {"mid": meeting_id, "now": 1_700_000_000_000},
        )
        conn.execute(
            text("UPDATE meetings SET primary_transcript_artifact_id='art-tr-1' WHERE id=:id"),
            {"id": meeting_id},
        )


class _StubSpeakerService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def reassign_meeting_speaker(self, opts: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(opts))
        return {
            "updatedSegmentsCount": 2,
            "extractedVoiceprintsCount": 0,
            "localName": opts.get("toSpeakerName"),
        }

    async def get_speaker(self, speaker_id: str, options: dict[str, Any] | None = None) -> Any:
        return None

    async def list_speakers(self, options: dict[str, Any] | None = None) -> list[Any]:
        return []


class _StubTranscriptionWithProgress:
    def get_transcription_progress(self, meeting_id: str) -> Any:
        from olive.schemas import TranscriptionProgressUpdate

        return TranscriptionProgressUpdate(stage="transcribing", percent=42, message="decoding")

    def get_all_active_transcription_progress(self) -> dict[str, Any]:
        from olive.schemas import TranscriptionProgressUpdate

        return {"m-a": TranscriptionProgressUpdate(stage="transcribing", percent=10, message="hi")}


class _StubSummaryWithStatus:
    def get_summary_generation_status(self, meeting_id: str) -> Any:
        from olive.schemas import SummaryGenerationStatus

        return SummaryGenerationStatus(meetingId=meeting_id, stage="running", startedAt=1_700_000_000_000)


class _StubTemplateService:
    async def list_templates(self) -> list[Any]:
        from olive.schemas import Template

        return [
            Template(
                id="tpl-1",
                name="Default",
                description="test",
                systemPrompt="sys",
                userPrompt="{{transcript}}",
                isDefault=True,
                isBuiltin=True,
                createdAt=1_700_000_000_000,
                updatedAt=1_700_000_000_000,
            )
        ]

    async def get_template(self, template_id: str) -> Any:
        from olive.schemas import Template

        if template_id != "tpl-1":
            return None
        return Template(
            id="tpl-1",
            name="Default",
            description="test",
            systemPrompt="sys",
            userPrompt="{{transcript}}",
            isDefault=True,
            isBuiltin=True,
            createdAt=1_700_000_000_000,
            updatedAt=1_700_000_000_000,
        )


class _StubPlaudPoller:
    async def get_status(self) -> dict[str, Any]:
        return {"connected": True, "lastPollAt": 1_700_000_000_000, "pcsPending": 0}

    def trigger_background(self) -> bool:
        return True


def test_read_tools_include_new_batch(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    _seed_meeting(handle.engine, meetings_dir, "m-r-1", "Read batch", [])
    _seed_transcript_with_speakers(handle.engine, meetings_dir, "m-r-1")
    app = create_app(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "transcriptionService": _StubTranscriptionWithProgress(),
            "summaryService": _StubSummaryWithStatus(),
            "templateService": _StubTemplateService(),
            "plaudPoller": _StubPlaudPoller(),
        }
    )
    client = TestClient(app)
    _, body = _mcp_request(client, "tools/list")
    names = [t["name"] for t in body["result"]["tools"]]
    for expected in [
        "list_templates",
        "get_template",
        "get_transcription_status",
        "list_active_transcriptions",
        "get_summary_status",
        "plaud_status",
    ]:
        assert expected in names, f"missing read tool {expected}"

    # Exercise each read tool
    tpl_list = _parse_text(_call_tool(client, "list_templates", {}))
    assert len(tpl_list["templates"]) == 1
    tpl_one = _parse_text(_call_tool(client, "get_template", {"templateId": "tpl-1"}))
    assert tpl_one["template"]["id"] == "tpl-1"
    missing_tpl = _call_tool(client, "get_template", {"templateId": "nope"})
    assert missing_tpl.get("isError") is True

    trs = _parse_text(_call_tool(client, "get_transcription_status", {"meetingId": "m-r-1"}))
    assert trs["progress"]["percent"] == 42
    assert _call_tool(client, "get_transcription_status", {"meetingId": "missing"}).get("isError") is True

    active = _parse_text(_call_tool(client, "list_active_transcriptions", {}))
    assert "m-a" in active["active"]

    summ = _parse_text(_call_tool(client, "get_summary_status", {"meetingId": "m-r-1"}))
    assert summ["generating"] is True
    assert summ["status"]["stage"] == "running"

    plaud = _parse_text(_call_tool(client, "plaud_status", {}))
    assert plaud["connected"] is True
    handle.engine.dispose()


def test_reassign_and_plaud_trigger_write_flow(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    _seed_meeting(handle.engine, meetings_dir, "m-w-1", "Write batch", [])
    _seed_transcript_with_speakers(handle.engine, meetings_dir, "m-w-1")
    # seed a target speaker so toSpeakerId path has something to resolve (stub doesn't check DB)
    _seed_speaker(handle.engine, "spk-target", "Bob")
    speaker_stub = _StubSpeakerService()
    plaud_stub = _StubPlaudPoller()
    app = _write_app(tmp_path, handle.engine, speakerService=speaker_stub, plaudPoller=plaud_stub)
    client = TestClient(app)

    _, body = _mcp_request(client, "tools/list")
    names = [t["name"] for t in body["result"]["tools"]]
    assert "reassign_speaker" in names
    assert "plaud_sync_trigger" in names

    # dryRun paths
    dry = _parse_text(
        _call_tool(
            client,
            "reassign_speaker",
            {"meetingId": "m-w-1", "fromLabel": "Speaker 1", "toSpeakerName": "Alice", "dryRun": True},
        )
    )
    assert dry["dryRun"] is True
    assert dry["wouldReassign"] is True
    assert speaker_stub.calls == []

    dry2 = _parse_text(_call_tool(client, "plaud_sync_trigger", {"dryRun": True}))
    assert dry2["dryRun"] is True

    # expectedFromLabel guard
    err = _call_tool(
        client,
        "reassign_speaker",
        {
            "meetingId": "m-w-1",
            "fromLabel": "Speaker 1",
            "toSpeakerName": "Alice",
            "expectedFromLabel": "Nobody",
        },
    )
    assert err.get("isError") is True
    assert "Precondition failed" in err["content"][0]["text"]

    # missing target
    bad = _call_tool(client, "reassign_speaker", {"meetingId": "m-w-1", "fromLabel": "Speaker 1"})
    assert bad.get("isError") is True

    # real apply paths
    applied = _parse_text(
        _call_tool(
            client,
            "reassign_speaker",
            {"meetingId": "m-w-1", "fromLabel": "Speaker 1", "toSpeakerName": "Alice"},
        )
    )
    assert applied["operation"] == "reassign_speaker"
    assert applied["updatedSegmentsCount"] == 2
    assert len(speaker_stub.calls) == 1

    trig = _parse_text(_call_tool(client, "plaud_sync_trigger", {}))
    assert trig["status"] == "started"
    assert trig["alreadyRunning"] is False

    # without plaud poller
    handle2 = create_db(":memory:")
    _seed_meeting(handle2.engine, str(tmp_path / "meetings2"), "m-x", "No poller", [])
    app2 = _write_app(tmp_path, handle2.engine, speakerService=_StubSpeakerService())
    client2 = TestClient(app2)
    # plaud_status should error when poller missing (even though it's a read tool, it needs a poller)
    ps_err = _call_tool(client2, "plaud_status", {})
    assert ps_err.get("isError") is True
    handle2.engine.dispose()
    handle.engine.dispose()
