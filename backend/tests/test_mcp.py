"""MCP API endpoint and read-only tools tests (port of mcp.test.ts)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
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
    extra_headers: dict[str, str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    headers = {**MCP_HEADERS, **(extra_headers or {})}
    res = client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params or {},
        },
    )
    return res, res.json()


def test_mcp_is_available_without_token_on_loopback_and_lists_tools(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    app = create_app({"db": handle.engine, "meetingsDir": meetings_dir})
    client = TestClient(app)

    init_res, init_body = _mcp_request(
        client,
        "initialize",
        {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "olive-test", "version": "1.0.0"},
        },
    )
    assert init_res.status_code == 200
    assert init_body["result"]["serverInfo"]["name"] == "olive"

    tools_res, tools_body = _mcp_request(client, "tools/list")
    assert tools_res.status_code == 200
    tool_names = [t["name"] for t in tools_body["result"]["tools"]]
    assert tool_names == [
        "list_meetings",
        "list_tags",
        "get_meeting",
        "search_transcripts",
        "get_action_items",
        "list_speakers",
        "get_speaker_profile",
        "export_speaker_library",
        "list_templates",
        "get_template",
        "get_transcription_status",
        "list_active_transcriptions",
        "get_summary_status",
        "plaud_status",
        "insights_overview",
        "person_insights",
        "meeting_signals",
        "find_insight_events",
        "person_playbook",
    ]
    handle.engine.dispose()


def test_mcp_uses_token_authentication_when_configured() -> None:
    handle = create_db(":memory:")
    app = create_app({"db": handle.engine, "mcpToken": "mcp-secret"})
    client = TestClient(app)

    unauth_res, _ = _mcp_request(client, "tools/list")
    assert unauth_res.status_code == 401
    assert unauth_res.headers.get("www-authenticate") == "Bearer"

    auth_res, _ = _mcp_request(
        client,
        "tools/list",
        extra_headers={"Authorization": "Bearer mcp-secret"},
    )
    assert auth_res.status_code == 200
    handle.engine.dispose()


def test_mcp_falls_back_to_ingest_token_and_rejects_unauthenticated_non_loopback() -> None:
    handle = create_db(":memory:")
    app = create_app({"db": handle.engine, "ingestToken": "shared-secret"})
    client = TestClient(app)

    res, _ = _mcp_request(client, "tools/list")
    assert res.status_code == 401

    with pytest.raises(
        ValueError,
        match="OLIVE_MCP_TOKEN or OLIVE_INGEST_TOKEN is required when OLIVE_BIND_HOST is not loopback",
    ):
        create_app({"db": handle.engine, "bindHost": "0.0.0.0"})

    handle.engine.dispose()


def test_mcp_calls_meeting_transcript_action_item_and_speaker_tools(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    meeting_id = "meeting-mcp-1"
    speaker_id = "speaker-mcp-1"
    now = 1_700_000_000_000
    paths = meeting_paths(meetings_dir, now, "MCP fixture", meeting_id)

    Path(paths.transcripts_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.summaries_dir).mkdir(parents=True, exist_ok=True)
    (Path(paths.folder) / "transcripts/local.txt").write_text(
        "Discussed the launch plan and owner follow-up.", encoding="utf-8"
    )
    (Path(paths.folder) / "summaries/local.md").write_text(
        "- [ ] Matt: confirm launch date\n- [x] Alex: send agenda", encoding="utf-8"
    )

    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, 'MCP fixture', :now, :end, 'upload', 'ready', '[]',
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {"id": meeting_id, "now": now, "end": now + 60_000},
        )
        conn.execute(
            text(
                """
                INSERT INTO speakers (
                  id, name, provider_ids, enrolled_at, enrollment_clip_paths, created_at
                ) VALUES (
                  :sid, 'Matt', '{}', NULL, '[]', :now
                )
                """
            ),
            {"sid": speaker_id, "now": now},
        )
        conn.execute(
            text(
                """
                INSERT INTO meeting_speakers (meeting_id, speaker_id, evidence_artifact_id)
                VALUES (:mid, :sid, NULL)
                """
            ),
            {"mid": meeting_id, "sid": speaker_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO artifacts (
                  id, meeting_id, recording_id, kind, provider, format, path, created_at
                ) VALUES
                  ('artifact-mcp-transcript', :mid, NULL, 'transcript', 'local', 'txt', 'transcripts/local.txt', :now),
                  ('artifact-mcp-summary', :mid, NULL, 'summary', 'local', 'md', 'summaries/local.md', :now)
                """
            ),
            {"mid": meeting_id, "now": now},
        )
        conn.execute(
            text(
                """
                UPDATE meetings
                SET primary_transcript_artifact_id = 'artifact-mcp-transcript',
                    primary_summary_artifact_id = 'artifact-mcp-summary'
                WHERE id = :id
                """
            ),
            {"id": meeting_id},
        )

    app = create_app({"db": handle.engine, "meetingsDir": meetings_dir})
    client = TestClient(app)

    _, list_body = _mcp_request(
        client,
        "tools/call",
        {"name": "list_meetings", "arguments": {"search": "MCP fixture"}},
    )
    assert list_body["result"].get("isError") is not True
    slim_meeting = json.loads(list_body["result"]["content"][0]["text"])["meetings"][0]
    assert slim_meeting["id"] == meeting_id
    assert set(slim_meeting.keys()) == {"id", "title", "startTime", "endTime", "source", "tags"}

    _, full_body = _mcp_request(
        client,
        "tools/call",
        {
            "name": "list_meetings",
            "arguments": {"search": "MCP fixture", "includeDetails": True},
        },
    )
    full_meeting = json.loads(full_body["result"]["content"][0]["text"])["meetings"][0]
    assert full_meeting["id"] == meeting_id
    assert full_meeting["status"] == "ready"
    assert "audioStatus" in full_meeting
    assert "primaryTranscriptArtifactId" in full_meeting

    _, detail_body = _mcp_request(
        client,
        "tools/call",
        {"name": "get_meeting", "arguments": {"meetingId": meeting_id}},
    )
    detail = json.loads(detail_body["result"]["content"][0]["text"])
    assert "launch plan" in (detail["transcriptContent"] or "")
    assert detail["summaryContent"] is not None
    assert len(detail["summaries"]) == 1
    assert detail["summaries"][0]["content"]
    # Meeting speakers default to vector-free summaries.
    assert detail["speakers"][0]["name"] == "Matt"
    assert detail["speakers"][0]["providerIds"] == {}

    _, no_text_body = _mcp_request(
        client,
        "tools/call",
        {
            "name": "get_meeting",
            "arguments": {
                "meetingId": meeting_id,
                "includeTranscript": False,
                "includeSummaries": False,
            },
        },
    )
    slim_detail = json.loads(no_text_body["result"]["content"][0]["text"])
    assert slim_detail["transcriptContent"] is None
    assert slim_detail["summaryContent"] is None
    assert slim_detail["summaries"] == []
    assert slim_detail["meeting"]["id"] == meeting_id

    _, slim_spk_body = _mcp_request(
        client,
        "tools/call",
        {"name": "get_speaker_profile", "arguments": {"speakerId": speaker_id}},
    )
    slim_spk = json.loads(slim_spk_body["result"]["content"][0]["text"])
    assert slim_spk["speaker"]["name"] == "Matt"
    assert "providerIds" not in slim_spk["speaker"]
    assert slim_spk["speaker"]["voiceprintCounts"] == {}

    _, full_spk_body = _mcp_request(
        client,
        "tools/call",
        {
            "name": "get_speaker_profile",
            "arguments": {"speakerId": speaker_id, "includeVoiceprints": True},
        },
    )
    full_spk = json.loads(full_spk_body["result"]["content"][0]["text"])
    assert "providerIds" in full_spk["speaker"]

    _, tr_body = _mcp_request(
        client,
        "tools/call",
        {"name": "search_transcripts", "arguments": {"query": "launch plan"}},
    )
    assert json.loads(tr_body["result"]["content"][0]["text"])["matches"][0]["meeting"]["id"] == meeting_id

    _, act_body = _mcp_request(
        client,
        "tools/call",
        {
            "name": "get_action_items",
            "arguments": {"meetingId": meeting_id, "includeCompleted": False},
        },
    )
    assert json.loads(act_body["result"]["content"][0]["text"])["actionItems"] == [
        {
            "meetingId": meeting_id,
            "meetingTitle": "MCP fixture",
            "text": "Matt: confirm launch date",
            "completed": False,
        }
    ]

    _, spk_body = _mcp_request(
        client,
        "tools/call",
        {"name": "get_speaker_profile", "arguments": {"speakerId": speaker_id}},
    )
    assert json.loads(spk_body["result"]["content"][0]["text"])["speaker"]["name"] == "Matt"
    handle.engine.dispose()
