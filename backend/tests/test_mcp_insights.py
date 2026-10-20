"""MCP tools for Olive Insights: read tools, the insights-enabled gate, and the gated backfill write tool."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from olive.app import create_app
from olive.db import DbHandle, create_db
from olive.services.insights import extract, llm_query

MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "Host": "127.0.0.1",
}

ME = "spk-me"
ALICE = "spk-alice"

READ_TOOLS = [
    "insights_overview",
    "person_insights",
    "meeting_signals",
    "find_insight_events",
    "person_playbook",
]


def _ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, 12, tzinfo=UTC).timestamp() * 1000)


class Env:
    def __init__(self, tmp_path: Path, **opts: Any) -> None:
        self.db: DbHandle = create_db(":memory:")
        self.client = TestClient(
            create_app(
                {
                    "db": self.db.engine,
                    "meetingsDir": str(tmp_path / "meetings"),
                    "configDir": str(tmp_path / "cfg"),
                    **opts,
                }
            )
        )

    def run(self, sql: str, params: dict[str, Any] | None = None) -> None:
        with self.db.engine.begin() as conn:
            conn.execute(text(sql), params or {})

    def seed(self) -> None:
        for sid, name in ((ME, "Me"), (ALICE, "Alice")):
            self.run("INSERT INTO speakers (id, name, created_at) VALUES (:i, :n, 0)", {"i": sid, "n": name})
        for i in range(3):
            mid, start = f"m{i}", _ms(2025, i + 1, 5)
            self.run(
                "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, "
                "created_at, updated_at) VALUES (:id, :id, :s, :e, 'upload', 'ready', '[]', 0, 0)",
                {"id": mid, "s": start, "e": start + 60_000},
            )
            self.run(
                "INSERT INTO artifacts (id, meeting_id, kind, provider, format, path, created_at) "
                "VALUES (:a, :m, 'transcript', 'x', 'json', 't.json', 0)",
                {"a": f"art-{mid}", "m": mid},
            )
            self.run(
                "UPDATE meetings SET primary_transcript_artifact_id = :a WHERE id = :m",
                {"a": f"art-{mid}", "m": mid},
            )
            for j, sid in enumerate((ME, ALICE)):
                self.run(
                    "INSERT INTO meeting_speaker_signals (id, meeting_id, speaker_id, speaker_label, "
                    "identity_source, identity_confidence, dimensions, talk_ms, turn_count, interrupt_count, "
                    "created_at) VALUES (:id, :m, :s, :s, 'voiceprint', 0.9, :d, 1000, 4, 1, 0)",
                    {
                        "id": f"sig-{i}-{j}",
                        "m": mid,
                        "s": sid,
                        "d": json.dumps({"warmth": 0.4 + i * 0.1, "hostility": 0.1}),
                    },
                )


def _mcp_call(client: TestClient, name: str, args: dict[str, Any]) -> dict[str, Any]:
    res = client.post(
        "/mcp",
        headers=MCP_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        },
    )
    assert res.status_code == 200
    return res.json()["result"]


def _parse(result: dict[str, Any]) -> Any:
    assert result.get("isError") is not True, result
    return json.loads(result["content"][0]["text"])


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(extract.ENABLED_ENV, "1")


@pytest.fixture
def disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(extract.ENABLED_ENV, "0")


@pytest.fixture
def env(tmp_path: Path, enabled: None) -> Env:
    e = Env(tmp_path)
    e.seed()
    return e


def test_read_tools_listed_without_write_flag(env: Env) -> None:
    res = env.client.post(
        "/mcp",
        headers=MCP_HEADERS,
        json={"jsonrpc": "2.0", "id": "1", "method": "tools/list", "params": {}},
    )
    names = [t["name"] for t in res.json()["result"]["tools"]]
    for tool in READ_TOOLS:
        assert tool in names
    assert "backfill_insights" not in names


def test_overview_person_and_meeting_return_json_text(env: Env) -> None:
    overview = _parse(
        _mcp_call(env.client, "insights_overview", {"from": _ms(2025, 2, 1), "speakerId": ALICE})
    )
    assert overview["speakerId"] == ALICE
    assert overview["from"] == _ms(2025, 2, 1)
    assert overview["coverage"]["totalMeetings"] == 2
    assert overview["series"]["warmth"]

    person = _parse(_mcp_call(env.client, "person_insights", {"speakerId": ALICE}))
    assert person["speakerId"] == ALICE
    assert person["name"] == "Alice"
    assert person["meetings"]

    meeting = _parse(_mcp_call(env.client, "meeting_signals", {"meetingId": "m0"}))
    assert meeting["meetingId"] == "m0"
    assert len(meeting["speakers"]) == 2


def test_read_tool_argument_errors(env: Env) -> None:
    assert _mcp_call(env.client, "person_insights", {}).get("isError") is True
    assert _mcp_call(env.client, "meeting_signals", {}).get("isError") is True
    missing = _mcp_call(env.client, "meeting_signals", {"meetingId": "nope"})
    assert missing.get("isError") is True
    assert "not found" in missing["content"][0]["text"].lower()
    assert _mcp_call(env.client, "find_insight_events", {"query": "  "}).get("isError") is True
    assert _mcp_call(env.client, "person_playbook", {}).get("isError") is True
    assert _mcp_call(env.client, "insights_overview", {"from": "not-a-date"}).get("isError") is True


def test_find_events_and_playbook_use_llm_service(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def find(self: Any, query: str, limit: int = 5) -> list[dict[str, Any]]:
        seen["find"] = (query, limit)
        return [{"meetingId": "m0", "startMs": 1000, "quote": "q"}]

    async def playbook(self: Any, speaker_id: str) -> dict[str, Any]:
        seen["playbook"] = speaker_id
        return {"speakerId": speaker_id, "summary": "s", "doMore": [], "easeOff": []}

    monkeypatch.setattr(llm_query.InsightsLlmService, "find_events", find)
    monkeypatch.setattr(llm_query.InsightsLlmService, "playbook", playbook)

    found = _parse(_mcp_call(env.client, "find_insight_events", {"query": "when did I snap", "limit": 3}))
    assert found["query"] == "when did I snap"
    assert found["results"][0]["meetingId"] == "m0"
    assert seen["find"] == ("when did I snap", 3)

    pb = _parse(_mcp_call(env.client, "person_playbook", {"speakerId": ALICE}))
    assert pb["speakerId"] == ALICE
    assert seen["playbook"] == ALICE


def test_find_events_limit_is_clamped(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    async def find(self: Any, query: str, limit: int = 5) -> list[dict[str, Any]]:
        seen.append(limit)
        return []

    monkeypatch.setattr(llm_query.InsightsLlmService, "find_events", find)
    _mcp_call(env.client, "find_insight_events", {"query": "x", "limit": 10**9})
    _mcp_call(env.client, "find_insight_events", {"query": "x", "limit": -5})
    assert seen == [llm_query.MAX_FIND_EVENTS_LIMIT, 1]


def test_tools_error_when_insights_disabled(tmp_path: Path, disabled: None) -> None:
    e = Env(tmp_path, mcpWriteEnabled=True)
    e.seed()
    calls = {
        "insights_overview": {},
        "person_insights": {"speakerId": ALICE},
        "meeting_signals": {"meetingId": "m0"},
        "find_insight_events": {"query": "x"},
        "person_playbook": {"speakerId": ALICE},
        "backfill_insights": {},
    }
    for name, args in calls.items():
        res = _mcp_call(e.client, name, args)
        assert res.get("isError") is True, name
        assert "disabled" in res["content"][0]["text"].lower()


def test_backfill_blocked_without_write_flag(env: Env) -> None:
    res = _mcp_call(env.client, "backfill_insights", {})
    assert res.get("isError") is True
    assert "OLIVE_MCP_WRITE=1" in res["content"][0]["text"]


def test_backfill_dry_run_counts_candidates_without_enqueuing(
    tmp_path: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = Env(tmp_path, mcpWriteEnabled=True)
    e.seed()
    enqueued: list[str] = []
    monkeypatch.setattr(extract, "get_enqueue_hook", lambda: enqueued.append)

    names = [
        t["name"]
        for t in e.client.post(
            "/mcp", headers=MCP_HEADERS, json={"jsonrpc": "2.0", "id": "1", "method": "tools/list"}
        ).json()["result"]["tools"]
    ]
    assert "backfill_insights" in names

    plan = _parse(_mcp_call(e.client, "backfill_insights", {"dryRun": True, "from": _ms(2025, 2, 1)}))
    assert plan["dryRun"] is True
    assert plan["candidates"] == 2
    assert enqueued == []

    capped = _parse(_mcp_call(e.client, "backfill_insights", {"dryRun": True, "limit": 1}))
    assert capped["candidates"] == 1
    assert enqueued == []


def test_backfill_enqueues_when_not_dry_run(
    tmp_path: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = Env(tmp_path, mcpWriteEnabled=True)
    e.seed()
    enqueued: list[str] = []

    def hook(meeting_id: str, force: bool = False) -> bool:
        assert force is True
        enqueued.append(meeting_id)
        return True

    monkeypatch.setattr(extract, "get_enqueue_hook", lambda: hook)
    res = _parse(_mcp_call(e.client, "backfill_insights", {"force": True}))
    assert res == {"dryRun": False, "queued": 3, "skipped": 0}
    assert enqueued == ["m0", "m1", "m2"]
    assert _mcp_call(e.client, "backfill_insights", {"limit": "x"}).get("isError") is True
