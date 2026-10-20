"""HTTP API tests for Olive Insights (/api/insights/*) and POST /api/transcripts/import."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from olive import schemas as s
from olive.app import create_app, export_openapi_schema
from olive.db import DbHandle, create_db
from olive.services.insights import extract, llm_query

ME = "spk-me"
ALICE = "spk-alice"
BOB = "spk-bob"

ZOOM_VTT = """WEBVTT

1
00:00:01.000 --> 00:00:04.500
Alice Smith: Hello everyone, thanks for joining.

2
00:00:06.500 --> 00:00:09.250
Bob Jones: Sounds good.
"""
ZOOM_NAME = "GMT20240315-150000_Recording.transcript.vtt"


def _ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, 12, tzinfo=UTC).timestamp() * 1000)


class Env:
    def __init__(self, tmp_path: Path, **opts: Any) -> None:
        self.db: DbHandle = create_db(":memory:")
        self.tmp = tmp_path
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

    def speaker(self, sid: str, name: str) -> None:
        self.run("INSERT INTO speakers (id, name, created_at) VALUES (:i, :n, 0)", {"i": sid, "n": name})

    def meeting(self, mid: str, start: int, *, transcribed: bool = True) -> None:
        self.run(
            "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, created_at, updated_at) "
            "VALUES (:id, :id, :s, :e, 'upload', 'ready', '[]', 0, 0)",
            {"id": mid, "s": start, "e": start + 60_000},
        )
        if transcribed:
            self.run(
                "INSERT INTO artifacts (id, meeting_id, kind, provider, format, path, created_at) "
                "VALUES (:a, :m, 'transcript', 'x', 'json', 't.json', 0)",
                {"a": f"art-{mid}", "m": mid},
            )
            self.run(
                "UPDATE meetings SET primary_transcript_artifact_id = :a WHERE id = :m",
                {"a": f"art-{mid}", "m": mid},
            )

    def signal(self, n: int, mid: str, sid: str, dims: dict[str, float]) -> None:
        self.run(
            "INSERT INTO meeting_speaker_signals (id, meeting_id, speaker_id, speaker_label, identity_source, "
            "identity_confidence, dimensions, talk_ms, turn_count, interrupt_count, created_at) "
            "VALUES (:id, :m, :s, :s, 'voiceprint', 0.9, :d, 1000, 4, 1, 0)",
            {"id": f"sig-{n}", "m": mid, "s": sid, "d": json.dumps(dims)},
        )

    def moment(self, n: int, mid: str, start: int, sid: str, toward: str, valence: str) -> None:
        self.run(
            "INSERT INTO meeting_moments (id, meeting_id, speaker_id, toward_speaker_id, start_ms, end_ms, "
            "event_type, valence, dimension, severity, quote, verified, created_at) "
            "VALUES (:id, :m, :s, :t, :st, :st, 'evt', :v, 'hostility', 0.8, 'q', 1, 0)",
            {"id": f"mom-{n}", "m": mid, "s": sid, "t": toward, "st": start, "v": valence},
        )

    def seed(self) -> None:
        for sid, name in ((ME, "Me"), (ALICE, "Alice"), (BOB, "Bob")):
            self.speaker(sid, name)
        n = 0
        for i, (mo, day) in enumerate(((1, 5), (2, 6), (3, 7), (4, 8), (5, 9))):
            mid = f"m{i}"
            start = _ms(2025, mo, day)
            self.meeting(mid, start, transcribed=i < 4)
            if i == 4:
                continue
            for sid, warmth in ((ME, 0.5), (ALICE, 0.4 + i * 0.05)):
                n += 1
                self.signal(n, mid, sid, {"warmth": warmth, "hostility": 0.1})
            n += 1
            self.moment(n, mid, start + 1000, ME, ALICE, "good" if i % 2 else "bad")
            n += 1
            self.moment(n, mid, start + 2000, ALICE, ME, "good")


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


# -- read routes --------------------------------------------------------------


def test_overview_shape(env: Env) -> None:
    res = env.client.get("/api/insights/overview", params={"bucket": "month"})
    assert res.status_code == 200
    body = s.InsightsOverview.model_validate(res.json())
    assert body.bucket == "month"
    assert body.coverage.totalMeetings == 5
    assert body.coverage.analyzedMeetings == 4
    assert body.series["warmth"]
    assert all(p.meetingIds and isinstance(p.startMs, int) for p in body.series["warmth"])
    assert "from" in res.json()


def test_overview_filters_and_bad_input(env: Env) -> None:
    res = env.client.get("/api/insights/overview", params={"from": _ms(2025, 3, 1), "speaker": ALICE})
    assert res.status_code == 200
    assert res.json()["speakerId"] == ALICE
    assert res.json()["from"] == _ms(2025, 3, 1)
    assert env.client.get("/api/insights/overview", params={"bucket": "year"}).status_code == 400
    assert env.client.get("/api/insights/overview", params={"from": "not-a-date"}).status_code == 400


def test_person_shape(env: Env) -> None:
    res = env.client.get(f"/api/insights/person/{ALICE}", params={"bucket": "week"})
    assert res.status_code == 200
    body = s.InsightsPerson.model_validate(res.json())
    assert body.speakerId == ALICE
    assert body.name == "Alice"
    assert body.bucket == "week"
    assert body.meetings and all(isinstance(m.startMs, int) for m in body.meetings)


def test_meeting_shape_and_unknown(env: Env) -> None:
    res = env.client.get("/api/insights/meeting/m0")
    assert res.status_code == 200
    body = s.InsightsMeetingSignals.model_validate(res.json())
    assert body.meetingId == "m0"
    assert len(body.speakers) == 2
    assert all(m.meetingId == "m0" for m in body.moments)
    missing = env.client.get("/api/insights/meeting/nope")
    assert missing.status_code == 404
    assert "error" in missing.json()


def test_dyads_shape(env: Env) -> None:
    assert env.client.put("/api/insights/me", json={"speakerId": ME, "aliases": ["phil"]}).status_code == 200
    res = env.client.get("/api/insights/dyads")
    assert res.status_code == 200
    body = s.InsightsDyads.model_validate(res.json())
    assert body.meId == ME
    assert {d.direction for d in body.dyads} == {"meToThem", "themToMe"}
    assert all(d.meetingIds and isinstance(d.startMs, int) for d in body.dyads)


def test_me_round_trip(env: Env) -> None:
    assert s.InsightsMe.model_validate(env.client.get("/api/insights/me").json()).speakerId is None
    put = env.client.put("/api/insights/me", json={"speakerId": ME, "aliases": ["Phil", "P"]})
    assert put.status_code == 200
    assert put.json() == {"speakerId": ME, "aliases": ["Phil", "P"]}
    assert env.client.get("/api/insights/me").json() == {"speakerId": ME, "aliases": ["Phil", "P"]}


def test_me_rejects_bad_body(env: Env) -> None:
    assert env.client.put("/api/insights/me", json={"aliases": "nope"}).status_code == 400
    assert env.client.put("/api/insights/me", content="not json").status_code == 400


def test_status_counts(env: Env) -> None:
    res = env.client.get("/api/insights/status")
    assert res.status_code == 200
    body = s.InsightsStatus.model_validate(res.json())
    assert (body.transcribedMeetings, body.analyzedMeetings, body.pendingMeetings) == (4, 4, 0)
    env.meeting("fresh", _ms(2025, 6, 1))
    body = s.InsightsStatus.model_validate(env.client.get("/api/insights/status").json())
    assert (body.transcribedMeetings, body.analyzedMeetings, body.pendingMeetings) == (5, 4, 1)


# -- llm routes (service stubbed: the model is out of scope here) -------------


def test_find_playbook_coaching(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    row = {
        "momentId": "mom-1",
        "meetingId": "m0",
        "meetingTitle": "m0",
        "startMs": 1000,
        "endMs": 2000,
        "quote": "q",
        "eventType": "evt",
        "valence": "bad",
        "speaker": "Me",
        "speakerId": ME,
        "toward": "Alice",
        "towardSpeakerId": ALICE,
        "score": 0.9,
        "enacted": True,
        "reason": "r",
    }
    seen: dict[str, Any] = {}

    async def find(self: Any, query: str, limit: int = 5) -> list[dict[str, Any]]:
        seen["find"] = (query, limit)
        return [row]

    async def playbook(self: Any, speaker_id: str) -> dict[str, Any]:
        seen["playbook"] = speaker_id
        return {
            "speakerId": speaker_id,
            "cached": False,
            "summary": "s",
            "doMore": [{"point": "p", "evidence": [{"quote": "q", "meetingId": "m0", "startMs": 5}]}],
            "easeOff": [],
        }

    async def coaching(self: Any, scope: str = "all", speaker_id: str | None = None) -> dict[str, Any]:
        seen["coaching"] = (scope, speaker_id)
        return {"scope": scope, "speakerId": speaker_id, "cached": False, "strengths": [], "growthAreas": []}

    monkeypatch.setattr(llm_query.InsightsLlmService, "find_events", find)
    monkeypatch.setattr(llm_query.InsightsLlmService, "playbook", playbook)
    monkeypatch.setattr(llm_query.InsightsLlmService, "coaching", coaching)

    res = env.client.get("/api/insights/find", params={"q": "when did I snap", "limit": 3})
    assert res.status_code == 200
    body = s.InsightsFindResponse.model_validate(res.json())
    assert body.results[0].meetingId == "m0" and body.results[0].startMs == 1000
    assert seen["find"] == ("when did I snap", 3)
    assert env.client.get("/api/insights/find").status_code == 400
    assert env.client.get("/api/insights/find", params={"q": "x", "limit": "abc"}).status_code == 400

    pb = env.client.get(f"/api/insights/playbook/{ALICE}")
    assert pb.status_code == 200
    assert s.InsightsPlaybook.model_validate(pb.json()).doMore[0].evidence[0].startMs == 5
    assert seen["playbook"] == ALICE

    all_scope = env.client.get("/api/insights/coaching")
    assert all_scope.status_code == 200
    assert s.InsightsCoaching.model_validate(all_scope.json()).scope == "all"
    assert seen["coaching"] == ("all", None)
    spk = env.client.get("/api/insights/coaching", params={"speaker": ALICE})
    assert spk.status_code == 200
    assert seen["coaching"] == ("speaker", ALICE)


def test_llm_value_errors_are_400(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(self: Any, *a: Any, **k: Any) -> Any:
        raise ValueError("bad request")

    monkeypatch.setattr(llm_query.InsightsLlmService, "playbook", boom)
    res = env.client.get(f"/api/insights/playbook/{ALICE}")
    assert res.status_code == 400
    assert res.json() == {"error": "bad request"}


# -- backfill -------------------------------------------------------------------


def test_backfill_returns_202(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_backfill(db: Any, meetings_dir: Any, config_dir: Any, **kw: Any) -> dict[str, int]:
        captured.update(kw)
        return {"queued": 3, "skipped": 1}

    monkeypatch.setattr("olive.services.insights.backfill.backfill", fake_backfill)
    res = env.client.post(
        "/api/insights/backfill", json={"from": "2025-01-01", "to": 99, "force": True, "limit": 2}
    )
    assert res.status_code == 202
    assert s.InsightsBackfillResult.model_validate(res.json()).queued == 3
    assert captured == {
        "frm": int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000),
        "to": 99,
        "force": True,
        "limit": 2,
    }
    assert env.client.post("/api/insights/backfill").status_code == 202
    assert env.client.post("/api/insights/backfill", json={"limit": "x"}).status_code == 400


# -- gating ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/insights/overview"),
        ("get", f"/api/insights/person/{ALICE}"),
        ("get", "/api/insights/meeting/m0"),
        ("get", "/api/insights/dyads"),
        ("get", "/api/insights/find?q=x"),
        ("get", f"/api/insights/playbook/{ALICE}"),
        ("get", "/api/insights/coaching"),
        ("get", "/api/insights/me"),
        ("put", "/api/insights/me"),
        ("post", "/api/insights/backfill"),
        ("get", "/api/insights/status"),
    ],
)
def test_routes_404_when_disabled(tmp_path: Path, disabled: None, method: str, path: str) -> None:
    e = Env(tmp_path)
    res = getattr(e.client, method)(path)
    assert res.status_code == 404
    assert res.json() == {"error": "insights disabled"}


def test_enabled_by_settings_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(extract.ENABLED_ENV, raising=False)
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "settings.json").write_text(json.dumps({"insightsEnabled": True}))
    e = Env(tmp_path)
    assert e.client.get("/api/insights/status").status_code == 200


# -- transcript import ------------------------------------------------------------


def _import_multipart(client: TestClient, **extra: Any) -> Any:
    return client.post(
        "/api/transcripts/import",
        files={"file": (ZOOM_NAME, ZOOM_VTT.encode(), "text/vtt")},
        data=extra.pop("data", {"meAliases": "Alice Smith"}),
        **extra,
    )


def test_import_multipart_201(tmp_path: Path, enabled: None) -> None:
    e = Env(tmp_path)
    res = _import_multipart(e.client)
    assert res.status_code == 201
    body = s.TranscriptImportResult.model_validate(res.json())
    assert body.deduped is False
    assert body.format == "vtt"
    assert body.segmentCount == 2
    assert body.startTime == int(datetime(2024, 3, 15, 15, tzinfo=UTC).timestamp() * 1000)
    with e.db.engine.connect() as conn:
        src = conn.execute(text("SELECT source FROM meetings WHERE id = :i"), {"i": body.meetingId}).scalar()
    assert src == "import"


def test_import_multipart_form_fields(tmp_path: Path, enabled: None) -> None:
    e = Env(tmp_path)
    res = _import_multipart(
        e.client,
        data={
            "title": "Weekly sync",
            "startTime": "2024-05-01T10:00:00Z",
            "subtype": "zoom",
            "format": "vtt",
        },
    )
    assert res.status_code == 201
    assert res.json()["startTime"] == int(datetime(2024, 5, 1, 10, tzinfo=UTC).timestamp() * 1000)
    assert res.json()["subtype"] == "zoom"
    with e.db.engine.connect() as conn:
        title = conn.execute(
            text("SELECT title FROM meetings WHERE id = :i"), {"i": res.json()["meetingId"]}
        ).scalar()
    assert title == "Weekly sync"


def test_import_json_201_then_dedup_200(tmp_path: Path, enabled: None) -> None:
    e = Env(tmp_path)
    payload = {"content": ZOOM_VTT, "filename": ZOOM_NAME, "meAliases": ["Alice Smith"]}
    first = e.client.post("/api/transcripts/import", json=payload)
    assert first.status_code == 201
    again = e.client.post("/api/transcripts/import", json=payload)
    assert again.status_code == 200
    assert again.json()["deduped"] is True
    assert again.json()["meetingId"] == first.json()["meetingId"]
    # The UI reads warnings/unmappedNames/speakers without checking `deduped` first, so a
    # deduplicated import has the same complete shape as a fresh one.
    assert set(again.json()) == set(first.json())
    assert again.json()["warnings"] == [] and again.json()["unmappedNames"] == []
    assert again.json()["format"] == first.json()["format"]
    assert again.json()["segmentCount"] == first.json()["segmentCount"]
    assert again.json()["startTime"] == first.json()["startTime"]
    multipart_again = _import_multipart(e.client)
    assert multipart_again.status_code == 200


def test_import_enqueues_analysis(tmp_path: Path, enabled: None) -> None:
    class StubInsights:
        def __init__(self) -> None:
            self.queued: list[str] = []

        def is_enabled(self) -> bool:
            return True

        async def enqueue_analyze(self, meeting_id: str) -> bool:
            self.queued.append(meeting_id)
            return True

    stub = StubInsights()
    e = Env(tmp_path, insightsService=stub)
    res = e.client.post("/api/transcripts/import", json={"content": ZOOM_VTT, "filename": ZOOM_NAME})
    assert res.status_code == 201
    assert stub.queued == [res.json()["meetingId"]]


@pytest.mark.parametrize("content", ["", "   \n\n", "\x00\x01\x02 not a transcript"])
def test_import_garbage_400(tmp_path: Path, enabled: None, content: str) -> None:
    e = Env(tmp_path)
    res = e.client.post(
        "/api/transcripts/import", json={"content": content, "filename": "x.vtt", "startTime": 1700000000000}
    )
    assert res.status_code == 400
    assert "error" in res.json()


def test_import_bad_request_shapes_400(tmp_path: Path, enabled: None) -> None:
    e = Env(tmp_path)
    assert e.client.post("/api/transcripts/import", json={"filename": "x.vtt"}).status_code == 400
    assert e.client.post("/api/transcripts/import", json=["nope"]).status_code == 400
    assert (
        e.client.post(
            "/api/transcripts/import", content="plain", headers={"content-type": "text/plain"}
        ).status_code
        == 400
    )
    assert (
        e.client.post(
            "/api/transcripts/import", data={"title": "no file"}, files={"x": ("x", b"y")}
        ).status_code
        == 400
    )
    bad_format = e.client.post(
        "/api/transcripts/import", json={"content": ZOOM_VTT, "filename": ZOOM_NAME, "format": "docx"}
    )
    assert bad_format.status_code == 400
    bad_start = e.client.post(
        "/api/transcripts/import", json={"content": ZOOM_VTT, "filename": ZOOM_NAME, "startTime": "yesterday"}
    )
    assert bad_start.status_code == 400


def test_import_missing_date_422(tmp_path: Path, enabled: None) -> None:
    e = Env(tmp_path)
    res = e.client.post(
        "/api/transcripts/import", json={"content": "Alice: hi\nBob: yo\n", "filename": "notes.txt"}
    )
    assert res.status_code == 422
    assert "startTime" in res.json()["error"]
    ok = e.client.post(
        "/api/transcripts/import",
        json={"content": "Alice: hi\nBob: yo\n", "filename": "notes.txt", "startTime": "2024-01-02"},
    )
    assert ok.status_code == 201


def test_import_too_large_413(tmp_path: Path, enabled: None) -> None:
    e = Env(tmp_path, maxTranscriptImportBytes=100)
    res = e.client.post("/api/transcripts/import", json={"content": "x" * 500, "filename": "big.txt"})
    assert res.status_code == 413
    big_form = e.client.post("/api/transcripts/import", files={"file": ("big.txt", b"x" * 500, "text/plain")})
    assert big_form.status_code == 413


def test_import_multibyte_json_over_byte_cap_413(tmp_path: Path, enabled: None) -> None:
    e = Env(tmp_path, maxTranscriptImportBytes=1000)
    content = "é" * 600  # 600 characters but 1200 bytes
    res = e.client.post("/api/transcripts/import", json={"content": content, "filename": "big.txt"})
    assert res.status_code == 413


def test_import_chunked_body_without_content_length_413(tmp_path: Path, enabled: None) -> None:
    e = Env(tmp_path, maxTranscriptImportBytes=1000)

    def chunks() -> Any:
        for _ in range(200):
            yield b"x" * 1024

    json_res = e.client.post(
        "/api/transcripts/import", content=chunks(), headers={"content-type": "application/json"}
    )
    assert json_res.status_code == 413
    multi = e.client.post(
        "/api/transcripts/import",
        content=chunks(),
        headers={"content-type": "multipart/form-data; boundary=zzz"},
    )
    assert multi.status_code == 413


def test_import_huge_digit_start_time_400(tmp_path: Path, enabled: None) -> None:
    e = Env(tmp_path)
    res = e.client.post(
        "/api/transcripts/import",
        json={"content": ZOOM_VTT, "filename": ZOOM_NAME, "startTime": "9" * 400},
    )
    assert res.status_code == 400


def test_import_internal_error_not_leaked_500(tmp_path: Path, enabled: None) -> None:
    class Boom:
        async def import_transcript(self, opts: Any) -> Any:
            raise RuntimeError("secret internal detail /var/db")

    e = Env(tmp_path, transcriptImportService=Boom())
    res = e.client.post("/api/transcripts/import", json={"content": ZOOM_VTT, "filename": ZOOM_NAME})
    assert res.status_code == 500
    assert "secret" not in res.text


def test_insights_find_limit_is_clamped(
    tmp_path: Path, enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []

    async def find(self: Any, query: str, limit: int = 5) -> list[dict[str, Any]]:
        seen.append(limit)
        return []

    monkeypatch.setattr(llm_query.InsightsLlmService, "find_events", find)
    e = Env(tmp_path)
    assert e.client.get("/api/insights/find", params={"q": "x", "limit": 10**9}).status_code == 200
    assert seen == [llm_query.MAX_FIND_EVENTS_LIMIT]


def test_import_bearer_token(tmp_path: Path, enabled: None) -> None:
    e = Env(tmp_path, ingestToken="sekret")
    payload = {"content": ZOOM_VTT, "filename": ZOOM_NAME}
    assert e.client.post("/api/transcripts/import", json=payload).status_code == 401
    wrong = e.client.post("/api/transcripts/import", json=payload, headers={"authorization": "Bearer nope"})
    assert wrong.status_code == 401
    right = e.client.post("/api/transcripts/import", json=payload, headers={"authorization": "Bearer sekret"})
    assert right.status_code == 201


def test_import_works_when_insights_disabled(tmp_path: Path, disabled: None) -> None:
    e = Env(tmp_path)
    res = e.client.post("/api/transcripts/import", json={"content": ZOOM_VTT, "filename": ZOOM_NAME})
    assert res.status_code == 201
    assert e.client.get("/api/insights/status").status_code == 404


# -- wiring + openapi --------------------------------------------------------------


def test_create_app_registers_queue_hook(tmp_path: Path, enabled: None) -> None:
    extract.set_enqueue_hook(None)
    try:
        e = Env(tmp_path)
        e.client.get("/api/health")
        e.client.get("/api/insights/status")
        e.client.post("/api/transcripts/import", json={"content": ZOOM_VTT, "filename": ZOOM_NAME})
        assert extract.get_enqueue_hook() is not None
    finally:
        extract.set_enqueue_hook(None)


def test_openapi_export_includes_insights_models() -> None:
    schemas = export_openapi_schema()["components"]["schemas"]
    for name in (
        "InsightsOverview",
        "InsightsPerson",
        "InsightsMeetingSignals",
        "InsightsDyads",
        "InsightsFindResponse",
        "InsightsPlaybook",
        "InsightsCoaching",
        "InsightsBackfillResult",
        "InsightsStatus",
        "InsightsMe",
        "TranscriptImportResult",
    ):
        assert name in schemas, name
    assert "from" in schemas["InsightsOverview"]["properties"]
