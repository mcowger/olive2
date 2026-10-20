"""Olive Insights query layer: event retrieval, per-person playbooks, and coaching (fake LLM, no ML)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from olive.db import create_db
from olive.layout import meeting_paths
from olive.schemas import LlmSettings
from olive.services.insights.extract import EgressBlockedError
from olive.services.insights.llm_query import (
    MAX_FILTER_TERMS,
    MAX_FIND_EVENTS_LIMIT,
    EventFilter,
    InsightsLlmService,
)
from olive.services.insights.validate import InsightsValidationError

PROXY_HOST = "llm-proxy.local:3020"
PROXY_URL = f"http://{PROXY_HOST}/v1"
NOW = int(time.time() * 1000)

M1_SEGMENTS = [
    ("Phil", "Let's go over the launch retro."),
    ("Bob", "You sabotaged the launch on purpose."),
    ("Phil", "That is not what happened."),
    ("Phil", "Moving on to the schedule."),
    ("Bob", "Send me the TLDR and I will approve."),
    ("Phil", "TLDR: we ship Friday."),
    ("Bob", "Great, approved."),
    ("Phil", "Thanks."),
    ("Bob", "One more thing about budget."),
    ("Phil", "Go ahead."),
]
M2_SEGMENTS = [
    ("Phil", "Good morning."),
    ("Bob", "Thanks for covering my shift last week, that really helped."),
    ("Phil", "Anytime."),
    ("Bob", "Quick update on the vendors."),
    ("Phil", "Go on."),
]


class FakeLlm:
    """Stands in for LlmService: records calls, resolves a base URL, returns canned text."""

    def __init__(self, responder: Callable[[int, dict[str, Any]], str], base_url: str = PROXY_URL) -> None:
        self.responder = responder
        self.base_url = base_url
        self.calls: list[dict[str, Any]] = []

    async def get_settings(self) -> LlmSettings:
        return LlmSettings(defaultProvider="openai", defaultModel="default-model")

    async def resolve_model(self, provider: str, model_id: str, base_url_override: str | None = None):
        return {"id": model_id, "provider": provider, "baseUrl": base_url_override or self.base_url}

    async def generate_text(self, options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(options)
        return {
            "text": self.responder(len(self.calls), options),
            "provider": options.get("provider") or "openai",
            "model": options.get("model") or "default-model",
            "durationMs": 5,
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }


def _write_meeting(
    engine, meetings_dir: str, mid: str, title: str, segs, *, with_transcript: bool = True
) -> None:
    paths = meeting_paths(meetings_dir, NOW, title, mid)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id, last_error,
                  created_at, updated_at)
                VALUES (:id, :title, :now, :end, 'upload', 'ready', '[]', NULL, NULL, NULL, :now, :now)
                """
            ),
            {"id": mid, "title": title, "now": NOW, "end": NOW + 600_000},
        )
        for name in ("Phil", "Bob"):
            conn.execute(
                text("INSERT OR IGNORE INTO speakers (id, name, created_at) VALUES (:id, :n, :now)"),
                {"id": f"spk-{name.lower()}", "n": name, "now": NOW},
            )
            conn.execute(
                text("INSERT INTO meeting_speakers (meeting_id, speaker_id) VALUES (:m, :s)"),
                {"m": mid, "s": f"spk-{name.lower()}"},
            )
        if not with_transcript:
            return
        Path(paths.transcripts_dir).mkdir(parents=True, exist_ok=True)
        segments = [
            {"startMs": i * 5000, "endMs": i * 5000 + 4000, "speaker": who, "text": body}
            for i, (who, body) in enumerate(segs)
        ]
        (Path(paths.folder) / "transcripts/transcript.json").write_text(
            json.dumps({"segments": segments}), encoding="utf-8"
        )
        conn.execute(
            text(
                "INSERT INTO artifacts (id, meeting_id, recording_id, kind, provider, format, path, created_at) "
                "VALUES (:a, :m, NULL, 'transcript', 'local', 'json', 'transcripts/transcript.json', :now)"
            ),
            {"a": f"art-{mid}", "m": mid, "now": NOW},
        )
        conn.execute(
            text("UPDATE meetings SET primary_transcript_artifact_id = :a WHERE id = :m"),
            {"a": f"art-{mid}", "m": mid},
        )


def _moment(
    engine,
    mid: str,
    meeting: str,
    *,
    speaker: str,
    toward: str,
    turn: int,
    event: str,
    valence: str,
    severity: float,
    quote: str,
    rationale: str = "",
    verified: int = 1,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meeting_moments (id, meeting_id, speaker_id, toward_speaker_id, start_ms, end_ms,
                  turn_start_idx, turn_end_idx, event_type, valence, severity, quote, rationale, verified,
                  created_at)
                VALUES (:id, :m, :s, :t, :start, :end, :turn, :turn, :ev, :val, :sev, :q, :r, :ver, :now)
                """
            ),
            {
                "id": mid,
                "m": meeting,
                "s": speaker,
                "t": toward,
                "start": turn * 5000,
                "end": turn * 5000 + 4000,
                "turn": turn,
                "ev": event,
                "val": valence,
                "sev": severity,
                "q": quote,
                "r": rationale,
                "ver": verified,
                "now": NOW,
            },
        )


def _make(
    tmp_path: Path,
    responder: Callable[[int, dict[str, Any]], str],
    *,
    allowed_hosts: list[str] | None = None,
    config: dict[str, Any] | None = None,
    prompts: dict[str, str] | None = None,
):
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    config_dir = tmp_path / "config"
    Path(meetings_dir).mkdir(parents=True, exist_ok=True)
    (config_dir / "insights").mkdir(parents=True, exist_ok=True)
    if config:
        (config_dir / "insights" / "config.json").write_text(json.dumps(config))
    if prompts:
        (config_dir / "insights" / "prompts.json").write_text(json.dumps(prompts))
    llm = FakeLlm(responder)
    service = InsightsLlmService(
        db=handle.engine,
        meetings_dir=meetings_dir,
        config_dir=str(config_dir),
        llm_service=llm,
        allowed_hosts=[PROXY_HOST] if allowed_hosts is None else allowed_hosts,
        self_speaker_id="spk-phil",
    )
    _write_meeting(handle.engine, meetings_dir, "m1", "Launch Retro", M1_SEGMENTS)
    _write_meeting(handle.engine, meetings_dir, "m2", "Shift Sync", M2_SEGMENTS)
    _write_meeting(handle.engine, meetings_dir, "m3", "HR Meeting", [], with_transcript=False)
    _moment(
        handle.engine, "mom-sab", "m1", speaker="spk-bob", toward="spk-phil", turn=1, event="accusation",
        valence="bad", severity=0.9, quote="You sabotaged the launch on purpose.",
        rationale="Bob blames Phil directly for deliberately undermining the launch.",
    )  # fmt: skip
    _moment(
        handle.engine, "mom-praise", "m2", speaker="spk-bob", toward="spk-phil", turn=1, event="praise",
        valence="good", severity=0.4, quote="Thanks for covering my shift last week, that really helped.",
    )  # fmt: skip
    _moment(
        handle.engine, "mom-fire", "m3", speaker="spk-bob", toward="spk-phil", turn=0, event="termination",
        valence="bad", severity=0.95, quote="We are letting you go effective today.",
    )  # fmt: skip
    return handle, llm, service


def _filter_json(**over: Any) -> str:
    base: dict[str, Any] = {
        "eventTypes": ["accusation"],
        "valence": "bad",
        "towardMe": True,
        "dateRange": {"start": "2020-01-01", "end": "2099-01-01"},
        "keywords": ["sabotage"],
    }
    base.update(over)
    return json.dumps(base)


def _rerank_json(cid: str = "c1", **over: Any) -> str:
    row: dict[str, Any] = {
        "candidateId": cid,
        "score": 0.97,
        "enacted": True,
        "reason": "Bob accuses Phil of deliberate sabotage as it happens.",
    }
    row.update(over)
    return json.dumps({"results": [row]})


# ------------------------------------------------------------------------- find_events


async def test_find_events_returns_seeded_sabotage_accusation(tmp_path: Path) -> None:
    def responder(n: int, opts: dict[str, Any]) -> str:
        return _filter_json() if n == 1 else _rerank_json("c1")

    handle, llm, service = _make(tmp_path, responder)
    results = await service.find_events("the call where someone accused me of sabotage")

    assert len(llm.calls) == 2  # filter + rerank, no per-transcript scan
    assert len(results) == 1
    top = results[0]
    assert top["meetingId"] == "m1"
    assert top["startMs"] == 5000
    assert top["quote"] == "You sabotaged the launch on purpose."
    assert top["eventType"] == "accusation"
    assert top["toward"] == "Phil"
    assert top["towardSpeakerId"] == "spk-phil"
    assert top["speaker"] == "Bob"
    assert top["score"] == pytest.approx(0.97)
    assert top["enacted"] is True
    assert "sabotage" in top["reason"]
    rerank_prompt = llm.calls[1]["prompt"]
    assert "Let's go over the launch retro." in rerank_prompt  # short transcript context
    assert "We are letting you go" not in rerank_prompt  # filtered out by SQL tier
    assert "discussed" in rerank_prompt.lower()  # enacted-vs-discussed is part of the ask
    handle.engine.dispose()


async def test_find_events_falls_back_to_most_severe_when_filter_matches_nothing(tmp_path: Path) -> None:
    def responder(n: int, opts: dict[str, Any]) -> str:
        if n == 1:
            return _filter_json(eventTypes=["nonexistent"], keywords=["zzz"], valence=None, dateRange=None)
        return _rerank_json("c1", reason="closest severe event")

    handle, llm, service = _make(tmp_path, responder)
    results = await service.find_events("something that never happened", limit=2)

    rerank_prompt = llm.calls[1]["prompt"]
    assert "We are letting you go effective today." in rerank_prompt  # most severe moment, no transcript file
    assert "You sabotaged the launch on purpose." in rerank_prompt
    assert rerank_prompt.index("We are letting you go") < rerank_prompt.index("You sabotaged")
    assert [r["meetingId"] for r in results] == ["m3"]
    handle.engine.dispose()


async def test_find_events_drops_candidate_ids_the_model_invented(tmp_path: Path) -> None:
    def responder(n: int, opts: dict[str, Any]) -> str:
        return _filter_json() if n == 1 else _rerank_json("c99")

    handle, _, service = _make(tmp_path, responder)
    assert await service.find_events("sabotage") == []
    handle.engine.dispose()


async def test_find_events_malformed_rerank_errors_cleanly_after_one_repair(tmp_path: Path) -> None:
    def responder(n: int, opts: dict[str, Any]) -> str:
        return _filter_json() if n == 1 else "sorry, no JSON for you"

    handle, llm, service = _make(tmp_path, responder)
    with pytest.raises(InsightsValidationError):
        await service.find_events("sabotage")
    assert len(llm.calls) == 3  # filter, rerank, exactly one repair re-ask
    assert "rejected" in llm.calls[2]["prompt"]
    handle.engine.dispose()


async def test_find_events_repair_can_recover_a_bad_first_filter(tmp_path: Path) -> None:
    def responder(n: int, opts: dict[str, Any]) -> str:
        if n == 1:
            return json.dumps({"valence": "furious"})  # not a valid valence
        return _filter_json() if n == 2 else _rerank_json("c1")

    handle, llm, service = _make(tmp_path, responder)
    results = await service.find_events("sabotage")
    assert len(llm.calls) == 3
    assert results[0]["meetingId"] == "m1"
    handle.engine.dispose()


async def test_find_events_uses_user_prompt_override(tmp_path: Path) -> None:
    marker = "USER-OVERRIDDEN FILTER PROMPT"
    prompts = {"queryFilterPrompt": f"{marker} for: {{{{query}}}}"}

    def responder(n: int, opts: dict[str, Any]) -> str:
        return _filter_json() if n == 1 else _rerank_json("c1")

    handle, llm, service = _make(tmp_path, responder, prompts=prompts)
    await service.find_events("accused me of sabotage")
    assert marker in llm.calls[0]["prompt"]
    assert "accused me of sabotage" in llm.calls[0]["prompt"]
    handle.engine.dispose()


async def test_egress_guard_blocks_non_allowlisted_host_before_any_call(tmp_path: Path) -> None:
    handle, llm, service = _make(tmp_path, lambda n, o: _filter_json(), allowed_hosts=["other.example:1"])
    with pytest.raises(EgressBlockedError):
        await service.find_events("sabotage")
    with pytest.raises(EgressBlockedError):
        await service.playbook("spk-bob")
    with pytest.raises(EgressBlockedError):
        await service.coaching("all")
    assert llm.calls == []
    handle.engine.dispose()


# ------------------------------------------------------------------------- playbook


def _playbook_json() -> str:
    return json.dumps(
        {
            "summary": "Bob responds to brevity and concrete asks.",
            "doMore": [
                {
                    "point": "Lead with a TLDR.",
                    "evidence": [{"quote": "Send me the TLDR and I will approve.", "meetingId": "m1"}],
                    "counterExample": {"quote": "You sabotaged the launch on purpose.", "meetingId": "m1"},
                }
            ],
            "easeOff": [
                {
                    "point": "Long unprompted backstory.",
                    "evidence": [{"quote": "a line that exists nowhere in the data"}],
                }
            ],
        }
    )


async def test_playbook_uses_dyad_moments_and_sampled_windows_and_caches(tmp_path: Path) -> None:
    handle, llm, service = _make(tmp_path, lambda n, o: _playbook_json(), config={"playbookWindows": 50})
    result = await service.playbook("spk-bob")

    assert len(llm.calls) == 1
    prompt = llm.calls[0]["prompt"]
    assert "You sabotaged the launch on purpose." in prompt  # bad dyad moment
    assert "Thanks for covering my shift" in prompt  # good dyad moment
    assert "Send me the TLDR and I will approve." in prompt  # ordinary window (not a notable turn)
    assert "Quick update on the vendors." in prompt

    assert result["speakerId"] == "spk-bob"
    assert result["cached"] is False
    assert result["doMore"][0]["point"] == "Lead with a TLDR."
    assert result["doMore"][0]["evidence"][0]["quote"] == "Send me the TLDR and I will approve."
    assert result["doMore"][0]["counterExample"]["quote"].startswith("You sabotaged")
    assert result["easeOff"] == []  # evidence quote not found in the supplied data is dropped
    size = result["sampleSize"]
    assert size["goodMoments"] == 1
    assert size["badMoments"] == 2
    assert size["ordinaryWindows"] >= 1

    again = await service.playbook("spk-bob")
    assert len(llm.calls) == 1  # served from sync_state cache
    assert again["cached"] is True
    assert again["doMore"] == result["doMore"]
    with handle.engine.connect() as conn:
        stored = conn.execute(
            text("SELECT value FROM sync_state WHERE key = 'insights.playbook:spk-bob'")
        ).scalar()
    assert stored and json.loads(stored)["signature"]

    # New data for the dyad changes the signature, so the playbook is recomputed.
    _moment(
        handle.engine, "mom-new", "m2", speaker="spk-bob", toward="spk-phil", turn=3, event="apology",
        valence="good", severity=0.5, quote="Sorry, that came out wrong.",
    )  # fmt: skip
    fresh = await service.playbook("spk-bob")
    assert len(llm.calls) == 2
    assert fresh["cached"] is False
    assert "Sorry, that came out wrong." in llm.calls[1]["prompt"]
    handle.engine.dispose()


async def test_playbook_recomputes_when_prompt_changes(tmp_path: Path) -> None:
    handle, llm, service = _make(tmp_path, lambda n, o: _playbook_json())
    await service.playbook("spk-bob")
    (Path(service.config_dir) / "insights" / "prompts.json").write_text(
        json.dumps({"playbookPrompt": "Different instructions. {{evidence}}"})
    )
    await service.playbook("spk-bob")
    assert len(llm.calls) == 2
    handle.engine.dispose()


async def test_playbook_without_any_data_makes_no_llm_call(tmp_path: Path) -> None:
    handle, llm, service = _make(tmp_path, lambda n, o: _playbook_json())
    with handle.engine.begin() as conn:
        conn.execute(
            text("INSERT INTO speakers (id, name, created_at) VALUES ('spk-new', 'Newcomer', :n)"), {"n": NOW}
        )
    result = await service.playbook("spk-new")
    assert result["insufficientData"] is True
    assert llm.calls == []
    handle.engine.dispose()


async def test_playbook_malformed_json_errors_after_one_repair(tmp_path: Path) -> None:
    handle, llm, service = _make(tmp_path, lambda n, o: "nope")
    with pytest.raises(InsightsValidationError):
        await service.playbook("spk-bob")
    assert len(llm.calls) == 2
    handle.engine.dispose()


# ------------------------------------------------------------------------- coaching


def _coaching_json() -> str:
    return json.dumps(
        {
            "summary": "Mostly constructive, with one heated exchange.",
            "strengths": [
                {
                    "point": "Colleagues openly thank you.",
                    "evidence": [{"quote": "Thanks for covering my shift last week, that really helped."}],
                }
            ],
            "growthAreas": [
                {
                    "point": "A launch dispute escalated into blame.",
                    "suggestion": "Name the specific behavior and the request.",
                    "evidence": [{"quote": "You sabotaged the launch on purpose."}],
                }
            ],
        }
    )


async def test_coaching_builds_from_top_moments_and_caches(tmp_path: Path) -> None:
    handle, llm, service = _make(tmp_path, lambda n, o: _coaching_json())
    result = await service.coaching("all")

    assert len(llm.calls) == 1
    prompt = llm.calls[0]["prompt"]
    assert "You sabotaged the launch on purpose." in prompt
    assert "Thanks for covering my shift" in prompt
    assert "mental" in llm.calls[0]["systemPrompt"].lower()  # no-diagnosis framing is in the system prompt
    assert result["scope"] == "all"
    assert result["strengths"][0]["evidence"][0]["quote"].startswith("Thanks for covering")
    assert result["growthAreas"][0]["suggestion"].startswith("Name the specific")

    again = await service.coaching("all")
    assert len(llm.calls) == 1
    assert again["cached"] is True
    with handle.engine.connect() as conn:
        assert (
            conn.execute(text("SELECT COUNT(*) FROM sync_state WHERE key = 'insights.coaching:all'")).scalar()
            == 1
        )

    _moment(
        handle.engine, "mom-new", "m2", speaker="spk-phil", toward="spk-bob", turn=2, event="repair",
        valence="good", severity=0.6, quote="Let me restate that more kindly.",
    )  # fmt: skip
    await service.coaching("all")
    assert len(llm.calls) == 2
    handle.engine.dispose()


async def test_coaching_speaker_scope_requires_id_and_caches_per_speaker(tmp_path: Path) -> None:
    handle, llm, service = _make(tmp_path, lambda n, o: _coaching_json())
    with pytest.raises(ValueError):
        await service.coaching("speaker")
    await service.coaching("speaker", speaker_id="spk-bob")
    await service.coaching("speaker", speaker_id="spk-bob")
    assert len(llm.calls) == 1
    with handle.engine.connect() as conn:
        keys = [
            r[0]
            for r in conn.execute(text("SELECT key FROM sync_state WHERE key LIKE 'insights.coaching:%'"))
        ]
    assert keys == ["insights.coaching:spk-bob"]
    handle.engine.dispose()


# ------------------------------------------------------------- trust + input bounds


async def test_find_events_excludes_unverified_moments(tmp_path: Path) -> None:
    def responder(n: int, opts: dict[str, Any]) -> str:
        if n == 1:
            return _filter_json(
                eventTypes=["hallucination"], keywords=["ghost"], valence=None, dateRange=None
            )
        return _rerank_json("c1")

    handle, llm, service = _make(tmp_path, responder)
    _moment(
        handle.engine, "mom-ghost", "m1", speaker="spk-bob", toward="spk-phil", turn=2, event="hallucination",
        valence="bad", severity=1.0, quote="A ghost quote nobody said ghost.", verified=0,
    )  # fmt: skip
    await service.find_events("ghost")
    prompt = llm.calls[1]["prompt"]
    assert "ghost quote" not in prompt  # neither the SQL match nor the fallback surfaces it
    assert "We are letting you go" in prompt  # verified moments still do

    llm.calls.clear()
    await service.find_events("ghost", include_unverified=True)
    assert "ghost quote" in llm.calls[1]["prompt"]
    handle.engine.dispose()


async def test_find_events_clamps_limit(tmp_path: Path) -> None:
    def responder(n: int, opts: dict[str, Any]) -> str:
        return _filter_json() if n == 1 else _rerank_json("c1")

    handle, llm, service = _make(tmp_path, responder)
    seen: list[int] = []
    real = service._select_candidates

    def spy(flt: Any, self_id: Any, limit: int, include_unverified: bool = False) -> Any:
        seen.append(limit)
        return real(flt, self_id, limit, include_unverified)

    service._select_candidates = spy  # type: ignore[method-assign]
    await service.find_events("sabotage", limit=10**6)
    assert seen and seen[0] <= MAX_FIND_EVENTS_LIMIT * 4
    assert f"{MAX_FIND_EVENTS_LIMIT}" in llm.calls[1]["prompt"]
    assert str(10**6) not in llm.calls[1]["prompt"]
    handle.engine.dispose()


def test_select_candidates_truncates_oversized_term_lists(tmp_path: Path) -> None:
    handle, _, service = _make(tmp_path, lambda n, o: "{}")
    captured: dict[str, Any] = {}
    real = service._query_moments

    def spy(where: str, params: dict[str, Any], limit: int, include_unverified: bool = False) -> Any:
        captured.setdefault("params", params)
        return real(where, params, limit, include_unverified)

    service._query_moments = spy  # type: ignore[method-assign]
    flt = EventFilter(eventTypes=[f"t{i}" for i in range(5000)], keywords=[f"k{i}" for i in range(5000)])
    service._select_candidates(flt, None, 5)
    params = captured["params"]
    assert sum(1 for k in params if k.startswith("et")) == MAX_FILTER_TERMS
    assert sum(1 for k in params if k.startswith("kw")) == MAX_FILTER_TERMS
    handle.engine.dispose()
