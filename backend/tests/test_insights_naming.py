"""Olive Insights speaker naming: proposals, grounding, confirmations, batch, schema."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from olive.db import MeetingSpeakerNameProposalModel, create_db, run_migrations
from olive.layout import meeting_paths
from olive.schemas import LlmSettings
from olive.services.insights.extract import EgressBlockedError, InsightsExtractionError
from olive.services.insights.naming import (
    ProposedName,
    SpeakerNamingService,
    ground_name,
    parse_proposals_with_repair,
    validate_proposals,
)
from olive.services.insights.validate import InsightsValidationError
from olive.services.meetings import delete_meeting

PROXY_HOST = "llm-proxy.local:3020"
PROXY_URL = f"http://{PROXY_HOST}/v1"

# Turn 1 addresses "Speaker 2" as Vineet; turn 4 addresses "Speaker 1" as Dana; turn 5 mentions a
# third party (Zed) who is not at the meeting.
SEGMENTS = [
    {
        "startMs": 0,
        "endMs": 4000,
        "speaker": "Speaker 1",
        "text": "Okay, let's start. Vineet, what is the status?",
    },
    {"startMs": 4200, "endMs": 9000, "speaker": "Speaker 2", "text": "We shipped the beta last week."},
    {"startMs": 9200, "endMs": 12000, "speaker": "Speaker 1", "text": "Great, thanks."},
    {"startMs": 12200, "endMs": 15000, "speaker": "Speaker 2", "text": "Dana, can you review the budget?"},
    {"startMs": 15200, "endMs": 18000, "speaker": "Speaker 1", "text": "Sure. Zed asked about it too."},
    {"startMs": 18200, "endMs": 20000, "speaker": "Speaker 3", "text": "I have nothing to add."},
]


def _reply(*items: dict[str, Any]) -> str:
    return json.dumps({"proposals": list(items)})


def _item(
    speaker: str, name: str | None, quote: str | None, turn: int | None, confidence: float = 0.9
) -> dict[str, Any]:
    return {
        "speaker": speaker,
        "proposedName": name,
        "confidence": confidence,
        "evidenceQuote": quote,
        "evidenceTurnIdx": turn,
    }


GOOD_REPLY = _reply(
    _item("Speaker 2", "Vineet", "Vineet, what is the status?", 0, 0.92),
    _item("Speaker 1", "Dana", "Dana, can you review the budget?", 3, 0.85),
    _item("Speaker 3", None, None, None, 0.0),
)


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


def _make_service(
    tmp_path: Path,
    responder: Callable[[int, dict[str, Any]], str] | str = GOOD_REPLY,
    *,
    me_aliases: list[str] | None = None,
    allowed_hosts: list[str] | None = None,
    base_url: str = PROXY_URL,
):
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    config_dir = str(tmp_path / "config")
    Path(meetings_dir).mkdir(parents=True, exist_ok=True)
    Path(config_dir).mkdir(parents=True, exist_ok=True)
    fn = (lambda _n, _o: responder) if isinstance(responder, str) else responder
    llm = FakeLlm(fn, base_url)
    service = SpeakerNamingService(
        db=handle.engine,
        meetings_dir=meetings_dir,
        config_dir=config_dir,
        llm_service=llm,
        allowed_hosts=[PROXY_HOST] if allowed_hosts is None else allowed_hosts,
        me_aliases=me_aliases,
    )
    return handle, meetings_dir, llm, service


def _seed(
    engine,
    meetings_dir: str,
    segments: list[dict[str, Any]] | None = None,
    *,
    meeting_id: str = "meeting-nm-1",
    start: int | None = None,
) -> str:
    now = start if start is not None else int(time.time() * 1000)
    title = "Budget Sync"
    paths = meeting_paths(meetings_dir, now, title, meeting_id)
    Path(paths.transcripts_dir).mkdir(parents=True, exist_ok=True)
    rel = "transcripts/transcript.json"
    (Path(paths.folder) / rel).write_text(json.dumps({"segments": segments or SEGMENTS}), encoding="utf-8")
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
            {"id": meeting_id, "title": title, "now": now, "end": now + 600_000},
        )
        conn.execute(
            text(
                """
                INSERT INTO artifacts (id, meeting_id, recording_id, kind, provider, format, path, created_at)
                VALUES (:aid, :mid, NULL, 'transcript', 'local', 'json', :path, :now)
                """
            ),
            {"aid": f"art-{meeting_id}", "mid": meeting_id, "path": rel, "now": now},
        )
        conn.execute(
            text("UPDATE meetings SET primary_transcript_artifact_id = :aid WHERE id = :mid"),
            {"aid": f"art-{meeting_id}", "mid": meeting_id},
        )
    return meeting_id


def _add_speaker(
    engine, speaker_id: str, name: str, *, created: int = 1, enrolled: int | None = None
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO speakers (id, name, enrolled_at, created_at) VALUES (:i, :n, :e, :c)"),
            {"i": speaker_id, "n": name, "e": enrolled, "c": created},
        )


def _rows(engine, sql: str, **params: Any) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(sql), params).mappings().all()]


def _proposals(engine, meeting_id: str = "meeting-nm-1") -> dict[str, dict[str, Any]]:
    rows = _rows(engine, "SELECT * FROM meeting_speaker_name_proposals WHERE meeting_id = :m", m=meeting_id)
    return {r["speaker_label"]: r for r in rows}


# --------------------------------------------------------------------------- propose


async def test_propose_stores_pending_proposals_with_confidence_and_evidence(tmp_path: Path) -> None:
    handle, meetings_dir, llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir)

    res = await service.propose_names(mid)

    assert res["status"] == "proposed"
    stored = _proposals(handle.engine)
    assert set(stored) == {"Speaker 1", "Speaker 2", "Speaker 3"}
    vineet = stored["Speaker 2"]
    assert vineet["proposed_name"] == "Vineet"
    assert vineet["status"] == "pending"
    assert vineet["confidence"] == pytest.approx(0.92)
    assert vineet["evidence_quote"] == "Vineet, what is the status?"
    assert vineet["evidence_turn_idx"] == 0
    assert vineet["model"] == "default-model" and vineet["provider"] == "openai"
    assert stored["Speaker 1"]["proposed_name"] == "Dana"
    assert stored["Speaker 1"]["evidence_turn_idx"] == 3
    assert res["counts"] == {"pending": 2, "none": 1, "ungrounded": 0}
    # The prompt cites turns and asks only about generic labels.
    prompt = llm.calls[0]["prompt"]
    assert "[0] Speaker 1: Okay, let's start." in prompt
    assert "Speaker 1, Speaker 2, Speaker 3" in prompt


async def test_known_speakers_are_offered_as_candidates_and_canonicalized(tmp_path: Path) -> None:
    reply = _reply(_item("Speaker 2", "vineet", "Vineet, what is the status?", 0))
    handle, meetings_dir, llm, service = _make_service(tmp_path, reply)
    mid = _seed(handle.engine, meetings_dir)
    _add_speaker(handle.engine, "spk-v", "Vineet Rao")
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO speaker_aliases (id, alias_norm, speaker_id, kind, confirmed, created_at) "
                "VALUES ('al1', 'vineet', 'spk-v', 'nickname', 1, 1)"
            )
        )

    await service.propose_names(mid, candidate_names=["Priya"])

    assert "Vineet Rao" in llm.calls[0]["prompt"] and "Priya" in llm.calls[0]["prompt"]
    # The spoken first name resolves, via the alias, to the known speaker's spelling.
    assert _proposals(handle.engine)["Speaker 2"]["proposed_name"] == "Vineet Rao"


async def test_non_verbatim_evidence_quote_is_flagged_not_offered(tmp_path: Path) -> None:
    reply = _reply(
        _item("Speaker 2", "Vineet", "Vineet, please give us the status report", 0),  # paraphrase
        _item("Speaker 1", "Dana", "Dana, can you review the budget?", 3),
    )
    handle, meetings_dir, _llm, service = _make_service(tmp_path, reply)
    mid = _seed(handle.engine, meetings_dir)

    await service.propose_names(mid)

    stored = _proposals(handle.engine)
    assert stored["Speaker 2"]["status"] == "ungrounded"
    assert stored["Speaker 1"]["status"] == "pending"
    assert [
        r["speaker_label"]
        for r in _rows(
            handle.engine, "SELECT speaker_label FROM meeting_speaker_name_proposals WHERE status = 'pending'"
        )
    ] == ["Speaker 1"]


async def test_quote_cited_at_the_wrong_turn_is_not_grounded(tmp_path: Path) -> None:
    reply = _reply(_item("Speaker 2", "Vineet", "Vineet, what is the status?", 2))  # quote is in turn 0
    handle, meetings_dir, _llm, service = _make_service(tmp_path, reply)
    mid = _seed(handle.engine, meetings_dir)

    await service.propose_names(mid)

    assert _proposals(handle.engine)["Speaker 2"]["status"] == "ungrounded"


async def test_out_of_range_turn_is_not_grounded(tmp_path: Path) -> None:
    reply = _reply(_item("Speaker 2", "Vineet", "Vineet, what is the status?", 99))
    handle, meetings_dir, _llm, service = _make_service(tmp_path, reply)
    mid = _seed(handle.engine, meetings_dir)

    await service.propose_names(mid)

    assert _proposals(handle.engine)["Speaker 2"]["status"] == "ungrounded"


async def test_name_neither_known_nor_spoken_is_never_proposed(tmp_path: Path) -> None:
    # "Zed" is verbatim-quoted nowhere in turn 1; the model also invented "Morgan" with a real quote.
    reply = _reply(
        _item("Speaker 2", "Morgan", "We shipped the beta last week.", 1),
        _item("Speaker 3", "Zed", "I have nothing to add.", 5),
    )
    handle, meetings_dir, _llm, service = _make_service(tmp_path, reply)
    mid = _seed(handle.engine, meetings_dir)

    await service.propose_names(mid)

    stored = _proposals(handle.engine)
    for label in ("Speaker 2", "Speaker 3"):
        assert stored[label]["status"] == "ungrounded"
    assert _rows(handle.engine, "SELECT 1 FROM meeting_speaker_name_proposals WHERE status = 'pending'") == []


def test_ground_name_accepts_only_whole_word_names_in_the_quote() -> None:
    from olive.schemas import TranscriptSegment

    segs = [TranscriptSegment(startMs=0, endMs=1, speaker="Speaker 1", text="Thanks Alexander, that helps.")]
    ok = ProposedName(
        speaker="Speaker 2", proposedName="Alexander", evidenceQuote="thanks alexander", evidenceTurnIdx=0
    )
    assert ground_name(ok, segs, {"alexander": "Alexander"})[2] is True
    partial = ProposedName(
        speaker="Speaker 2", proposedName="Alex", evidenceQuote="Thanks Alexander", evidenceTurnIdx=0
    )
    assert ground_name(partial, segs, {})[2] is False
    generic = ProposedName(
        speaker="Speaker 2", proposedName="Speaker 4", evidenceQuote="Thanks", evidenceTurnIdx=0
    )
    assert ground_name(generic, segs, {})[2] is False


async def test_generic_label_without_support_stays_unproposed(tmp_path: Path) -> None:
    handle, meetings_dir, _llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir)

    await service.propose_names(mid)

    silent = _proposals(handle.engine)["Speaker 3"]
    assert silent["status"] == "none"
    assert silent["proposed_name"] is None and silent["evidence_quote"] is None
    assert silent["confidence"] is None


async def test_unknown_string_and_unrequested_labels_are_handled(tmp_path: Path) -> None:
    reply = _reply(
        _item("Speaker 3", "unknown", None, None),
        _item("Speaker 9", "Dana", "Dana, can you review the budget?", 3),  # never asked about
    )
    handle, meetings_dir, _llm, service = _make_service(tmp_path, reply)
    mid = _seed(handle.engine, meetings_dir)

    await service.propose_names(mid)

    stored = _proposals(handle.engine)
    assert "Speaker 9" not in stored
    assert stored["Speaker 3"]["status"] == "none"
    assert stored["Speaker 1"]["status"] == "none"  # the model omitted it; still recorded as asked


async def test_named_labels_are_context_not_targets(tmp_path: Path) -> None:
    segs = [
        {"startMs": 0, "endMs": 1000, "speaker": "Alice", "text": "Hi everyone."},
        {"startMs": 1100, "endMs": 2000, "speaker": "Speaker 1", "text": "Hi Alice, it is Bea."},
    ]
    reply = _reply(_item("Speaker 1", "Bea", "it is Bea", 1))
    handle, meetings_dir, llm, service = _make_service(tmp_path, reply)
    mid = _seed(handle.engine, meetings_dir, segs)

    await service.propose_names(mid)

    assert set(_proposals(handle.engine)) == {"Speaker 1"}
    assert "Alice" in llm.calls[0]["prompt"]  # offered as a candidate name


async def test_meeting_without_generic_labels_makes_no_llm_call(tmp_path: Path) -> None:
    segs = [{"startMs": 0, "endMs": 1000, "speaker": "Alice", "text": "Hello."}]
    handle, meetings_dir, llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir, segs)

    res = await service.propose_names(mid)

    assert res["status"] == "no_generic_labels"
    assert llm.calls == []


async def test_repropose_is_idempotent_unless_forced(tmp_path: Path) -> None:
    handle, meetings_dir, llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir)

    await service.propose_names(mid)
    assert len(llm.calls) == 1
    again = await service.propose_names(mid)
    assert again["status"] == "skipped"
    assert len(llm.calls) == 1

    forced = await service.propose_names(mid, force=True)
    assert forced["status"] == "proposed"
    assert len(llm.calls) == 2
    assert len(_proposals(handle.engine)) == 3  # upserted, not duplicated


async def test_force_never_overwrites_a_confirmed_proposal(tmp_path: Path) -> None:
    handle, meetings_dir, _llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir)
    await service.propose_names(mid)
    service.apply_confirmations(mid, {"Speaker 2": "Vineet"})

    await service.propose_names(mid, force=True)

    assert _proposals(handle.engine)["Speaker 2"]["status"] == "confirmed"


async def test_invalid_reply_is_repaired_once_then_fails_without_writes(tmp_path: Path) -> None:
    replies = iter(["not json", "still not json"])
    handle, meetings_dir, llm, service = _make_service(tmp_path, lambda _n, _o: next(replies))
    mid = _seed(handle.engine, meetings_dir)

    with pytest.raises(InsightsExtractionError):
        await service.propose_names(mid)

    assert len(llm.calls) == 2  # one ask + exactly one repair re-ask
    assert "rejected" in llm.calls[1]["prompt"]
    assert _proposals(handle.engine) == {}


async def test_repair_reask_can_recover(tmp_path: Path) -> None:
    replies = iter(["oops", GOOD_REPLY])
    handle, meetings_dir, llm, service = _make_service(tmp_path, lambda _n, _o: next(replies))
    mid = _seed(handle.engine, meetings_dir)

    await service.propose_names(mid)

    assert len(llm.calls) == 2
    assert _proposals(handle.engine)["Speaker 2"]["status"] == "pending"


async def test_parse_proposals_with_repair_and_validation_helpers() -> None:
    asked: list[str] = []

    async def ask(prompt: str) -> str:
        asked.append(prompt)
        return '{"proposals": "nope"}'

    with pytest.raises(InsightsValidationError):
        await parse_proposals_with_repair(ask, "P")
    assert len(asked) == 2
    parsed = validate_proposals('```json\n{"proposals": [{"speaker": "S", "confidence": "2.5"}]}\n```')
    assert parsed.proposals[0].confidence == 1.0  # clamped


async def test_egress_to_an_unlisted_host_is_blocked_before_any_call(tmp_path: Path) -> None:
    handle, meetings_dir, llm, service = _make_service(tmp_path, base_url="http://api.public.example/v1")
    mid = _seed(handle.engine, meetings_dir)

    with pytest.raises(EgressBlockedError):
        await service.propose_names(mid)

    assert llm.calls == []
    assert _proposals(handle.engine) == {}


async def test_prompt_text_comes_from_the_editable_prompts_config(tmp_path: Path) -> None:
    handle, meetings_dir, llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir)
    prompts_dir = tmp_path / "config" / "insights"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "prompts.json").write_text(
        json.dumps(
            {
                "namingSystemPrompt": "CUSTOM SYSTEM",
                "namingPrompt": "CUSTOM {{title}} :: {{labels}} :: {{candidates}}\n{{transcript}}",
            }
        ),
        encoding="utf-8",
    )

    await service.propose_names(mid)

    assert llm.calls[0]["systemPrompt"] == "CUSTOM SYSTEM"
    assert llm.calls[0]["prompt"].startswith("CUSTOM Budget Sync :: Speaker 1, Speaker 2, Speaker 3")


# --------------------------------------------------------------------------- confirm


async def test_apply_creates_and_links_a_speaker_with_name_identity_and_alias(tmp_path: Path) -> None:
    handle, meetings_dir, _llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir)
    await service.propose_names(mid)

    applied = service.apply_confirmations(mid, {"Speaker 2": "Vineet"})

    sid = applied["Speaker 2"]["speakerId"]
    assert applied["Speaker 2"]["created"] is True
    assert _rows(handle.engine, "SELECT name FROM speakers WHERE id = :i", i=sid) == [{"name": "Vineet"}]
    link = _rows(handle.engine, "SELECT * FROM meeting_speakers WHERE meeting_id = :m", m=mid)
    assert len(link) == 1
    assert link[0]["speaker_id"] == sid and link[0]["identity_source"] == "name"
    assert link[0]["identity_confidence"] == pytest.approx(0.92)
    alias = _rows(handle.engine, "SELECT * FROM speaker_aliases WHERE speaker_id = :i", i=sid)
    assert [(a["alias_norm"], a["kind"], a["confirmed"]) for a in alias] == [("vineet", "name", 1)]
    assert _proposals(handle.engine)["Speaker 2"]["status"] == "confirmed"
    assert _proposals(handle.engine)["Speaker 1"]["status"] == "pending"  # untouched


async def test_apply_dedupes_case_insensitively_against_speakers_and_aliases(tmp_path: Path) -> None:
    handle, meetings_dir, _llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir)
    _add_speaker(handle.engine, "spk-vineet", "Vineet")
    _add_speaker(handle.engine, "spk-dana", "Dana Whitfield")
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO speaker_aliases (id, alias_norm, speaker_id, kind, confirmed, created_at) "
                "VALUES ('al-d', 'dana', 'spk-dana', 'nickname', 1, 1)"
            )
        )

    applied = service.apply_confirmations(mid, {"Speaker 2": "  VINEET ", "Speaker 1": "Dana"})

    assert applied["Speaker 2"] == {"speakerId": "spk-vineet", "name": "VINEET", "created": False}
    assert applied["Speaker 1"]["speakerId"] == "spk-dana"
    assert _rows(handle.engine, "SELECT COUNT(*) AS n FROM speakers") == [{"n": 2}]
    linked = {r["speaker_id"] for r in _rows(handle.engine, "SELECT speaker_id FROM meeting_speakers")}
    assert linked == {"spk-vineet", "spk-dana"}


async def test_apply_maps_the_users_aliases_to_the_single_me_speaker(tmp_path: Path) -> None:
    handle, meetings_dir, _llm, service = _make_service(tmp_path, me_aliases=["pmcneely", "Phil"])
    mid = _seed(handle.engine, meetings_dir)
    _add_speaker(handle.engine, "spk-pm", "pmcneely", created=1, enrolled=500)
    _add_speaker(handle.engine, "spk-phil", "Phil", created=2)

    applied = service.apply_confirmations(mid, {"Speaker 1": "phil", "Speaker 3": "Pmcneely"})

    assert applied["Speaker 1"]["speakerId"] == "spk-pm"  # the enrolled row is "me"
    assert applied["Speaker 3"]["speakerId"] == "spk-pm"
    assert _rows(handle.engine, "SELECT COUNT(*) AS n FROM speakers") == [{"n": 2}]
    assert _rows(handle.engine, "SELECT speaker_id FROM meeting_speakers") == [{"speaker_id": "spk-pm"}]
    aliases = {r["alias_norm"] for r in _rows(handle.engine, "SELECT alias_norm FROM speaker_aliases")}
    assert aliases == {"phil", "pmcneely"}


async def test_me_aliases_can_come_from_insights_config(tmp_path: Path) -> None:
    handle, meetings_dir, _llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir)
    (tmp_path / "config" / "insights").mkdir(parents=True)
    (tmp_path / "config" / "insights" / "config.json").write_text(
        json.dumps({"meAliases": ["pmcneely", "phil"]}), encoding="utf-8"
    )
    _add_speaker(handle.engine, "spk-pm", "pmcneely")

    applied = service.apply_confirmations(mid, {"Speaker 1": "Phil"})

    assert applied["Speaker 1"]["speakerId"] == "spk-pm"


async def test_apply_does_not_downgrade_a_stronger_identity(tmp_path: Path) -> None:
    handle, meetings_dir, _llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir)
    _add_speaker(handle.engine, "spk-vineet", "Vineet")
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO meeting_speakers (meeting_id, speaker_id, identity_source, identity_confidence) "
                "VALUES (:m, 'spk-vineet', 'voiceprint', 0.97)"
            ),
            {"m": mid},
        )

    service.apply_confirmations(mid, {"Speaker 2": "Vineet"})

    row = _rows(handle.engine, "SELECT identity_source, identity_confidence FROM meeting_speakers")[0]
    assert row == {"identity_source": "voiceprint", "identity_confidence": 0.97}


async def test_apply_without_a_prior_proposal_records_a_confirmed_row(tmp_path: Path) -> None:
    handle, meetings_dir, _llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir)

    service.apply_confirmations(mid, {"Speaker 2": "Vineet"})

    row = _proposals(handle.engine)["Speaker 2"]
    assert row["status"] == "confirmed" and row["proposed_name"] == "Vineet"


async def test_apply_is_all_or_nothing_and_rejects_generic_names(tmp_path: Path) -> None:
    handle, meetings_dir, _llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir)

    with pytest.raises(ValueError):
        service.apply_confirmations(mid, {"Speaker 1": "Vineet", "Speaker 2": "Speaker 4"})
    with pytest.raises(ValueError):
        service.apply_confirmations("no-such-meeting", {"Speaker 1": "Vineet"})

    assert _rows(handle.engine, "SELECT COUNT(*) AS n FROM speakers") == [{"n": 0}]
    assert _rows(handle.engine, "SELECT COUNT(*) AS n FROM meeting_speakers") == [{"n": 0}]
    assert _proposals(handle.engine) == {}


async def test_apply_is_repeatable(tmp_path: Path) -> None:
    handle, meetings_dir, _llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir)

    first = service.apply_confirmations(mid, {"Speaker 2": "Vineet"})
    second = service.apply_confirmations(mid, {"Speaker 2": "Vineet"})

    assert first["Speaker 2"]["speakerId"] == second["Speaker 2"]["speakerId"]
    assert _rows(handle.engine, "SELECT COUNT(*) AS n FROM speakers") == [{"n": 1}]
    assert _rows(handle.engine, "SELECT COUNT(*) AS n FROM speaker_aliases") == [{"n": 1}]


# --------------------------------------------------------------------------- batch


async def test_propose_all_loops_meetings_filters_and_summarizes(tmp_path: Path) -> None:
    handle, meetings_dir, llm, service = _make_service(tmp_path)
    day = 86_400_000
    base = 1_700_000_000_000
    _seed(handle.engine, meetings_dir, meeting_id="m-old", start=base)
    _seed(handle.engine, meetings_dir, meeting_id="m-mid", start=base + 10 * day)
    _seed(handle.engine, meetings_dir, meeting_id="m-new", start=base + 20 * day)
    named_only = [{"startMs": 0, "endMs": 1000, "speaker": "Alice", "text": "Hello."}]
    _seed(handle.engine, meetings_dir, named_only, meeting_id="m-named", start=base + 30 * day)

    summary = await service.propose_all()

    assert summary["meetings"] == 4
    assert summary["processed"] == 3 and summary["noGenericLabels"] == 1
    assert summary["pending"] == 6 and summary["none"] == 3
    assert len(llm.calls) == 3

    again = await service.propose_all()
    assert again["skipped"] == 3 and again["processed"] == 0
    assert len(llm.calls) == 3

    forced_limited = await service.propose_all(limit=1, force=True)
    assert forced_limited["processed"] == 1
    assert len(llm.calls) == 4


async def test_propose_all_date_bounds_and_limit(tmp_path: Path) -> None:
    handle, meetings_dir, llm, service = _make_service(tmp_path)
    day = 86_400_000
    jan1 = 1_767_225_600_000  # 2026-01-01T00:00:00Z
    _seed(handle.engine, meetings_dir, meeting_id="m-a", start=jan1)
    _seed(handle.engine, meetings_dir, meeting_id="m-b", start=jan1 + 5 * day + 3_600_000)
    _seed(handle.engine, meetings_dir, meeting_id="m-c", start=jan1 + 9 * day)

    only_b = await service.propose_all("2026-01-03", "2026-01-06")
    assert only_b["meetings"] == 1 and only_b["processed"] == 1

    first = await service.propose_all(limit=1)
    assert first["processed"] == 1 and first["skipped"] == 0  # oldest m-a runs; stops at the limit
    rest = await service.propose_all(limit=1)
    assert rest["processed"] == 1 and rest["skipped"] == 2  # already-done meetings do not use the limit
    assert len(llm.calls) == 3


async def test_propose_all_records_a_failing_meeting_and_continues(tmp_path: Path) -> None:
    handle, meetings_dir, _llm, service = _make_service(
        tmp_path, lambda n, _o: "garbage" if n <= 2 else GOOD_REPLY
    )
    base = 1_700_000_000_000
    _seed(handle.engine, meetings_dir, meeting_id="m-bad", start=base)
    _seed(handle.engine, meetings_dir, meeting_id="m-good", start=base + 1000)

    summary = await service.propose_all()

    assert summary["failed"] == 1 and summary["processed"] == 1
    assert summary["errors"][0]["meetingId"] == "m-bad"


async def test_propose_all_aborts_when_egress_is_blocked(tmp_path: Path) -> None:
    handle, meetings_dir, llm, service = _make_service(tmp_path, allowed_hosts=[])
    _seed(handle.engine, meetings_dir)

    with pytest.raises(EgressBlockedError):
        await service.propose_all()
    assert llm.calls == []


# --------------------------------------------------------------------------- schema


def test_migration_adds_proposals_table_idempotently_with_indexes() -> None:
    handle = create_db(":memory:")
    run_migrations(handle.engine)
    run_migrations(handle.engine)

    with handle.engine.connect() as conn:
        tables = set(conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).scalars())
        indexes = {
            r["name"]: r["sql"]
            for r in conn.execute(
                text("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name = :t"),
                {"t": "meeting_speaker_name_proposals"},
            ).mappings()
        }
        cols = {
            r["name"]: r
            for r in conn.execute(text("PRAGMA table_info(meeting_speaker_name_proposals)")).mappings()
        }
    assert "meeting_speaker_name_proposals" in tables
    assert set(cols) == {
        "id", "meeting_id", "speaker_label", "proposed_name", "confidence", "evidence_quote",
        "evidence_turn_idx", "status", "model", "provider", "created_at",
    }  # fmt: skip
    assert "UNIQUE" in indexes["idx_name_proposals_label"].upper()
    assert "idx_name_proposals_meeting" in indexes
    assert cols["status"]["dflt_value"] == "'pending'" and cols["status"]["notnull"] == 1


def test_proposals_table_round_trips_defaults_uniqueness_and_foreign_key() -> None:
    handle = create_db(":memory:")
    now = 1_700_000_000_000
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, created_at, updated_at) "
                "VALUES ('m1', 't', :n, :n, 'upload', 'ready', '[]', :n, :n)"
            ),
            {"n": now},
        )
        conn.execute(
            text(
                "INSERT INTO meeting_speaker_name_proposals (id, meeting_id, speaker_label, created_at) "
                "VALUES ('p1', 'm1', 'Speaker 1', :n)"
            ),
            {"n": now},
        )
        row = (
            conn.execute(text("SELECT * FROM meeting_speaker_name_proposals WHERE id = 'p1'"))
            .mappings()
            .one()
        )
    assert row["status"] == "pending" and row["proposed_name"] is None and row["confidence"] is None

    with pytest.raises(IntegrityError), handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO meeting_speaker_name_proposals (id, meeting_id, speaker_label, created_at) "
                "VALUES ('p2', 'm1', 'Speaker 1', :n)"
            ),
            {"n": now},
        )
    with pytest.raises(IntegrityError), handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO meeting_speaker_name_proposals (id, meeting_id, speaker_label, created_at) "
                "VALUES ('p3', 'ghost', 'Speaker 1', :n)"
            ),
            {"n": now},
        )
    assert MeetingSpeakerNameProposalModel.__tablename__ == "meeting_speaker_name_proposals"


async def test_delete_meeting_removes_proposals(tmp_path: Path) -> None:
    handle, meetings_dir, _llm, service = _make_service(tmp_path)
    mid = _seed(handle.engine, meetings_dir)
    keep = _seed(handle.engine, meetings_dir, meeting_id="meeting-keep")
    await service.propose_names(mid)
    await service.propose_names(keep)

    assert delete_meeting(handle.engine, mid, meetings_dir) is True

    assert _proposals(handle.engine, mid) == {}
    assert len(_proposals(handle.engine, keep)) == 3


def test_naming_module_does_not_import_the_ml_stack() -> None:
    code = (
        "import sys, olive.services.insights.naming; "
        "bad = sorted(m for m in ('torch', 'pyannote', 'pyannote.audio', 'olive.ml.identify') if m in sys.modules); "
        "sys.exit(1 if bad else 0)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
