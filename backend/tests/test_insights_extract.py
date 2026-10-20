"""Olive Insights Phase 1a: signal-extraction core (prompts, chunking, validation, identity, service)."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from olive.db import create_db
from olive.layout import meeting_paths
from olive.schemas import LlmSettings, TranscriptSegment
from olive.services.insights import extract as insights_extract
from olive.services.insights.chunking import chunk_segments, dedup_moments, format_turn
from olive.services.insights.extract import (
    ChunkResult,
    EgressBlockedError,
    InsightsExtractionError,
    InsightsService,
    compute_turn_stats,
    enforce_egress,
    ground_moment,
    map_ordinal,
    merge_chunks,
)
from olive.services.insights.identity import resolve_labels
from olive.services.insights.naming import SpeakerNamingService
from olive.services.insights.prompts import (
    DEFAULT_CONFIG,
    DEFAULT_PROMPTS,
    compute_prompt_version,
    load_insights_config,
    load_insights_prompts,
)
from olive.services.insights.validate import (
    ExtractedMoment,
    InsightsValidationError,
    parse_llm_json,
    parse_with_repair,
)
from olive.settings import AppConfig, mask_app_config

PROXY_HOST = "llm-proxy.local:3020"
PROXY_URL = f"http://{PROXY_HOST}/v1"


def _seg(start: int, end: int, speaker: str, body: str) -> TranscriptSegment:
    return TranscriptSegment(startMs=start, endMs=end, speaker=speaker, text=body)


MAIN_SEGMENTS = [
    {"startMs": 0, "endMs": 5000, "speaker": "Alice", "text": "Welcome everyone. Let's discuss the roadmap."},
    {
        "startMs": 5200,
        "endMs": 12000,
        "speaker": "Bob",
        "text": "You always sabotage this project and I am sick of it.",
    },
    {"startMs": 11500, "endMs": 14000, "speaker": "Alice", "text": "That is not fair, I did my part."},
    {"startMs": 14500, "endMs": 16000, "speaker": "Speaker 1", "text": "Okay, moving on."},
]


def _chunk_json(speakers: list[dict[str, Any]], moments: list[dict[str, Any]] | None = None) -> str:
    return json.dumps({"speakers": speakers, "moments": moments or []})


def _main_response() -> str:
    return _chunk_json(
        [
            {"speaker": "Alice", "dimensions": {"hostility": "low", "warmth": "high"}, "stance": "assertive"},
            {"speaker": "Bob", "dimensions": {"hostility": "high"}, "stance": "aggressive"},
        ],
        [
            {
                "speaker": "Bob",
                "toward": "Alice",
                "turnStartIdx": 1,
                "eventType": "accusation",
                "valence": "bad",
                "framework": "gottman:criticism",
                "dimension": "hostility",
                "severity": "high",
                "quote": "You always sabotage this project",
                "rationale": "Global blame aimed at a person rather than a behavior.",
                "guidance": "Name the specific behavior and the request.",
                "startMs": 99999,  # an LLM-supplied timestamp must never be trusted
            }
        ],
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
        body = self.responder(len(self.calls), options)
        return {
            "text": body,
            "provider": options.get("provider") or "openai",
            "model": options.get("model") or "default-model",
            "durationMs": 5,
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }


def _make_service(
    tmp_path: Path,
    responder: Callable[[int, dict[str, Any]], str],
    *,
    base_url: str = PROXY_URL,
    allowed_hosts: list[str] | None = None,
):
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    config_dir = str(tmp_path / "config")
    Path(meetings_dir).mkdir(parents=True, exist_ok=True)
    Path(config_dir).mkdir(parents=True, exist_ok=True)
    llm = FakeLlm(responder, base_url)
    service = InsightsService(
        db=handle.engine,
        meetings_dir=meetings_dir,
        config_dir=config_dir,
        llm_service=llm,
        allowed_hosts=[PROXY_HOST] if allowed_hosts is None else allowed_hosts,
    )
    return handle, meetings_dir, config_dir, llm, service


def _seed(
    engine,
    meetings_dir: str,
    segments: list[dict[str, Any]] | None = None,
    *,
    meeting_id: str = "meeting-ins-1",
    people: tuple[str, ...] = ("Alice", "Bob"),
) -> str:
    now = int(time.time() * 1000)
    title = "Roadmap Sync"
    paths = meeting_paths(meetings_dir, now, title, meeting_id)
    Path(paths.transcripts_dir).mkdir(parents=True, exist_ok=True)
    rel = "transcripts/transcript.json"
    (Path(paths.folder) / rel).write_text(
        json.dumps({"segments": segments or MAIN_SEGMENTS}), encoding="utf-8"
    )
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
        for name in people:
            sid = f"spk-{name.lower()}"
            conn.execute(
                text("INSERT OR IGNORE INTO speakers (id, name, created_at) VALUES (:id, :name, :now)"),
                {"id": sid, "name": name, "now": now},
            )
            conn.execute(
                text("INSERT INTO meeting_speakers (meeting_id, speaker_id) VALUES (:m, :s)"),
                {"m": meeting_id, "s": sid},
            )
    return meeting_id


def _folder(engine, meetings_dir: str, meeting_id: str) -> str:
    row = _rows(engine, "SELECT start_time, title FROM meetings WHERE id = :m", m=meeting_id)[0]
    return meeting_paths(meetings_dir, row["start_time"], row["title"], meeting_id).folder


def _rows(engine, sql: str, **params: Any) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(sql), params).mappings().all()]


# --------------------------------------------------------------------------- parsing


def test_parse_llm_json_strips_think_fences_and_prose() -> None:
    raw = (
        "<think>I should output {not json} here</think>\n"
        "Sure! Here is the result:\n```json\n"
        '{"speakers": [], "moments": [{"quote": "a } brace in a string"}]}\n'
        "```\nHope that helps {really}."
    )
    parsed = parse_llm_json(raw)
    assert parsed["speakers"] == []
    assert parsed["moments"][0]["quote"] == "a } brace in a string"


def test_parse_llm_json_skips_unparseable_brace_runs_and_rejects_non_json() -> None:
    assert parse_llm_json('note {oops} then {"a": 1}') == {"a": 1}
    with pytest.raises(InsightsValidationError):
        parse_llm_json("there is no object here")
    with pytest.raises(InsightsValidationError):
        parse_llm_json('{"unterminated": ')


async def test_parse_with_repair_makes_exactly_one_repair_ask() -> None:
    prompts: list[str] = []

    async def ask_bad(prompt: str) -> str:
        prompts.append(prompt)
        return "not json at all"

    with pytest.raises(InsightsValidationError):
        await parse_with_repair(ask_bad, "ORIGINAL")
    assert len(prompts) == 2
    assert prompts[0] == "ORIGINAL"
    assert prompts[1].startswith("ORIGINAL") and len(prompts[1]) > len("ORIGINAL")

    answers = iter(["garbage", _chunk_json([{"speaker": "A", "dimensions": {"x": "low"}}])])

    async def ask_fix(prompt: str) -> str:
        return next(answers)

    model, raw = await parse_with_repair(ask_fix, "ORIGINAL")
    assert model.speakers[0].speaker == "A"
    assert "speakers" in raw


def test_moment_model_normalizes_valence_and_rejects_unknown() -> None:
    base = {"speaker": "A", "turnStartIdx": 0, "eventType": "praise", "quote": "great", "severity": 0.5}
    assert ExtractedMoment.model_validate({**base, "valence": "GOOD"}).valence == "good"
    with pytest.raises(ValueError):
        ExtractedMoment.model_validate({**base, "valence": "terrible"})


# --------------------------------------------------------------------------- chunking


def test_chunking_respects_budget_never_splits_and_covers_everything() -> None:
    segs = [_seg(i * 1000, i * 1000 + 900, "A" if i % 2 else "B", "word " * 10) for i in range(12)]
    budget = 3 * len(format_turn(0, segs[0])) + 3
    chunks = chunk_segments(segs, budget, overlap_turns=0)
    assert len(chunks) > 1
    covered: list[int] = []
    for ch in chunks:
        cost = sum(len(format_turn(i, segs[i])) + 1 for i in range(ch.start, ch.end))
        assert cost <= budget
        covered.extend(range(ch.start, ch.end))
    assert covered == list(range(12))  # contiguous, no gaps, no duplicates
    assert chunk_segments(segs, budget, overlap_turns=0) == chunks  # deterministic


def test_chunking_overlap_and_core_start_and_oversize_segment() -> None:
    segs = [_seg(i * 1000, i * 1000 + 900, "A", "word " * 10) for i in range(10)]
    budget = 4 * (len(format_turn(0, segs[0])) + 1) + 2
    chunks = chunk_segments(segs, budget, overlap_turns=1)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.start == prev.end - 1  # one turn of overlap
        assert nxt.core_start == prev.end  # weighting only counts new turns
    assert chunks[0].core_start == 0 and chunks[-1].end == 10

    big = [_seg(0, 1000, "A", "x" * 500), _seg(1000, 2000, "B", "short")]
    only = chunk_segments(big, 100, overlap_turns=0)
    assert [(c.start, c.end) for c in only] == [(0, 1), (1, 2)]  # oversize turn stays whole

    assert chunk_segments([], 100) == []
    assert len(chunk_segments(segs, 10**9)) == 1


def test_dedup_moments_by_speaker_turn_range_and_event_type() -> None:
    def mk(speaker: str, start: int, end: int | None, event: str, quote: str = "q") -> ExtractedMoment:
        return ExtractedMoment(
            speaker=speaker,
            turnStartIdx=start,
            turnEndIdx=end,
            eventType=event,
            valence="bad",
            severity="high",
            quote=quote,
        )

    items = [
        mk("Bob", 3, None, "Accusation", "first"),
        mk("bob ", 3, 3, "accusation", "dup"),  # same key after normalization
        mk("Bob", 3, None, "threat"),
        mk("Alice", 3, None, "accusation"),
        mk("Bob", 4, None, "accusation"),
    ]
    kept = dedup_moments(items)
    assert [m.quote for m in kept][0] == "first"
    assert len(kept) == 4


# --------------------------------------------------------------------------- stats / grounding / merge


def test_compute_turn_stats_counts_talk_turns_and_interruptions() -> None:
    segs = [_seg(0, 5000, "A", "x"), _seg(4000, 8000, "B", "y"), _seg(9000, 10000, "A", "z")]
    stats = compute_turn_stats(segs)
    assert stats["A"] == {"talkMs": 6000, "turnCount": 2, "interruptCount": 0}
    assert stats["B"] == {"talkMs": 4000, "turnCount": 1, "interruptCount": 1}


def _moment(**over: Any) -> ExtractedMoment:
    data: dict[str, Any] = {
        "speaker": "Bob",
        "turnStartIdx": 1,
        "eventType": "accusation",
        "valence": "bad",
        "severity": "high",
        "quote": "you always sabotage this project",
    }
    data.update(over)
    return ExtractedMoment.model_validate(data)


def _main_segments() -> list[TranscriptSegment]:
    return [TranscriptSegment.model_validate(s) for s in MAIN_SEGMENTS]


def test_ground_moment_verifies_and_snaps_timestamp_to_segment() -> None:
    g = ground_moment(_moment(quote="You ALWAYS sabotage this project!"), _main_segments())
    assert g is not None and g.verified is True
    assert (g.start_ms, g.end_ms) == (5200, 12000)


def test_ground_moment_flags_hallucinated_quote_wrong_speaker_and_drops_bad_index() -> None:
    segs = _main_segments()
    fake = ground_moment(_moment(quote="You never listen to anyone"), segs)
    assert fake is not None and fake.verified is False
    wrong = ground_moment(_moment(speaker="Alice"), segs)  # quote is Bob's line
    assert wrong is not None and wrong.verified is False
    assert ground_moment(_moment(turnStartIdx=99), segs) is None
    assert ground_moment(_moment(turnStartIdx=-1), segs) is None


def test_ground_moment_accepts_multi_turn_range() -> None:
    g = ground_moment(
        _moment(speaker="Alice", turnStartIdx=2, turnEndIdx=3, quote="did my part"), _main_segments()
    )
    assert g is not None and g.verified is True and g.end_ms == 16000


def test_map_ordinal_uses_configured_scale() -> None:
    scale = DEFAULT_CONFIG["dimensionScale"]
    assert map_ordinal("None", scale) == scale["none"]
    assert map_ordinal("Very High", scale) == scale["very high"]
    assert map_ordinal("very_high", scale) == scale["very high"]
    assert map_ordinal(0.3, scale) == 0.3
    assert map_ordinal(7, scale) == 1.0  # clamped
    assert map_ordinal("0.4", scale) == 0.4
    assert map_ordinal("sorta", scale) is None
    assert map_ordinal("high", {"high": 0.9}) == 0.9  # config, not code, owns the numbers


def test_merge_chunks_weights_dimensions_by_talk_time() -> None:
    from olive.services.insights.validate import ChunkExtraction

    scale = {"low": 0.2, "high": 0.8}

    def ext(level: str) -> ChunkExtraction:
        return ChunkExtraction.model_validate(
            {"speakers": [{"speaker": "Alice", "dimensions": {"hostility": level}, "stance": level}]}
        )

    merged = merge_chunks(
        [
            ChunkResult(ext("low"), {"Alice": 3000}),
            ChunkResult(ext("high"), {"alice": 1000}),
        ],
        scale,
    )
    alice = merged["alice"]
    assert alice.label == "Alice"
    assert alice.dimensions["hostility"] == pytest.approx((3000 * 0.2 + 1000 * 0.8) / 4000)
    assert alice.stance == "low"  # stance from the chunk where they spoke most

    zero = merge_chunks([ChunkResult(ext("low"), {}), ChunkResult(ext("high"), {})], scale)
    assert zero["alice"].dimensions["hostility"] == pytest.approx(0.5)  # plain mean when no weights


# --------------------------------------------------------------------------- identity


def test_resolve_labels_keeps_voiceprint_provenance_when_matched_via_alias() -> None:
    handle = create_db(":memory:")
    now = 1_700_000_000_000
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, created_at, updated_at) "
                "VALUES ('m', 't', :n, :n, 'upload', 'ready', '[]', :n, :n)"
            ),
            {"n": now},
        )
        conn.execute(
            text("INSERT INTO speakers (id, name, created_at) VALUES ('s-rob', 'Robert Smith', :n)"),
            {"n": now},
        )
        conn.execute(
            text(
                "INSERT INTO meeting_speakers (meeting_id, speaker_id, identity_source, identity_confidence) "
                "VALUES ('m', 's-rob', 'voiceprint', 0.87)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO speaker_aliases (id, alias_norm, speaker_id, kind, confirmed, created_at) "
                "VALUES ('a1', 'bobby', 's-rob', 'nickname', 1, :n)"
            ),
            {"n": now},
        )
    with handle.engine.connect() as conn:
        ids = resolve_labels(conn, "m", ["Bobby"])
    assert ids["Bobby"].speaker_id == "s-rob"
    assert ids["Bobby"].identity_source == "voiceprint"
    assert ids["Bobby"].identity_confidence == pytest.approx(0.87)
    handle.engine.dispose()


def test_resolve_labels_leaves_ambiguous_names_and_aliases_unresolved() -> None:
    handle = create_db(":memory:")
    now = 1_700_000_000_000
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, created_at, updated_at) "
                "VALUES ('m', 't', :n, :n, 'upload', 'ready', '[]', :n, :n)"
            ),
            {"n": now},
        )
        for sid, name in (("s-a1", "Sam"), ("s-a2", "sam"), ("s-b", "Beth"), ("s-c", "Carol")):
            conn.execute(
                text("INSERT INTO speakers (id, name, created_at) VALUES (:i, :n, :now)"),
                {"i": sid, "n": name, "now": now},
            )
        for sid in ("s-a1", "s-a2"):
            conn.execute(
                text(
                    "INSERT INTO meeting_speakers (meeting_id, speaker_id, identity_source) VALUES ('m', :s, 'name')"
                ),
                {"s": sid},
            )
        for aid, sid in (("a1", "s-b"), ("a2", "s-c")):
            conn.execute(
                text(
                    "INSERT INTO speaker_aliases (id, alias_norm, speaker_id, kind, confirmed, created_at) "
                    "VALUES (:a, 'bc', :s, 'nickname', 1, :n)"
                ),
                {"a": aid, "s": sid, "n": now},
            )
    with handle.engine.connect() as conn:
        ids = resolve_labels(conn, "m", ["Sam", "BC", "Beth"])
    assert ids["Sam"].speaker_id is None  # two meeting speakers share the name
    assert ids["BC"].speaker_id is None  # alias held by two people
    assert ids["Beth"].speaker_id is None  # not linked to this meeting and no alias of its own
    handle.engine.dispose()


def test_validate_chunk_rejects_objects_that_are_not_extractions() -> None:
    from olive.services.insights.validate import validate_chunk

    for bad in ('{"error": "I cannot do that"}', "{}", '{"speaker": []}', '{"speakers": "none"}'):
        with pytest.raises(InsightsValidationError):
            validate_chunk(bad)
    empty = validate_chunk('{"speakers": [], "moments": []}')  # validly empty is still fine
    assert empty.speakers == [] and empty.moments == []
    assert validate_chunk('{"moments": []}').speakers == []


async def test_malformed_extraction_object_fails_the_stage_run_not_silent_empty(tmp_path: Path) -> None:
    handle, meetings_dir, _, llm, service = _make_service(tmp_path, lambda n, o: '{"unexpected": true}')
    meeting_id = _seed(handle.engine, meetings_dir)
    with pytest.raises(InsightsExtractionError):
        await service.analyze_meeting(meeting_id)
    assert len(llm.calls) == 2  # one repair re-ask, then fail
    run = _rows(handle.engine, "SELECT status FROM stage_runs WHERE stage = 'insights_analyze'")
    assert run[0]["status"] == "error"
    assert _rows(handle.engine, "SELECT id FROM meeting_speaker_signals") == []
    handle.engine.dispose()


def test_resolve_labels_names_aliases_and_generic() -> None:
    handle = create_db(":memory:")
    now = 1_700_000_000_000
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, created_at, updated_at) "
                "VALUES ('m', 't', :n, :n, 'upload', 'ready', '[]', :n, :n)"
            ),
            {"n": now},
        )
        for sid, name in (("s-alice", "Alice"), ("s-rob", "Robert Smith")):
            conn.execute(
                text("INSERT INTO speakers (id, name, created_at) VALUES (:i, :n, :now)"),
                {"i": sid, "n": name, "now": now},
            )
        conn.execute(
            text(
                "INSERT INTO meeting_speakers (meeting_id, speaker_id, identity_source, identity_confidence) "
                "VALUES ('m', 's-alice', 'voiceprint', 0.91)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO speaker_aliases (id, alias_norm, speaker_id, kind, confirmed, created_at) "
                "VALUES ('a1', 'bobby', 's-rob', 'nickname', 1, :n)"
            ),
            {"n": now},
        )
    with handle.engine.connect() as conn:
        ids = resolve_labels(conn, "m", ["  alice ", "Bobby", "Speaker 1", "Stranger", ""])
    assert ids["  alice "].speaker_id == "s-alice"
    assert ids["  alice "].identity_source == "voiceprint"
    assert ids["  alice "].identity_confidence == pytest.approx(0.91)
    assert ids["Bobby"].speaker_id == "s-rob" and ids["Bobby"].identity_source == "manual"
    assert ids["Speaker 1"].speaker_id is None and ids["Speaker 1"].label == "Speaker 1"
    assert ids["Stranger"].speaker_id is None
    assert ids[""].speaker_id is None


CONFIRMED_SEGMENTS = [
    {"startMs": 0, "endMs": 5000, "speaker": "Alice", "text": "Welcome everyone, Vineet, over to you."},
    {
        "startMs": 5200,
        "endMs": 12000,
        "speaker": "Speaker 2",
        "text": "You always sabotage this project and I am sick of it.",
    },
    {"startMs": 12500, "endMs": 14000, "speaker": "Speaker 3", "text": "Okay, moving on."},
]


def _naming(handle, meetings_dir: str, config_dir: str) -> SpeakerNamingService:
    return SpeakerNamingService(
        db=handle.engine, meetings_dir=meetings_dir, config_dir=config_dir, llm_service=None
    )


def _confirmed_response() -> str:
    return _chunk_json(
        [{"speaker": "Speaker 2", "dimensions": {"hostility": "high"}, "stance": "aggressive"}],
        [
            {
                "speaker": "Speaker 2",
                "toward": "Alice",
                "turnStartIdx": 1,
                "eventType": "accusation",
                "valence": "bad",
                "dimension": "hostility",
                "severity": "high",
                "quote": "You always sabotage this project",
                "rationale": "Global blame.",
            }
        ],
    )


def test_resolve_labels_honors_confirmed_name_proposals(tmp_path: Path) -> None:
    handle, meetings_dir, config_dir, _llm, _service = _make_service(tmp_path, lambda n, o: "{}")
    mid = _seed(handle.engine, meetings_dir, CONFIRMED_SEGMENTS, people=("Alice",))
    with handle.engine.connect() as conn:
        before = resolve_labels(conn, mid, ["Speaker 2", "Speaker 3"])
    assert before["Speaker 2"].speaker_id is None

    applied = _naming(handle, meetings_dir, config_dir).apply_confirmations(mid, {"Speaker 2": "Vineet"})
    with handle.engine.connect() as conn:
        after = resolve_labels(conn, mid, ["Speaker 2", "Speaker 3"])
    assert after["Speaker 2"].speaker_id == applied["Speaker 2"]["speakerId"]
    assert after["Speaker 2"].identity_source == "name"
    assert after["Speaker 3"].speaker_id is None  # unconfirmed generic label stays unresolved
    handle.engine.dispose()


def test_resolve_labels_ignores_unconfirmed_and_other_meeting_proposals(tmp_path: Path) -> None:
    handle, meetings_dir, _config_dir, _llm, _service = _make_service(tmp_path, lambda n, o: "{}")
    mid = _seed(handle.engine, meetings_dir, CONFIRMED_SEGMENTS, people=("Alice", "Vineet"))
    other = _seed(
        handle.engine, meetings_dir, CONFIRMED_SEGMENTS, meeting_id="meeting-ins-2", people=("Vineet",)
    )
    with handle.engine.begin() as conn:
        for pid, meeting, status in (("p1", mid, "pending"), ("p2", other, "confirmed")):
            conn.execute(
                text(
                    "INSERT INTO meeting_speaker_name_proposals (id, meeting_id, speaker_label, proposed_name, "
                    "status, created_at) VALUES (:i, :m, 'Speaker 2', 'Vineet', :s, 1)"
                ),
                {"i": pid, "m": meeting, "s": status},
            )
    with handle.engine.connect() as conn:
        assert resolve_labels(conn, mid, ["Speaker 2"])["Speaker 2"].speaker_id is None
        assert resolve_labels(conn, other, ["Speaker 2"])["Speaker 2"].speaker_id == "spk-vineet"
    handle.engine.dispose()


def test_confirmed_name_does_not_override_a_voiceprint_link(tmp_path: Path) -> None:
    handle, meetings_dir, config_dir, _llm, _service = _make_service(tmp_path, lambda n, o: "{}")
    mid = _seed(handle.engine, meetings_dir, CONFIRMED_SEGMENTS, people=("Alice", "Vineet"))
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE meeting_speakers SET identity_source = 'voiceprint', identity_confidence = 0.95 "
                "WHERE meeting_id = :m AND speaker_id = 'spk-vineet'"
            ),
            {"m": mid},
        )
    _naming(handle, meetings_dir, config_dir).apply_confirmations(mid, {"Speaker 2": "Vineet"})
    with handle.engine.connect() as conn:
        hit = resolve_labels(conn, mid, ["Speaker 2"])["Speaker 2"]
    assert (hit.speaker_id, hit.identity_source) == ("spk-vineet", "voiceprint")
    assert hit.identity_confidence == pytest.approx(0.95)
    handle.engine.dispose()


async def test_extraction_attributes_confirmed_label_to_the_confirmed_speaker(tmp_path: Path) -> None:
    handle, meetings_dir, config_dir, _llm, service = _make_service(
        tmp_path, lambda n, o: _confirmed_response()
    )
    (Path(config_dir) / "settings.json").write_text(
        json.dumps({"insightsProvider": "openai", "insightsModel": "pinned-model"})
    )
    mid = _seed(handle.engine, meetings_dir, CONFIRMED_SEGMENTS, people=("Alice",))
    applied = _naming(handle, meetings_dir, config_dir).apply_confirmations(mid, {"Speaker 2": "Vineet"})
    vineet = applied["Speaker 2"]["speakerId"]

    assert (await service.analyze_meeting(mid))["status"] == "done"

    signals = {r["speaker_label"]: r for r in _rows(handle.engine, "SELECT * FROM meeting_speaker_signals")}
    assert signals["Speaker 2"]["speaker_id"] == vineet
    assert signals["Speaker 2"]["identity_source"] == "name"
    assert signals["Speaker 3"]["speaker_id"] is None
    moment = _rows(handle.engine, "SELECT * FROM meeting_moments")[0]
    assert (moment["speaker_id"], moment["speaker_label"]) == (vineet, "Speaker 2")
    assert moment["toward_speaker_id"] == "spk-alice"
    handle.engine.dispose()


# --------------------------------------------------------------------------- prompts / config


def test_prompts_seed_on_first_use_and_user_keys_override_per_key(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    prompts = load_insights_prompts(str(config_dir))
    assert prompts == DEFAULT_PROMPTS
    seeded = json.loads((config_dir / "insights" / "prompts.json").read_text())
    assert set(seeded) == set(DEFAULT_PROMPTS)

    (config_dir / "insights" / "prompts.json").write_text(json.dumps({"systemPrompt": "Custom lens."}))
    over = load_insights_prompts(str(config_dir))
    assert over["systemPrompt"] == "Custom lens."
    assert over["extractPrompt"] == DEFAULT_PROMPTS["extractPrompt"]
    assert over["frameworks"] == DEFAULT_PROMPTS["frameworks"]
    assert compute_prompt_version(over) != compute_prompt_version(DEFAULT_PROMPTS)
    assert compute_prompt_version(over) == compute_prompt_version(dict(over))


def test_naming_prompts_are_seeded_versioned_and_user_overridable(tmp_path: Path) -> None:
    assert {"namingPrompt", "namingSystemPrompt"} <= set(DEFAULT_PROMPTS)
    config_dir = tmp_path / "cfg"
    seeded = load_insights_prompts(str(config_dir))
    assert "{{labels}}" in seeded["namingPrompt"] and seeded["namingSystemPrompt"]
    on_disk = json.loads((config_dir / "insights" / "prompts.json").read_text())
    assert on_disk["namingPrompt"] == DEFAULT_PROMPTS["namingPrompt"]

    for key in ("namingPrompt", "namingSystemPrompt"):
        tweaked = {**DEFAULT_PROMPTS, key: "changed"}
        assert compute_prompt_version(tweaked) != compute_prompt_version(DEFAULT_PROMPTS)

    (config_dir / "insights" / "prompts.json").write_text(json.dumps({"namingPrompt": "Mine {{labels}}"}))
    over = load_insights_prompts(str(config_dir))
    assert over["namingPrompt"] == "Mine {{labels}}"
    assert over["namingSystemPrompt"] == DEFAULT_PROMPTS["namingSystemPrompt"]


def test_malformed_prompt_and_config_files_fall_back_to_defaults(tmp_path: Path) -> None:
    (tmp_path / "insights").mkdir()
    (tmp_path / "insights" / "prompts.json").write_text("{ nope")
    (tmp_path / "insights" / "config.json").write_text("[1, 2")
    assert load_insights_prompts(str(tmp_path)) == DEFAULT_PROMPTS
    assert load_insights_config(str(tmp_path)) == DEFAULT_CONFIG

    (tmp_path / "insights" / "config.json").write_text(json.dumps({"chunkChars": 500}))
    cfg = load_insights_config(str(tmp_path))
    assert cfg["chunkChars"] == 500
    assert cfg["dimensionScale"] == DEFAULT_CONFIG["dimensionScale"]


def test_default_prompt_is_behavioral_with_named_frameworks() -> None:
    blob = " ".join(str(v) for v in DEFAULT_PROMPTS.values()).lower()
    for term in ("criticism", "contempt", "defensiveness", "stonewalling", "repair", "nonviolent"):
        assert term in blob
    for term in ("assertive", "aggressive", "passive"):
        assert term in blob
    assert not any(ch.isdigit() for ch in DEFAULT_PROMPTS["frameworks"])  # no numeric thresholds
    assert "{{transcript}}" in DEFAULT_PROMPTS["extractPrompt"]


def test_app_config_has_pinned_insights_model_fields() -> None:
    cfg = AppConfig()
    assert cfg.insightsProvider == "" and cfg.insightsModel == ""
    masked = mask_app_config(AppConfig(insightsProvider="openai", insightsModel="qwen"))
    assert masked["insightsProvider"] == "openai" and masked["insightsModel"] == "qwen"


# --------------------------------------------------------------------------- egress guard


def test_enforce_egress_matches_host_or_host_port_and_fails_closed() -> None:
    enforce_egress("http://llm-proxy.local:3020/v1", ["llm-proxy.local:3020"])
    enforce_egress("http://LLM-Proxy.local:3020/v1/", ["llm-proxy.local"])
    for bad_url, allow in (
        ("https://api.openai.com/v1", ["llm-proxy.local:3020"]),
        ("http://llm-proxy.local:9999/v1", ["llm-proxy.local:3020"]),
        ("http://llm-proxy.local:3020/v1", []),  # empty allowlist blocks everything
        (None, ["llm-proxy.local"]),
        ("not a url", ["llm-proxy.local"]),
    ):
        with pytest.raises(EgressBlockedError):
            enforce_egress(bad_url, allow)


async def test_egress_guard_blocks_before_any_llm_call(tmp_path: Path) -> None:
    handle, meetings_dir, _, llm, service = _make_service(
        tmp_path, lambda n, o: _main_response(), base_url="https://api.openai.com/v1"
    )
    meeting_id = _seed(handle.engine, meetings_dir)
    with pytest.raises(EgressBlockedError):
        await service.analyze_meeting(meeting_id)
    assert llm.calls == []
    run = _rows(handle.engine, "SELECT status, last_error FROM stage_runs WHERE stage = 'insights_analyze'")
    assert run[0]["status"] == "error" and "api.openai.com" in run[0]["last_error"]
    assert _rows(handle.engine, "SELECT id FROM meeting_speaker_signals") == []
    handle.engine.dispose()


async def test_allowlist_comes_from_env_when_not_injected(tmp_path: Path, monkeypatch) -> None:
    handle, meetings_dir, config_dir, llm, _ = _make_service(tmp_path, lambda n, o: _main_response())
    service = InsightsService(
        db=handle.engine, meetings_dir=meetings_dir, config_dir=config_dir, llm_service=llm
    )
    meeting_id = _seed(handle.engine, meetings_dir)
    monkeypatch.delenv("OLIVE_INSIGHTS_ALLOWED_HOSTS", raising=False)
    with pytest.raises(EgressBlockedError):
        await service.analyze_meeting(meeting_id)
    monkeypatch.setenv("OLIVE_INSIGHTS_ALLOWED_HOSTS", f"other.host, {PROXY_HOST}")
    result = await service.analyze_meeting(meeting_id)
    assert result["status"] == "done"
    handle.engine.dispose()


# --------------------------------------------------------------------------- analyze_meeting


async def test_analyze_persists_signals_moments_and_artifact(tmp_path: Path) -> None:
    handle, meetings_dir, config_dir, llm, service = _make_service(tmp_path, lambda n, o: _main_response())
    (Path(config_dir) / "settings.json").write_text(
        json.dumps({"insightsProvider": "openai", "insightsModel": "pinned-model"})
    )
    meeting_id = _seed(handle.engine, meetings_dir)

    result = await service.analyze_meeting(meeting_id)
    assert result["status"] == "done"
    assert len(llm.calls) == 1
    assert llm.calls[0]["provider"] == "openai" and llm.calls[0]["model"] == "pinned-model"

    signals = {r["speaker_label"]: r for r in _rows(handle.engine, "SELECT * FROM meeting_speaker_signals")}
    assert set(signals) == {"Alice", "Bob", "Speaker 1"}
    assert signals["Alice"]["speaker_id"] == "spk-alice"
    assert signals["Alice"]["identity_source"] == "name"
    assert signals["Bob"]["speaker_id"] == "spk-bob"
    assert signals["Speaker 1"]["speaker_id"] is None  # generic label stays unresolved
    scale = DEFAULT_CONFIG["dimensionScale"]
    assert json.loads(signals["Bob"]["dimensions"]) == {"hostility": scale["high"]}
    assert json.loads(signals["Alice"]["dimensions"]) == {
        "hostility": scale["low"],
        "warmth": scale["high"],
    }
    assert signals["Bob"]["stance"] == "aggressive"
    assert signals["Alice"]["talk_ms"] == 5000 + 2500
    assert signals["Alice"]["turn_count"] == 2 and signals["Alice"]["interrupt_count"] == 1
    assert signals["Bob"]["interrupt_count"] == 0
    assert signals["Speaker 1"]["dimensions"] == "{}"
    assert signals["Bob"]["prompt_version"] == compute_prompt_version(load_insights_prompts(config_dir))
    assert signals["Bob"]["provider"] == "openai" and signals["Bob"]["model"] == "pinned-model"
    transcript_raw = (
        Path(_folder(handle.engine, meetings_dir, meeting_id)) / "transcripts/transcript.json"
    ).read_text(encoding="utf-8")
    assert signals["Bob"]["transcript_hash"] == hashlib.sha256(transcript_raw.encode()).hexdigest()

    moments = _rows(handle.engine, "SELECT * FROM meeting_moments")
    assert len(moments) == 1
    m = moments[0]
    assert (m["speaker_id"], m["toward_speaker_id"]) == ("spk-bob", "spk-alice")
    assert (m["speaker_label"], m["toward_label"]) == ("Bob", "Alice")
    assert m["start_ms"] == 5200 and m["end_ms"] == 12000  # snapped to the segment, not 99999
    assert m["turn_start_idx"] == 1 and m["verified"] == 1
    assert m["event_type"] == "accusation" and m["valence"] == "bad"
    assert m["framework"] == "gottman:criticism" and m["severity"] == scale["high"]
    assert m["guidance"] and m["rationale"]

    artifacts = _rows(handle.engine, "SELECT * FROM artifacts WHERE kind = 'signals'")
    assert len(artifacts) == 1 and artifacts[0]["format"] == "json"
    folder = _folder(handle.engine, meetings_dir, meeting_id)
    payload = json.loads((Path(folder) / artifacts[0]["path"]).read_text())
    assert payload["transcriptHash"] == signals["Bob"]["transcript_hash"]
    assert payload["chunks"][0]["raw"]

    run = _rows(handle.engine, "SELECT * FROM stage_runs WHERE stage = 'insights_analyze'")
    assert len(run) == 1 and run[0]["status"] == "done" and run[0]["last_error"] is None
    handle.engine.dispose()


async def test_hallucinated_quote_is_stored_unverified(tmp_path: Path) -> None:
    bad = _chunk_json(
        [{"speaker": "Bob", "dimensions": {"hostility": "high"}}],
        [
            {
                "speaker": "Bob",
                "turnStartIdx": 1,
                "eventType": "threat",
                "valence": "bad",
                "severity": "high",
                "quote": "I will end this partnership tomorrow",
            }
        ],
    )
    handle, meetings_dir, _, _, service = _make_service(tmp_path, lambda n, o: bad)
    meeting_id = _seed(handle.engine, meetings_dir)
    await service.analyze_meeting(meeting_id)
    rows = _rows(handle.engine, "SELECT verified, start_ms FROM meeting_moments")
    assert rows == [{"verified": 0, "start_ms": 5200}]
    handle.engine.dispose()


async def test_moment_citing_a_missing_turn_is_dropped(tmp_path: Path) -> None:
    resp = _chunk_json(
        [],
        [
            {
                "speaker": "Bob",
                "turnStartIdx": 500,
                "eventType": "praise",
                "valence": "good",
                "severity": "low",
                "quote": "nice work",
            }
        ],
    )
    handle, meetings_dir, _, _, service = _make_service(tmp_path, lambda n, o: resp)
    meeting_id = _seed(handle.engine, meetings_dir)
    await service.analyze_meeting(meeting_id)
    assert _rows(handle.engine, "SELECT id FROM meeting_moments") == []
    handle.engine.dispose()


async def test_multi_chunk_dimensions_are_talk_weighted_and_each_chunk_is_called(tmp_path: Path) -> None:
    segs = [
        {
            "startMs": 0,
            "endMs": 3000,
            "speaker": "Alice",
            "text": "Alice opens with a long first remark here.",
        },
        {"startMs": 3000, "endMs": 4000, "speaker": "Bob", "text": "Bob replies with a medium-length reply."},
        {"startMs": 5000, "endMs": 6000, "speaker": "Alice", "text": "Alice follows up with another remark."},
        {"startMs": 6000, "endMs": 7000, "speaker": "Bob", "text": "Bob closes out with a final remark."},
    ]
    cost = max(len(format_turn(i, TranscriptSegment.model_validate(s))) + 1 for i, s in enumerate(segs))

    def responder(n: int, _o: dict[str, Any]) -> str:
        level = "low" if n == 1 else "high"
        return _chunk_json([{"speaker": "Alice", "dimensions": {"hostility": level}}])

    handle, meetings_dir, config_dir, llm, service = _make_service(tmp_path, responder)
    (Path(config_dir) / "insights").mkdir(parents=True, exist_ok=True)
    (Path(config_dir) / "insights" / "config.json").write_text(
        json.dumps({"chunkChars": 2 * cost + 2, "chunkOverlapTurns": 0})
    )
    meeting_id = _seed(handle.engine, meetings_dir, segs)
    await service.analyze_meeting(meeting_id)

    assert len(llm.calls) == 2
    assert "[2]" in llm.calls[1]["prompt"] and "[0]" not in llm.calls[1]["prompt"]  # global turn indices
    scale = DEFAULT_CONFIG["dimensionScale"]
    expected = (3000 * scale["low"] + 1000 * scale["high"]) / 4000
    alice = _rows(handle.engine, "SELECT dimensions FROM meeting_speaker_signals WHERE speaker_label='Alice'")
    assert json.loads(alice[0]["dimensions"])["hostility"] == pytest.approx(expected)
    handle.engine.dispose()


async def test_reanalyze_replaces_rows_and_skips_unless_forced(tmp_path: Path) -> None:
    handle, meetings_dir, config_dir, llm, service = _make_service(tmp_path, lambda n, o: _main_response())
    meeting_id = _seed(handle.engine, meetings_dir)

    first = await service.analyze_meeting(meeting_id)
    first_ids = {r["id"] for r in _rows(handle.engine, "SELECT id FROM meeting_speaker_signals")}
    assert first["status"] == "done" and len(llm.calls) == 1

    skipped = await service.analyze_meeting(meeting_id)
    assert skipped["status"] == "skipped" and len(llm.calls) == 1  # same transcript + prompt

    forced = await service.analyze_meeting(meeting_id, force=True)
    assert forced["status"] == "done" and len(llm.calls) == 2
    second_ids = {r["id"] for r in _rows(handle.engine, "SELECT id FROM meeting_speaker_signals")}
    assert len(second_ids) == len(first_ids) == 3 and second_ids.isdisjoint(first_ids)
    assert len(_rows(handle.engine, "SELECT id FROM meeting_moments")) == 1
    arts = _rows(handle.engine, "SELECT path FROM artifacts WHERE kind = 'signals'")
    assert len(arts) == 1  # old signals artifact replaced, not accumulated

    # A prompt edit changes the idempotency key and re-runs without force.
    (Path(config_dir) / "insights").mkdir(parents=True, exist_ok=True)
    (Path(config_dir) / "insights" / "prompts.json").write_text(json.dumps({"systemPrompt": "New lens."}))
    rerun = await service.analyze_meeting(meeting_id)
    assert rerun["status"] == "done" and len(llm.calls) == 3

    # So does a changed transcript.
    tpath = next(Path(meetings_dir).rglob("transcript.json"))
    data = json.loads(tpath.read_text())
    data["segments"][3]["text"] = "Okay, moving on to the next item."
    tpath.write_text(json.dumps(data))
    changed = await service.analyze_meeting(meeting_id)
    assert changed["status"] == "done" and len(llm.calls) == 4
    handle.engine.dispose()


async def test_invalid_json_gets_one_repair_then_fails_stage_run(tmp_path: Path) -> None:
    handle, meetings_dir, _, llm, service = _make_service(tmp_path, lambda n, o: "I cannot do that.")
    meeting_id = _seed(handle.engine, meetings_dir)
    with pytest.raises(InsightsExtractionError):
        await service.analyze_meeting(meeting_id)
    assert len(llm.calls) == 2
    assert "rejected" in llm.calls[1]["prompt"].lower() or "invalid" in llm.calls[1]["prompt"].lower()
    run = _rows(handle.engine, "SELECT status, last_error FROM stage_runs WHERE stage = 'insights_analyze'")
    assert run[0]["status"] == "error" and run[0]["last_error"]
    assert _rows(handle.engine, "SELECT id FROM meeting_speaker_signals") == []
    assert _rows(handle.engine, "SELECT id FROM artifacts WHERE kind = 'signals'") == []
    handle.engine.dispose()


async def test_failed_reanalysis_keeps_previous_rows(tmp_path: Path) -> None:
    answers = {"mode": "good"}

    def responder(n: int, _o: dict[str, Any]) -> str:
        return _main_response() if answers["mode"] == "good" else "nope"

    handle, meetings_dir, _, _, service = _make_service(tmp_path, responder)
    meeting_id = _seed(handle.engine, meetings_dir)
    await service.analyze_meeting(meeting_id)
    answers["mode"] = "bad"
    with pytest.raises(InsightsExtractionError):
        await service.analyze_meeting(meeting_id, force=True)
    assert len(_rows(handle.engine, "SELECT id FROM meeting_speaker_signals")) == 3
    assert len(_rows(handle.engine, "SELECT id FROM meeting_moments")) == 1
    handle.engine.dispose()


async def test_repair_success_after_one_bad_reply(tmp_path: Path) -> None:
    def responder(n: int, _o: dict[str, Any]) -> str:
        return "```\nnot valid\n```" if n == 1 else _main_response()

    handle, meetings_dir, _, llm, service = _make_service(tmp_path, responder)
    meeting_id = _seed(handle.engine, meetings_dir)
    result = await service.analyze_meeting(meeting_id)
    assert result["status"] == "done" and len(llm.calls) == 2
    handle.engine.dispose()


async def test_missing_meeting_and_missing_transcript_raise(tmp_path: Path) -> None:
    handle, meetings_dir, _, _, service = _make_service(tmp_path, lambda n, o: _main_response())
    with pytest.raises(ValueError, match="Meeting not found"):
        await service.analyze_meeting("nope")
    meeting_id = _seed(handle.engine, meetings_dir)
    with handle.engine.begin() as conn:
        conn.execute(text("UPDATE meetings SET primary_transcript_artifact_id = NULL"))
        conn.execute(text("DELETE FROM artifacts WHERE meeting_id = :m"), {"m": meeting_id})
    with pytest.raises(ValueError, match="No transcript found"):
        await service.analyze_meeting(meeting_id)
    handle.engine.dispose()


# --------------------------------------------------------------------------- enqueue seam (P1b)


async def test_enqueue_analyze_respects_enabled_flag_and_hook(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OLIVE_INSIGHTS_ENABLED", raising=False)
    handle, _, config_dir, _, service = _make_service(tmp_path, lambda n, o: _main_response())
    queued: list[str] = []

    async def hook(meeting_id: str) -> bool:
        queued.append(meeting_id)
        return True

    insights_extract.set_enqueue_hook(hook)
    try:
        assert await service.enqueue_analyze("m1") is False  # insights disabled by default
        assert queued == []

        (Path(config_dir) / "settings.json").write_text(json.dumps({"insightsEnabled": True}))
        assert await service.enqueue_analyze("m1") is True
        assert queued == ["m1"]

        insights_extract.set_enqueue_hook(None)
        assert await service.enqueue_analyze("m2") is False  # no queue wired yet (P1b)
    finally:
        insights_extract.set_enqueue_hook(None)
    handle.engine.dispose()


def test_parse_segments_accepts_all_stored_shapes() -> None:
    """Regression: extraction must handle the bare-list transcript shape (the
    majority of the real corpus), not only the Transcript-object shape."""
    import json as _json

    import pytest as _pytest

    from olive.services.insights.extract import InsightsService

    seg = {"startMs": 0, "endMs": 1000, "speaker": "Phil", "text": "Hello there.", "words": []}
    # Bare list of segments (was previously rejected).
    out_list = InsightsService._parse_segments(_json.dumps([seg]), "json")
    assert len(out_list) == 1 and out_list[0].speaker == "Phil"
    # Transcript object.
    out_obj = InsightsService._parse_segments(_json.dumps({"segments": [seg]}), "json")
    assert len(out_obj) == 1 and out_obj[0].text == "Hello there."
    # Non-JSON / unusable input still raises ValueError (mapped to a clear error).
    with _pytest.raises(ValueError):
        InsightsService._parse_segments("not json at all", "json")
    with _pytest.raises(ValueError):
        InsightsService._parse_segments(_json.dumps([]), "json")
