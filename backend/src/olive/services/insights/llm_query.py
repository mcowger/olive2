"""LLM query layer over extracted Insights: event retrieval, per-person playbooks, coaching.

These are *queries over already-extracted signals* (``meeting_moments``), never a re-scan of
every transcript. Retrieval is three tiers so the expensive model only sees a handful of rows:
the LLM turns the question into a structured filter, SQL selects and orders candidates, and the
LLM re-ranks just those candidates with a little transcript context. Playbooks and coaching are
synthesized by the LLM from evidence the code gathers, then cached in ``sync_state`` under a
data+prompt signature so they are recomputed only when something they depend on changes.

Division of labor matches extraction: the LLM supplies judgment (filters, ranking, advice);
code supplies everything checkable (ids, timestamps, names, sample sizes, and verbatim-quote
verification of every cited example). All prompts are defined here as local defaults and can be
overridden per key in ``<configDir>/insights/prompts.json``; tuning numbers can be overridden in
``insights/config.json``.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import ValidationError, field_validator
from sqlalchemy import text

from ...layout import meeting_paths
from ...logging_db import logger
from ...schemas import CamelModel, TranscriptSegment
from ..census import load_transcript, recording_duration, resolve_transcript_row
from .chunking import format_turn
from .extract import (
    InsightsService,
    _call_llm_generate,
    enforce_egress,
    normalize_text,
)
from .identity import resolve_labels
from .prompts import load_insights_config, load_insights_prompts
from .validate import InsightsValidationError, parse_llm_json

# ---------------------------------------------------------------------------
# Local prompt defaults (user-overridable via insights/prompts.json) and tuning defaults
# ---------------------------------------------------------------------------

QUERY_SYSTEM_PROMPT = """\
You are a careful analyst helping a person look back over their own recorded meetings. You work \
only from the evidence you are shown, you never invent quotes, people, or events, and you do not \
diagnose anyone or infer mental-health conditions. You reply with a single JSON object and \
nothing else."""

QUERY_FILTER_PROMPT = """\
Today's date is {{today}}.

A person is looking for a moment in their recorded meetings. Turn their request into a search \
filter over a table of extracted moments. Each moment has an open-vocabulary event label chosen \
by an earlier model (for example accusation, termination, threat, apology, praise, repair), a \
valence (good, bad, neutral, ambiguous), a speaker, a person it was directed toward, a verbatim \
quote, and a one-line rationale.

Request: {{query}}

Reply with exactly one JSON object shaped like this, and nothing else. Leave a field empty or \
null when the request does not say:
{"eventTypes": ["short event labels the moment might carry, including close synonyms"], \
"valence": "good|bad|neutral|ambiguous|null", \
"towardMe": true when the event was directed at the person asking (words like "me", "my", "I got"), else false, \
"dateRange": {"start": "YYYY-MM-DD or null", "end": "YYYY-MM-DD or null"} or null, \
"keywords": ["distinctive words likely to appear in the quote itself"]}"""

QUERY_RERANK_PROMPT = """\
A person asked: {{query}}

Below are candidate moments from their recorded meetings, each with the quote, who said it to \
whom, the earlier model's label and rationale, and a little surrounding transcript. Rank the \
candidates by how well each matches the request, best first, and keep at most {{limit}}.

Distinguish an event that was ENACTED in the meeting (it really happened there, for example \
someone actually accusing, firing, or praising) from one that was merely DISCUSSED, quoted, \
hypothesized, or reported secondhand. Set "enacted" accordingly, and rank enacted events above \
discussed ones unless the request is about discussion. Judge only from the text shown.

CANDIDATES
{{candidates}}
END CANDIDATES

Reply with exactly one JSON object shaped like this, and nothing else. Use the candidate ids \
exactly as given and omit candidates that do not match at all:
{"results": [{"candidateId": "c1", "score": 0.0, "enacted": true, "reason": "one sentence"}]}"""

PLAYBOOK_SYSTEM_PROMPT = """\
You are a careful communication analyst. You compare how conversations with one person went \
well versus badly, using real quotes as evidence, and you describe observable behavior only: \
you never diagnose, never infer a mental-health condition, and never judge character. Every \
recommendation must cite verbatim quotes copied exactly from the evidence you are given. You \
reply with a single JSON object and nothing else."""

PLAYBOOK_PROMPT = """\
{{frameworks}}

The evidence below is about conversations with one person ({{person}}). Part of it is moments \
that were flagged as notable and went well or badly. The rest is ordinary exchanges that were \
not flagged. Contrast them. Work out what the reader does or says that tends to land well with \
this person, and what tends to backfire, such as directness versus context, framing and word \
choices, tone and assertiveness, and timing or format. Only claim a pattern the evidence \
supports; if the sample is thin, say so in the summary. Where a pattern has a clear \
counter-example in the evidence, include it.

{{evidence}}

Reply with exactly one JSON object shaped like this, and nothing else. Every item needs at \
least one evidence quote copied exactly from above:
{"summary": "two sentences, including how much evidence there is", \
"doMore": [{"point": "short actionable advice", "evidence": [{"quote": "verbatim"}], \
"counterExample": {"quote": "verbatim"} or null}], \
"easeOff": [{"point": "short actionable advice", "evidence": [{"quote": "verbatim"}], \
"counterExample": {"quote": "verbatim"} or null}]}"""

COACHING_SYSTEM_PROMPT = """\
You are a supportive, evidence-first communication coach. You describe what the transcripts \
show and what might help, in a constructive and non-judgmental tone. Your observations are \
model-inferred reads of short excerpts, not facts about a person: you never diagnose, never use \
mental-health labels, and never assess character. Every point cites verbatim quotes copied \
exactly from the evidence. You reply with a single JSON object and nothing else."""

COACHING_PROMPT = """\
{{frameworks}}

{{scope}}

Below are the most notable moments, those that went well and those that went badly. Write \
coaching for the reader, addressed as "you" where the reader was a participant: what is working \
and worth keeping, and what could be tried differently, each tied to quotes. Lead with strengths. \
Phrase growth areas as options to try, not faults. Stay within what the quotes support.

{{evidence}}

Reply with exactly one JSON object shaped like this, and nothing else:
{"summary": "two or three constructive sentences", \
"strengths": [{"point": "what is working", "evidence": [{"quote": "verbatim"}]}], \
"growthAreas": [{"point": "what could be tried differently", "suggestion": "a concrete alternative", \
"evidence": [{"quote": "verbatim"}]}]}"""

DEFAULT_QUERY_PROMPTS: dict[str, str] = {
    "queryFilterPrompt": QUERY_FILTER_PROMPT,
    "queryRerankPrompt": QUERY_RERANK_PROMPT,
    "querySystemPrompt": QUERY_SYSTEM_PROMPT,
    "playbookPrompt": PLAYBOOK_PROMPT,
    "playbookSystemPrompt": PLAYBOOK_SYSTEM_PROMPT,
    "coachingPrompt": COACHING_PROMPT,
    "coachingSystemPrompt": COACHING_SYSTEM_PROMPT,
}

# Integer tuning knobs; ``insights/config.json`` may override any of them.
DEFAULT_QUERY_CONFIG: dict[str, int] = {
    "retrievalCandidates": 25,  # rows SQL hands to the re-ranker
    "retrievalContextTurns": 2,  # turns of transcript shown either side of a candidate
    "retrievalContextChars": 900,  # cap on context text per candidate
    "playbookMoments": 12,  # good and bad moments each, per dyad
    "playbookWindows": 4,  # ordinary transcript windows sampled per playbook
    "playbookWindowTurns": 3,  # turns per sampled window
    "coachingMoments": 12,  # good and bad moments each
}

PLAYBOOK_KEY_PREFIX = "insights.playbook:"
COACHING_KEY_PREFIX = "insights.coaching:"
# Hard bounds on caller/LLM-controlled sizes: the result count, and how many LLM-emitted
# event types / keywords reach the SQL (each becomes a bound variable and a LIKE clause).
MAX_FIND_EVENTS_LIMIT = 50
MAX_FILTER_TERMS = 32

T = TypeVar("T", bound=CamelModel)


# ---------------------------------------------------------------------------
# LLM-facing schemas
# ---------------------------------------------------------------------------


class DateRange(CamelModel):
    start: str | None = None
    end: str | None = None


class EventFilter(CamelModel):
    """Structured filter the model derives from a natural-language request."""

    eventTypes: list[str] = []
    valence: Literal["good", "bad", "neutral", "ambiguous"] | None = None
    towardMe: bool = False
    dateRange: DateRange | None = None
    keywords: list[str] = []

    @field_validator("valence", mode="before")
    @classmethod
    def _normalize_valence(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().lower()
            return None if value in {"", "null", "none", "any"} else value
        return value

    @field_validator("eventTypes", "keywords", mode="before")
    @classmethod
    def _as_list(cls, value: Any) -> Any:
        if value is None:
            return []
        return [value] if isinstance(value, str) else value


class RankedCandidate(CamelModel):
    candidateId: str
    score: float = 0.0
    enacted: bool = True
    reason: str = ""


class RerankResult(CamelModel):
    results: list[RankedCandidate] = []


class EvidenceIn(CamelModel):
    quote: str

    @classmethod
    def _coerce(cls, value: Any) -> Any:
        return {"quote": value} if isinstance(value, str) else value


def _evidence_list(value: Any) -> Any:
    if value is None:
        return []
    return [EvidenceIn._coerce(v) for v in value] if isinstance(value, list) else value


class GuidanceItem(CamelModel):
    point: str
    suggestion: str | None = None
    evidence: list[EvidenceIn] = []
    counterExample: EvidenceIn | None = None

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence(cls, value: Any) -> Any:
        return _evidence_list(value)

    @field_validator("counterExample", mode="before")
    @classmethod
    def _counter(cls, value: Any) -> Any:
        return EvidenceIn._coerce(value)


class PlaybookOut(CamelModel):
    summary: str = ""
    doMore: list[GuidanceItem] = []
    easeOff: list[GuidanceItem] = []


class CoachingOut(CamelModel):
    summary: str = ""
    strengths: list[GuidanceItem] = []
    growthAreas: list[GuidanceItem] = []


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _date_to_ms(value: str | None, *, end: bool = False) -> int | None:
    """``YYYY-MM-DD`` to UTC epoch ms (an end date is inclusive); None when unparseable."""
    if not value:
        return None
    try:
        day = datetime.strptime(value.strip()[:10], "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None
    if end:
        day += timedelta(days=1)
    return int(day.timestamp() * 1000)


def _digest(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _render(template: str, **values: str) -> str:
    """Fill ``{{name}}`` placeholders; later values are substituted last so quoted text is inert."""
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def _clip(value: str, limit: int) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "…"


@dataclass
class _Piece:
    """One citable chunk of evidence; used to verify and attribute quotes the model cites."""

    meeting_id: str
    normalized: str
    start_ms: int | None = None


@dataclass
class _Evidence:
    text: str = ""
    pieces: list[_Piece] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class InsightsLlmService(InsightsService):
    """Retrieval, playbooks, and coaching over extracted moments.

    Shares ``InsightsService``'s construction, egress allowlist, and route resolution so the
    same fail-closed privacy rules apply. ``self_speaker_id`` (or ``selfSpeakerId`` in
    ``insights/config.json``) names the canonical speaker that "me" refers to in requests.
    """

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        db: Any = None,
        meetings_dir: str | None = None,
        config_dir: str | None = None,
        llm_service: Any | None = None,
        allowed_hosts: Sequence[str] | None = None,
        self_speaker_id: str | None = None,
    ) -> None:
        super().__init__(
            options,
            db=db,
            meetings_dir=meetings_dir,
            config_dir=config_dir,
            llm_service=llm_service,
            allowed_hosts=allowed_hosts,
        )
        opts = options or {}
        self._self_speaker_id = self_speaker_id or opts.get("selfSpeakerId") or opts.get("self_speaker_id")

    # -- configuration -------------------------------------------------------

    def _query_prompts(self) -> dict[str, str]:
        resolved = load_insights_prompts(self.config_dir)
        out = dict(DEFAULT_QUERY_PROMPTS)
        out["frameworks"] = resolved.get("frameworks", "")
        for key in DEFAULT_QUERY_PROMPTS:
            value = resolved.get(key)
            if isinstance(value, str) and value.strip():
                out[key] = value
        return out

    def _query_config(self) -> tuple[dict[str, Any], dict[str, int]]:
        config = load_insights_config(self.config_dir)
        knobs: dict[str, int] = {}
        for key, default in DEFAULT_QUERY_CONFIG.items():
            value = config.get(key, default)
            valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
            if not valid:
                logger.warn("Invalid insights config value, using default", {"key": key})
            knobs[key] = value if valid else default
        return config, knobs

    def _self_id(self, config: Mapping[str, Any]) -> str | None:
        value = self._self_speaker_id or config.get("selfSpeakerId")
        return str(value) if value else None

    # -- LLM plumbing ----------------------------------------------------------

    async def _open_session(self, config: Mapping[str, Any]) -> Callable[[str, str], Awaitable[str]]:
        """Resolve the insights route and enforce egress; returns ``ask(system, prompt)``.

        The guard runs here, before any request is made, and fails closed.
        """
        app_cfg = self._app_config()
        provider, model, base_url = await self._resolve_route(
            app_cfg.insightsProvider.strip(), app_cfg.insightsModel.strip()
        )
        enforce_egress(base_url, self._allowed_hosts(config))

        async def ask(system: str, prompt: str) -> str:
            res = await _call_llm_generate(
                self.llm_service,
                {"provider": provider, "model": model, "systemPrompt": system, "prompt": prompt},
            )
            return res.text

        return ask

    @staticmethod
    async def _ask_validated(
        ask: Callable[[str, str], Awaitable[str]], system: str, prompt: str, model: type[T]
    ) -> T:
        """Ask once; on a parse/validation failure re-ask exactly once with the error appended."""

        def check(reply: str) -> T:
            data = parse_llm_json(reply)
            try:
                return model.model_validate(data)
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8]
                )
                raise InsightsValidationError(f"JSON does not match the expected shape ({problems})") from exc

        first = await ask(system, prompt)
        try:
            return check(first)
        except InsightsValidationError as exc:
            error = str(exc)
        repair = (
            f"{prompt}\n\n"
            "Your previous reply was rejected and could not be used.\n"
            f"Problem: {error}\n"
            f"Previous reply (truncated):\n{first[:1500]}\n\n"
            "Reply again with ONLY the corrected JSON object, with no commentary."
        )
        second = await ask(system, repair)
        try:
            return check(second)
        except InsightsValidationError as exc:
            raise InsightsValidationError(f"{exc} (after one repair attempt; first error: {error})") from exc

    # -- transcript reading ------------------------------------------------------

    def _load_segments(self, meeting_id: str) -> list[TranscriptSegment] | None:
        """Primary transcript segments for a meeting, or None when absent or unreadable."""
        with self.db.connect() as conn:
            meeting = (
                conn.execute(
                    text(
                        "SELECT id, title, start_time, primary_transcript_artifact_id "
                        "FROM meetings WHERE id = :id"
                    ),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not meeting:
                return None
            row = resolve_transcript_row(conn, dict(meeting))
            duration = recording_duration(conn, meeting_id)
        if not row:
            return None
        folder = meeting_paths(
            self.meetings_dir, int(meeting["start_time"]), str(meeting["title"]), meeting_id
        ).folder
        loaded = load_transcript(Path(folder) / str(row["path"]), duration)
        if isinstance(loaded, str):
            return None
        return list(loaded[1].segments)

    # -- moment queries ------------------------------------------------------------

    _MOMENT_SELECT = """
        SELECT mm.id, mm.meeting_id, mm.speaker_id, mm.toward_speaker_id, mm.start_ms, mm.end_ms,
               mm.turn_start_idx, mm.turn_end_idx, mm.event_type, mm.valence, mm.severity,
               mm.quote, mm.rationale,
               COALESCE(s.name, mm.speaker_label, mm.speaker_id) AS speaker_name,
               COALESCE(t.name, mm.toward_label, mm.toward_speaker_id) AS toward_name,
               m.title AS meeting_title, m.start_time AS meeting_start
        FROM meeting_moments mm
        JOIN meetings m ON m.id = mm.meeting_id
        LEFT JOIN speakers s ON s.id = mm.speaker_id
        LEFT JOIN speakers t ON t.id = mm.toward_speaker_id
    """
    _ORDER = " ORDER BY COALESCE(mm.severity, 0) DESC, m.start_time DESC, mm.start_ms ASC"

    def _query_moments(
        self, where: str, params: dict[str, Any], limit: int, include_unverified: bool = False
    ) -> list[dict[str, Any]]:
        """Moments matching ``where``; ungrounded (``verified = 0``) ones only when asked for."""
        if not include_unverified:
            where = f"{where} AND mm.verified = 1" if where.strip() else "WHERE mm.verified = 1"
        sql = f"{self._MOMENT_SELECT} {where} {self._ORDER} LIMIT :limit"
        with self.db.connect() as conn:
            return [dict(r) for r in conn.execute(text(sql), {**params, "limit": limit}).mappings().all()]

    def _select_candidates(
        self, flt: EventFilter, self_id: str | None, limit: int, include_unverified: bool = False
    ) -> list[dict[str, Any]]:
        """Tier 2: SQL candidates for ``flt`` by severity; most severe overall when none match."""
        clauses: list[str] = []
        params: dict[str, Any] = {}

        terms: list[str] = []
        labels = [t.strip().lower() for t in flt.eventTypes if t.strip()][:MAX_FILTER_TERMS]
        for i, label in enumerate(labels):
            params[f"et{i}"] = f"%{_like_escape(label)}%"
            terms.append(f"LOWER(mm.event_type) LIKE :et{i} ESCAPE '\\'")
        words = [k.strip().lower() for k in flt.keywords if k.strip()][:MAX_FILTER_TERMS]
        for i, word in enumerate(words):
            params[f"kw{i}"] = f"%{_like_escape(word)}%"
            terms.append(
                f"(LOWER(mm.quote) LIKE :kw{i} ESCAPE '\\' "
                f"OR LOWER(COALESCE(mm.rationale, '')) LIKE :kw{i} ESCAPE '\\')"
            )
        if terms:
            clauses.append("(" + " OR ".join(terms) + ")")
        if flt.valence:
            clauses.append("mm.valence = :valence")
            params["valence"] = flt.valence
        if flt.towardMe and self_id:
            clauses.append("mm.toward_speaker_id = :self_id")
            params["self_id"] = self_id
        if flt.dateRange:
            start = _date_to_ms(flt.dateRange.start)
            end = _date_to_ms(flt.dateRange.end, end=True)
            if start is not None:
                clauses.append("m.start_time >= :d0")
                params["d0"] = start
            if end is not None:
                clauses.append("m.start_time < :d1")
                params["d1"] = end

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._query_moments(where, params, limit, include_unverified)
        return rows or self._query_moments("", {}, limit, include_unverified)

    def _context_for(
        self,
        row: Mapping[str, Any],
        cache: dict[str, list[TranscriptSegment] | None],
        turns: int,
        max_chars: int,
    ) -> str:
        """A few transcript turns around the moment; empty when no transcript is on disk."""
        meeting_id = str(row["meeting_id"])
        if meeting_id not in cache:
            cache[meeting_id] = self._load_segments(meeting_id)
        segments = cache[meeting_id]
        if not segments:
            return ""
        idx = row["turn_start_idx"]
        if idx is None or not 0 <= int(idx) < len(segments):
            idx = min(range(len(segments)), key=lambda i: abs(segments[i].startMs - int(row["start_ms"])))
        first = int(idx)
        last = int(row["turn_end_idx"]) if row["turn_end_idx"] is not None else first
        last = min(max(last, first), len(segments) - 1)
        lines = [
            format_turn(i, segments[i])
            for i in range(max(0, first - turns), min(len(segments), last + turns + 1))
        ]
        return _clip("\n".join(lines), max_chars)

    # -- find_events -------------------------------------------------------------

    async def find_events(
        self, query: str, limit: int = 5, include_unverified: bool = False
    ) -> list[dict[str, Any]]:
        """Find the moments matching a natural-language description, best first.

        Returns ``{meetingId, startMs, quote, eventType, toward, score, reason, enacted, ...}``
        rows. Only the candidate moments (and a little context around them) reach the model.
        ``limit`` is clamped to ``MAX_FIND_EVENTS_LIMIT``; ungrounded moments are excluded unless
        ``include_unverified``.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("query is required")
        limit = min(max(1, int(limit)), MAX_FIND_EVENTS_LIMIT)
        prompts = self._query_prompts()
        config, knobs = self._query_config()
        ask = await self._open_session(config)
        system = prompts["querySystemPrompt"]

        flt = await self._ask_validated(
            ask,
            system,
            _render(
                prompts["queryFilterPrompt"],
                today=datetime.now(UTC).strftime("%Y-%m-%d"),
                query=query,
            ),
            EventFilter,
        )

        candidates = self._select_candidates(
            flt,
            self._self_id(config),
            min(max(knobs["retrievalCandidates"], limit), MAX_FIND_EVENTS_LIMIT * 4),
            include_unverified,
        )
        if not candidates:
            return []

        cache: dict[str, list[TranscriptSegment] | None] = {}
        blocks: list[str] = []
        by_id: dict[str, dict[str, Any]] = {}
        for n, row in enumerate(candidates, start=1):
            cid = f"c{n}"
            by_id[cid] = row
            context = self._context_for(
                row, cache, knobs["retrievalContextTurns"], knobs["retrievalContextChars"]
            )
            block = (
                f'[{cid}] meeting "{row["meeting_title"]}" | {row["speaker_name"] or "Unknown"} -> '
                f"{row['toward_name'] or 'unclear'} | label: {row['event_type']} ({row['valence']})\n"
                f'quote: "{row["quote"]}"\n'
                f"rationale: {row['rationale'] or 'n/a'}"
            )
            if context:
                block += f"\ncontext:\n{context}"
            blocks.append(block)

        ranked = await self._ask_validated(
            ask,
            system,
            _render(
                prompts["queryRerankPrompt"],
                limit=str(limit),
                query=query,
                candidates="\n\n".join(blocks),
            ),
            RerankResult,
        )

        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in sorted(ranked.results, key=lambda r: r.score, reverse=True):
            row = by_id.get(item.candidateId.strip())
            if row is None or item.candidateId in seen:
                continue  # the model may not invent or repeat candidates
            seen.add(item.candidateId)
            out.append(
                {
                    "momentId": row["id"],
                    "meetingId": row["meeting_id"],
                    "meetingTitle": row["meeting_title"],
                    "startMs": int(row["start_ms"]),
                    "endMs": row["end_ms"],
                    "quote": row["quote"],
                    "eventType": row["event_type"],
                    "valence": row["valence"],
                    "speaker": row["speaker_name"],
                    "speakerId": row["speaker_id"],
                    "toward": row["toward_name"],
                    "towardSpeakerId": row["toward_speaker_id"],
                    "score": item.score,
                    "enacted": item.enacted,
                    "reason": item.reason,
                }
            )
            if len(out) >= limit:
                break
        return out

    # -- cache -------------------------------------------------------------------

    def _cache_get(self, key: str, signature: str) -> dict[str, Any] | None:
        with self.db.connect() as conn:
            raw = conn.execute(text("SELECT value FROM sync_state WHERE key = :k"), {"k": key}).scalar()
        if not raw:
            return None
        try:
            stored = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(stored, dict) or stored.get("signature") != signature:
            return None
        result = stored.get("result")
        return result if isinstance(result, dict) else None

    def _cache_put(self, key: str, signature: str, result: dict[str, Any]) -> None:
        value = json.dumps({"signature": signature, "result": result}, ensure_ascii=False)
        with self.db.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO sync_state (key, value) VALUES (:k, :v)
                    ON CONFLICT(key) DO UPDATE SET value = :v
                    """
                ),
                {"k": key, "v": value},
            )

    # -- evidence rendering + verification ----------------------------------------

    @staticmethod
    def _moment_lines(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[_Piece]]:
        lines: list[str] = []
        pieces: list[_Piece] = []
        for row in rows:
            who = f"{row['speaker_name'] or 'Unknown'} -> {row['toward_name'] or 'unclear'}"
            line = f'- [{row["meeting_title"]}] {who}, {row["event_type"]}: "{row["quote"]}"'
            if row["rationale"]:
                line += f" ({row['rationale']})"
            lines.append(line)
            pieces.append(
                _Piece(str(row["meeting_id"]), normalize_text(str(row["quote"])), int(row["start_ms"]))
            )
        return lines, pieces

    @staticmethod
    def _verified_items(items: Sequence[GuidanceItem], pieces: Sequence[_Piece]) -> list[dict[str, Any]]:
        """Keep only items whose cited quotes appear verbatim in the supplied evidence."""

        def ground(quote: str) -> dict[str, Any] | None:
            needle = normalize_text(quote)
            if not needle:
                return None
            for piece in pieces:
                if needle in piece.normalized:
                    found: dict[str, Any] = {"quote": quote.strip(), "meetingId": piece.meeting_id}
                    if piece.start_ms is not None:
                        found["startMs"] = piece.start_ms
                    return found
            return None

        out: list[dict[str, Any]] = []
        for item in items:
            evidence = [g for g in (ground(e.quote) for e in item.evidence) if g]
            if not evidence:
                continue  # never a bare recommendation without a real example
            entry: dict[str, Any] = {"point": item.point.strip(), "evidence": evidence}
            if item.suggestion:
                entry["suggestion"] = item.suggestion.strip()
            counter = ground(item.counterExample.quote) if item.counterExample else None
            if counter:
                entry["counterExample"] = counter
            out.append(entry)
        return out

    # -- playbook ------------------------------------------------------------------

    def _dyad_moments(
        self, speaker_id: str, per_side: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        where = "WHERE (mm.speaker_id = :sid OR mm.toward_speaker_id = :sid) AND mm.valence = :v"
        good = self._query_moments(where, {"sid": speaker_id, "v": "good"}, per_side)
        bad = self._query_moments(where, {"sid": speaker_id, "v": "bad"}, per_side)
        return good, bad

    def _dyad_meetings(self, speaker_id: str) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT m.id AS id, m.title AS title, m.primary_transcript_artifact_id AS transcript_id
                    FROM meetings m
                    WHERE m.id IN (SELECT meeting_id FROM meeting_speakers WHERE speaker_id = :sid)
                       OR m.id IN (SELECT meeting_id FROM meeting_moments
                                   WHERE speaker_id = :sid OR toward_speaker_id = :sid)
                    ORDER BY m.id
                    """
                ),
                {"sid": speaker_id},
            ).mappings()
            return [dict(r) for r in rows]

    def _sample_windows(
        self, speaker_id: str, meetings: Sequence[Mapping[str, Any]], count: int, size: int
    ) -> tuple[list[str], list[_Piece]]:
        """Random ordinary windows where the person speaks and nothing notable was flagged.

        Sampling is seeded by the speaker so a given dataset always yields the same windows.
        """
        if count <= 0 or size <= 0:
            return [], []
        pool: list[tuple[str, str, list[TranscriptSegment], int, int]] = []
        for meeting in meetings:
            mid = str(meeting["id"])
            segments = self._load_segments(mid)
            if not segments:
                continue
            labels = list(dict.fromkeys(s.speaker for s in segments))
            with self.db.connect() as conn:
                identities = resolve_labels(conn, mid, labels)
                notable = [
                    (int(r["start_ms"]), int(r["end_ms"] if r["end_ms"] is not None else r["start_ms"]))
                    for r in conn.execute(
                        text("SELECT start_ms, end_ms FROM meeting_moments WHERE meeting_id = :m"), {"m": mid}
                    ).mappings()
                ]
            theirs = {lbl for lbl, ident in identities.items() if ident.speaker_id == speaker_id}
            if not theirs:
                continue

            def is_notable(seg: TranscriptSegment, spans: list[tuple[int, int]] = notable) -> bool:
                return any(seg.startMs <= end and seg.endMs >= start for start, end in spans)

            seen: set[tuple[int, int]] = set()
            for i, seg in enumerate(segments):
                if seg.speaker not in theirs:
                    continue
                lo = max(0, i - size // 2)
                hi = min(len(segments), lo + size)
                if (lo, hi) in seen or any(is_notable(s) for s in segments[lo:hi]):
                    continue
                seen.add((lo, hi))
                pool.append((mid, str(meeting["title"]), segments, lo, hi))

        chosen = random.Random(f"playbook:{speaker_id}").sample(pool, min(count, len(pool)))
        blocks: list[str] = []
        pieces: list[_Piece] = []
        for n, (mid, title, segments, lo, hi) in enumerate(chosen, start=1):
            window = segments[lo:hi]
            blocks.append(
                f'--- window {n} (meeting "{title}") ---\n'
                + "\n".join(format_turn(i, segments[i]) for i in range(lo, hi))
            )
            pieces.extend(_Piece(mid, normalize_text(s.text), s.startMs) for s in window)
        return blocks, pieces

    async def playbook(self, speaker_id: str) -> dict[str, Any]:
        """ "Do more of / ease off on" guide for communicating with ``speaker_id``.

        Re-reads at synthesis time: the dyad's good and bad moments are contrasted with a small
        random sample of ordinary windows. Cached per speaker until the data or prompts change.
        """
        if not speaker_id:
            raise ValueError("speaker_id is required")
        prompts = self._query_prompts()
        config, knobs = self._query_config()
        good, bad = self._dyad_moments(speaker_id, knobs["playbookMoments"])
        meetings = self._dyad_meetings(speaker_id)
        base: dict[str, Any] = {"speakerId": speaker_id}

        if not good and not bad and not meetings:
            return {**base, "insufficientData": True, "cached": False, "doMore": [], "easeOff": []}

        signature = _digest(
            [(r["id"], r["quote"], r["valence"], r["severity"]) for r in good + bad],
            [(m["id"], m["transcript_id"]) for m in meetings],
            {k: knobs[k] for k in ("playbookWindows", "playbookWindowTurns")},
            [prompts[k] for k in ("playbookSystemPrompt", "playbookPrompt", "frameworks")],
            self._app_config().insightsModel,
        )
        key = f"{PLAYBOOK_KEY_PREFIX}{speaker_id}"
        cached = self._cache_get(key, signature)
        if cached is not None:
            return {**cached, "cached": True}

        ask = await self._open_session(config)
        good_lines, good_pieces = self._moment_lines(good)
        bad_lines, bad_pieces = self._moment_lines(bad)
        window_blocks, window_pieces = self._sample_windows(
            speaker_id, meetings, knobs["playbookWindows"], knobs["playbookWindowTurns"]
        )
        evidence = "\n\n".join(
            [
                "MOMENTS THAT WENT WELL\n" + ("\n".join(good_lines) or "(none)"),
                "MOMENTS THAT WENT BADLY\n" + ("\n".join(bad_lines) or "(none)"),
                "ORDINARY EXCHANGES (not flagged as notable)\n" + ("\n\n".join(window_blocks) or "(none)"),
            ]
        )
        name = next(
            (r["speaker_name"] if r["speaker_id"] == speaker_id else r["toward_name"] for r in good + bad),
            speaker_id,
        )
        out = await self._ask_validated(
            ask,
            prompts["playbookSystemPrompt"],
            _render(
                prompts["playbookPrompt"],
                frameworks=prompts["frameworks"],
                person=str(name),
                evidence=evidence,
            ),
            PlaybookOut,
        )
        pieces = good_pieces + bad_pieces + window_pieces
        result = {
            **base,
            "person": name,
            "summary": out.summary.strip(),
            "doMore": self._verified_items(out.doMore, pieces),
            "easeOff": self._verified_items(out.easeOff, pieces),
            "sampleSize": {
                "goodMoments": len(good),
                "badMoments": len(bad),
                "ordinaryWindows": len(window_blocks),
                "meetings": len(meetings),
            },
            "generatedAt": int(time.time() * 1000),
        }
        self._cache_put(key, signature, result)
        return {**result, "cached": False}

    # -- coaching ------------------------------------------------------------------

    async def coaching(self, scope: str = "all", speaker_id: str | None = None) -> dict[str, Any]:
        """Constructive, evidence-first coaching from the top good and bad moments.

        ``scope='all'`` uses every moment; ``scope='speaker'`` uses moments involving
        ``speaker_id``. Cached per scope until the moments or prompts change.
        """
        if scope not in {"all", "speaker"}:
            raise ValueError("scope must be 'all' or 'speaker'")
        if scope == "speaker" and not speaker_id:
            raise ValueError("speaker_id is required for scope='speaker'")
        prompts = self._query_prompts()
        config, knobs = self._query_config()

        where = "WHERE mm.valence = :v"
        params: dict[str, Any] = {}
        if scope == "speaker":
            where += " AND (mm.speaker_id = :sid OR mm.toward_speaker_id = :sid)"
            params["sid"] = speaker_id
        good = self._query_moments(where, {**params, "v": "good"}, knobs["coachingMoments"])
        bad = self._query_moments(where, {**params, "v": "bad"}, knobs["coachingMoments"])
        key = f"{COACHING_KEY_PREFIX}{'all' if scope == 'all' else speaker_id}"
        base: dict[str, Any] = {"scope": scope, "speakerId": speaker_id if scope == "speaker" else None}

        if not good and not bad:
            return {**base, "insufficientData": True, "cached": False, "strengths": [], "growthAreas": []}

        signature = _digest(
            [(r["id"], r["quote"], r["valence"], r["severity"]) for r in good + bad],
            [prompts[k] for k in ("coachingSystemPrompt", "coachingPrompt", "frameworks")],
            self._app_config().insightsModel,
        )
        cached = self._cache_get(key, signature)
        if cached is not None:
            return {**cached, "cached": True}

        ask = await self._open_session(config)
        good_lines, good_pieces = self._moment_lines(good)
        bad_lines, bad_pieces = self._moment_lines(bad)
        evidence = "MOMENTS THAT WENT WELL\n" + ("\n".join(good_lines) or "(none)")
        evidence += "\n\nMOMENTS THAT WENT BADLY\n" + ("\n".join(bad_lines) or "(none)")
        scope_text = (
            "Scope: all of the reader's recorded meetings."
            if scope == "all"
            else "Scope: the reader's interactions with one person."
        )
        out = await self._ask_validated(
            ask,
            prompts["coachingSystemPrompt"],
            _render(
                prompts["coachingPrompt"],
                frameworks=prompts["frameworks"],
                scope=scope_text,
                evidence=evidence,
            ),
            CoachingOut,
        )
        pieces = good_pieces + bad_pieces
        result = {
            **base,
            "summary": out.summary.strip(),
            "strengths": self._verified_items(out.strengths, pieces),
            "growthAreas": self._verified_items(out.growthAreas, pieces),
            "sampleSize": {"goodMoments": len(good), "badMoments": len(bad)},
            "generatedAt": int(time.time() * 1000),
        }
        self._cache_put(key, signature, result)
        return {**result, "cached": False}


__all__ = [
    "DEFAULT_QUERY_CONFIG",
    "DEFAULT_QUERY_PROMPTS",
    "CoachingOut",
    "EventFilter",
    "MAX_FILTER_TERMS",
    "MAX_FIND_EVENTS_LIMIT",
    "InsightsLlmService",
    "PlaybookOut",
    "RerankResult",
]
