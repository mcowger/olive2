"""LLM speaker naming: map generic diarization labels to real names from how people are addressed.

Most stored transcripts carry placeholder labels ("Speaker 1"). Cross-meeting, per-person
analysis needs real people, but the audio does not have to be re-processed: the conversation
itself usually says who is who ("Thanks, Vineet", "I'm Dana"). :class:`SpeakerNamingService`
asks the LLM to read that evidence and propose a name per generic label, then verifies it in
code. Division of labor mirrors extraction: the LLM supplies judgment (a name, a confidence, a
verbatim quote, a turn number); code decides whether the proposal is grounded.

A proposal is stored as ``pending`` only when the cited turn exists, the quote is verbatim in
that turn, and the proposed name is actually spoken in the quote, so a name can never be
invented. Failing proposals are kept as ``ungrounded`` for audit and never offered. Nothing is
linked to a person until :meth:`SpeakerNamingService.apply_confirmations` is called with names a
human confirmed.

Transcripts are read through the no-ML reader in :mod:`olive.services.census`, so this module
(and the ``insights name`` CLI) never imports the pyannote/torch stack.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError, field_validator
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from ...db import DbHandle
from ...layout import meeting_paths
from ...logging_db import logger
from ...schemas import CamelModel, LlmGenerateResult, TranscriptSegment
from ...speaker_labels import is_generic_speaker_label
from ..census import _load_transcript, _recording_duration, _resolve_transcript_row
from .chunking import chunk_segments, format_turn
from .extract import (
    InsightsExtractionError,
    InsightsService,
    _call_llm_generate,
    enforce_egress,
    normalize_text,
)
from .identity import normalize_alias
from .prompts import load_insights_config, load_insights_prompts
from .validate import InsightsValidationError, parse_llm_json

NAMING_SYSTEM_PROMPT = """\
You are a careful reader of meeting transcripts. Your only job is to work out who the \
unidentified speakers are, using what the people in the conversation say to and about each \
other. You never guess a name from a voice, a role, or a topic. You only name a speaker when \
the words on the page support it, and every name you give is tied to a short verbatim quote \
copied exactly from a cited turn. When the transcript does not say who someone is, you say so \
by returning a null name. You reply with a single JSON object and nothing else."""

NAMING_PROMPT = """\
Meeting: {{title}}
Known people (use these exact spellings when one matches): {{candidates}}
Speaker labels to name: {{labels}}

Below is part of a diarized transcript. Each line is "[turn number] Speaker: words". Turn numbers \
are the identifiers you must cite.

TRANSCRIPT
{{transcript}}
END TRANSCRIPT

Task: for EACH speaker label listed above, decide whether the transcript itself reveals the real \
name of the person behind that label. Good evidence is another speaker addressing that person by \
name and that person replying, the person introducing themselves, or someone else introducing \
them. A name that is only mentioned in passing, or that belongs to a third party being talked \
about, is not evidence. Propose a name only when it is supported; otherwise give null. Prefer a \
spelling from the known people when one matches, and otherwise use the name exactly as it is \
spoken. The quote must be copied exactly from the cited turn and must contain the name.

Reply with exactly one JSON object shaped like this, with one entry per label, and nothing else:
{"proposals": [{"speaker": "Speaker 1", "proposedName": "Name or null", "confidence": 0.0, \
"evidenceQuote": "verbatim words containing the name", "evidenceTurnIdx": 0}]}"""

#: Keys read from the insights prompts config; the defaults apply until a key is overridden.
DEFAULT_NAMING_PROMPTS: dict[str, str] = {
    "namingSystemPrompt": NAMING_SYSTEM_PROMPT,
    "namingPrompt": NAMING_PROMPT,
}

#: Proposal lifecycle. ``pending`` awaits a human; ``none`` means the transcript did not support a
#: name; ``ungrounded`` means the model proposed one but the evidence failed verification.
STATUS_PENDING = "pending"
STATUS_NONE = "none"
STATUS_UNGROUNDED = "ungrounded"
STATUS_CONFIRMED = "confirmed"

_STATUS_RANK = {STATUS_PENDING: 2, STATUS_UNGROUNDED: 1, STATUS_NONE: 0}
_NULL_NAMES = {"", "null", "none", "unknown", "n/a", "na"}
_REPAIR_ECHO_LIMIT = 1500


class ProposedName(CamelModel):
    """The model's answer for one speaker label."""

    speaker: str
    proposedName: str | None = None
    confidence: float | None = None
    evidenceQuote: str | None = None
    evidenceTurnIdx: int | None = None

    @field_validator("proposedName", mode="before")
    @classmethod
    def _null_name(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("proposedName must be a string or null")
        stripped = " ".join(value.split())
        return None if stripped.casefold() in _NULL_NAMES else stripped

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return None
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return None

    @field_validator("evidenceTurnIdx", mode="before")
    @classmethod
    def _lenient_turn(cls, value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


class NameProposals(CamelModel):
    """The validated reply for one transcript chunk."""

    proposals: list[ProposedName] = []


def validate_proposals(raw: str) -> NameProposals:
    """Parse an LLM reply and validate it against :class:`NameProposals`."""
    data = parse_llm_json(raw)
    try:
        return NameProposals.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8])
        raise InsightsValidationError(f"JSON does not match the expected shape ({problems})") from exc


async def parse_proposals_with_repair(
    ask: Callable[[str], Awaitable[str]], prompt: str
) -> tuple[NameProposals, str]:
    """Ask once; on a parse/validation failure re-ask exactly once with the error appended.

    Same contract as :func:`olive.services.insights.validate.parse_with_repair`, which is bound to
    the extraction shape. A second failure raises :class:`InsightsValidationError`.
    """
    first = await ask(prompt)
    try:
        return validate_proposals(first), first
    except InsightsValidationError as exc:
        error = str(exc)

    repair_prompt = (
        f"{prompt}\n\n"
        "Your previous reply was rejected and could not be used.\n"
        f"Problem: {error}\n"
        f"Previous reply (truncated):\n{first[:_REPAIR_ECHO_LIMIT]}\n\n"
        "Reply again with ONLY the corrected JSON object, with no commentary."
    )
    second = await ask(repair_prompt)
    try:
        return validate_proposals(second), second
    except InsightsValidationError as exc:
        raise InsightsValidationError(f"{exc} (after one repair attempt; first error: {error})") from exc


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def _contains_phrase(haystack: str, phrase: str) -> bool:
    """Whole-word containment of ``phrase`` in ``haystack``, both already normalized."""
    return bool(phrase) and f" {phrase} " in f" {haystack} "


def ground_name(
    item: ProposedName,
    segments: Sequence[TranscriptSegment],
    known: Mapping[str, str],
) -> tuple[str | None, str | None, bool]:
    """Verify one proposal; return ``(canonical_name, evidence_quote, grounded)``.

    Grounded means: the cited turn exists, the normalized quote appears in that turn's text,
    and the proposed name is spoken inside the quote. ``known`` maps normalized names and
    aliases to their canonical display spelling; a spoken name that matches it is canonicalized,
    anything else is kept as spoken. A name that is neither known nor spoken is never accepted.
    """
    name = item.proposedName
    quote = (item.evidenceQuote or "").strip()
    idx = item.evidenceTurnIdx
    if not name or is_generic_speaker_label(name):
        return None, quote or None, False
    if idx is None or idx < 0 or idx >= len(segments) or not quote:
        return name, quote or None, False
    norm_quote = normalize_text(quote)
    if not norm_quote or norm_quote not in normalize_text(segments[idx].text):
        return name, quote, False
    norm_name = normalize_text(name)
    if not _contains_phrase(norm_quote, norm_name):
        return name, quote, False
    return known.get(normalize_alias(name), name), quote, True


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


def _parse_bound(value: int | str | date | datetime | None, *, end: bool) -> int | None:
    """Epoch milliseconds for a ``--from``/``--to`` bound given as ms, ISO text, or a date.

    A bare date (or midnight ISO text) as the upper bound includes that whole day.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text_value = value.strip()
        value = datetime.fromisoformat(text_value)
        date_only = "T" not in text_value and " " not in text_value
    else:
        date_only = not isinstance(value, datetime)
    if not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    if end and date_only:
        value += timedelta(days=1)
    return int(value.timestamp() * 1000)


class SpeakerNamingService(InsightsService):
    """Proposes real names for generic speaker labels and applies human-confirmed ones.

    Shares configuration, LLM routing, and the egress guard with :class:`InsightsService`.
    ``me_aliases`` (or ``meAliases`` in ``insights/config.json``) lists the names the user goes
    by; a confirmed name matching one of them resolves to the single existing "me" speaker
    rather than creating a duplicate.
    """

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        db: Engine | DbHandle | None = None,
        meetings_dir: str | None = None,
        config_dir: str | None = None,
        llm_service: Any | None = None,
        allowed_hosts: Sequence[str] | None = None,
        me_aliases: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            options,
            db=db,
            meetings_dir=meetings_dir,
            config_dir=config_dir,
            llm_service=llm_service,
            allowed_hosts=allowed_hosts,
        )
        injected = me_aliases if me_aliases is not None else (options or {}).get("meAliases")
        self._injected_me_aliases = None if injected is None else [str(a) for a in injected]

    # -- configuration -----------------------------------------------------

    def _me_aliases(self) -> set[str]:
        raw = self._injected_me_aliases
        if raw is None:
            configured = load_insights_config(self.config_dir).get("meAliases")
            raw = [str(a) for a in configured] if isinstance(configured, list) else []
        return {normalize_alias(a) for a in raw if normalize_alias(a)}

    def _naming_prompts(self) -> dict[str, str]:
        merged = {**DEFAULT_NAMING_PROMPTS}
        stored = load_insights_prompts(self.config_dir)
        for key in DEFAULT_NAMING_PROMPTS:
            value = stored.get(key)
            if isinstance(value, str) and value.strip():
                merged[key] = value
        return merged

    # -- loading -----------------------------------------------------------

    def _load_segments(self, meeting_id: str) -> tuple[Mapping[Any, Any], list[TranscriptSegment]]:
        """Load the stored transcript through the census reader (no ML imports)."""
        with self.db.connect() as conn:
            meeting = (
                conn.execute(
                    text(
                        "SELECT id, title, start_time, end_time, primary_transcript_artifact_id "
                        "FROM meetings WHERE id = :id"
                    ),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")
            artifact = _resolve_transcript_row(conn, dict(meeting))
            duration_ms = _recording_duration(conn, meeting_id)
        if artifact is None:
            raise ValueError("No transcript found for meeting. Please transcribe the recording first.")
        folder = meeting_paths(
            self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]
        ).folder
        loaded = _load_transcript(Path(folder) / artifact["path"], duration_ms)
        if isinstance(loaded, str):
            raise ValueError(f"Transcript could not be read ({loaded})")
        return meeting, list(loaded[1].segments)

    def _known_names(self, conn: Connection) -> dict[str, str]:
        """Normalized name or alias -> canonical display name, for every known speaker."""
        known: dict[str, str] = {}
        speakers = (
            conn.execute(text("SELECT id, name FROM speakers ORDER BY created_at, id")).mappings().all()
        )
        by_id = {row["id"]: row["name"] for row in speakers}
        for row in speakers:
            known.setdefault(normalize_alias(row["name"]), row["name"])
        aliases = conn.execute(
            text("SELECT alias_norm, speaker_id FROM speaker_aliases WHERE speaker_id IS NOT NULL")
        ).mappings()
        for row in aliases:
            display = by_id.get(row["speaker_id"])
            if display:
                known.setdefault(row["alias_norm"], display)
        return known

    def _has_proposals(self, meeting_id: str) -> bool:
        with self.db.connect() as conn:
            return (
                conn.execute(
                    text("SELECT 1 FROM meeting_speaker_name_proposals WHERE meeting_id = :mid LIMIT 1"),
                    {"mid": meeting_id},
                ).first()
                is not None
            )

    # -- proposing ---------------------------------------------------------

    async def propose_names(
        self,
        meeting_id: str,
        *,
        candidate_names: Sequence[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Propose a real name for each generic speaker label in one meeting.

        Idempotent: a meeting that already has proposal rows returns ``{"status": "skipped"}``
        unless ``force``. A forced re-run refreshes proposals but never overwrites one a human
        already confirmed. A failed run (egress blocked, invalid reply) stores nothing.
        """
        meeting, segments = self._load_segments(meeting_id)
        if not force and self._has_proposals(meeting_id):
            return {"status": "skipped", "meetingId": meeting_id}

        labels = list(dict.fromkeys(s.speaker for s in segments))
        targets = [label for label in labels if is_generic_speaker_label(label)]
        if not targets:
            return {"status": "no_generic_labels", "meetingId": meeting_id, "proposals": []}

        prompts = self._naming_prompts()
        config = load_insights_config(self.config_dir)
        app_cfg = self._app_config()
        provider, model, base_url = await self._resolve_route(
            app_cfg.insightsProvider.strip(), app_cfg.insightsModel.strip()
        )
        enforce_egress(base_url, self._allowed_hosts(config))

        with self.db.connect() as conn:
            known = self._known_names(conn)
        spoken_labels = [label for label in labels if not is_generic_speaker_label(label)]
        for name in [*spoken_labels, *(candidate_names or [])]:
            if name and name.strip():
                known.setdefault(normalize_alias(name), " ".join(name.split()))
        candidates = list(dict.fromkeys(known.values()))

        usage_log: list[tuple[int, LlmGenerateResult]] = []
        best: dict[str, dict[str, Any]] = {}
        chunks = chunk_segments(segments, int(config["chunkChars"]), int(config["chunkOverlapTurns"]))
        target_by_norm = {normalize_alias(label): label for label in targets}

        for chunk in chunks:
            transcript_text = "\n".join(format_turn(i, segments[i]) for i in range(chunk.start, chunk.end))
            prompt = self._render(
                prompts["namingPrompt"],
                title=str(meeting["title"]),
                candidates=candidates,
                labels=targets,
                transcript=transcript_text,
            )

            async def ask(user_prompt: str, _chunk_index: int = chunk.index) -> str:
                res = await _call_llm_generate(
                    self.llm_service,
                    {
                        "provider": provider,
                        "model": model,
                        "systemPrompt": prompts["namingSystemPrompt"],
                        "prompt": user_prompt,
                    },
                )
                usage_log.append((_chunk_index, res))
                return res.text

            try:
                reply, _raw = await parse_proposals_with_repair(ask, prompt)
            except InsightsValidationError as exc:
                raise InsightsExtractionError(
                    f"Chunk {chunk.index + 1}/{len(chunks)} produced no valid name proposals: {exc}"
                ) from exc

            for item in reply.proposals:
                label = target_by_norm.get(normalize_alias(item.speaker))
                if label is None:
                    continue  # the model named a label we did not ask about
                name, quote, grounded = ground_name(item, segments, known)
                if not name:
                    status = STATUS_NONE
                else:
                    status = STATUS_PENDING if grounded else STATUS_UNGROUNDED
                row = {
                    "label": label,
                    "name": name if status != STATUS_NONE else None,
                    "conf": item.confidence if status != STATUS_NONE else None,
                    "quote": quote if status != STATUS_NONE else None,
                    "turn": item.evidenceTurnIdx if status != STATUS_NONE else None,
                    "status": status,
                }
                current = best.get(label)
                if current is None or self._better(row, current):
                    best[label] = row

        last = usage_log[-1][1] if usage_log else None
        now = int(time.time() * 1000)
        out_model = last.model if last else model
        out_provider = last.provider if last else provider
        for label in targets:
            best.setdefault(
                label,
                {
                    "label": label,
                    "name": None,
                    "conf": None,
                    "quote": None,
                    "turn": None,
                    "status": STATUS_NONE,
                },
            )

        with self.db.begin() as conn:
            for label in targets:
                row = best[label]
                conn.execute(
                    text(
                        """
                        INSERT INTO meeting_speaker_name_proposals (
                          id, meeting_id, speaker_label, proposed_name, confidence, evidence_quote,
                          evidence_turn_idx, status, model, provider, created_at
                        ) VALUES (
                          :id, :mid, :label, :name, :conf, :quote, :turn, :status, :model, :provider, :now
                        )
                        ON CONFLICT(meeting_id, speaker_label) DO UPDATE SET
                          proposed_name = excluded.proposed_name,
                          confidence = excluded.confidence,
                          evidence_quote = excluded.evidence_quote,
                          evidence_turn_idx = excluded.evidence_turn_idx,
                          status = excluded.status,
                          model = excluded.model,
                          provider = excluded.provider,
                          created_at = excluded.created_at
                        WHERE meeting_speaker_name_proposals.status != 'confirmed'
                        """
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "mid": meeting_id,
                        "model": out_model,
                        "provider": out_provider,
                        "now": now,
                        **row,
                    },
                )
        self._record_usage(meeting_id, "", "naming", usage_log, now)

        stored = self.list_proposals(meeting_id)
        counts = {
            s: sum(1 for p in stored if p["status"] == s)
            for s in (STATUS_PENDING, STATUS_NONE, STATUS_UNGROUNDED)
        }
        return {"status": "proposed", "meetingId": meeting_id, "proposals": stored, "counts": counts}

    @staticmethod
    def _better(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
        """Chunk merge rule: a more trustworthy status wins, then the higher confidence."""

        def rank(r: dict[str, Any]) -> tuple[int, float]:
            return _STATUS_RANK[r["status"]], r["conf"] or 0.0

        return rank(candidate) > rank(current)

    @staticmethod
    def _render(
        template: str, *, title: str, candidates: Sequence[str], labels: Sequence[str], transcript: str
    ) -> str:
        """Fill the naming prompt placeholders; the transcript goes last so its text stays untouched."""
        body = template.replace("{{title}}", title)
        body = body.replace("{{candidates}}", ", ".join(candidates) or "(none on record)")
        body = body.replace("{{labels}}", ", ".join(labels))
        return body.replace("{{transcript}}", transcript)

    def list_proposals(self, meeting_id: str) -> list[dict[str, Any]]:
        """Stored proposals for a meeting, in label order, as plain dicts."""
        with self.db.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT speaker_label, proposed_name, confidence, evidence_quote,
                               evidence_turn_idx, status, model, provider
                        FROM meeting_speaker_name_proposals
                        WHERE meeting_id = :mid
                        ORDER BY speaker_label
                        """
                    ),
                    {"mid": meeting_id},
                )
                .mappings()
                .all()
            )
        return [dict(r) for r in rows]

    async def propose_all(
        self,
        frm: int | str | date | datetime | None = None,
        to: int | str | date | datetime | None = None,
        limit: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Propose names across every transcribed meeting, oldest first.

        ``frm``/``to`` bound the meeting start time (epoch ms, ISO text, or a date; a bare
        date ``to`` includes that whole day). ``limit`` caps meetings actually sent to the
        model; already-proposed meetings are skipped without counting. A failure on one
        meeting is recorded and the loop continues, except a blocked egress, which aborts:
        every remaining call would be blocked too.
        """
        lower = _parse_bound(frm, end=False)
        upper = _parse_bound(to, end=True)
        sql = (
            "SELECT m.id FROM meetings m WHERE EXISTS "
            "(SELECT 1 FROM artifacts a WHERE a.meeting_id = m.id AND a.kind = 'transcript' "
            "AND a.format = 'json')"
        )
        params: dict[str, Any] = {}
        if lower is not None:
            sql += " AND m.start_time >= :lo"
            params["lo"] = lower
        if upper is not None:
            sql += " AND m.start_time < :hi"
            params["hi"] = upper
        with self.db.connect() as conn:
            ids = list(conn.execute(text(sql + " ORDER BY m.start_time, m.id"), params).scalars())

        summary: dict[str, Any] = {
            "meetings": len(ids),
            "processed": 0,
            "skipped": 0,
            "noGenericLabels": 0,
            "failed": 0,
            "pending": 0,
            "ungrounded": 0,
            "none": 0,
            "errors": [],
        }
        for meeting_id in ids:
            if limit is not None and summary["processed"] >= limit:
                break
            try:
                res = await self.propose_names(meeting_id, force=force)
            except InsightsExtractionError as exc:
                summary["failed"] += 1
                summary["errors"].append({"meetingId": meeting_id, "error": str(exc)})
                logger.warn("Speaker naming failed", {"meetingId": meeting_id, "error": str(exc)})
                continue
            except ValueError as exc:
                summary["failed"] += 1
                summary["errors"].append({"meetingId": meeting_id, "error": str(exc)})
                continue
            if res["status"] == "skipped":
                summary["skipped"] += 1
            elif res["status"] == "no_generic_labels":
                summary["noGenericLabels"] += 1
            else:
                summary["processed"] += 1
                for key, value in res["counts"].items():
                    summary[key] += value
        return summary

    # -- confirming --------------------------------------------------------

    def apply_confirmations(self, meeting_id: str, confirmations: Mapping[str, str]) -> dict[str, Any]:
        """Link confirmed names to speakers for one meeting, all-or-nothing.

        ``confirmations`` maps a speaker label to the confirmed real name. Each name resolves,
        case-insensitively, to an existing speaker by name or confirmed alias; the user's own
        ``me_aliases`` resolve to the single existing "me" speaker; otherwise a speaker is
        created. The speaker is linked via ``meeting_speakers`` with ``identity_source='name'``
        (a stronger existing voiceprint or manual link is not downgraded), the name is recorded
        as a confirmed ``speaker_aliases`` row, and the proposal row becomes ``confirmed``.
        """
        cleaned: dict[str, str] = {}
        for label, name in confirmations.items():
            display = " ".join((name or "").split())
            if not display or is_generic_speaker_label(display):
                raise ValueError(f"Confirmed name for {label!r} must be a real name, got {name!r}")
            cleaned[label] = display

        me_aliases = self._me_aliases()
        now = int(time.time() * 1000)
        applied: dict[str, Any] = {}
        with self.db.begin() as conn:
            if not conn.execute(text("SELECT 1 FROM meetings WHERE id = :id"), {"id": meeting_id}).first():
                raise ValueError(f"Meeting not found: {meeting_id}")
            speakers = [
                dict(r)
                for r in conn.execute(
                    text("SELECT id, name, enrolled_at FROM speakers ORDER BY created_at, id")
                ).mappings()
            ]
            aliases = {
                r["alias_norm"]: r["speaker_id"]
                for r in conn.execute(
                    text(
                        "SELECT alias_norm, speaker_id FROM speaker_aliases "
                        "WHERE speaker_id IS NOT NULL ORDER BY confirmed DESC, created_at, id"
                    )
                ).mappings()
            }
            by_name: dict[str, str] = {}
            for sp in speakers:
                by_name.setdefault(normalize_alias(sp["name"]), sp["id"])
            me_id = self._find_me(speakers, aliases, me_aliases)

            for label, display in cleaned.items():
                norm = normalize_alias(display)
                created = False
                if norm in me_aliases and me_id is not None:
                    speaker_id = me_id
                elif norm in by_name:
                    speaker_id = by_name[norm]
                elif norm in aliases:
                    speaker_id = aliases[norm]
                else:
                    speaker_id = str(uuid.uuid4())
                    created = True
                    conn.execute(
                        text("INSERT INTO speakers (id, name, created_at) VALUES (:id, :name, :now)"),
                        {"id": speaker_id, "name": display, "now": now},
                    )
                    by_name[norm] = speaker_id
                    if norm in me_aliases:
                        me_id = speaker_id

                proposal = (
                    conn.execute(
                        text(
                            "SELECT confidence FROM meeting_speaker_name_proposals "
                            "WHERE meeting_id = :mid AND speaker_label = :label"
                        ),
                        {"mid": meeting_id, "label": label},
                    )
                    .mappings()
                    .first()
                )
                conn.execute(
                    text(
                        """
                        INSERT INTO meeting_speakers (meeting_id, speaker_id, identity_source, identity_confidence)
                        VALUES (:mid, :sid, 'name', :conf)
                        ON CONFLICT(meeting_id, speaker_id) DO UPDATE SET
                          identity_source = 'name',
                          identity_confidence = excluded.identity_confidence
                        WHERE meeting_speakers.identity_source IS NULL
                           OR meeting_speakers.identity_source = 'name'
                        """
                    ),
                    {
                        "mid": meeting_id,
                        "sid": speaker_id,
                        "conf": proposal["confidence"] if proposal else None,
                    },
                )
                conn.execute(
                    text(
                        """
                        INSERT INTO speaker_aliases (id, alias_norm, speaker_id, kind, confirmed, created_at)
                        VALUES (:id, :alias, :sid, 'name', 1, :now)
                        ON CONFLICT(alias_norm, speaker_id) DO UPDATE SET confirmed = 1
                        """
                    ),
                    {"id": str(uuid.uuid4()), "alias": norm, "sid": speaker_id, "now": now},
                )
                aliases.setdefault(norm, speaker_id)
                conn.execute(
                    text(
                        """
                        INSERT INTO meeting_speaker_name_proposals (
                          id, meeting_id, speaker_label, proposed_name, status, created_at
                        ) VALUES (:id, :mid, :label, :name, 'confirmed', :now)
                        ON CONFLICT(meeting_id, speaker_label) DO UPDATE SET status = 'confirmed'
                        """
                    ),
                    {"id": str(uuid.uuid4()), "mid": meeting_id, "label": label, "name": display, "now": now},
                )
                applied[label] = {"speakerId": speaker_id, "name": display, "created": created}
        return applied

    @staticmethod
    def _find_me(
        speakers: Sequence[Mapping[str, Any]], aliases: Mapping[str, str], me_aliases: set[str]
    ) -> str | None:
        """The one speaker standing for the user: enrolled first, then oldest, matching a me alias."""
        if not me_aliases:
            return None
        matches = [
            sp
            for sp in speakers
            if normalize_alias(sp["name"]) in me_aliases
            or any(aliases.get(a) == sp["id"] for a in me_aliases)
        ]
        if not matches:
            return None
        matches.sort(key=lambda sp: sp["enrolled_at"] is None)  # stable: keeps created_at order
        return str(matches[0]["id"])


__all__ = [
    "DEFAULT_NAMING_PROMPTS",
    "NAMING_PROMPT",
    "NAMING_SYSTEM_PROMPT",
    "NameProposals",
    "ProposedName",
    "SpeakerNamingService",
    "ground_name",
    "parse_proposals_with_repair",
    "validate_proposals",
]
