"""Per-meeting signal extraction: turn stats, grounding, chunk merging, and ``InsightsService``.

Division of labor: the LLM supplies judgment (ordinal ratings, event labels, verbatim quotes,
turn numbers). Code supplies everything checkable: talk time, turn and interruption counts,
timestamps (snapped from the cited turn, never taken from the model), quote and speaker
verification, and the ordinal-to-number mapping (configured, not hardcoded).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import time
import unicodedata
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ...db import DbHandle, get_db
from ...layout import meeting_paths
from ...logging_db import logger
from ...schemas import LlmGenerateResult, TranscriptSegment
from ...settings import AppConfig, resolve_paths
from .chunking import chunk_segments, dedup_moments, format_turn
from .identity import ResolvedIdentity, normalize_alias, resolve_labels
from .prompts import (
    compute_prompt_version,
    load_insights_config,
    load_insights_prompts,
    render_extract_prompt,
)
from .validate import (
    ChunkExtraction,
    ExtractedMoment,
    InsightsValidationError,
    parse_with_repair,
)

STAGE_NAME = "insights_analyze"
ALLOWED_HOSTS_ENV = "OLIVE_INSIGHTS_ALLOWED_HOSTS"
ENABLED_ENV = "OLIVE_INSIGHTS_ENABLED"
_TRUTHY = {"1", "true", "yes", "on"}


class EgressBlockedError(RuntimeError):
    """The resolved LLM base URL is not on the insights allowlist; no call was made."""


class InsightsExtractionError(RuntimeError):
    """The LLM never produced a valid extraction for a chunk (after the one repair re-ask)."""


# ---------------------------------------------------------------------------
# Egress guard
# ---------------------------------------------------------------------------


def parse_allowed_hosts(raw: str | Sequence[str] | None) -> list[str]:
    """Normalize an allowlist from a comma-separated string or a list into lowercase entries."""
    if raw is None:
        return []
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    return [str(i).strip().lower() for i in items if str(i).strip()]


def enforce_egress(base_url: str | None, allowed_hosts: Sequence[str]) -> None:
    """Raise :class:`EgressBlockedError` unless ``base_url``'s host (or host:port) is allowlisted.

    Fails closed: an unresolvable URL or an empty allowlist blocks the call. Insights reads
    sensitive interpersonal content, so it may only travel to explicitly approved hosts.
    """
    if not base_url:
        raise EgressBlockedError("Insights egress blocked: could not resolve the LLM base URL")
    try:
        parsed = urlparse(base_url.strip())
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        host, port = "", None
    if not host:
        raise EgressBlockedError(f"Insights egress blocked: unparseable LLM base URL {base_url!r}")
    allowed = {a.lower() for a in allowed_hosts}
    if host in allowed or (port is not None and f"{host}:{port}" in allowed):
        return
    shown = f"{host}:{port}" if port is not None else host
    raise EgressBlockedError(
        f"Insights egress blocked: {shown} is not in the allowlist "
        f"(set {ALLOWED_HOSTS_ENV} or insights/config.json allowedHosts)"
    )


# ---------------------------------------------------------------------------
# Deterministic transcript statistics
# ---------------------------------------------------------------------------


def compute_turn_stats(segments: Sequence[TranscriptSegment]) -> dict[str, dict[str, int]]:
    """Per-label ``talkMs`` / ``turnCount`` / ``interruptCount`` from segment timing alone.

    Each segment is one turn. An interruption is credited to the speaker whose segment starts
    before the previous (different) speaker's segment ended.
    """
    stats: dict[str, dict[str, int]] = {}
    prev: TranscriptSegment | None = None
    for seg in segments:
        entry = stats.setdefault(seg.speaker, {"talkMs": 0, "turnCount": 0, "interruptCount": 0})
        entry["talkMs"] += max(0, seg.endMs - seg.startMs)
        entry["turnCount"] += 1
        if prev is not None and prev.speaker != seg.speaker and seg.startMs < prev.endMs:
            entry["interruptCount"] += 1
        prev = seg
    return stats


# ---------------------------------------------------------------------------
# Ordinal mapping and chunk merging
# ---------------------------------------------------------------------------


def map_ordinal(value: str | float | int | None, scale: Mapping[str, float]) -> float | None:
    """Map an ordinal word (via the configured ``scale``) or a bare number to a stored float.

    Numbers pass through clamped to 0..1; unknown words return None so they are dropped
    rather than invented.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return min(1.0, max(0.0, float(value)))
    word = re.sub(r"[\s_-]+", " ", str(value)).strip().casefold()
    if not word:
        return None
    normalized_scale = {re.sub(r"[\s_-]+", " ", k).strip().casefold(): v for k, v in scale.items()}
    if word in normalized_scale:
        return float(normalized_scale[word])
    try:
        return min(1.0, max(0.0, float(word)))
    except ValueError:
        return None


@dataclass(frozen=True)
class ChunkResult:
    """One validated chunk reply plus each speaker's talk time within that chunk's new turns."""

    extraction: ChunkExtraction
    talk_ms: Mapping[str, int] = field(default_factory=dict)


@dataclass
class MergedSpeaker:
    """A speaker's dimensions (numeric, talk-weighted across chunks) and stance."""

    label: str
    dimensions: dict[str, float] = field(default_factory=dict)
    stance: str | None = None


def merge_chunks(results: Sequence[ChunkResult], scale: Mapping[str, float]) -> dict[str, MergedSpeaker]:
    """Aggregate per-chunk speaker ratings into one entry per speaker, keyed by normalized label.

    Each ordinal word is mapped to a number via ``scale``, then averaged across chunks weighted
    by that speaker's talk time in each chunk (plain mean if no talk time is known). The stance
    is taken from the chunk where the speaker talked most.
    """
    sums: dict[str, dict[str, list[float]]] = {}  # key -> dim -> [weighted_sum, weight, plain_sum, count]
    merged: dict[str, MergedSpeaker] = {}
    best_stance_weight: dict[str, float] = {}

    for result in results:
        talk = {normalize_alias(k): v for k, v in result.talk_ms.items()}
        for sp in result.extraction.speakers:
            key = normalize_alias(sp.speaker)
            if not key:
                continue
            entry = merged.setdefault(key, MergedSpeaker(label=sp.speaker.strip()))
            weight = float(talk.get(key, 0))
            acc = sums.setdefault(key, {})
            for dim, word in sp.dimensions.items():
                num = map_ordinal(word, scale)
                if num is None:
                    continue
                cell = acc.setdefault(dim, [0.0, 0.0, 0.0, 0.0])
                cell[0] += weight * num
                cell[1] += weight
                cell[2] += num
                cell[3] += 1
            if sp.stance and (key not in best_stance_weight or weight > best_stance_weight[key]):
                entry.stance = sp.stance
                best_stance_weight[key] = weight

    for key, dims in sums.items():
        for dim, (wsum, weight, psum, count) in dims.items():
            merged[key].dimensions[dim] = wsum / weight if weight > 0 else psum / count
    return merged


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def normalize_text(value: str) -> str:
    """Case/punctuation/whitespace-insensitive form for quote matching."""
    folded = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[\W_]+", " ", folded).split())


@dataclass(frozen=True)
class GroundedMoment:
    """A moment anchored to transcript timing, with the result of the quote/speaker checks."""

    moment: ExtractedMoment
    start_ms: int
    end_ms: int
    turn_start_idx: int
    turn_end_idx: int
    verified: bool


def ground_moment(moment: ExtractedMoment, segments: Sequence[TranscriptSegment]) -> GroundedMoment | None:
    """Anchor ``moment`` to the transcript, or return None if it cites a turn that does not exist.

    Timestamps come from the cited segments, never from the model. ``verified`` is True only if
    the claimed speaker spoke in the cited range and the normalized quote appears in what that
    speaker said there; otherwise the moment is kept but flagged.
    """
    start = moment.turnStartIdx
    if start < 0 or start >= len(segments):
        return None
    end = start if moment.turnEndIdx is None else min(max(moment.turnEndIdx, start), len(segments) - 1)

    claimed = normalize_alias(moment.speaker)
    spoken = [
        normalize_text(segments[i].text)
        for i in range(start, end + 1)
        if normalize_alias(segments[i].speaker) == claimed
    ]
    quote = normalize_text(moment.quote)
    verified = bool(quote) and any(quote in body for body in spoken)
    return GroundedMoment(
        moment=moment,
        start_ms=segments[start].startMs,
        end_ms=segments[end].endMs,
        turn_start_idx=start,
        turn_end_idx=end,
        verified=verified,
    )


# ---------------------------------------------------------------------------
# P1b queue seam
# ---------------------------------------------------------------------------

EnqueueHook = Callable[[str], Awaitable[bool] | bool]
_enqueue_hook: EnqueueHook | None = None


def set_enqueue_hook(hook: EnqueueHook | None) -> None:
    """Register the function that durably enqueues ``analyze_meeting`` for a meeting.

    The workflows layer registers a hook built on ``enqueue_stage_job`` and the ``insights``
    queue (``configure_insights_queue``), keeping this module free of DBOS imports. Until one is
    registered :meth:`InsightsService.enqueue_analyze` is a no-op.
    """
    global _enqueue_hook
    _enqueue_hook = hook


def get_enqueue_hook() -> EnqueueHook | None:
    """The currently registered queue hook, if any (used by the backfill)."""
    return _enqueue_hook


async def _call_llm_generate(llm_service: Any, options: dict[str, Any]) -> LlmGenerateResult:
    inst_dict = getattr(llm_service, "__dict__", {})
    fn = (
        inst_dict.get("generate_text")
        or inst_dict.get("generateText")
        or getattr(llm_service, "generate_text", None)
        or getattr(llm_service, "generateText", None)
    )
    if not callable(fn):
        raise AttributeError("llm_service has no generate_text/generateText method")
    res = fn(options)
    if inspect.isawaitable(res):
        res = await res
    return res if isinstance(res, LlmGenerateResult) else LlmGenerateResult.model_validate(res)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class InsightsService:
    """Extracts per-speaker signals and notable moments from a meeting's primary transcript."""

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        db: Engine | DbHandle | None = None,
        meetings_dir: str | None = None,
        config_dir: str | None = None,
        llm_service: Any | None = None,
        allowed_hosts: Sequence[str] | None = None,
    ) -> None:
        opts = options or {}
        raw_db = db or opts.get("db")
        if isinstance(raw_db, DbHandle):
            self.db: Engine = raw_db.engine
        elif isinstance(raw_db, Engine):
            self.db = raw_db
        else:
            self.db = get_db()

        paths = resolve_paths()
        self.meetings_dir = str(
            meetings_dir or opts.get("meetingsDir") or opts.get("meetings_dir") or paths.meetings_dir
        )
        self.config_dir = str(
            config_dir or opts.get("configDir") or opts.get("config_dir") or paths.config_dir
        )
        if llm_service is None:
            llm_service = opts.get("llmService") or opts.get("llm_service")
        if llm_service is None:
            from ..llm import LlmService

            llm_service = LlmService(db=self.db, config_dir=self.config_dir)
        self.llm_service = llm_service
        injected = allowed_hosts if allowed_hosts is not None else opts.get("allowedHosts")
        self._injected_allowed_hosts = None if injected is None else parse_allowed_hosts(injected)

    # -- configuration -----------------------------------------------------

    def _app_config(self) -> AppConfig:
        """Read ``settings.json`` from this service's config dir (defaults when absent/invalid)."""
        path = Path(self.config_dir) / "settings.json"
        try:
            return AppConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return AppConfig()

    def is_enabled(self) -> bool:
        """Insights privacy gate: ``OLIVE_INSIGHTS_ENABLED`` wins, else the ``insightsEnabled`` setting."""
        env = os.environ.get(ENABLED_ENV)
        if env is not None and env.strip():
            return env.strip().lower() in _TRUTHY
        return self._app_config().insightsEnabled

    def _allowed_hosts(self, config: Mapping[str, Any]) -> list[str]:
        if self._injected_allowed_hosts is not None:
            return self._injected_allowed_hosts
        return parse_allowed_hosts(config.get("allowedHosts")) + parse_allowed_hosts(
            os.environ.get(ALLOWED_HOSTS_ENV)
        )

    async def _resolve_route(self, provider: str, model: str) -> tuple[str, str, str | None]:
        """Resolve the effective (provider, model, base URL) the LLM service would use."""
        llm = self.llm_service
        get_settings = getattr(llm, "get_settings", None) or getattr(llm, "getSettings", None)
        resolve_model = getattr(llm, "resolve_model", None) or getattr(llm, "resolveModel", None)
        if not callable(get_settings) or not callable(resolve_model):
            return provider, model, None
        settings = await get_settings()
        eff_provider = provider or settings.defaultProvider
        eff_model = model or settings.defaultModel
        info = await resolve_model(eff_provider, eff_model)
        return eff_provider, eff_model, (info or {}).get("baseUrl")

    # -- loading -----------------------------------------------------------

    def _load_meeting_with_transcript(
        self, meeting_id: str
    ) -> tuple[Mapping[Any, Any], str, Mapping[Any, Any]]:
        with self.db.connect() as conn:
            meeting = (
                conn.execute(text("SELECT * FROM meetings WHERE id = :id"), {"id": meeting_id})
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")
            artifacts = (
                conn.execute(text("SELECT * FROM artifacts WHERE meeting_id = :id"), {"id": meeting_id})
                .mappings()
                .all()
            )

        folder = meeting_paths(
            self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]
        ).folder

        primary_tid = meeting.get("primary_transcript_artifact_id")
        transcript_artifact = next((a for a in artifacts if a["id"] == primary_tid), None)
        if not transcript_artifact:
            transcript_artifact = next((a for a in artifacts if a["kind"] == "transcript"), None)
        if not transcript_artifact:
            raise ValueError("No transcript found for meeting. Please transcribe the recording first.")

        full_path = Path(folder) / transcript_artifact["path"]
        if not full_path.exists():
            raise ValueError(f"Transcript artifact not found on disk at: {transcript_artifact['path']}")
        return meeting, folder, transcript_artifact

    # -- stage run ---------------------------------------------------------

    def _start_stage_run(self, meeting_id: str) -> str:
        stage_run_id = str(uuid.uuid4())
        now = int(time.time() * 1000)
        with self.db.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO stage_runs (
                      id, meeting_id, stage, status, attempts, started_at, created_at, updated_at
                    ) VALUES (:id, :mid, :stage, 'running', 1, :now, :now, :now)
                    ON CONFLICT(meeting_id, stage) DO UPDATE SET
                      id = :id,
                      status = 'running',
                      attempts = stage_runs.attempts + 1,
                      started_at = :now,
                      finished_at = NULL,
                      last_error = NULL,
                      updated_at = :now
                    """
                ),
                {"id": stage_run_id, "mid": meeting_id, "stage": STAGE_NAME, "now": now},
            )
        return stage_run_id

    def _finish_stage_run(self, stage_run_id: str, error: str | None) -> None:
        now = int(time.time() * 1000)
        with self.db.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE stage_runs
                    SET status = :status, last_error = :err, finished_at = :now, updated_at = :now
                    WHERE id = :id
                    """
                ),
                {"status": "error" if error else "done", "err": error, "now": now, "id": stage_run_id},
            )

    # -- public API --------------------------------------------------------

    async def enqueue_analyze(self, meeting_id: str) -> bool:
        """Request a durable analysis job for ``meeting_id``; True if one was queued.

        Returns False when insights is disabled or no queue hook is registered yet.
        The hook is the DBOS ``insights`` queue wiring registered via :func:`set_enqueue_hook`.
        """
        if not self.is_enabled():
            return False
        hook = _enqueue_hook
        if hook is None:
            logger.info("Insights enqueue skipped: no queue hook registered", {"meetingId": meeting_id})
            return False
        res = hook(meeting_id)
        if inspect.isawaitable(res):
            res = await res
        return bool(res)

    async def analyze_meeting(self, meeting_id: str, *, force: bool = False) -> dict[str, Any]:
        """Extract and persist signals + moments for one meeting.

        Idempotency key is ``meeting_id + transcript_hash + prompt_version``: if rows for that key
        exist the call returns ``{"status": "skipped"}`` unless ``force``. A re-run replaces the
        meeting's previous rows atomically; a failed run leaves the previous rows untouched.
        """
        meeting, folder, transcript_artifact = self._load_meeting_with_transcript(meeting_id)
        raw_transcript = (Path(folder) / transcript_artifact["path"]).read_text(encoding="utf-8")
        transcript_hash = hashlib.sha256(raw_transcript.encode("utf-8")).hexdigest()

        prompts = load_insights_prompts(self.config_dir)
        config = load_insights_config(self.config_dir)
        prompt_version = compute_prompt_version(prompts)

        if not force and self._has_current_signals(meeting_id, transcript_hash, prompt_version):
            return {
                "status": "skipped",
                "meetingId": meeting_id,
                "transcriptHash": transcript_hash,
                "promptVersion": prompt_version,
            }

        segments = self._parse_segments(raw_transcript, transcript_artifact["format"])
        stage_run_id = self._start_stage_run(meeting_id)
        try:
            result = await self._analyze(
                meeting=meeting,
                folder=folder,
                transcript_artifact_id=transcript_artifact["id"],
                segments=segments,
                transcript_hash=transcript_hash,
                prompts=prompts,
                config=config,
                prompt_version=prompt_version,
            )
        except Exception as exc:
            self._finish_stage_run(stage_run_id, str(exc) or exc.__class__.__name__)
            logger.error("Insights analysis failed", {"meetingId": meeting_id, "error": str(exc)})
            raise
        self._finish_stage_run(stage_run_id, None)
        return result

    # -- internals ---------------------------------------------------------

    def _has_current_signals(self, meeting_id: str, transcript_hash: str, prompt_version: str) -> bool:
        with self.db.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT 1 FROM meeting_speaker_signals
                    WHERE meeting_id = :mid AND transcript_hash = :h AND prompt_version = :v
                    LIMIT 1
                    """
                ),
                {"mid": meeting_id, "h": transcript_hash, "v": prompt_version},
            ).first()
        return row is not None

    @staticmethod
    def _parse_segments(raw: str, fmt: str) -> list[TranscriptSegment]:
        if fmt != "json":
            raise ValueError("Insights requires a JSON transcript with speaker-attributed segments")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Transcript is not valid JSON: {exc}") from exc
        # Stored transcripts come in several shapes: a Transcript object, a bare
        # list of segments, and a raw Plaud list. Normalize them all the same way
        # the rest of the app does (census mirrors services.transcription here).
        try:
            from ..census import _parse_transcript

            _shape, transcript = _parse_transcript(data, None)
            segments = transcript.segments
        except ValueError as exc:
            raise ValueError(f"Transcript is not valid speaker-segment JSON: {exc}") from exc
        if not segments:
            raise ValueError("Transcript has no segments to analyze")
        return segments

    async def _analyze(
        self,
        *,
        meeting: Mapping[Any, Any],
        folder: str,
        transcript_artifact_id: str,
        segments: list[TranscriptSegment],
        transcript_hash: str,
        prompts: dict[str, str],
        config: dict[str, Any],
        prompt_version: str,
    ) -> dict[str, Any]:
        meeting_id = str(meeting["id"])
        scale: Mapping[str, float] = config["dimensionScale"]

        app_cfg = self._app_config()
        provider, model, base_url = await self._resolve_route(
            app_cfg.insightsProvider.strip(), app_cfg.insightsModel.strip()
        )
        enforce_egress(base_url, self._allowed_hosts(config))

        labels = list(dict.fromkeys(s.speaker for s in segments))
        label_by_norm = {normalize_alias(label): label for label in labels}
        with self.db.connect() as conn:
            identities = resolve_labels(conn, meeting_id, labels)

        usage_log: list[tuple[int, LlmGenerateResult]] = []
        chunk_payloads: list[dict[str, Any]] = []
        chunk_results: list[ChunkResult] = []
        extracted_moments: list[ExtractedMoment] = []
        last_result: LlmGenerateResult | None = None

        chunks = chunk_segments(segments, int(config["chunkChars"]), int(config["chunkOverlapTurns"]))
        for chunk in chunks:
            transcript_text = "\n".join(format_turn(i, segments[i]) for i in range(chunk.start, chunk.end))
            chunk_speakers = list(dict.fromkeys(s.speaker for s in segments[chunk.start : chunk.end]))
            prompt = render_extract_prompt(
                prompts,
                config,
                title=str(meeting["title"]),
                speakers=chunk_speakers,
                transcript=transcript_text,
            )

            async def ask(user_prompt: str, _chunk_index: int = chunk.index) -> str:
                nonlocal last_result
                res = await _call_llm_generate(
                    self.llm_service,
                    {
                        "provider": provider,
                        "model": model,
                        "systemPrompt": prompts["systemPrompt"],
                        "prompt": user_prompt,
                    },
                )
                last_result = res
                usage_log.append((_chunk_index, res))
                return res.text

            try:
                extraction, raw = await parse_with_repair(ask, prompt)
            except InsightsValidationError as exc:
                raise InsightsExtractionError(
                    f"Chunk {chunk.index + 1}/{len(chunks)} produced no valid extraction: {exc}"
                ) from exc

            core = compute_turn_stats(segments[chunk.core_start : chunk.end])
            chunk_results.append(ChunkResult(extraction, {k: v["talkMs"] for k, v in core.items()}))
            extracted_moments.extend(extraction.moments)
            chunk_payloads.append(
                {
                    "index": chunk.index,
                    "startTurn": chunk.start,
                    "endTurn": chunk.end,
                    "raw": raw,
                    "parsed": extraction.model_dump(),
                }
            )

        merged = merge_chunks(chunk_results, scale)
        stats = compute_turn_stats(segments)
        grounded = [g for m in dedup_moments(extracted_moments) if (g := ground_moment(m, segments))]
        grounded.sort(key=lambda g: (g.start_ms, g.turn_start_idx))

        now = int(time.time() * 1000)
        out_provider = last_result.provider if last_result else provider
        out_model = last_result.model if last_result else model

        signal_rows = []
        for label in labels:
            ident = identities[label]
            entry = merged.get(normalize_alias(label))
            st = stats[label]
            signal_rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "mid": meeting_id,
                    "sid": ident.speaker_id,
                    "label": label,
                    "src": ident.identity_source,
                    "conf": ident.identity_confidence,
                    "dims": json.dumps(entry.dimensions if entry else {}),
                    "talk": st["talkMs"],
                    "turns": st["turnCount"],
                    "intr": st["interruptCount"],
                    "stance": entry.stance if entry else None,
                    "th": transcript_hash,
                    "pv": prompt_version,
                    "model": out_model,
                    "provider": out_provider,
                    "taid": transcript_artifact_id,
                    "now": now,
                }
            )

        def identity_for(raw_label: str | None) -> tuple[str | None, ResolvedIdentity | None]:
            if not raw_label or not raw_label.strip():
                return None, None
            canonical = label_by_norm.get(normalize_alias(raw_label))
            if canonical is None:
                return raw_label.strip(), None
            return canonical, identities[canonical]

        moment_rows = []
        for g in grounded:
            m = g.moment
            speaker_label, speaker_ident = identity_for(m.speaker)
            toward_label, toward_ident = identity_for(m.toward)
            severity = map_ordinal(m.severity, scale)
            moment_rows.append(
                {
                    "id": str(uuid.uuid4()),
                    "mid": meeting_id,
                    "sid": speaker_ident.speaker_id if speaker_ident else None,
                    "slabel": speaker_label,
                    "tid": toward_ident.speaker_id if toward_ident else None,
                    "tlabel": toward_label,
                    "start": g.start_ms,
                    "end": g.end_ms,
                    "tsi": g.turn_start_idx,
                    "tei": g.turn_end_idx,
                    "etype": m.eventType.strip(),
                    "valence": m.valence,
                    "framework": m.framework,
                    "dimension": m.dimension,
                    "severity": severity,
                    "quote": m.quote,
                    "rationale": m.rationale or None,
                    "guidance": m.guidance,
                    "verified": 1 if g.verified else 0,
                    "src": speaker_ident.identity_source if speaker_ident else None,
                    "conf": speaker_ident.identity_confidence if speaker_ident else None,
                    "th": transcript_hash,
                    "pv": prompt_version,
                    "now": now,
                }
            )

        artifact_id = str(uuid.uuid4())
        relative_path = f"signals/signals_{now}_{artifact_id[:8]}.json"
        signals_dir = Path(folder) / "signals"
        signals_dir.mkdir(parents=True, exist_ok=True)
        full_path = Path(folder) / relative_path
        full_path.write_text(
            json.dumps(
                {
                    "meetingId": meeting_id,
                    "transcriptHash": transcript_hash,
                    "promptVersion": prompt_version,
                    "provider": out_provider,
                    "model": out_model,
                    "createdAt": now,
                    "turnStats": stats,
                    "chunks": chunk_payloads,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        stale_paths: list[str] = []
        try:
            with self.db.begin() as conn:
                stale = (
                    conn.execute(
                        text("SELECT id, path FROM artifacts WHERE meeting_id = :mid AND kind = 'signals'"),
                        {"mid": meeting_id},
                    )
                    .mappings()
                    .all()
                )
                stale_paths = [r["path"] for r in stale]
                conn.execute(
                    text("DELETE FROM artifacts WHERE meeting_id = :mid AND kind = 'signals'"),
                    {"mid": meeting_id},
                )
                conn.execute(text("DELETE FROM meeting_moments WHERE meeting_id = :mid"), {"mid": meeting_id})
                conn.execute(
                    text("DELETE FROM meeting_speaker_signals WHERE meeting_id = :mid"), {"mid": meeting_id}
                )
                conn.execute(
                    text(
                        """
                        INSERT INTO artifacts (id, meeting_id, recording_id, kind, provider, format, path, created_at)
                        VALUES (:id, :mid, NULL, 'signals', :provider, 'json', :path, :now)
                        """
                    ),
                    {
                        "id": artifact_id,
                        "mid": meeting_id,
                        "provider": f"{out_provider}:{out_model}:{prompt_version}",
                        "path": relative_path,
                        "now": now,
                    },
                )
                for row in signal_rows:
                    conn.execute(
                        text(
                            """
                            INSERT INTO meeting_speaker_signals (
                              id, meeting_id, speaker_id, speaker_label, identity_source, identity_confidence,
                              dimensions, talk_ms, turn_count, interrupt_count, stance, transcript_hash,
                              prompt_version, model, provider, transcript_artifact_id, created_at
                            ) VALUES (
                              :id, :mid, :sid, :label, :src, :conf, :dims, :talk, :turns, :intr, :stance,
                              :th, :pv, :model, :provider, :taid, :now
                            )
                            """
                        ),
                        row,
                    )
                for row in moment_rows:
                    conn.execute(
                        text(
                            """
                            INSERT INTO meeting_moments (
                              id, meeting_id, speaker_id, speaker_label, toward_speaker_id, toward_label,
                              start_ms, end_ms, turn_start_idx, turn_end_idx, event_type, valence, framework,
                              dimension, severity, quote, rationale, guidance, verified, identity_source,
                              identity_confidence, transcript_hash, prompt_version, created_at
                            ) VALUES (
                              :id, :mid, :sid, :slabel, :tid, :tlabel, :start, :end, :tsi, :tei, :etype,
                              :valence, :framework, :dimension, :severity, :quote, :rationale, :guidance,
                              :verified, :src, :conf, :th, :pv, :now
                            )
                            """
                        ),
                        row,
                    )
        except Exception:
            full_path.unlink(missing_ok=True)
            raise

        for stale_path in stale_paths:
            if stale_path != relative_path:
                (Path(folder) / stale_path).unlink(missing_ok=True)

        self._record_usage(meeting_id, transcript_hash, prompt_version, usage_log, now)
        logger.info(
            "Extracted insights for meeting",
            {
                "meetingId": meeting_id,
                "speakers": len(signal_rows),
                "moments": len(moment_rows),
                "chunks": len(chunks),
                "provider": out_provider,
                "model": out_model,
            },
        )
        return {
            "status": "done",
            "meetingId": meeting_id,
            "artifactId": artifact_id,
            "transcriptHash": transcript_hash,
            "promptVersion": prompt_version,
            "provider": out_provider,
            "model": out_model,
            "chunks": len(chunks),
            "speakers": len(signal_rows),
            "moments": len(moment_rows),
            "unverifiedMoments": sum(1 for r in moment_rows if not r["verified"]),
        }

    def _record_usage(
        self,
        meeting_id: str,
        transcript_hash: str,
        prompt_version: str,
        usage_log: Sequence[tuple[int, LlmGenerateResult]],
        now: int,
    ) -> None:
        try:
            from ..usage import record_llm_usage
        except Exception:
            return
        for n, (chunk_index, res) in enumerate(usage_log):
            try:
                record_llm_usage(
                    self.db,
                    meeting_id=meeting_id,
                    kind="insights",
                    provider=res.provider,
                    model=res.model,
                    input_tokens=res.usage.inputTokens,
                    output_tokens=res.usage.outputTokens,
                    total_tokens=res.usage.totalTokens,
                    created_at=now,
                    source=f"insights:{meeting_id}:{transcript_hash[:8]}:{prompt_version}:{now}:{n}",
                    detail={"chunk": chunk_index, "promptVersion": prompt_version},
                )
            except Exception:
                pass


__all__ = [
    "ChunkResult",
    "EgressBlockedError",
    "GroundedMoment",
    "InsightsExtractionError",
    "InsightsService",
    "MergedSpeaker",
    "compute_turn_stats",
    "enforce_egress",
    "ground_moment",
    "map_ordinal",
    "merge_chunks",
    "normalize_text",
    "parse_allowed_hosts",
    "get_enqueue_hook",
    "set_enqueue_hook",
]
