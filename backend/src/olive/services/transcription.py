"""Transcription orchestration service for Local & Speechmatics providers (port of transcription/service.ts)."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import time
import urllib.parse
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..db import DbHandle, get_db
from ..layout import ensure_meeting_folder, meeting_paths
from ..logging_db import Logger
from ..logging_db import logger as default_logger
from ..ml import (
    AbortController,
    DiscoveredSpeakerVoiceprint,
    LocalTranscriptionWorkerRunner,
)
from ..ml.identify import identify_cloud_transcript_speakers
from ..providers.assemblyai import AssemblyAIClient, parse_assemblyai_transcript
from ..providers.muse import (
    MuseClient,
    MuseTranscribeResponse,
    merge_muse_chunk_transcripts,
    prepare_muse_wav_chunks,
)
from ..providers.speechmatics import (
    SpeechmaticsClient,
    parse_speechmatics_json_v2,
    transcript_to_text,
)
from ..schemas import (
    EnrolledSpeaker,
    SpeakerReidentificationStatus,
    Transcript,
    TranscriptionProgressUpdate,
    TranscriptSegment,
)
from ..speaker_labels import is_generic_speaker_label
from ..workflows import (
    QUEUE_ASSEMBLYAI,
    QUEUE_DRIVER,
    QUEUE_MUSE,
    QUEUE_PLAUD_SYNC,
    QUEUE_SPEAKER_EMBED,
    QUEUE_SPEECHMATICS,
    QUEUE_SUMMARIZE,
    cancel_dbos_workflows_for_meeting,
    enqueue_driver_job,
    enqueue_stage_job,
    ensure_dbos_launched_async,
    get_dbos_active_progress_map,
)

STAGE_SPEECHMATICS_TRANSCRIBE = "speechmatics_transcribe"
STAGE_ASSEMBLYAI_TRANSCRIBE = "assemblyai_transcribe"
STAGE_MUSE_TRANSCRIBE = "muse_transcribe"
STAGE_LOCAL_TRANSCRIBE = "local_transcribe"
STAGE_SPEAKER_REIDENTIFY = "speaker_reidentify"
PROVIDER_SPEECHMATICS = "speechmatics"
PROVIDER_ASSEMBLYAI = "assemblyai"
PROVIDER_MUSE = "muse"
PROVIDER_LOCAL = "local"

TranscriptionProviderName = Literal["speechmatics", "assemblyai", "muse", "local"]

# Durable-queue routing for background transcription work (see AGENTS.md:
# the DBOS job queue is the SOLE acceptable path for enqueuing work).
TRANSCRIBE_QUEUE_BY_PROVIDER: dict[str, str] = {
    # local driver runs on QUEUE_DRIVER, not QUEUE_PIPELINE: the driver awaits its
    # child local_transcription_workflow (which runs on QUEUE_PIPELINE), so sharing
    # that 2-slot queue deadlocks at >=2 concurrent meetings. See QUEUE_DRIVER note.
    PROVIDER_LOCAL: QUEUE_DRIVER,
    PROVIDER_SPEECHMATICS: QUEUE_SPEECHMATICS,
    PROVIDER_ASSEMBLYAI: QUEUE_ASSEMBLYAI,
    PROVIDER_MUSE: QUEUE_MUSE,
}


def normalize_transcription_provider(value: Any, default: str = PROVIDER_LOCAL) -> str:
    """Maps engine labels (e.g. ``local:cohere``) to a queue-routable provider.

    Mirrors :meth:`TranscriptionService.transcribe_meeting` dispatch: ``None``
    falls back to ``default``; anything unrecognized executes as
    ``speechmatics`` there, so it routes to the speechmatics queue here.
    """
    if value is None or str(value).strip() == "":
        return str(default)
    raw = str(value).strip().lower()
    if raw.startswith(PROVIDER_LOCAL):
        return PROVIDER_LOCAL
    if raw == PROVIDER_ASSEMBLYAI:
        return PROVIDER_ASSEMBLYAI
    if raw == PROVIDER_MUSE:
        return PROVIDER_MUSE
    return PROVIDER_SPEECHMATICS


def is_known_transcription_provider(value: Any) -> bool:
    """Whether ``value`` names a real transcription provider (``local:*`` allowed)."""
    if value is None:
        return True
    raw = str(value).strip().lower()
    return raw.startswith(PROVIDER_LOCAL) or raw in (
        PROVIDER_SPEECHMATICS,
        PROVIDER_ASSEMBLYAI,
        PROVIDER_MUSE,
    )


def _parse_json_field(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _signal_aborted(signal: Any) -> bool:
    if signal is None:
        return False
    if getattr(signal, "aborted", False):
        return True
    is_set = getattr(signal, "is_set", None)
    return bool(callable(is_set) and is_set())


def _speaker_has_local_voiceprint_vectors(speaker: EnrolledSpeaker) -> bool:
    """Whether an enrolled speaker carries usable local voiceprint vectors."""
    items = (speaker.providerIds or {}).get("local") or []
    for item in items:
        try:
            parsed = json.loads(item) if isinstance(item, str) else item
        except Exception:
            continue
        if isinstance(parsed, list) and parsed:
            if isinstance(parsed[0], list):
                if any(isinstance(sub, list) and sub for sub in parsed):
                    return True
            else:
                return True
    return False


def _is_placeholder_speaker_name(name: str) -> bool:
    return is_generic_speaker_label(name)


def _parse_stored_transcript(raw: Any, duration_ms: int | float | None = None) -> Transcript:
    """Parse a persisted transcript JSON artifact in any known shape.

    Canonical artifacts are ``{"segments": [...]}`` objects, but speaker
    identification rewrites Plaud transcripts in place preserving their
    bare-array shape (``transcripts/plaud.json`` must stay a list for the
    Plaud poller), and not-yet-identified Plaud payloads use raw
    ``{start_time, end_time, speaker, content}`` dicts. Accept all three so
    re-identification works regardless of which shape is on disk.

    Raises ``ValueError`` when nothing usable remains.
    """
    if isinstance(raw, dict):
        return Transcript.model_validate(raw)
    if isinstance(raw, list) and raw:
        first = next((item for item in raw if isinstance(item, dict)), None)
        if first is not None and ("start_time" in first or "end_time" in first or "content" in first):
            from ..plaud.poller import plaud_segments_to_transcript

            converted = plaud_segments_to_transcript(raw, duration_ms)
            if converted is not None:
                return converted
        segments: list[TranscriptSegment] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text_value = str(item.get("text") or item.get("content") or "")
            if not text_value.strip():
                continue
            try:
                start = int(item.get("startMs") or item.get("start_time") or 0)
                end = int(item.get("endMs") or item.get("end_time") or 0)
            except (TypeError, ValueError):
                continue
            segments.append(
                TranscriptSegment(
                    startMs=start,
                    endMs=end,
                    speaker=str(item.get("speaker") or "Speaker"),
                    text=text_value,
                    verified=bool(item.get("verified")),
                )
            )
        if segments:
            return Transcript(segments=segments)
    raise ValueError("unrecognized transcript shape")


@dataclass
class _ActiveTranscriptionJob:
    abort_controller: AbortController
    provider: TranscriptionProviderName
    stage_run_id: str | None = None
    job_id: str | None = None


class TranscriptionService:
    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        db: Engine | DbHandle | None = None,
        meetings_dir: str | None = None,
        speechmatics_client: SpeechmaticsClient | None = None,
        assemblyai_client: AssemblyAIClient | None = None,
        muse_client: MuseClient | None = None,
        local_pipeline: Any = None,
        default_provider: TranscriptionProviderName | None = None,
        logger: Logger | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
        now: Callable[[], int] | None = None,
        summary_service: Any = None,
        plaud_poller: Any = None,
        config_dir: str | None = None,
    ) -> None:
        opts = options or {}
        raw_db = db or opts.get("db")
        if isinstance(raw_db, DbHandle):
            self.db: Engine = raw_db.engine
        elif isinstance(raw_db, Engine):
            self.db = raw_db
        else:
            self.db = get_db()

        self.meetings_dir = str(meetings_dir or opts.get("meetingsDir") or opts.get("meetings_dir") or "")
        self.config_dir = config_dir or opts.get("configDir") or opts.get("config_dir")
        passed_sm = speechmatics_client or opts.get("speechmaticsClient") or opts.get("speechmatics_client")
        self.client: SpeechmaticsClient = passed_sm or SpeechmaticsClient()
        passed_aai = assemblyai_client or opts.get("assemblyaiClient") or opts.get("assemblyai_client")
        self.assemblyai_client: AssemblyAIClient = passed_aai or AssemblyAIClient()
        passed_muse = muse_client or opts.get("museClient") or opts.get("muse_client")
        self.muse_client: MuseClient = passed_muse or MuseClient()
        # Injectable for tests: async (audio_path, transcript, enrolled) ->
        # (renamed_transcript, discovered) | None. None = use local ML default.
        self.cloud_speaker_identifier = opts.get("cloudSpeakerIdentifier") or opts.get(
            "cloud_speaker_identifier"
        )
        self.local_pipeline = (
            local_pipeline
            or opts.get("localPipeline")
            or opts.get("local_pipeline")
            or LocalTranscriptionWorkerRunner()
        )
        self.default_provider: TranscriptionProviderName = (
            default_provider
            or opts.get("defaultProvider")
            or opts.get("default_provider")
            or (
                "speechmatics"
                if (passed_sm or os.environ.get("SPEECHMATICS_API_KEY"))
                else "muse"
                if (passed_muse or os.environ.get("MUSE_API_KEY") or os.environ.get("MODEL_API_KEY"))
                else "assemblyai"
                if (passed_aai or os.environ.get("ASSEMBLY_API_KEY") or os.environ.get("ASSEMBLYAI_API_KEY"))
                else "local"
            )
        )
        self.logger: Logger = logger or opts.get("logger") or default_logger
        self.webhook_url: str | None = webhook_url or opts.get("webhookUrl") or opts.get("webhook_url")
        self.webhook_secret: str | None = (
            webhook_secret or opts.get("webhookSecret") or opts.get("webhook_secret")
        )
        self.now: Callable[[], int] = now or opts.get("now") or (lambda: int(time.time() * 1000))
        self.summary_service = summary_service or opts.get("summaryService") or opts.get("summary_service")
        self.plaud_poller = plaud_poller or opts.get("plaudPoller") or opts.get("plaud_poller")
        self._active_jobs: dict[str, _ActiveTranscriptionJob] = {}
        self._active_progress: dict[str, TranscriptionProgressUpdate] = {}
        self._progress_listeners: dict[str, set[Callable[[TranscriptionProgressUpdate], Any]]] = {}
        self._reidentify_statuses: dict[str, SpeakerReidentificationStatus] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def speechmatics(self) -> SpeechmaticsClient:
        return self.client

    @property
    def assemblyai(self) -> AssemblyAIClient:
        return self.assemblyai_client

    @property
    def muse(self) -> MuseClient:
        return self.muse_client

    @property
    def local(self) -> Any:
        return self.local_pipeline

    def emit_progress(self, meeting_id: str, update: TranscriptionProgressUpdate | dict[str, Any]) -> None:
        upd = (
            update
            if isinstance(update, TranscriptionProgressUpdate)
            else TranscriptionProgressUpdate.model_validate(update)
        )
        if upd.stage not in ("done", "error", "cancelled"):
            # The in-flight job can drain queued updates for a few hundred ms
            # after cancellation; drop them so a cancel stays cancelled instead
            # of flipping back to in-progress.
            active = self._active_jobs.get(meeting_id)
            if active is not None and _signal_aborted(active.abort_controller.signal):
                return
        self._active_progress[meeting_id] = upd
        try:
            current_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
            self._loop = current_loop
        except RuntimeError:
            current_loop = None
        target_loop = current_loop or self._loop

        if upd.stage in ("done", "error", "cancelled") and target_loop and not target_loop.is_closed():

            def _cleanup() -> None:
                if self._active_progress.get(meeting_id) is upd:
                    self._active_progress.pop(meeting_id, None)

            target_loop.call_soon_threadsafe(lambda: target_loop.call_later(10.0, _cleanup))

        listeners = self._progress_listeners.get(meeting_id)
        if listeners:
            for listener in list(listeners):
                try:
                    res = listener(upd)
                    if asyncio.iscoroutine(res):
                        if current_loop is not None:
                            current_loop.create_task(res)
                        elif target_loop is not None and not target_loop.is_closed():
                            asyncio.run_coroutine_threadsafe(res, target_loop)
                        else:
                            res.close()
                except Exception:
                    pass

    emitProgress = emit_progress

    def get_transcription_progress(self, meeting_id: str) -> TranscriptionProgressUpdate | None:
        live = self._active_progress.get(meeting_id)
        if live is not None:
            return live
        dbos_prog = get_dbos_active_progress_map(self.db).get(meeting_id)
        return TranscriptionProgressUpdate.model_validate(dbos_prog) if dbos_prog else None

    getTranscriptionProgress = get_transcription_progress

    def get_all_active_transcription_progress(self) -> dict[str, dict[str, Any]]:
        merged = get_dbos_active_progress_map(self.db)
        merged.update(
            {mid: prog.model_dump(exclude_none=True) for mid, prog in self._active_progress.items()}
        )
        return merged

    getAllActiveTranscriptionProgress = get_all_active_transcription_progress

    def subscribe_transcription_progress(
        self,
        meeting_id: str,
        listener: Callable[[TranscriptionProgressUpdate], Any],
    ) -> Callable[[], None]:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        s = self._progress_listeners.setdefault(meeting_id, set())
        s.add(listener)

        def _unsub() -> None:
            s.discard(listener)
            if not s:
                self._progress_listeners.pop(meeting_id, None)

        return _unsub

    subscribeTranscriptionProgress = subscribe_transcription_progress

    def set_summary_service(self, service: Any) -> None:
        self.summary_service = service

    setSummaryService = set_summary_service

    def set_plaud_poller(self, poller: Any) -> None:
        self.plaud_poller = poller

    setPlaudPoller = set_plaud_poller

    def set_default_provider(self, provider: TranscriptionProviderName) -> None:
        self.default_provider = provider

    setDefaultProvider = set_default_provider

    def stop(self) -> None:
        for job in list(self._active_jobs.values()):
            try:
                job.abort_controller.abort()
            except Exception:
                pass
        self._active_jobs.clear()
        stop_fn = getattr(self.local_pipeline, "stop", None)
        if callable(stop_fn):
            try:
                stop_fn()
            except Exception:
                pass

    async def _auto_summarize(self, meeting_id: str) -> None:
        if not self.summary_service:
            return
        try:
            with self.db.connect() as conn:
                meeting = (
                    conn.execute(
                        text("SELECT id, primary_summary_artifact_id FROM meetings WHERE id = :id"),
                        {"id": meeting_id},
                    )
                    .mappings()
                    .first()
                )
            if meeting and not meeting.get("primary_summary_artifact_id"):
                self.logger.debug(
                    "Triggering auto-summarization on transcript completion",
                    {"category": "summary", "meetingId": meeting_id},
                )
                gen_fn = getattr(self.summary_service, "generate_summary", None) or getattr(
                    self.summary_service, "generateSummary", None
                )
                if callable(gen_fn):
                    res = gen_fn({"meetingId": meeting_id, "setPrimary": True})
                    if inspect.isawaitable(res):
                        await res
                self.logger.info(
                    "Auto-generated summary on transcript completion",
                    {"category": "summary", "meetingId": meeting_id},
                )
        except Exception as exc:
            self.logger.warn(
                "Auto-summarization skipped or failed on transcript completion",
                {"category": "summary", "meetingId": meeting_id, "error": str(exc)},
            )

    async def _enqueue_auto_summarize(self, meeting_id: str) -> None:
        if not self.summary_service:
            return
        try:
            await ensure_dbos_launched_async(
                config_dir=self.config_dir,
                meetings_dir=self.meetings_dir,
            )
            await enqueue_stage_job(
                QUEUE_SUMMARIZE,
                lambda: self._auto_summarize(meeting_id),
                deduplication_id=f"auto-summarize:{meeting_id}",
            )
        except Exception:
            await self._auto_summarize(meeting_id)

    async def cancel_transcription(self, meeting_id: str) -> dict[str, Any]:
        self.logger.info(
            "Cancelling transcription for meeting",
            {"category": "transcription", "meetingId": meeting_id},
        )
        active = self._active_jobs.get(meeting_id)
        if active:
            active.abort_controller.abort()
            if active.provider == "speechmatics" and active.job_id:
                try:
                    await self.client.delete_job(active.job_id)
                except Exception as exc:
                    self.logger.warn(
                        "Failed to delete Speechmatics job during cancellation",
                        {"jobId": active.job_id, "error": str(exc)},
                    )

        # Staged local jobs run as DBOS workflows; aborting the in-memory
        # job alone leaves them running, so cancel them explicitly.
        try:
            await cancel_dbos_workflows_for_meeting(
                meeting_id,
                db=self.db,
                config_dir=self.config_dir,
                meetings_dir=self.meetings_dir,
            )
        except Exception as exc:
            self.logger.warn(
                "Failed to cancel DBOS workflows during cancellation",
                {"meetingId": meeting_id, "error": str(exc)},
            )

        cancel_time = self.now()
        self._mark_reidentify_cancelled(meeting_id, cancel_time)
        self.emit_progress(
            meeting_id,
            {
                "stage": "cancelled",
                "percent": 0,
                "message": "Transcription was cancelled.",
            },
        )

        with self.db.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE stage_runs
                    SET status = 'cancelled', last_error = 'Cancelled by user',
                        finished_at = :t, updated_at = :t
                    WHERE meeting_id = :mid AND status = 'running'
                    """
                ),
                {"t": cancel_time, "mid": meeting_id},
            )
            meeting = (
                conn.execute(
                    text("SELECT primary_transcript_artifact_id, status FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if meeting:
                next_status = "ready" if meeting.get("primary_transcript_artifact_id") else "pending"
                conn.execute(
                    text(
                        "UPDATE meetings SET status = :st, last_error = NULL, updated_at = :t WHERE id = :id"
                    ),
                    {"st": next_status, "t": cancel_time, "id": meeting_id},
                )

        return {"success": True, "message": "Transcription cancelled successfully"}

    cancelTranscription = cancel_transcription

    async def start_transcription(
        self,
        meeting_id: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = dict(options or {})
        with self.db.connect() as conn:
            meeting = (
                conn.execute(
                    text("SELECT id FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")
            recording = (
                conn.execute(
                    text("SELECT id FROM recordings WHERE meeting_id = :mid ORDER BY created_at ASC LIMIT 1"),
                    {"mid": meeting_id},
                )
                .mappings()
                .first()
            )
            if not recording:
                raise ValueError(f"No recording found for meeting {meeting_id}")

        existing = self._active_jobs.get(meeting_id)
        if existing is not None and _signal_aborted(existing.abort_controller.signal):
            # A cancelled job lingers until the in-flight task drains; evict it
            # so a fresh start is not blocked and its "queued" update is kept.
            self._active_jobs.pop(meeting_id, None)
            existing = None
        if existing is not None and not existing.abort_controller.signal.aborted:
            return {
                "status": "running",
                "jobId": existing.job_id or f"active-{meeting_id[:8]}",
                "meetingId": meeting_id,
                "alreadyRunning": True,
            }

        provider = normalize_transcription_provider(
            opts.get("provider") or self.default_provider,
            default=self.default_provider,
        )
        queue_name = TRANSCRIBE_QUEUE_BY_PROVIDER[provider]
        job_id = f"transcribe:{meeting_id}:{uuid.uuid4().hex[:8]}"
        await ensure_dbos_launched_async(
            config_dir=self.config_dir,
            meetings_dir=self.meetings_dir,
        )
        await enqueue_driver_job(
            queue_name,
            lambda: self._run_queued_transcription(meeting_id, opts),
            deduplication_id=f"transcribe:{meeting_id}",
            workflow_id=job_id,
            label=f"transcribe:{provider}",
            meeting_id=meeting_id,
        )
        self.emit_progress(
            meeting_id,
            {
                "stage": "queued",
                "percent": 2,
                "message": "Queued transcription...",
            },
        )
        return {
            "status": "running",
            "jobId": job_id,
            "meetingId": meeting_id,
            "alreadyRunning": False,
        }

    startTranscription = start_transcription

    # --- Post-facto speaker re-identification ------------------------------
    # Re-match a persisted transcript's speaker labels against the current
    # enrolled voiceprints (e.g. after enrolling new speakers or improving
    # voiceprints). Leaf work: everything runs inline in one
    # ``speaker_embed`` stage job (see AGENTS.md rule 5).

    @staticmethod
    def _current_transcript_artifact(conn: Any, meeting_id: str) -> Any | None:
        """Primary transcript JSON artifact, falling back to the latest one."""
        meeting = (
            conn.execute(
                text("SELECT primary_transcript_artifact_id FROM meetings WHERE id = :id"),
                {"id": meeting_id},
            )
            .mappings()
            .first()
        )
        primary_id = meeting["primary_transcript_artifact_id"] if meeting else None
        if primary_id:
            art = (
                conn.execute(
                    text(
                        "SELECT * FROM artifacts WHERE id = :id AND kind = 'transcript' AND format = 'json'"
                    ),
                    {"id": primary_id},
                )
                .mappings()
                .first()
            )
            if art:
                return art
        return (
            conn.execute(
                text(
                    "SELECT * FROM artifacts WHERE meeting_id = :mid AND kind = 'transcript' "
                    "AND format = 'json' ORDER BY created_at DESC LIMIT 1"
                ),
                {"mid": meeting_id},
            )
            .mappings()
            .first()
        )

    def _reidentify_targets(self, meeting_id: str) -> dict[str, Any]:
        """Resolve meeting + transcript + audio for re-identification.

        Raises ValueError with a user-facing message when anything is missing.
        """
        with self.db.connect() as conn:
            meeting = (
                conn.execute(text("SELECT * FROM meetings WHERE id = :id"), {"id": meeting_id})
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")
            artifact = self._current_transcript_artifact(conn, meeting_id)
            if artifact is None:
                raise ValueError(f"No transcript found for meeting {meeting_id}")
            recording = (
                conn.execute(
                    text("SELECT * FROM recordings WHERE meeting_id = :mid ORDER BY created_at ASC LIMIT 1"),
                    {"mid": meeting_id},
                )
                .mappings()
                .first()
            )
        folder = meeting_paths(
            self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]
        ).folder
        audio_path = str(Path(folder) / recording["path"]) if recording and recording.get("path") else None
        if not audio_path or not Path(audio_path).exists():
            raise ValueError(f"No audio recording available for meeting {meeting_id}")
        transcript_file = Path(folder) / str(artifact["path"])
        if not transcript_file.exists():
            raise ValueError(f"Transcript file is missing for meeting {meeting_id}")
        try:
            raw = json.loads(transcript_file.read_text(encoding="utf-8"))
            duration_ms = (recording.get("duration_ms") if recording else None) or None
            transcript = _parse_stored_transcript(raw, duration_ms)
        except Exception:
            raise ValueError(f"Stored transcript for meeting {meeting_id} could not be parsed") from None
        if not transcript.segments:
            raise ValueError(f"Transcript for meeting {meeting_id} has no segments")
        return {
            "meeting": meeting,
            "artifact": artifact,
            "recording": recording,
            "audio_path": audio_path,
            "transcript": transcript,
        }

    def get_speaker_reidentification_status(self, meeting_id: str) -> SpeakerReidentificationStatus | None:
        status = self._reidentify_statuses.get(meeting_id)
        return status.model_copy() if status else None

    getSpeakerReidentificationStatus = get_speaker_reidentification_status

    def _write_reidentify_stage_run(self, meeting_id: str, status: str, error: str | None = None) -> None:
        now_ms = self.now()
        try:
            with self.db.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO stage_runs (
                          id, meeting_id, stage, status, attempts, last_error,
                          started_at, finished_at, created_at, updated_at
                        ) VALUES (
                          :id, :mid, :stage, :st, 1,
                          :err, :now, :fin, :now, :now
                        )
                        ON CONFLICT(meeting_id, stage) DO UPDATE SET
                          status = :st,
                          attempts = stage_runs.attempts + 1,
                          last_error = :err,
                          started_at = CASE WHEN :st IN ('running', 'queued') THEN :now ELSE stage_runs.started_at END,
                          finished_at = :fin,
                          updated_at = :now
                        """
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "mid": meeting_id,
                        "stage": STAGE_SPEAKER_REIDENTIFY,
                        "st": status,
                        "err": error,
                        "now": now_ms,
                        "fin": None if status in ("running", "queued") else now_ms,
                    },
                )
        except Exception:
            pass

    def _prune_reidentify_status_later(self, meeting_id: str, delay_s: float = 300.0) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        def _cleanup() -> None:
            cur = self._reidentify_statuses.get(meeting_id)
            if cur and cur.stage not in ("queued", "running"):
                self._reidentify_statuses.pop(meeting_id, None)

        loop.call_later(delay_s, _cleanup)

    def _mark_reidentify_cancelled(self, meeting_id: str, cancel_time: int | None = None) -> None:
        cur = self._reidentify_statuses.get(meeting_id)
        if cur is not None and cur.stage in ("queued", "running"):
            self._reidentify_statuses[meeting_id] = SpeakerReidentificationStatus(
                meetingId=meeting_id,
                stage="cancelled",
                startedAt=cur.startedAt,
                finishedAt=cancel_time or self.now(),
                jobId=cur.jobId,
                provider=cur.provider,
            )
            self._prune_reidentify_status_later(meeting_id)

    async def _run_tracked_reidentification(
        self, meeting_id: str, job_id: str, provider: str, plaud: bool
    ) -> dict[str, Any]:
        """DBOS job body: run the matcher and flip the tracked status to terminal."""
        cur = self._reidentify_statuses.get(meeting_id)
        if cur is not None:
            self._reidentify_statuses[meeting_id] = cur.model_copy(update={"stage": "running"})
        self._write_reidentify_stage_run(meeting_id, "running")
        try:
            if plaud:
                poller = self.plaud_poller
                run_fn = getattr(poller, "run_plaud_speaker_identification", None)
                if not callable(run_fn):
                    raise ValueError("Plaud speaker re-identification is unavailable")
                raw_res: Any = run_fn(meeting_id)
                if inspect.isawaitable(raw_res):
                    raw_res = await raw_res
                result: dict[str, Any] = (
                    {str(k): v for k, v in raw_res.items()}
                    if isinstance(raw_res, dict)
                    else {"meetingId": meeting_id, "status": "skipped"}
                )
            else:
                result = await self.run_speaker_reidentification(meeting_id)
        except Exception as exc:
            msg = str(exc)
            self.logger.warn(
                "Speaker re-identification job failed",
                {"category": "transcription", "meetingId": meeting_id, "error": msg},
            )
            self._reidentify_statuses[meeting_id] = SpeakerReidentificationStatus(
                meetingId=meeting_id,
                stage="error",
                startedAt=cur.startedAt if cur else self.now(),
                finishedAt=self.now(),
                jobId=job_id,
                provider=provider,
                error=msg,
            )
            self._write_reidentify_stage_run(meeting_id, "error", msg)
            self._prune_reidentify_status_later(meeting_id)
            return {
                "meetingId": meeting_id,
                "status": "error",
                "error": msg,
            }
        matched = result.get("matched")
        renamed = result.get("renamed")
        error_val = result.get("error")
        self._reidentify_statuses[meeting_id] = SpeakerReidentificationStatus(
            meetingId=meeting_id,
            stage="done",
            startedAt=cur.startedAt if cur else self.now(),
            finishedAt=self.now(),
            jobId=job_id,
            provider=provider,
            matched=[str(m) for m in matched] if isinstance(matched, list) else None,
            renamed={str(k): str(v) for k, v in renamed.items()} if isinstance(renamed, dict) else None,
            error=str(error_val) if isinstance(error_val, str) else None,
        )
        self._write_reidentify_stage_run(meeting_id, "done")
        self._prune_reidentify_status_later(meeting_id)
        return result

    async def request_speaker_reidentification(self, meeting_id: str) -> dict[str, Any]:
        """Validate and enqueue a forced speaker re-identification as a tracked job.

        Both providers funnel through one status lifecycle
        (``queued`` -> ``running`` -> ``done``/``error``/``cancelled``)
        exposed via :meth:`get_speaker_reidentification_status`, the
        ``GET /api/meetings/{id}/speakers/reidentify/status`` route, and the
        ``speakerReidentification`` field on the meeting detail — mirroring
        transcription progress and summary generation. The work itself stays
        on the DBOS queues (``speaker_embed`` for local/cloud transcripts,
        ``plaud_sync`` for Plaud) with a ``reidentify-speakers:{meeting}``
        deduplication id, so it keeps bounded concurrency, dedup, restart
        recovery, and ``cancel_dbos_workflows_for_meeting`` coverage.
        Unlike the ingest-time hooks, a manual re-run always re-matches
        *all* labels so improved voiceprints can fix previously assigned
        names too.
        """
        existing = self._reidentify_statuses.get(meeting_id)
        if existing is not None and existing.stage in ("queued", "running"):
            return {
                "status": "queued",
                "jobId": existing.jobId,
                "meetingId": meeting_id,
                "provider": existing.provider,
                "alreadyRunning": True,
            }
        with self.db.connect() as conn:
            meeting = (
                conn.execute(text("SELECT id FROM meetings WHERE id = :id"), {"id": meeting_id})
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")
            provider = self._current_transcript_provider(conn, meeting_id)
        plaud = (provider or "").lower() == "plaud"
        if plaud:
            poller = self.plaud_poller
            if poller is None:
                raise ValueError("Plaud sync is not configured; cannot re-identify this meeting")
            if not callable(getattr(poller, "run_plaud_speaker_identification", None)):
                raise ValueError("Plaud speaker re-identification is unavailable")
            targets = self._reidentify_targets(meeting_id)
            enrolled = [
                s for s in await self._get_enrolled_speakers() if _speaker_has_local_voiceprint_vectors(s)
            ]
            if not enrolled:
                raise ValueError("No enrolled speakers with voiceprints yet — enroll a speaker first")
            provider_label = str(targets["artifact"]["provider"] or "plaud")
            queue_name = QUEUE_PLAUD_SYNC
        else:
            targets = self._reidentify_targets(meeting_id)
            enrolled = [
                s for s in await self._get_enrolled_speakers() if _speaker_has_local_voiceprint_vectors(s)
            ]
            if not enrolled:
                raise ValueError("No enrolled speakers with voiceprints yet — enroll a speaker first")
            provider_label = str(targets["artifact"]["provider"] or "")
            queue_name = QUEUE_SPEAKER_EMBED
        job_id = f"reidentify-speakers:{meeting_id}:{uuid.uuid4().hex[:8]}"
        started_at = self.now()
        self._reidentify_statuses[meeting_id] = SpeakerReidentificationStatus(
            meetingId=meeting_id,
            stage="queued",
            startedAt=started_at,
            jobId=job_id,
            provider=provider_label,
        )
        self._write_reidentify_stage_run(meeting_id, "running")
        await ensure_dbos_launched_async(
            config_dir=self.config_dir,
            meetings_dir=self.meetings_dir,
        )
        await enqueue_stage_job(
            queue_name,
            lambda: self._run_tracked_reidentification(meeting_id, job_id, provider_label, plaud),
            deduplication_id=f"reidentify-speakers:{meeting_id}",
            workflow_id=job_id,
            label="reidentify-speakers",
            meeting_id=meeting_id,
        )
        return {
            "status": "queued",
            "jobId": job_id,
            "meetingId": meeting_id,
            "provider": provider_label,
            "alreadyRunning": False,
        }

    requestSpeakerReidentification = request_speaker_reidentification

    async def cancel_speaker_reidentification(self, meeting_id: str) -> dict[str, Any]:
        """Cancel a running re-identify job for one meeting."""
        try:
            await cancel_dbos_workflows_for_meeting(
                meeting_id,
                db=self.db,
                config_dir=self.config_dir,
                meetings_dir=self.meetings_dir,
            )
        except Exception as exc:
            self.logger.warn(
                "Failed to cancel re-identify workflows",
                {"meetingId": meeting_id, "error": str(exc)},
            )
        cancel_time = self.now()
        self._mark_reidentify_cancelled(meeting_id, cancel_time)
        self._write_reidentify_stage_run(meeting_id, "cancelled", "Cancelled by user")
        return {"success": True, "meetingId": meeting_id}

    cancelSpeakerReidentification = cancel_speaker_reidentification

    def _touch_meeting_updated_at(self, meeting_id: str) -> None:
        with self.db.begin() as conn:
            conn.execute(
                text("UPDATE meetings SET updated_at = :now WHERE id = :id"),
                {"now": self.now(), "id": meeting_id},
            )

    async def _prune_stale_transcript_speaker_links(
        self, meeting_id: str, evidence_artifact_id: str, transcript: Transcript
    ) -> None:
        """Drop links evidencing an artifact whose speaker label is gone.

        After a re-match renames labels away, name-only links pointing at
        the rewritten artifact would otherwise linger (e.g. an unverified
        name replaced by an enrolled speaker). Labels still present —
        matched or not — keep their links.
        """
        final_names = {seg.speaker.strip() for seg in transcript.segments if seg.speaker.strip()}
        with self.db.begin() as conn:
            linked = (
                conn.execute(
                    text(
                        "SELECT ms.speaker_id AS sid, s.name AS name FROM meeting_speakers ms "
                        "JOIN speakers s ON s.id = ms.speaker_id "
                        "WHERE ms.meeting_id = :mid AND ms.evidence_artifact_id = :aid"
                    ),
                    {"mid": meeting_id, "aid": evidence_artifact_id},
                )
                .mappings()
                .all()
            )
            stale = [row["sid"] for row in linked if str(row["name"] or "").strip() not in final_names]
            if stale:
                placeholders = ", ".join(f":s{i}" for i in range(len(stale)))
                conn.execute(
                    text(
                        "DELETE FROM meeting_speakers WHERE meeting_id = :mid "
                        f"AND evidence_artifact_id = :aid AND speaker_id IN ({placeholders})"
                    ),
                    {
                        "mid": meeting_id,
                        "aid": evidence_artifact_id,
                        **{f"s{i}": sid for i, sid in enumerate(stale)},
                    },
                )

    async def run_speaker_reidentification(self, meeting_id: str) -> dict[str, Any]:
        """DBOS ``speaker_embed`` job: re-match transcript labels to voiceprints.

        Loads the primary persisted transcript, re-embeds every speaker
        cluster from the meeting audio, and rewrites the transcript
        artifacts with current enrolled names. Never raises: any failure
        just keeps the existing labels.
        """
        targets = self._reidentify_targets(meeting_id)
        meeting, artifact = targets["meeting"], targets["artifact"]
        transcript: Transcript = targets["transcript"]
        recording = targets["recording"]
        rec_id = recording["id"] if recording else None
        provider = str(artifact["provider"] or "")

        enrolled = [
            s for s in await self._get_enrolled_speakers() if _speaker_has_local_voiceprint_vectors(s)
        ]
        if not enrolled:
            return {"meetingId": meeting_id, "status": "skipped", "reason": "no-enrolled-speakers"}
        try:
            if self.cloud_speaker_identifier is not None:
                res = self.cloud_speaker_identifier(targets["audio_path"], transcript, enrolled)
                if inspect.isawaitable(res):
                    res = await res
                result = res
            else:
                threshold = self._configured_similarity_threshold()
                result = await identify_cloud_transcript_speakers(
                    targets["audio_path"],
                    transcript,
                    enrolled,
                    similarity_threshold=float(threshold) if threshold is not None else 0.45,
                    meeting_id=meeting_id,
                )
        except Exception as exc:
            self.logger.warn(
                "Speaker re-identification failed; keeping existing labels",
                {"category": "transcription", "meetingId": meeting_id, "error": str(exc)},
            )
            self._touch_meeting_updated_at(meeting_id)
            return {"meetingId": meeting_id, "status": "kept-labels"}
        if result is None:
            self._touch_meeting_updated_at(meeting_id)
            return {"meetingId": meeting_id, "status": "kept-labels"}
        renamed, discovered = result
        rename_map: dict[str, str] = {}
        for old_seg, new_seg in zip(transcript.segments, renamed.segments, strict=False):
            if new_seg.speaker != old_seg.speaker:
                rename_map[old_seg.speaker] = new_seg.speaker
        if not rename_map:
            self._touch_meeting_updated_at(meeting_id)
            return {"meetingId": meeting_id, "status": "kept-labels"}

        rel_path = str(artifact["path"])
        t_json = f"{json.dumps(renamed.model_dump(exclude_none=True), indent=2)}\n"
        t_text = f"{transcript_to_text(renamed)}\n"
        paths = ensure_meeting_folder(
            meeting_paths(self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"])
        )
        json_art_id = await self._ensure_artifact(
            meeting["id"], rec_id, "transcript", "json", rel_path, t_json, paths.transcripts_dir, provider
        )
        base, slash, filename = rel_path.rpartition("/")
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        txt_rel = f"{base}{slash}{stem}.txt"
        await self._ensure_artifact(
            meeting["id"], rec_id, "transcript", "txt", txt_rel, t_text, paths.transcripts_dir, provider
        )

        await self._persist_local_discovered_speakers(meeting["id"], json_art_id, discovered)
        await self._link_transcript_speakers(meeting["id"], json_art_id, renamed)
        await self._prune_stale_transcript_speaker_links(meeting["id"], json_art_id, renamed)
        self._touch_meeting_updated_at(meeting["id"])

        matched = sorted({d.name for d in discovered if d.isEnrolled})
        self.logger.info(
            "Speaker re-identification renamed speakers",
            {
                "category": "transcription",
                "meetingId": meeting_id,
                "matched": matched,
                "renamed": rename_map,
            },
        )
        return {
            "meetingId": meeting_id,
            "status": "renamed",
            "matched": matched,
            "renamed": rename_map,
        }

    async def _run_queued_transcription(
        self,
        meeting_id: str,
        options: dict[str, Any],
    ) -> None:
        """DBOS-queue worker body for one meeting's transcription.

        This is the SOLE path by which transcription work executes in the
        background: :meth:`start_transcription` enqueues it via
        ``enqueue_driver_job`` (workflow context, so nested stage enqueues
        stay legal) and never via ``asyncio.create_task``.
        """
        try:
            await self.transcribe_meeting(meeting_id, options)
        except Exception as exc:
            self.logger.error(
                "Background transcription failed",
                {"category": "transcription", "meetingId": meeting_id, "error": str(exc)},
            )

    async def start_bulk_transcription(
        self,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Queue background re-transcription for many meetings at once.

        Accepts either an explicit ``meetingIds`` list or ``filters``
        (``search``/``startTime``/``endTime``/``currentProvider``) that are
        resolved server-side so callers don't need to paginate through the
        whole library. Each eligible meeting is enqueued as a durable job on
        its provider's DBOS queue via :meth:`start_transcription`, so bulk
        runs get bounded worker concurrency, deduplication, restart
        recovery, and per-meeting progress/cancel — never bare asyncio tasks.
        """
        opts = dict(options or {})
        if not is_known_transcription_provider(opts.get("provider")):
            raise ValueError("provider must be one of 'local', 'speechmatics', 'assemblyai', 'muse'")
        provider = normalize_transcription_provider(
            opts.get("provider") or self.default_provider,
            default=self.default_provider,
        )
        force = opts.get("force", True)
        if force is None:
            force = True
        force = bool(force)
        language = opts.get("language")

        meeting_ids = opts.get("meetingIds") or opts.get("meeting_ids")
        filters = opts.get("filters") or {}
        if not isinstance(filters, dict):
            filters = {}
        search = filters.get("search") or opts.get("search")
        start_time = filters.get("startTime", filters.get("start_time", opts.get("startTime")))
        end_time = filters.get("endTime", filters.get("end_time", opts.get("endTime")))
        current_provider = filters.get("currentProvider") or filters.get("current_provider")
        if current_provider is not None:
            current_provider = str(current_provider).strip().lower() or None

        if meeting_ids is not None and not isinstance(meeting_ids, list):
            raise ValueError("meetingIds must be an array of meeting id strings")
        if meeting_ids:
            seen: list[str] = []
            for mid in meeting_ids:
                if isinstance(mid, str) and mid.strip() and mid not in seen:
                    seen.append(mid)
            target_ids = seen
        else:
            target_ids = self._resolve_bulk_meeting_ids(
                search=search,
                tags=filters.get("tags", opts.get("tags")),
                status=filters.get("status", opts.get("status")),
                speaker_id=filters.get("speakerId", filters.get("speaker_id", opts.get("speakerId"))),
                has_summary=filters.get("hasSummary", filters.get("has_summary", opts.get("hasSummary"))),
                has_transcript=filters.get(
                    "hasTranscript",
                    filters.get("has_transcript", opts.get("hasTranscript")),
                ),
                start_time=start_time,
                end_time=end_time,
                current_provider=current_provider,
            )
        if not target_ids:
            raise ValueError("No meetings matched: provide meetingIds or matching filters")
        if len(target_ids) > 2000:
            raise ValueError("Bulk transcribe supports at most 2000 meetings per request")

        per_meeting_opts: dict[str, Any] = {
            "provider": provider,
            "force": force,
        }
        if language:
            per_meeting_opts["language"] = language
        for key in (
            "poll",
            "pollIntervalMs",
            "maxPollWaitMs",
            "similarityThreshold",
            "clusteringThreshold",
            "modelId",
        ):
            if opts.get(key) is not None:
                per_meeting_opts[key] = opts[key]

        bulk_id = f"bulk-{uuid.uuid4().hex[:8]}"
        results: list[dict[str, Any]] = []
        for meeting_id in target_ids:
            try:
                started = await self.start_transcription(meeting_id, dict(per_meeting_opts))
                results.append(
                    {
                        "meetingId": meeting_id,
                        "status": str(started.get("status", "running")),
                        "jobId": started.get("jobId"),
                        "alreadyRunning": bool(started.get("alreadyRunning", False)),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "meetingId": meeting_id,
                        "status": "skipped",
                        "error": str(exc),
                    }
                )

        queued = sum(1 for r in results if r["status"] == "running" and not r.get("alreadyRunning"))
        already_running = sum(1 for r in results if r.get("alreadyRunning"))
        skipped = sum(1 for r in results if r["status"] == "skipped")
        self.logger.info(
            "Bulk transcription queued",
            {
                "category": "transcription",
                "bulkId": bulk_id,
                "provider": provider,
                "total": len(results),
                "queued": queued,
            },
        )
        return {
            "bulkId": bulk_id,
            "provider": provider,
            "total": len(results),
            "queued": queued,
            "alreadyRunning": already_running,
            "skipped": skipped,
            "results": results,
        }

    startBulkTranscription = start_bulk_transcription

    def _resolve_bulk_meeting_ids(
        self,
        *,
        search: Any = None,
        tags: Any = None,
        status: Any = None,
        speaker_id: Any = None,
        has_summary: Any = None,
        has_transcript: Any = None,
        start_time: Any = None,
        end_time: Any = None,
        current_provider: str | None = None,
    ) -> list[str]:
        from .meetings import normalize_tag_list, parse_bool_filter

        clauses: list[str] = []
        params: dict[str, Any] = {}
        if search is not None and str(search).strip():
            clauses.append("(title LIKE :search OR tags LIKE :search)")
            params["search"] = f"%{str(search).strip()}%"
        for idx, tag in enumerate(normalize_tag_list(tags)):
            pname = f"tag_{idx}"
            clauses.append(
                f"EXISTS (SELECT 1 FROM json_each(meetings.tags) WHERE lower(value) = lower(:{pname}))"
            )
            params[pname] = tag
        if status is not None and str(status).strip():
            clauses.append("status = :status")
            params["status"] = str(status).strip()
        if speaker_id is not None and str(speaker_id).strip():
            clauses.append(
                "EXISTS (SELECT 1 FROM meeting_speakers "
                "WHERE meeting_speakers.meeting_id = meetings.id "
                "AND meeting_speakers.speaker_id = :speaker_id)"
            )
            params["speaker_id"] = str(speaker_id).strip()
        summary_flag = parse_bool_filter(has_summary)
        if summary_flag is True:
            clauses.append("primary_summary_artifact_id IS NOT NULL")
        elif summary_flag is False:
            clauses.append("primary_summary_artifact_id IS NULL")
        transcript_flag = parse_bool_filter(has_transcript)
        if transcript_flag is True:
            clauses.append("primary_transcript_artifact_id IS NOT NULL")
        elif transcript_flag is False:
            clauses.append("primary_transcript_artifact_id IS NULL")
        try:
            start_ms = (
                int(float(start_time)) if start_time is not None and str(start_time).strip() != "" else None
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("startTime must be an epoch-milliseconds number") from exc
        try:
            end_ms = int(float(end_time)) if end_time is not None and str(end_time).strip() != "" else None
        except (TypeError, ValueError) as exc:
            raise ValueError("endTime must be an epoch-milliseconds number") from exc
        if start_ms is not None:
            clauses.append("start_time >= :start_time")
            params["start_time"] = start_ms
        if end_ms is not None:
            clauses.append("start_time <= :end_time")
            params["end_time"] = end_ms
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.db.connect() as conn:
            rows = (
                conn.execute(
                    text(f"SELECT id FROM meetings {where_sql} ORDER BY start_time DESC"),
                    params,
                )
                .mappings()
                .all()
            )
            candidate_ids = [r["id"] for r in rows]
            if current_provider is None or current_provider in ("", "any", "all"):
                return candidate_ids
            resolved: list[str] = []
            for mid in candidate_ids:
                prov = self._current_transcript_provider(conn, mid)
                if current_provider == "none":
                    if prov is None:
                        resolved.append(mid)
                elif prov == current_provider:
                    resolved.append(mid)
            return resolved

    @staticmethod
    def _current_transcript_provider(conn: Any, meeting_id: str) -> str | None:
        meeting = (
            conn.execute(
                text("SELECT primary_transcript_artifact_id FROM meetings WHERE id = :id"),
                {"id": meeting_id},
            )
            .mappings()
            .first()
        )
        primary_id = meeting["primary_transcript_artifact_id"] if meeting else None
        if primary_id:
            art = (
                conn.execute(
                    text("SELECT provider FROM artifacts WHERE id = :id"),
                    {"id": primary_id},
                )
                .mappings()
                .first()
            )
            if art and art["provider"]:
                return str(art["provider"]).strip().lower()
        latest = (
            conn.execute(
                text(
                    "SELECT provider FROM artifacts WHERE meeting_id = :mid AND kind = 'transcript' "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"mid": meeting_id},
            )
            .mappings()
            .first()
        )
        if latest and latest["provider"]:
            return str(latest["provider"]).strip().lower()
        return None

    async def transcribe_meeting(
        self,
        meeting_id: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = dict(options or {})
        provider: TranscriptionProviderName = opts.get("provider") or self.default_provider

        self.logger.debug(
            "Initiating meeting transcription",
            {
                "category": "transcription",
                "meetingId": meeting_id,
                "provider": provider,
                "language": opts.get("language"),
                "force": bool(opts.get("force")),
            },
        )

        abort_controller = AbortController()
        ext_signal = opts.get("signal")
        if ext_signal is not None:
            if getattr(ext_signal, "aborted", False) or (
                hasattr(ext_signal, "is_set") and ext_signal.is_set()
            ):
                abort_controller.abort()
            elif hasattr(ext_signal, "add_event_listener"):
                ext_signal.add_event_listener("abort", abort_controller.abort)

        active_job = _ActiveTranscriptionJob(
            abort_controller=abort_controller,
            provider=provider,
        )
        self._active_jobs[meeting_id] = active_job
        user_on_progress = opts.get("onProgress") or opts.get("on_progress")

        async def on_progress_handler(update: Any) -> None:
            try:
                update_stage = (
                    update.get("stage") if isinstance(update, dict) else getattr(update, "stage", None)
                )
            except Exception:
                update_stage = None
            if update_stage not in ("done", "error", "cancelled") and (
                self._active_jobs.get(meeting_id) is not active_job
                or _signal_aborted(abort_controller.signal)
            ):
                # Stale update from a cancelled or superseded job; must not
                # overwrite the terminal state (or a newer job's progress).
                return
            self.emit_progress(meeting_id, update)
            if callable(user_on_progress):
                upd_obj = (
                    update
                    if isinstance(update, TranscriptionProgressUpdate)
                    else TranscriptionProgressUpdate.model_validate(update)
                )
                r = user_on_progress(upd_obj)
                if inspect.isawaitable(r):
                    await r

        merged_opts = {
            **opts,
            "signal": abort_controller.signal,
            "onProgress": on_progress_handler,
        }

        try:
            if provider == "local":
                return await self.transcribe_meeting_local(meeting_id, merged_opts)
            if provider == "assemblyai":
                return await self.transcribe_meeting_assemblyai(meeting_id, merged_opts)
            if provider == "muse":
                return await self.transcribe_meeting_muse(meeting_id, merged_opts)
            return await self.transcribe_meeting_speechmatics(meeting_id, merged_opts)
        finally:
            if self._active_jobs.get(meeting_id) is active_job:
                self._active_jobs.pop(meeting_id, None)

    transcribeMeeting = transcribe_meeting

    async def transcribe_meeting_local(
        self,
        meeting_id: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = options or {}
        with self.db.connect() as conn:
            meeting = (
                conn.execute(
                    text("SELECT * FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            recording = (
                conn.execute(
                    text("SELECT * FROM recordings WHERE meeting_id = :mid ORDER BY created_at ASC LIMIT 1"),
                    {"mid": meeting_id},
                )
                .mappings()
                .first()
            )
            if not recording:
                raise ValueError(f"No recording found for meeting {meeting_id}")

            stage_run = (
                conn.execute(
                    text("SELECT * FROM stage_runs WHERE meeting_id = :mid AND stage = :stage LIMIT 1"),
                    {"mid": meeting_id, "stage": STAGE_LOCAL_TRANSCRIBE},
                )
                .mappings()
                .first()
            )

        folder = meeting_paths(
            self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]
        ).folder
        audio_full_path = str(Path(folder) / recording["path"])
        if not Path(audio_full_path).exists():
            raise FileNotFoundError(f"Audio file not found on disk at {audio_full_path}")

        if stage_run and stage_run["status"] == "done" and not opts.get("force"):
            with self.db.connect() as conn:
                existing_art = (
                    conn.execute(
                        text(
                            """
                        SELECT id FROM artifacts
                        WHERE meeting_id = :mid AND provider = :prov AND kind = 'transcript' AND format = 'json'
                        LIMIT 1
                        """
                        ),
                        {"mid": meeting_id, "prov": PROVIDER_LOCAL},
                    )
                    .mappings()
                    .first()
                )
            return {
                "stageRunId": stage_run["id"],
                "jobId": stage_run.get("provider_job_id") or "",
                "status": "done",
                "transcriptArtifactId": existing_art["id"] if existing_art else None,
            }

        stage_run_id = stage_run["id"] if stage_run else str(uuid.uuid4())
        current_time = self.now()

        with self.db.begin() as conn:
            if stage_run:
                conn.execute(
                    text(
                        """
                        UPDATE stage_runs
                        SET status = 'running', attempts = :attempts, last_error = NULL,
                            started_at = :now, updated_at = :now
                        WHERE id = :id
                        """
                    ),
                    {
                        "attempts": int(stage_run["attempts"]) + 1,
                        "now": current_time,
                        "id": stage_run["id"],
                    },
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO stage_runs (
                          id, meeting_id, stage, status, provider_job_id, attempts,
                          last_error, started_at, finished_at, created_at, updated_at
                        ) VALUES (
                          :id, :mid, :stage, 'running', :job_id, 1,
                          NULL, :now, NULL, :now, :now
                        )
                        """
                    ),
                    {
                        "id": stage_run_id,
                        "mid": meeting_id,
                        "stage": STAGE_LOCAL_TRANSCRIBE,
                        "job_id": f"local-{stage_run_id[:8]}",
                        "now": current_time,
                    },
                )

        try:
            enrolled_speakers = await self._get_enrolled_speakers()
            candidate_speakers: list[str] = list(opts.get("candidateSpeakers") or [])
            title_lower = meeting["title"].lower()
            if not candidate_speakers:
                for spk in enrolled_speakers:
                    parts = [p for p in re.split(r"\s+", spk.name.strip().lower()) if len(p) >= 3]
                    if any(part in title_lower for part in parts):
                        candidate_speakers.append(spk.name)

            expected_speaker_count = opts.get("expectedSpeakerCount")
            if not expected_speaker_count:
                if re.search(r"1[:\-_/]?1|1-on-1|one-on-one|interview", meeting["title"], re.I):
                    expected_speaker_count = 2
                elif candidate_speakers:
                    expected_speaker_count = len(candidate_speakers)

            transcribe_fn = self.local_pipeline.transcribe
            pipe_res: Any = await transcribe_fn(
                {
                    "meetingId": meeting_id,
                    "audioPath": audio_full_path,
                    "language": opts.get("language"),
                    "enrolledSpeakers": enrolled_speakers,
                    "candidateSpeakers": candidate_speakers if candidate_speakers else None,
                    "expectedSpeakerCount": expected_speaker_count,
                    "similarityThreshold": opts.get("similarityThreshold"),
                    "clusteringThreshold": opts.get("clusteringThreshold"),
                    "modelId": opts.get("modelId"),
                    "signal": opts.get("signal"),
                    "onProgress": opts.get("onProgress"),
                    "db": self.db,
                    "now_fn": self.now,
                    "configDir": self.config_dir,
                    "meetingsDir": self.meetings_dir,
                    "workflow_id": f"local-{stage_run_id[:8]}",
                }
            )
            transcript: Transcript = (
                pipe_res.transcript
                if hasattr(pipe_res, "transcript")
                else Transcript.model_validate(pipe_res["transcript"])
            )
            discovered_speakers: list[DiscoveredSpeakerVoiceprint] = (
                pipe_res.discoveredSpeakers
                if hasattr(pipe_res, "discoveredSpeakers")
                else [
                    d if isinstance(d, DiscoveredSpeakerVoiceprint) else DiscoveredSpeakerVoiceprint(**d)
                    for d in pipe_res.get("discoveredSpeakers", [])
                ]
            )

            paths = ensure_meeting_folder(
                meeting_paths(self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"])
            )
            t_text = transcript_to_text(transcript)
            t_json = f"{json.dumps(transcript.model_dump(exclude_none=True), indent=2)}\n"

            json_art_id = await self._ensure_artifact(
                meeting["id"],
                recording["id"],
                "transcript",
                "json",
                "transcripts/local.json",
                t_json,
                paths.transcripts_dir,
                PROVIDER_LOCAL,
            )
            txt_art_id = await self._ensure_artifact(
                meeting["id"],
                recording["id"],
                "transcript",
                "txt",
                "transcripts/local.txt",
                f"{t_text}\n",
                paths.transcripts_dir,
                PROVIDER_LOCAL,
            )

            await self._persist_local_discovered_speakers(meeting["id"], json_art_id, discovered_speakers)

            completed_at = self.now()
            with self.db.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE meetings
                        SET primary_transcript_artifact_id = :aid, status = 'ready',
                            last_error = NULL, updated_at = :now
                        WHERE id = :id
                        """
                    ),
                    {"aid": json_art_id, "now": completed_at, "id": meeting["id"]},
                )
                conn.execute(
                    text(
                        """
                        UPDATE stage_runs
                        SET status = 'done', last_error = NULL,
                            finished_at = :now, updated_at = :now
                        WHERE id = :id
                        """
                    ),
                    {"now": completed_at, "id": stage_run_id},
                )

            asyncio.create_task(self._enqueue_auto_summarize(meeting["id"]))

            self.emit_progress(
                meeting["id"],
                {
                    "stage": "done",
                    "percent": 100,
                    "message": "Transcription completed successfully.",
                },
            )

            return {
                "stageRunId": stage_run_id,
                "jobId": f"local-{stage_run_id[:8]}",
                "status": "done",
                "transcriptArtifactId": json_art_id,
                "transcriptTextArtifactId": txt_art_id,
            }
        except Exception as exc:
            err_msg = str(exc)
            sig = opts.get("signal")
            is_set_fn = getattr(sig, "is_set", None)
            is_cancelled = bool(
                getattr(sig, "aborted", False)
                or (callable(is_set_fn) and is_set_fn())
                or "cancel" in err_msg.lower()
            )
            err_time = self.now()

            self.emit_progress(
                meeting_id,
                {
                    "stage": "cancelled" if is_cancelled else "error",
                    "percent": 0,
                    "message": "Transcription was cancelled." if is_cancelled else err_msg,
                    "error": None if is_cancelled else err_msg,
                },
            )

            if is_cancelled:
                self.logger.info("Local transcription cancelled by user", {"meetingId": meeting_id})
                with self.db.begin() as conn:
                    conn.execute(
                        text(
                            """
                            UPDATE stage_runs
                            SET status = 'cancelled', last_error = 'Cancelled by user',
                                finished_at = :t, updated_at = :t
                            WHERE id = :id
                            """
                        ),
                        {"t": err_time, "id": stage_run_id},
                    )
                    existing_m = (
                        conn.execute(
                            text("SELECT primary_transcript_artifact_id FROM meetings WHERE id = :id"),
                            {"id": meeting_id},
                        )
                        .mappings()
                        .first()
                    )
                    next_st = (
                        "ready"
                        if (existing_m and existing_m.get("primary_transcript_artifact_id"))
                        else "pending"
                    )
                    conn.execute(
                        text(
                            "UPDATE meetings SET status = :st, last_error = NULL, updated_at = :t WHERE id = :id"
                        ),
                        {"st": next_st, "t": err_time, "id": meeting_id},
                    )
                return {
                    "stageRunId": stage_run_id,
                    "jobId": "",
                    "status": "cancelled",
                    "error": "Cancelled by user",
                }

            self.logger.error(
                "Failed to run local transcription", {"meetingId": meeting_id, "error": err_msg}
            )
            with self.db.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE stage_runs
                        SET status = 'error', last_error = :err,
                            finished_at = :t, updated_at = :t
                        WHERE id = :id
                        """
                    ),
                    {"err": err_msg, "t": err_time, "id": stage_run_id},
                )
                conn.execute(
                    text("UPDATE meetings SET last_error = :err, updated_at = :t WHERE id = :id"),
                    {"err": err_msg, "t": err_time, "id": meeting_id},
                )
            return {
                "stageRunId": stage_run_id,
                "jobId": "",
                "status": "error",
                "error": err_msg,
            }

    transcribeMeetingLocal = transcribe_meeting_local

    async def complete_recovered_local_transcript(
        self,
        meeting_id: str,
        result: dict[str, Any],
        workflow_id: str | None,
    ) -> None:
        """Persist a local transcript that DBOS finished after a restart (no live caller)."""
        with self.db.connect() as conn:
            meeting = (
                conn.execute(text("SELECT * FROM meetings WHERE id = :id"), {"id": meeting_id})
                .mappings()
                .first()
            )
            recording = (
                conn.execute(
                    text("SELECT id FROM recordings WHERE meeting_id = :mid ORDER BY created_at ASC LIMIT 1"),
                    {"mid": meeting_id},
                )
                .mappings()
                .first()
            )
        if not meeting:
            return

        transcript = Transcript.model_validate(result["transcript"])
        discovered = [
            d if isinstance(d, DiscoveredSpeakerVoiceprint) else DiscoveredSpeakerVoiceprint(**d)
            for d in result.get("discoveredSpeakers", [])
        ]
        paths = ensure_meeting_folder(
            meeting_paths(self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"])
        )
        rec_id = recording["id"] if recording else None
        json_art_id = await self._ensure_artifact(
            meeting_id,
            rec_id,
            "transcript",
            "json",
            "transcripts/local.json",
            f"{json.dumps(transcript.model_dump(exclude_none=True), indent=2)}\n",
            paths.transcripts_dir,
            PROVIDER_LOCAL,
        )
        await self._ensure_artifact(
            meeting_id,
            rec_id,
            "transcript",
            "txt",
            "transcripts/local.txt",
            f"{transcript_to_text(transcript)}\n",
            paths.transcripts_dir,
            PROVIDER_LOCAL,
        )
        await self._persist_local_discovered_speakers(meeting_id, json_art_id, discovered)

        now = self.now()
        with self.db.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE meetings
                    SET primary_transcript_artifact_id = :aid, status = 'ready',
                        last_error = NULL, updated_at = :now
                    WHERE id = :id
                    """
                ),
                {"aid": json_art_id, "now": now, "id": meeting_id},
            )
            conn.execute(
                text(
                    """
                    UPDATE stage_runs
                    SET status = 'done', last_error = NULL, finished_at = :now, updated_at = :now
                    WHERE meeting_id = :mid AND stage = :stage AND status = 'running'
                    """
                ),
                {"now": now, "mid": meeting_id, "stage": STAGE_LOCAL_TRANSCRIBE},
            )
        self.emit_progress(
            meeting_id,
            {"stage": "done", "percent": 100, "message": "Transcription completed successfully."},
        )
        self.logger.info(
            "Persisted recovered local transcript",
            {"category": "transcription", "meetingId": meeting_id, "workflowId": workflow_id},
        )

    async def transcribe_meeting_speechmatics(
        self,
        meeting_id: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = options or {}
        on_progress = opts.get("onProgress")

        with self.db.connect() as conn:
            meeting = (
                conn.execute(
                    text("SELECT * FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            recording = (
                conn.execute(
                    text("SELECT * FROM recordings WHERE meeting_id = :mid ORDER BY created_at ASC LIMIT 1"),
                    {"mid": meeting_id},
                )
                .mappings()
                .first()
            )
            if not recording:
                raise ValueError(f"No recording found for meeting {meeting_id}")

            stage_run = (
                conn.execute(
                    text("SELECT * FROM stage_runs WHERE meeting_id = :mid AND stage = :stage LIMIT 1"),
                    {"mid": meeting_id, "stage": STAGE_SPEECHMATICS_TRANSCRIBE},
                )
                .mappings()
                .first()
            )

        folder = meeting_paths(
            self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]
        ).folder
        audio_full_path = str(Path(folder) / recording["path"])
        if not Path(audio_full_path).exists():
            raise FileNotFoundError(f"Audio file not found on disk at {audio_full_path}")

        if stage_run and stage_run["status"] == "done" and not opts.get("force"):
            with self.db.connect() as conn:
                existing_art = (
                    conn.execute(
                        text(
                            """
                        SELECT id FROM artifacts
                        WHERE meeting_id = :mid AND provider = :prov AND kind = 'transcript' AND format = 'json'
                        LIMIT 1
                        """
                        ),
                        {"mid": meeting_id, "prov": PROVIDER_SPEECHMATICS},
                    )
                    .mappings()
                    .first()
                )
            return {
                "stageRunId": stage_run["id"],
                "jobId": stage_run.get("provider_job_id") or "",
                "status": "done",
                "transcriptArtifactId": existing_art["id"] if existing_art else None,
            }

        if callable(on_progress):
            r = on_progress(
                TranscriptionProgressUpdate(
                    stage="decoding",
                    percent=5,
                    message="Preparing audio and speaker profiles for Speechmatics...",
                )
            )
            if inspect.isawaitable(r):
                await r

        stage_run_id = stage_run["id"] if stage_run else str(uuid.uuid4())
        current_time = self.now()

        # Speaker identity is resolved locally from voiceprints after ASR
        # (see _maybe_identify_cloud_speakers); Speechmatics is only asked
        # for diarized labels, not its native speaker_identifiers.
        webhook_url = (
            f"{self.webhook_url}?meetingId={urllib.parse.quote(meeting_id)}"
            f"&stageRunId={urllib.parse.quote(stage_run_id)}"
            if self.webhook_url
            else None
        )

        if callable(on_progress):
            r = on_progress(
                TranscriptionProgressUpdate(
                    stage="uploading",
                    percent=15,
                    message="Uploading audio to Speechmatics Cloud API...",
                )
            )
            if inspect.isawaitable(r):
                await r

        try:
            submit_res = await self.client.submit_job(
                {
                    "audio": {"path": audio_full_path},
                    "filename": Path(recording["path"]).name,
                    "mime": recording["mime"],
                    "language": opts.get("language") or "en",
                    "getSpeakers": True,
                    "webhookUrl": webhook_url,
                    "webhookSecret": self.webhook_secret,
                }
            )
        except Exception as exc:
            err_msg = str(exc)
            self.logger.error(
                "Failed to submit Speechmatics job", {"meetingId": meeting_id, "error": err_msg}
            )
            with self.db.begin() as conn:
                if stage_run:
                    conn.execute(
                        text(
                            """
                            UPDATE stage_runs
                            SET status = 'error', last_error = :err,
                                attempts = :attempts, updated_at = :now
                            WHERE id = :id
                            """
                        ),
                        {
                            "err": err_msg,
                            "attempts": int(stage_run["attempts"]) + 1,
                            "now": current_time,
                            "id": stage_run["id"],
                        },
                    )
                else:
                    conn.execute(
                        text(
                            """
                            INSERT INTO stage_runs (
                              id, meeting_id, stage, status, provider_job_id, attempts,
                              last_error, started_at, finished_at, created_at, updated_at
                            ) VALUES (
                              :id, :mid, :stage, 'error', NULL, 1,
                              :err, :now, :now, :now, :now
                            )
                            """
                        ),
                        {
                            "id": stage_run_id,
                            "mid": meeting_id,
                            "stage": STAGE_SPEECHMATICS_TRANSCRIBE,
                            "err": err_msg,
                            "now": current_time,
                        },
                    )
                conn.execute(
                    text("UPDATE meetings SET last_error = :err, updated_at = :now WHERE id = :id"),
                    {"err": err_msg, "now": current_time, "id": meeting_id},
                )
            if callable(on_progress):
                r = on_progress(
                    TranscriptionProgressUpdate(
                        stage="error",
                        percent=0,
                        message=err_msg,
                        error=err_msg,
                    )
                )
                if inspect.isawaitable(r):
                    await r
            return {
                "stageRunId": stage_run_id,
                "jobId": "",
                "status": "error",
                "error": err_msg,
            }

        job_id = str(submit_res["id"])
        with self.db.begin() as conn:
            if stage_run:
                conn.execute(
                    text(
                        """
                        UPDATE stage_runs
                        SET status = 'running', provider_job_id = :jid,
                            attempts = :attempts, last_error = NULL,
                            started_at = :now, updated_at = :now
                        WHERE id = :id
                        """
                    ),
                    {
                        "jid": job_id,
                        "attempts": int(stage_run["attempts"]) + 1,
                        "now": current_time,
                        "id": stage_run["id"],
                    },
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO stage_runs (
                          id, meeting_id, stage, status, provider_job_id, attempts,
                          last_error, started_at, finished_at, created_at, updated_at
                        ) VALUES (
                          :id, :mid, :stage, 'running', :jid, 1,
                          NULL, :now, NULL, :now, :now
                        )
                        """
                    ),
                    {
                        "id": stage_run_id,
                        "mid": meeting_id,
                        "stage": STAGE_SPEECHMATICS_TRANSCRIBE,
                        "jid": job_id,
                        "now": current_time,
                    },
                )

        if callable(on_progress):
            r = on_progress(
                TranscriptionProgressUpdate(
                    stage="transcribing",
                    percent=25,
                    message=f"Job {job_id[:10]}... accepted by Speechmatics. Cloud processing...",
                )
            )
            if inspect.isawaitable(r):
                await r

        cur_active = self._active_jobs.get(meeting_id)
        if cur_active:
            cur_active.stage_run_id = stage_run_id
            cur_active.job_id = job_id

        if opts.get("poll"):
            return await self.poll_until_complete(
                meeting_id,
                stage_run_id,
                job_id,
                int(opts.get("pollIntervalMs") or 2000),
                int(opts.get("maxPollWaitMs") or 180_000),
                on_progress,
                opts.get("signal"),
            )

        return {
            "stageRunId": stage_run_id,
            "jobId": job_id,
            "status": "running",
        }

    transcribeMeetingSpeechmatics = transcribe_meeting_speechmatics

    async def poll_until_complete(
        self,
        meeting_id: str,
        stage_run_id: str,
        job_id: str,
        interval_ms: int = 2000,
        max_wait_ms: int = 180_000,
        on_progress: Callable[[TranscriptionProgressUpdate], Any] | None = None,
        signal: Any = None,
    ) -> dict[str, Any]:
        start_time = self.now()
        poll_count = 0

        while self.now() - start_time < max_wait_ms:
            if getattr(signal, "aborted", False) or (hasattr(signal, "is_set") and signal.is_set()):
                try:
                    await self.client.delete_job(job_id)
                except Exception:
                    pass
                cancel_time = self.now()
                with self.db.begin() as conn:
                    conn.execute(
                        text(
                            """
                            UPDATE stage_runs
                            SET status = 'cancelled', last_error = 'Cancelled by user',
                                finished_at = :t, updated_at = :t
                            WHERE id = :id
                            """
                        ),
                        {"t": cancel_time, "id": stage_run_id},
                    )
                return {
                    "stageRunId": stage_run_id,
                    "jobId": job_id,
                    "status": "cancelled",
                    "error": "Cancelled by user",
                }

            poll_count += 1
            elapsed_ms = self.now() - start_time
            elapsed_sec = int(round(elapsed_ms / 1000.0))
            pct = min(70, int(round(20 + min(50.0, (elapsed_sec / 45.0) * 50.0))))

            if callable(on_progress):
                r = on_progress(
                    TranscriptionProgressUpdate(
                        stage="transcribing",
                        percent=pct,
                        currentMs=elapsed_ms,
                        message=f"Speechmatics cloud processing (poll #{poll_count}, {elapsed_sec}s elapsed)...",
                    )
                )
                if inspect.isawaitable(r):
                    await r

            status = await self.client.get_job(job_id)
            if status.get("status") == "done":
                if callable(on_progress):
                    r = on_progress(
                        TranscriptionProgressUpdate(
                            stage="finalizing",
                            percent=72,
                            message="Downloading transcript JSON & identifying speakers...",
                        )
                    )
                    if inspect.isawaitable(r):
                        await r

                json_v2 = await self.client.get_transcript(job_id, "json-v2")
                res = await self.complete_transcription(meeting_id, stage_run_id, job_id, json_v2)

                if callable(on_progress):
                    r = on_progress(
                        TranscriptionProgressUpdate(
                            stage="done",
                            percent=100,
                            message="Speechmatics transcription complete!",
                        )
                    )
                    if inspect.isawaitable(r):
                        await r
                return res

            if status.get("status") in ("rejected", "deleted"):
                errors = status.get("errors") or []
                err_msg = (
                    "; ".join(e.get("message", "") for e in errors)
                    if errors
                    else f"Speechmatics job {status.get('status')}"
                )
                err_time = self.now()
                with self.db.begin() as conn:
                    conn.execute(
                        text(
                            """
                            UPDATE stage_runs
                            SET status = 'error', last_error = :err, finished_at = :t, updated_at = :t
                            WHERE id = :id
                            """
                        ),
                        {"err": err_msg, "t": err_time, "id": stage_run_id},
                    )
                    conn.execute(
                        text("UPDATE meetings SET last_error = :err, updated_at = :t WHERE id = :id"),
                        {"err": err_msg, "t": err_time, "id": meeting_id},
                    )
                return {
                    "stageRunId": stage_run_id,
                    "jobId": job_id,
                    "status": "error",
                    "error": err_msg,
                }

            await asyncio.sleep(interval_ms / 1000.0)

        timeout_err = f"Speechmatics job timed out after {max_wait_ms}ms"
        timeout_at = self.now()
        with self.db.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE stage_runs
                    SET status = 'error', last_error = :err, finished_at = :t, updated_at = :t
                    WHERE id = :id
                    """
                ),
                {"err": timeout_err, "t": timeout_at, "id": stage_run_id},
            )
            conn.execute(
                text("UPDATE meetings SET last_error = :err, updated_at = :t WHERE id = :id"),
                {"err": timeout_err, "t": timeout_at, "id": meeting_id},
            )
        return {
            "stageRunId": stage_run_id,
            "jobId": job_id,
            "status": "error",
            "error": timeout_err,
        }

    pollUntilComplete = poll_until_complete

    def _configured_similarity_threshold(self) -> float | None:
        if not self.config_dir:
            return None
        try:
            raw = json.loads((Path(self.config_dir) / "settings.json").read_text(encoding="utf-8"))
            value = raw.get("localSimilarityThreshold")
            return float(value) if value is not None else None
        except Exception:
            return None

    async def _link_transcript_speakers(
        self,
        meeting_id: str,
        evidence_artifact_id: str,
        transcript: Transcript,
    ) -> None:
        """Ensure every named speaker in the transcript is linked to the meeting.

        Used after local-voiceprint persistence so speakers whose turns were
        too short to embed still appear on the meeting. Generic labels
        (``Speaker 1``, ``S1``, ``Unknown``) stay in the transcript only --
        an unidentified cluster is not a person, so no speaker row/link is
        created for it.
        """
        unique = {
            seg.speaker.strip()
            for seg in transcript.segments
            if seg.speaker.strip() and not _is_placeholder_speaker_name(seg.speaker)
        }
        current_time = self.now()
        with self.db.begin() as conn:
            for name in unique:
                all_speakers = conn.execute(text("SELECT * FROM speakers")).mappings().all()
                existing = next(
                    (s for s in all_speakers if s["name"].strip().lower() == name.lower()),
                    None,
                )
                speaker_id = existing["id"] if existing else str(uuid.uuid4())
                if not existing:
                    conn.execute(
                        text(
                            """
                            INSERT INTO speakers (
                              id, name, provider_ids, enrolled_at, enrollment_clip_paths, created_at
                            ) VALUES (:id, :name, '{}', NULL, '[]', :now)
                            """
                        ),
                        {"id": speaker_id, "name": name, "now": current_time},
                    )
                conn.execute(
                    text(
                        """
                        INSERT INTO meeting_speakers (meeting_id, speaker_id, evidence_artifact_id)
                        VALUES (:mid, :sid, :aid)
                        ON CONFLICT(meeting_id, speaker_id) DO UPDATE SET evidence_artifact_id = :aid
                        """
                    ),
                    {"mid": meeting_id, "sid": speaker_id, "aid": evidence_artifact_id},
                )

    async def _maybe_identify_cloud_speakers(
        self,
        meeting_id: str,
        audio_path: str | None,
        transcript: Transcript,
        options: dict[str, Any] | None = None,
    ) -> tuple[Transcript, list[DiscoveredSpeakerVoiceprint]] | None:
        """ASR-first speaker ID: match cloud speaker turns to local voiceprints.

        Returns ``(renamed_transcript, discovered)`` or ``None`` when local
        identification is unavailable — callers then keep cloud labels.
        """
        if not audio_path:
            return None
        opts = options or {}

        async def _identify_progress(update: dict[str, Any]) -> None:
            phase = str(update.get("phase") or "")
            raw_msg = str(update.get("message") or "")
            detail = raw_msg.removeprefix("Diarization: ").strip()
            current = update.get("current")
            total = update.get("total")
            if phase == "decoding":
                percent, message = 75.0, "Identifying speakers: decoding audio..."
            elif phase == "matching":
                percent, message = 94.0, f"Identifying speakers: {detail}"
            elif current is not None and total:
                frac = min(1.0, float(current) / max(1.0, float(total)))
                percent = 78.0 + 14.0 * frac
                message = (
                    f"Identifying speakers: {detail} ({current}/{total} speaker(s))"
                    if detail
                    else "Identifying speakers..."
                )
            else:
                percent, message = (
                    85.0,
                    f"Identifying speakers: {detail}" if detail else "Identifying speakers...",
                )
            self.emit_progress(
                meeting_id,
                {
                    "stage": "finalizing",
                    "percent": float(round(percent)),
                    "message": message,
                },
            )

        try:
            enrolled = await self._get_enrolled_speakers()
            if self.cloud_speaker_identifier is not None:
                res = self.cloud_speaker_identifier(audio_path, transcript, enrolled)
                if inspect.isawaitable(res):
                    res = await res
                result = res
            else:
                threshold = opts.get("similarityThreshold")
                if threshold is None:
                    threshold = self._configured_similarity_threshold()
                result = await identify_cloud_transcript_speakers(
                    audio_path,
                    transcript,
                    enrolled,
                    similarity_threshold=float(threshold) if threshold is not None else 0.45,
                    meeting_id=meeting_id,
                    on_progress=_identify_progress,
                )
            if result is not None:
                _renamed, discovered = result
                matched = [d.name for d in discovered if d.isEnrolled]
                clusters = len(discovered)
                if matched:
                    msg = f"Identified {len(matched)} of {clusters} speaker(s): " + ", ".join(matched)
                else:
                    msg = "No enrolled speakers matched; keeping cloud speaker labels"
                self.emit_progress(
                    meeting_id,
                    {"stage": "finalizing", "percent": 95, "message": msg},
                )
            return result
        except Exception as exc:
            self.logger.warn(
                "Cloud speaker identification failed; keeping cloud speaker labels",
                {"category": "transcription", "meetingId": meeting_id, "error": str(exc)},
            )
            return None

    async def _complete_cloud_transcription(
        self,
        meeting_id: str,
        stage_run_id: str,
        job_id: str,
        canonical: Transcript,
        provider: str,
        raw_payload: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a normalized cloud transcript + identified speakers."""
        with self.db.connect() as conn:
            meeting = (
                conn.execute(
                    text("SELECT * FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting {meeting_id} not found")

            recording = (
                conn.execute(
                    text(
                        "SELECT id, path FROM recordings WHERE meeting_id = :mid ORDER BY created_at ASC LIMIT 1"
                    ),
                    {"mid": meeting_id},
                )
                .mappings()
                .first()
            )

        paths = ensure_meeting_folder(
            meeting_paths(self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"])
        )
        audio_path = (
            str(Path(paths.folder) / recording["path"]) if recording and recording.get("path") else None
        )

        identified = await self._maybe_identify_cloud_speakers(meeting_id, audio_path, canonical, options)
        discovered: list[DiscoveredSpeakerVoiceprint] | None = None
        if identified is not None:
            canonical, discovered = identified

        if _signal_aborted((options or {}).get("signal")):
            cancelled_at = self.now()
            with self.db.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE stage_runs
                        SET status = 'cancelled', last_error = 'Cancelled by user',
                            finished_at = :t, updated_at = :t
                        WHERE id = :id
                        """
                    ),
                    {"t": cancelled_at, "id": stage_run_id},
                )
            return {
                "stageRunId": stage_run_id,
                "jobId": job_id,
                "status": "cancelled",
                "error": "Cancelled by user",
            }

        t_text = transcript_to_text(canonical)
        t_json = f"{json.dumps(canonical.model_dump(exclude_none=True), indent=2)}\n"

        rec_id = recording["id"] if recording else None
        json_art_id = await self._ensure_artifact(
            meeting["id"],
            rec_id,
            "transcript",
            "json",
            f"transcripts/{provider}.json",
            t_json,
            paths.transcripts_dir,
            provider,
        )
        txt_art_id = await self._ensure_artifact(
            meeting["id"],
            rec_id,
            "transcript",
            "txt",
            f"transcripts/{provider}.txt",
            f"{t_text}\n",
            paths.transcripts_dir,
            provider,
        )

        if discovered is not None:
            await self._persist_local_discovered_speakers(meeting["id"], json_art_id, discovered)
            await self._link_transcript_speakers(meeting["id"], json_art_id, canonical)
        elif provider == PROVIDER_SPEECHMATICS and raw_payload is not None:
            await self._persist_discovered_speakers(meeting["id"], json_art_id, canonical, raw_payload)
        else:
            await self._persist_discovered_speakers(meeting["id"], json_art_id, canonical, {})

        completed_at = self.now()
        with self.db.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE meetings
                    SET primary_transcript_artifact_id = :aid, status = 'ready',
                        last_error = NULL, updated_at = :now
                    WHERE id = :id
                    """
                ),
                {"aid": json_art_id, "now": completed_at, "id": meeting["id"]},
            )
            conn.execute(
                text(
                    """
                    UPDATE stage_runs
                    SET status = 'done', last_error = NULL,
                        finished_at = :now, updated_at = :now
                    WHERE id = :id
                    """
                ),
                {"now": completed_at, "id": stage_run_id},
            )

        asyncio.create_task(self._enqueue_auto_summarize(meeting["id"]))

        return {
            "stageRunId": stage_run_id,
            "jobId": job_id,
            "status": "done",
            "transcriptArtifactId": json_art_id,
            "transcriptTextArtifactId": txt_art_id,
        }

    async def complete_transcription(
        self,
        meeting_id: str,
        stage_run_id: str,
        job_id: str,
        json_v2: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        canonical = parse_speechmatics_json_v2(json_v2)
        res = await self._complete_cloud_transcription(
            meeting_id,
            stage_run_id,
            job_id,
            canonical,
            PROVIDER_SPEECHMATICS,
            raw_payload=json_v2,
            options=options,
        )

        try:
            if job_id:
                await self.client.delete_job(job_id)
        except Exception as cleanup_err:
            self.logger.warn(
                "Failed to delete Speechmatics job after completion",
                {"jobId": job_id, "error": str(cleanup_err)},
            )

        return res

    completeTranscription = complete_transcription

    async def transcribe_meeting_assemblyai(
        self,
        meeting_id: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = options or {}
        on_progress = opts.get("onProgress")

        with self.db.connect() as conn:
            meeting = (
                conn.execute(
                    text("SELECT * FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            recording = (
                conn.execute(
                    text("SELECT * FROM recordings WHERE meeting_id = :mid ORDER BY created_at ASC LIMIT 1"),
                    {"mid": meeting_id},
                )
                .mappings()
                .first()
            )
            if not recording:
                raise ValueError(f"No recording found for meeting {meeting_id}")

            stage_run = (
                conn.execute(
                    text("SELECT * FROM stage_runs WHERE meeting_id = :mid AND stage = :stage LIMIT 1"),
                    {"mid": meeting_id, "stage": STAGE_ASSEMBLYAI_TRANSCRIBE},
                )
                .mappings()
                .first()
            )

        folder = meeting_paths(
            self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]
        ).folder
        audio_full_path = str(Path(folder) / recording["path"])
        if not Path(audio_full_path).exists():
            raise FileNotFoundError(f"Audio file not found on disk at {audio_full_path}")

        if stage_run and stage_run["status"] == "done" and not opts.get("force"):
            with self.db.connect() as conn:
                existing_art = (
                    conn.execute(
                        text(
                            """
                        SELECT id FROM artifacts
                        WHERE meeting_id = :mid AND provider = :prov AND kind = 'transcript' AND format = 'json'
                        LIMIT 1
                        """
                        ),
                        {"mid": meeting_id, "prov": PROVIDER_ASSEMBLYAI},
                    )
                    .mappings()
                    .first()
                )
            return {
                "stageRunId": stage_run["id"],
                "jobId": stage_run.get("provider_job_id") or "",
                "status": "done",
                "transcriptArtifactId": existing_art["id"] if existing_art else None,
            }

        stage_run_id = stage_run["id"] if stage_run else str(uuid.uuid4())
        current_time = self.now()

        if callable(on_progress):
            r = on_progress(
                TranscriptionProgressUpdate(
                    stage="uploading",
                    percent=15,
                    message="Uploading audio to AssemblyAI (Universal-3.5 Pro)...",
                )
            )
            if inspect.isawaitable(r):
                await r

        with self.db.begin() as conn:
            if stage_run:
                conn.execute(
                    text(
                        """
                        UPDATE stage_runs
                        SET status = 'running', attempts = :attempts, last_error = NULL,
                            started_at = :now, updated_at = :now
                        WHERE id = :id
                        """
                    ),
                    {
                        "attempts": int(stage_run["attempts"]) + 1,
                        "now": current_time,
                        "id": stage_run["id"],
                    },
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO stage_runs (
                          id, meeting_id, stage, status, provider_job_id, attempts,
                          last_error, started_at, finished_at, created_at, updated_at
                        ) VALUES (
                          :id, :mid, :stage, 'running', NULL, 1,
                          NULL, :now, NULL, :now, :now
                        )
                        """
                    ),
                    {
                        "id": stage_run_id,
                        "mid": meeting_id,
                        "stage": STAGE_ASSEMBLYAI_TRANSCRIBE,
                        "now": current_time,
                    },
                )

        sig = opts.get("signal")
        if _signal_aborted(sig):
            return await self._fail_assemblyai(
                meeting_id,
                stage_run_id,
                "",
                "Transcription was cancelled.",
                cancelled=True,
                on_progress=on_progress,
            )

        async def _emit_aai(stage: str, percent: float, message: str) -> None:
            if callable(on_progress):
                r = on_progress(
                    TranscriptionProgressUpdate(
                        stage=stage,  # type: ignore[arg-type]
                        percent=float(round(percent)),
                        message=message,
                    )
                )
                if inspect.isawaitable(r):
                    await r

        # Upload + submit with a heartbeat, since the upload is a single
        # request with no byte-level progress from the SDK.
        submit_task = asyncio.create_task(
            self.assemblyai_client.submit_job(
                {
                    "audio": {"path": audio_full_path},
                    "filename": Path(recording["path"]).name,
                    "language": opts.get("language"),
                    "additionalVocab": opts.get("additionalVocab") or opts.get("additional_vocab"),
                }
            )
        )
        submit_start = self.now()
        submitted: Any
        try:
            while not submit_task.done():
                if _signal_aborted(sig):
                    submit_task.cancel()
                    return await self._fail_assemblyai(
                        meeting_id,
                        stage_run_id,
                        "",
                        "Transcription was cancelled.",
                        cancelled=True,
                        on_progress=on_progress,
                    )
                done, _pending = await asyncio.wait({submit_task}, timeout=5.0)
                if done:
                    break
                elapsed_s = int((self.now() - submit_start) / 1000)
                await _emit_aai("uploading", 15, f"Uploading audio to AssemblyAI... ({elapsed_s}s elapsed)")
            submitted = submit_task.result()
        except Exception as exc:
            return await self._fail_assemblyai(
                meeting_id, stage_run_id, "", str(exc), on_progress=on_progress
            )

        job_id = str(submitted.get("id") or "") if isinstance(submitted, dict) else ""
        with self.db.begin() as conn:
            conn.execute(
                text("UPDATE stage_runs SET provider_job_id = :jid, updated_at = :now WHERE id = :id"),
                {"jid": job_id or None, "now": self.now(), "id": stage_run_id},
            )
        await _emit_aai(
            "transcribing",
            25,
            f"Submitted to AssemblyAI (job {job_id[:8]}...). Processing...",
        )

        raw: Any = None
        if (
            isinstance(submitted, dict)
            and submitted.get("status") == "completed"
            and (submitted.get("words") or submitted.get("utterances"))
        ):
            raw = submitted
        else:
            interval_ms = int(opts.get("pollIntervalMs") or 5000)
            max_wait_ms = int(opts.get("maxPollWaitMs") or 3_600_000)
            duration_s = (recording.get("duration_ms") or 0) / 1000.0
            expected_s = min(900.0, max(45.0, duration_s / 8.0 if duration_s else 120.0))
            poll_start = self.now()
            poll_count = 0
            while self.now() - poll_start < max_wait_ms:
                if _signal_aborted(sig):
                    return await self._fail_assemblyai(
                        meeting_id,
                        stage_run_id,
                        job_id,
                        "Transcription was cancelled.",
                        cancelled=True,
                        on_progress=on_progress,
                    )
                state = await self.assemblyai_client.get_transcript(job_id)
                status = str(state.get("status") or "") if isinstance(state, dict) else ""
                if status == "completed":
                    raw = state
                    break
                if status == "error":
                    return await self._fail_assemblyai(
                        meeting_id,
                        stage_run_id,
                        job_id,
                        str(state.get("error") or "AssemblyAI transcription failed"),
                        on_progress=on_progress,
                    )
                poll_count += 1
                elapsed_s = int((self.now() - poll_start) / 1000)
                if status == "queued":
                    pct = 25.0
                else:
                    pct = min(70.0, 25.0 + 45.0 * min(1.0, (elapsed_s / expected_s) if expected_s else 1.0))
                await _emit_aai(
                    "transcribing",
                    pct,
                    f"AssemblyAI {status or 'processing'} (poll #{poll_count}, {elapsed_s}s elapsed)...",
                )
                await asyncio.sleep(interval_ms / 1000.0)
            if raw is None:
                return await self._fail_assemblyai(
                    meeting_id,
                    stage_run_id,
                    job_id,
                    f"AssemblyAI job timed out after {max_wait_ms}ms",
                    on_progress=on_progress,
                )

        if callable(on_progress):
            r = on_progress(
                TranscriptionProgressUpdate(
                    stage="finalizing",
                    percent=72,
                    message="AssemblyAI transcription complete. Identifying speakers...",
                )
            )
            if inspect.isawaitable(r):
                await r

        canonical = parse_assemblyai_transcript(raw)
        res = await self._complete_cloud_transcription(
            meeting_id,
            stage_run_id,
            job_id,
            canonical,
            PROVIDER_ASSEMBLYAI,
            raw_payload=None,
            options=opts,
        )
        if res.get("status") == "cancelled":
            return await self._fail_assemblyai(
                meeting_id,
                stage_run_id,
                job_id,
                "Transcription was cancelled.",
                cancelled=True,
                on_progress=on_progress,
            )
        if callable(on_progress):
            r = on_progress(
                TranscriptionProgressUpdate(
                    stage="done",
                    percent=100,
                    message="AssemblyAI transcription complete!",
                )
            )
            if inspect.isawaitable(r):
                await r
        return res

    transcribeMeetingAssemblyai = transcribe_meeting_assemblyai

    async def _fail_assemblyai(
        self,
        meeting_id: str,
        stage_run_id: str,
        job_id: str,
        err_msg: str,
        cancelled: bool = False,
        on_progress: Callable[[TranscriptionProgressUpdate], Any] | None = None,
    ) -> dict[str, Any]:
        err_time = self.now()
        with self.db.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE stage_runs
                    SET status = :st, last_error = :err, finished_at = :t, updated_at = :t
                    WHERE id = :id
                    """
                ),
                {
                    "st": "cancelled" if cancelled else "error",
                    "err": "Cancelled by user" if cancelled else err_msg,
                    "t": err_time,
                    "id": stage_run_id,
                },
            )
            if not cancelled:
                conn.execute(
                    text("UPDATE meetings SET last_error = :err, updated_at = :t WHERE id = :id"),
                    {"err": err_msg, "t": err_time, "id": meeting_id},
                )
        self.logger.error("AssemblyAI transcription failed", {"meetingId": meeting_id, "error": err_msg})
        if callable(on_progress):
            r = on_progress(
                TranscriptionProgressUpdate(
                    stage="cancelled" if cancelled else "error",
                    percent=0,
                    message="Transcription was cancelled." if cancelled else err_msg,
                    error=None if cancelled else err_msg,
                )
            )
            if inspect.isawaitable(r):
                await r
        return {
            "stageRunId": stage_run_id,
            "jobId": job_id,
            "status": "cancelled" if cancelled else "error",
            "error": "Cancelled by user" if cancelled else err_msg,
        }

    async def transcribe_meeting_muse(
        self,
        meeting_id: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = options or {}
        on_progress = opts.get("onProgress")

        with self.db.connect() as conn:
            meeting = (
                conn.execute(
                    text("SELECT * FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            recording = (
                conn.execute(
                    text("SELECT * FROM recordings WHERE meeting_id = :mid ORDER BY created_at ASC LIMIT 1"),
                    {"mid": meeting_id},
                )
                .mappings()
                .first()
            )
            if not recording:
                raise ValueError(f"No recording found for meeting {meeting_id}")

            stage_run = (
                conn.execute(
                    text("SELECT * FROM stage_runs WHERE meeting_id = :mid AND stage = :stage LIMIT 1"),
                    {"mid": meeting_id, "stage": STAGE_MUSE_TRANSCRIBE},
                )
                .mappings()
                .first()
            )

        folder = meeting_paths(
            self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]
        ).folder
        audio_full_path = str(Path(folder) / recording["path"])
        if not Path(audio_full_path).exists():
            raise FileNotFoundError(f"Audio file not found on disk at {audio_full_path}")

        if stage_run and stage_run["status"] == "done" and not opts.get("force"):
            with self.db.connect() as conn:
                existing_art = (
                    conn.execute(
                        text(
                            """
                        SELECT id FROM artifacts
                        WHERE meeting_id = :mid AND provider = :prov AND kind = 'transcript' AND format = 'json'
                        LIMIT 1
                        """
                        ),
                        {"mid": meeting_id, "prov": PROVIDER_MUSE},
                    )
                    .mappings()
                    .first()
                )
            return {
                "stageRunId": stage_run["id"],
                "jobId": stage_run.get("provider_job_id") or "",
                "status": "done",
                "transcriptArtifactId": existing_art["id"] if existing_art else None,
            }

        stage_run_id = stage_run["id"] if stage_run else str(uuid.uuid4())
        session_id = f"muse-{meeting_id[:8]}-{uuid.uuid4().hex[:8]}"
        current_time = self.now()

        async def _emit_muse(stage: str, percent: float, message: str) -> None:
            if callable(on_progress):
                r = on_progress(
                    TranscriptionProgressUpdate(
                        stage=stage,  # type: ignore[arg-type]
                        percent=float(round(percent)),
                        message=message,
                    )
                )
                if inspect.isawaitable(r):
                    await r

        with self.db.begin() as conn:
            if stage_run:
                conn.execute(
                    text(
                        """
                        UPDATE stage_runs
                        SET status = 'running', attempts = :attempts, last_error = NULL,
                            started_at = :now, updated_at = :now
                        WHERE id = :id
                        """
                    ),
                    {
                        "attempts": int(stage_run["attempts"]) + 1,
                        "now": current_time,
                        "id": stage_run["id"],
                    },
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO stage_runs (
                          id, meeting_id, stage, status, provider_job_id, attempts,
                          last_error, started_at, finished_at, created_at, updated_at
                        ) VALUES (
                          :id, :mid, :stage, 'running', :jid, 1,
                          NULL, :now, NULL, :now, :now
                        )
                        """
                    ),
                    {
                        "id": stage_run_id,
                        "mid": meeting_id,
                        "stage": STAGE_MUSE_TRANSCRIBE,
                        "jid": session_id,
                        "now": current_time,
                    },
                )

        sig = opts.get("signal")
        if _signal_aborted(sig):
            return await self._fail_muse(
                meeting_id,
                stage_run_id,
                session_id,
                "Transcription was cancelled.",
                cancelled=True,
                on_progress=on_progress,
            )

        await _emit_muse("uploading", 12, "Converting audio for Muse Voice (24 kHz WAV)...")
        try:
            chunks = await asyncio.to_thread(prepare_muse_wav_chunks, audio_full_path)
        except Exception as exc:
            return await self._fail_muse(
                meeting_id, stage_run_id, session_id, str(exc), on_progress=on_progress
            )

        if _signal_aborted(sig):
            return await self._fail_muse(
                meeting_id,
                stage_run_id,
                session_id,
                "Transcription was cancelled.",
                cancelled=True,
                on_progress=on_progress,
            )

        mode = str(opts.get("mode") or "DIARIZATION")
        raw_responses: list[MuseTranscribeResponse | dict[str, Any]] = []
        chunk_offsets: list[int] = []
        try:
            for index, (wav_bytes, offset_ms) in enumerate(chunks):
                if _signal_aborted(sig):
                    return await self._fail_muse(
                        meeting_id,
                        stage_run_id,
                        session_id,
                        "Transcription was cancelled.",
                        cancelled=True,
                        on_progress=on_progress,
                    )
                pct = 15.0 + 55.0 * (index / max(1, len(chunks)))
                await _emit_muse(
                    "transcribing",
                    pct,
                    f"Transcribing with Muse Voice (chunk {index + 1}/{len(chunks)})...",
                )
                raw = await self.muse_client.transcribe_wav(
                    wav_bytes,
                    {
                        "mode": mode,
                        "language": opts.get("language"),
                        "additionalVocab": opts.get("additionalVocab") or opts.get("additional_vocab"),
                        "sessionId": f"{session_id}-{index}",
                        "filename": f"muse-chunk-{index}.wav",
                    },
                )
                raw_responses.append(raw)
                chunk_offsets.append(offset_ms)
        except Exception as exc:
            return await self._fail_muse(
                meeting_id, stage_run_id, session_id, str(exc), on_progress=on_progress
            )

        await _emit_muse("finalizing", 72, "Muse transcription complete. Identifying speakers...")

        # Labels are request-scoped: the same letter in two chunks may be
        # different people, so merge assigns globally unique Speaker N
        # clusters and lets voiceprint matching reconcile identities.
        canonical = merge_muse_chunk_transcripts(raw_responses, chunk_offsets)
        res = await self._complete_cloud_transcription(
            meeting_id,
            stage_run_id,
            session_id,
            canonical,
            PROVIDER_MUSE,
            raw_payload=None,
            options=opts,
        )
        if res.get("status") == "cancelled":
            return await self._fail_muse(
                meeting_id,
                stage_run_id,
                session_id,
                "Transcription was cancelled.",
                cancelled=True,
                on_progress=on_progress,
            )
        if callable(on_progress):
            r = on_progress(
                TranscriptionProgressUpdate(
                    stage="done",
                    percent=100,
                    message="Muse transcription complete!",
                )
            )
            if inspect.isawaitable(r):
                await r
        return res

    transcribeMeetingMuse = transcribe_meeting_muse

    async def _fail_muse(
        self,
        meeting_id: str,
        stage_run_id: str,
        job_id: str,
        err_msg: str,
        cancelled: bool = False,
        on_progress: Callable[[TranscriptionProgressUpdate], Any] | None = None,
    ) -> dict[str, Any]:
        err_time = self.now()
        with self.db.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE stage_runs
                    SET status = :st, last_error = :err, finished_at = :t, updated_at = :t
                    WHERE id = :id
                    """
                ),
                {
                    "st": "cancelled" if cancelled else "error",
                    "err": "Cancelled by user" if cancelled else err_msg,
                    "t": err_time,
                    "id": stage_run_id,
                },
            )
            if not cancelled:
                conn.execute(
                    text("UPDATE meetings SET last_error = :err, updated_at = :t WHERE id = :id"),
                    {"err": err_msg, "t": err_time, "id": meeting_id},
                )
        self.logger.error("Muse transcription failed", {"meetingId": meeting_id, "error": err_msg})
        if callable(on_progress):
            r = on_progress(
                TranscriptionProgressUpdate(
                    stage="cancelled" if cancelled else "error",
                    percent=0,
                    message="Transcription was cancelled." if cancelled else err_msg,
                    error=None if cancelled else err_msg,
                )
            )
            if inspect.isawaitable(r):
                await r
        return {
            "stageRunId": stage_run_id,
            "jobId": job_id,
            "status": "cancelled" if cancelled else "error",
            "error": "Cancelled by user" if cancelled else err_msg,
        }

    async def _ensure_artifact(
        self,
        meeting_id: str,
        recording_id: str | None,
        kind: str,
        fmt: str,
        relative_path: str,
        content: str,
        directory: str,
        provider: str,
    ) -> str:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        filename = relative_path.split("/")[-1]
        (dir_path / filename).write_text(content, encoding="utf-8")

        with self.db.begin() as conn:
            existing = (
                conn.execute(
                    text(
                        """
                    SELECT id FROM artifacts
                    WHERE meeting_id = :mid AND provider = :prov AND kind = :kind
                      AND format = :fmt AND path = :path
                    LIMIT 1
                    """
                    ),
                    {
                        "mid": meeting_id,
                        "prov": provider,
                        "kind": kind,
                        "fmt": fmt,
                        "path": relative_path,
                    },
                )
                .mappings()
                .first()
            )

            if existing:
                art_id = str(existing["id"])
            else:
                art_id = str(uuid.uuid4())
                conn.execute(
                    text(
                        """
                    INSERT INTO artifacts (
                      id, meeting_id, recording_id, kind, provider, format, path, created_at
                    ) VALUES (
                      :id, :mid, :rid, :kind, :prov, :fmt, :path, :now
                    )
                    """
                    ),
                    {
                        "id": art_id,
                        "mid": meeting_id,
                        "rid": recording_id,
                        "kind": kind,
                        "prov": provider,
                        "fmt": fmt,
                        "path": relative_path,
                        "now": self.now(),
                    },
                )
        if kind == "transcript" and fmt == "json":
            # The JSON artifact is the canonical transcript record (a plain-text
            # twin is saved alongside it); track spend once against it. The
            # usage row is keyed by artifact id, so re-runs stay idempotent.
            try:
                from .usage import record_transcription_usage

                record_transcription_usage(
                    self.db,
                    meeting_id=meeting_id,
                    provider=provider,
                    recording_id=recording_id,
                    transcript_content=content,
                    created_at=self.now(),
                    source=f"transcript:{art_id}",
                    detail={"artifactId": art_id, "recordingId": recording_id},
                )
            except Exception:
                pass
        return art_id

    async def _persist_local_discovered_speakers(
        self,
        meeting_id: str,
        evidence_artifact_id: str,
        discovered_speakers: list[DiscoveredSpeakerVoiceprint],
    ) -> None:
        current_time = self.now()
        with self.db.begin() as conn:
            for disc in discovered_speakers:
                # Unmatched clusters are discovered under their generic label
                # ("Speaker 1", cluster id, ...). They remain in the transcript
                # for later assignment, but are not people: don't persist a
                # speaker row or meeting link for them.
                if _is_placeholder_speaker_name(disc.name):
                    continue
                all_speakers = conn.execute(text("SELECT * FROM speakers")).mappings().all()
                norm_name = disc.name.strip().lower()
                existing = next(
                    (s for s in all_speakers if s["name"].strip().lower() == norm_name),
                    None,
                )
                speaker_id = (
                    existing["id"] if existing else (disc.speakerId if disc.isEnrolled else str(uuid.uuid4()))
                )
                if not existing:
                    conn.execute(
                        text(
                            """
                            INSERT INTO speakers (
                              id, name, provider_ids, enrolled_at, enrollment_clip_paths, created_at
                            ) VALUES (:id, :name, '{}', NULL, '[]', :now)
                            """
                        ),
                        {"id": speaker_id, "name": disc.name, "now": current_time},
                    )
                conn.execute(
                    text(
                        """
                        INSERT INTO meeting_speakers (meeting_id, speaker_id, evidence_artifact_id)
                        VALUES (:mid, :sid, :aid)
                        ON CONFLICT(meeting_id, speaker_id) DO UPDATE SET evidence_artifact_id = :aid
                        """
                    ),
                    {"mid": meeting_id, "sid": speaker_id, "aid": evidence_artifact_id},
                )

    async def _persist_discovered_speakers(
        self,
        meeting_id: str,
        evidence_artifact_id: str,
        transcript: Transcript,
        json_v2: dict[str, Any],
    ) -> None:
        speaker_summary_map: dict[str, list[str]] = {}
        for s in json_v2.get("speakers") or []:
            if not isinstance(s, dict):
                continue
            name = str(s.get("label") or s.get("speaker") or "").strip()
            raw_ids = s.get("speaker_identifiers")
            ids = list(raw_ids) if isinstance(raw_ids, (list, set, tuple)) else []
            if name and ids:
                speaker_summary_map[name] = ids

        unique_speakers = {seg.speaker.strip() for seg in transcript.segments if seg.speaker.strip()}
        current_time = self.now()

        with self.db.begin() as conn:
            for name in unique_speakers:
                # Generic labels stay in the transcript only; they are not people.
                if _is_placeholder_speaker_name(name):
                    continue
                all_speakers = conn.execute(text("SELECT * FROM speakers")).mappings().all()
                norm_name = name.lower()
                existing = next(
                    (s for s in all_speakers if s["name"].strip().lower() == norm_name),
                    None,
                )
                sm_ids = speaker_summary_map.get(name, [])
                speaker_id = existing["id"] if existing else str(uuid.uuid4())

                if existing:
                    pids = _parse_json_field(existing.get("provider_ids"), {})
                    cur_sm = list(dict.fromkeys([*(pids.get(PROVIDER_SPEECHMATICS) or []), *sm_ids]))
                    pids[PROVIDER_SPEECHMATICS] = cur_sm
                    conn.execute(
                        text("UPDATE speakers SET provider_ids = :pids WHERE id = :id"),
                        {"pids": json.dumps(pids), "id": existing["id"]},
                    )
                else:
                    pids = {PROVIDER_SPEECHMATICS: sm_ids} if sm_ids else {}
                    enrolled_at = current_time if sm_ids else None
                    conn.execute(
                        text(
                            """
                            INSERT INTO speakers (
                              id, name, provider_ids, enrolled_at, enrollment_clip_paths, created_at
                            ) VALUES (:id, :name, :pids, :enrolled_at, '[]', :now)
                            """
                        ),
                        {
                            "id": speaker_id,
                            "name": name,
                            "pids": json.dumps(pids),
                            "enrolled_at": enrolled_at,
                            "now": current_time,
                        },
                    )

                conn.execute(
                    text(
                        """
                        INSERT INTO meeting_speakers (meeting_id, speaker_id, evidence_artifact_id)
                        VALUES (:mid, :sid, :aid)
                        ON CONFLICT(meeting_id, speaker_id) DO UPDATE SET evidence_artifact_id = :aid
                        """
                    ),
                    {"mid": meeting_id, "sid": speaker_id, "aid": evidence_artifact_id},
                )

    async def _get_enrolled_speakers(self) -> list[EnrolledSpeaker]:
        if self.config_dir:
            try:
                from .speakers import SpeakerService

                spk_svc = SpeakerService(
                    db=self.db,
                    config_dir=self.config_dir,
                    meetings_dir=self.meetings_dir,
                    logger=self.logger,
                )
                await spk_svc.ensure_embedding_model_current()
            except Exception as exc:
                self.logger.warn(
                    "Failed to ensure speaker embedding model current before transcription",
                    {"category": "speakers", "error": str(exc)},
                )

        with self.db.connect() as conn:
            rows = conn.execute(text("SELECT * FROM speakers")).mappings().all()
        out: list[EnrolledSpeaker] = []
        for r in rows:
            name = str(r["name"]).strip()
            if _is_placeholder_speaker_name(name):
                continue
            out.append(
                EnrolledSpeaker(
                    id=r["id"],
                    name=r["name"],
                    providerIds=_parse_json_field(r.get("provider_ids"), {}),
                )
            )
        return out
