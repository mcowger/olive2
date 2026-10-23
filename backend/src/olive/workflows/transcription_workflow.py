"""DBOS staged local transcription workflow and background queue tasks."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dbos import DBOS, SetEnqueueOptions, SetWorkflowID
from dbos._error import DBOSStepTimeoutError, DBOSWorkflowCancelledError
from dbos._serialization import DefaultSerializer
from sqlalchemy import Engine, text

from ..logging_db import logger
from ..ml.align import proportional_word_timestamps
from ..ml.asr_cohere import CohereAsrEngine
from ..ml.audio import enhance_audio_file, load_audio_samples
from ..ml.pipeline import LocalTranscriptionPipeline
from ..ml.protocols import (
    DiscoveredSpeakerVoiceprint,
    VoiceprintVector,
)
from ..ml.scoring import (
    merge_weighted_voiceprint_vectors,
    normalize_vector,
    score_against_trusted_vectors,
)
from ..schemas import (
    EnrolledSpeaker,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    coalesce_speaker_segments,
)
from .queues import (
    QUEUE_ALIGN,
    QUEUE_ASR,
    QUEUE_ASSEMBLYAI,
    QUEUE_DIARIZE,
    QUEUE_FFMPEG,
    QUEUE_PLAUD_SYNC,
    QUEUE_SPEAKER_EMBED,
    QUEUE_SPEECHMATICS,
    QUEUE_SUMMARIZE,
    STAGE_QUEUE_SPECS,
)
from .runtime import (
    ensure_dbos_launched_async,
    get_existing_dbos_sqlite_path,
    is_dbos_launched,
)

STAGE_FFMPEG_DECODE = "ffmpeg_decode"
STAGE_DIARIZE = "diarize"
STAGE_SPEAKER_EMBED = "speaker_embed"
STAGE_ASR = "asr"
STAGE_ALIGN = "align"

#: Flat local-transcription stage chain: decode -> diarize -> {embed || asr} -> align.
#:
#: Every stage is its own DBOS workflow on its own queue. A stage enqueues its
#: successors and **returns** — no stage ever awaits another workflow. That is
#: what keeps the pipeline deadlock-free: no slot holder ever waits on a slot,
#: so queue concurrency alone bounds concurrency and liveness never depends on
#: a supervision heuristics. Stuck work is bounded declaratively by per-stage
#: DBOS step timeouts (see ``_stage_timeout_sec``).
STAGE_SEQUENCE: tuple[str, ...] = (
    STAGE_FFMPEG_DECODE,
    STAGE_DIARIZE,
    STAGE_SPEAKER_EMBED,
    STAGE_ASR,
    STAGE_ALIGN,
)

#: Stage name in ``stage_runs`` used for the whole run (not a single stage).
STAGE_LOCAL_TRANSCRIBE = "local_transcribe"

_STAGE_QUEUE_BY_STAGE: dict[str, str] = {
    STAGE_FFMPEG_DECODE: QUEUE_FFMPEG,
    STAGE_DIARIZE: QUEUE_DIARIZE,
    STAGE_SPEAKER_EMBED: QUEUE_SPEAKER_EMBED,
    STAGE_ASR: QUEUE_ASR,
    STAGE_ALIGN: QUEUE_ALIGN,
}

_STAGE_SUCCESSORS: dict[str, tuple[str, ...]] = {
    STAGE_FFMPEG_DECODE: (STAGE_DIARIZE,),
    STAGE_DIARIZE: (STAGE_SPEAKER_EMBED, STAGE_ASR),
    STAGE_SPEAKER_EMBED: (STAGE_ALIGN,),
    STAGE_ASR: (STAGE_ALIGN,),
    STAGE_ALIGN: (),
}

#: Successor -> stages that must already be ``done`` before it may be claimed.
#: This is the fan-in join: ``align`` runs once, only after both parallel
#: siblings finished, and only from whichever of them finished second.
_STAGE_JOIN_DEPS: dict[str, tuple[str, ...]] = {
    STAGE_ALIGN: (STAGE_SPEAKER_EMBED, STAGE_ASR),
}

#: Absolute per-stage deadlines. A slow-but-healthy job that exceeds one of
#: these is failed as a retryable error rather than left to run forever; that is
#: the deliberate trade for never needing a progress watchdog.
_STAGE_DEFAULT_TIMEOUT_SEC: dict[str, float] = {
    STAGE_FFMPEG_DECODE: 600.0,
    STAGE_DIARIZE: 3600.0,
    STAGE_SPEAKER_EMBED: 600.0,
    STAGE_ASR: 7200.0,
    STAGE_ALIGN: 1800.0,
}

#: Overrides every stage deadline at process start (slow hardware, tests, ops).
STAGE_TIMEOUT_ENV = "OLIVE_STAGE_TIMEOUT_SEC"


def _stage_timeout_sec(stage: str) -> float:
    raw = (os.environ.get(STAGE_TIMEOUT_ENV) or "").strip()
    if raw:
        try:
            override = float(raw)
        except ValueError:
            override = 0.0
        if math.isfinite(override) and override > 0:
            return override
    return _STAGE_DEFAULT_TIMEOUT_SEC.get(stage, 1800.0)


@dataclass
class TaskExecutionContext:
    """Per-run state shared by a run's stage steps (pipeline, progress, clock).

    Local stages are separate workflows with no live caller, so this is created
    lazily by :func:`_ensure_task_context` from the process-wide recovery hooks,
    or registered up-front by the service when it has a pipeline to inject.
    """

    task_id: str
    meeting_id: str | None = None
    signal: Any = None
    on_progress: Callable[[dict[str, Any]], Any] | None = None
    pipeline: LocalTranscriptionPipeline | None = None
    pipeline_options: dict[str, Any] | None = None
    db: Engine | None = None
    now_fn: Callable[[], int] | None = None
    workflow_id: str | None = None


_TASK_CONTEXTS: dict[str, TaskExecutionContext] = {}
_GENERIC_JOBS: dict[str, Callable[[], Any]] = {}


def register_task_context(ctx: TaskExecutionContext) -> None:
    _TASK_CONTEXTS[ctx.task_id] = ctx


def unregister_task_context(task_id: str) -> None:
    _TASK_CONTEXTS.pop(task_id, None)


def get_task_context(task_id: str) -> TaskExecutionContext | None:
    return _TASK_CONTEXTS.get(task_id)


_RECOVERY_DB: Engine | None = None
_RECOVERY_PROGRESS_SINK: Callable[[str, dict[str, Any]], Any] | None = None
_RECOVERY_COMPLETE_SINK: Callable[[str, dict[str, Any], str | None], Any] | None = None
_RECOVERY_CONFIG_DIR: str | None = None
_RECOVERY_MEETINGS_DIR: str | None = None
_RECOVERY_NOW_FN: Callable[[], int] | None = None


def configure_recovery_runtime(
    db: Engine | None,
    progress_sink: Callable[[str, dict[str, Any]], Any] | None,
    complete_sink: Callable[[str, dict[str, Any], str | None], Any] | None = None,
    *,
    config_dir: str | None = None,
    meetings_dir: str | None = None,
    now_fn: Callable[[], int] | None = None,
) -> None:
    """Set process-wide hooks the flat stage workflows run against.

    Stages are fire-and-forget: there is no live caller holding a
    :class:`TaskExecutionContext`, so progress, persistence and clock access all
    come from these process-wide hooks instead of a per-request closure.
    """
    global _RECOVERY_DB, _RECOVERY_PROGRESS_SINK, _RECOVERY_COMPLETE_SINK
    global _RECOVERY_CONFIG_DIR, _RECOVERY_MEETINGS_DIR, _RECOVERY_NOW_FN
    _RECOVERY_DB = db
    _RECOVERY_PROGRESS_SINK = progress_sink
    if now_fn is not None:
        _RECOVERY_NOW_FN = now_fn
    if complete_sink is not None:
        _RECOVERY_COMPLETE_SINK = complete_sink
    if config_dir is not None:
        _RECOVERY_CONFIG_DIR = config_dir
    if meetings_dir is not None:
        _RECOVERY_MEETINGS_DIR = meetings_dir


def _now_ms() -> int:
    now_fn = _RECOVERY_NOW_FN
    if callable(now_fn):
        try:
            return int(now_fn())
        except Exception:
            pass
    return int(time.time() * 1000)


def local_run_workflow_prefix(meeting_id: str, task_id: str) -> str:
    """Deterministic workflow-id prefix for one local transcription run.

    Every stage id starts here (``<prefix>:<stage>``), which is what makes
    ``cancel_dbos_workflows_for_meeting``, the progress map and restart recovery
    able to find a run's stages without a parent workflow. The run id is unique
    per enqueue, so a forced re-transcription never collides with (and is never
    silently deduplicated onto) a previous run's completed workflows.
    """
    return f"stage:{meeting_id}:{task_id[:8]}"


def local_stage_workflow_id(meeting_id: str, task_id: str, stage: str) -> str:
    return f"{local_run_workflow_prefix(meeting_id, task_id)}:{stage}"


async def _stage_samples(payload: dict[str, Any], target_sr: int) -> Any:
    """Decode the run's audio for one stage, straight from the recording.

    Stages do not share a decoded WAV: it would live from decode until align, so
    a large bulk backlog would accumulate gigabytes of intermediate audio (a
    1-hour meeting is ~115 MB of 16 kHz mono). ffmpeg decoding is cheap next to
    diarization/ASR, and decoding per stage keeps every stage independently
    resumable — no checkpoint needs to carry audio.
    """
    audio_path = payload.get("audioPath")
    if not audio_path:
        raise ValueError("Local transcription payload has no audioPath")
    return await load_audio_samples({"audioPath": audio_path, "enhance": False}, target_sr)


class _InactiveLocalRunError(RuntimeError):
    """A cancelled or superseded run can no longer read or write checkpoints."""


def _stage_checkpoint_run_id(payload: dict[str, Any]) -> str:
    return str(payload["taskId"])


def _read_stage_checkpoint(payload: dict[str, Any], stage: str) -> dict[str, Any]:
    """Load one stage's output for this run (raises if it belongs to an older run).

    The checkpoint table is the hand-off medium between flat stage workflows:
    they have no shared parent to carry results, so durable state is the only way
    a stage can consume its predecessor's output. It lives in the application
    database rather than the meeting folder so it survives a folder rename,
    stays out of backups once the run ends, and is visible/enumerable by the DB.
    """
    db = _RECOVERY_DB
    if db is None:
        raise RuntimeError("Local transcription requires the application database")
    meeting_id = str(payload.get("meetingId") or "")
    run_id = _stage_checkpoint_run_id(payload)
    with db.connect() as conn:
        row = (
            conn.execute(
                text("SELECT run_id, doc FROM stage_checkpoints WHERE meeting_id = :mid AND stage = :stage"),
                {"mid": meeting_id, "stage": stage},
            )
            .mappings()
            .first()
        )
        if row is None or str(row["run_id"]) != run_id:
            active = conn.execute(
                text(
                    "SELECT 1 FROM stage_runs WHERE meeting_id = :mid AND stage = :stage"
                    " AND provider_job_id = :entry AND status = 'running'"
                ),
                {
                    "mid": meeting_id,
                    "stage": STAGE_LOCAL_TRANSCRIBE,
                    "entry": local_run_entry_workflow_id(meeting_id, run_id),
                },
            ).scalar()
            message = f"Missing '{stage}' checkpoint for this transcription run"
            if not active:
                raise _InactiveLocalRunError(message + " (cancelled or superseded run)")
            raise RuntimeError(message)
    doc = json.loads(str(row["doc"]))
    if not isinstance(doc, dict):
        raise ValueError(f"Corrupt '{stage}' checkpoint")
    return doc


def _write_stage_checkpoint(payload: dict[str, Any], stage: str, doc: dict[str, Any]) -> None:
    """Persist output only while this run owns the meeting's active run row."""
    db = _RECOVERY_DB
    if db is None:
        raise RuntimeError("Local transcription requires the application database")
    meeting_id = str(payload.get("meetingId") or "")
    run_id = _stage_checkpoint_run_id(payload)
    with db.begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO stage_checkpoints (meeting_id, stage, run_id, doc, updated_at)
                SELECT :mid, :stage, :run, :doc, :now
                WHERE EXISTS (
                  SELECT 1 FROM stage_runs
                  WHERE meeting_id = :mid AND stage = :run_stage
                    AND provider_job_id = :entry_id AND status = 'running'
                )
                ON CONFLICT(meeting_id, stage) DO UPDATE SET
                  run_id = excluded.run_id,
                  doc = excluded.doc,
                  updated_at = excluded.updated_at
                """
            ),
            {
                "mid": meeting_id,
                "stage": stage,
                "run": run_id,
                "doc": json.dumps(doc),
                "now": _now_ms(),
                "run_stage": STAGE_LOCAL_TRANSCRIBE,
                "entry_id": local_run_entry_workflow_id(meeting_id, run_id),
            },
        )
        if result.rowcount != 1:
            raise _InactiveLocalRunError("Transcription cancelled or superseded before checkpoint write")


def clear_local_stage_checkpoints(meeting_id: str | None, *, task_id: str) -> None:
    """Drop only this run's outputs, never those of a newer transcription."""
    db = _RECOVERY_DB
    if db is None or not meeting_id:
        return
    try:
        with db.begin() as conn:
            conn.execute(
                text("DELETE FROM stage_checkpoints WHERE meeting_id = :mid AND run_id = :run"),
                {"mid": meeting_id, "run": task_id},
            )
    except Exception as exc:
        logger.warn(
            "Failed to clear local transcription checkpoints",
            {"category": "transcription", "meetingId": meeting_id, "error": str(exc)},
        )


_STAGE_TERMINAL_STATUSES = ("done", "error", "cancelled")


def _mark_local_stage(
    meeting_id: str | None,
    stage: str,
    status: str,
    *,
    error: str | None = None,
    job_id: str | None = None,
) -> None:
    """Record one stage's lifecycle in ``stage_runs`` (+ meeting status/error).

    Called from workflow bodies, not steps: DBOS replays a workflow body on
    recovery, and these writes are idempotent upserts, so replay converges on
    the same state instead of being skipped as a cached step result.
    """
    db = _RECOVERY_DB
    if db is None or not meeting_id:
        return
    now = _now_ms()
    finished = now if status in _STAGE_TERMINAL_STATUSES else None
    attempts_inc = 1 if status == "running" else 0
    # Guard the upsert by the stage's own workflow id so a cancelled/superseded
    # run cannot write over the rows a newer run just reset. The first "running"
    # write of a stage legitimately stamps a row that has no id yet; terminal
    # writes must match the id this run claimed.
    guard = "1 = 1"
    if job_id:
        if status == "running":
            guard = (
                "(stage_runs.provider_job_id IS NULL OR stage_runs.provider_job_id = :jid)"
                " AND stage_runs.status != 'done'"
            )
        else:
            guard = "stage_runs.provider_job_id = :jid"
    try:
        with db.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO stage_runs (
                      id, meeting_id, stage, status, provider_job_id, attempts,
                      last_error, started_at, finished_at, created_at, updated_at
                    ) VALUES (
                      :id, :mid, :stage, :st, :jid, :att,
                      :err, :now, :fin, :now, :now
                    )
                    ON CONFLICT(meeting_id, stage) DO UPDATE SET
                      status = excluded.status,
                      provider_job_id = COALESCE(excluded.provider_job_id, stage_runs.provider_job_id),
                      attempts = stage_runs.attempts + :inc,
                      last_error = excluded.last_error,
                      started_at = CASE
                        WHEN excluded.status = 'running' THEN excluded.started_at
                        ELSE stage_runs.started_at END,
                      finished_at = excluded.finished_at,
                      updated_at = excluded.updated_at
                    WHERE {guard}
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "mid": meeting_id,
                    "stage": stage,
                    "st": status,
                    "jid": job_id,
                    "att": attempts_inc,
                    "inc": attempts_inc,
                    "err": error,
                    "now": now,
                    "fin": finished,
                },
            )
            if status == "running":
                conn.execute(
                    text(
                        """
                        UPDATE meetings
                        SET status = 'processing', last_error = NULL, updated_at = :now
                        WHERE id = :mid
                          AND primary_transcript_artifact_id IS NULL
                          AND status != 'processing'
                        """
                    ),
                    {"now": now, "mid": meeting_id},
                )
            elif status == "error":
                conn.execute(
                    text("UPDATE meetings SET last_error = :err, updated_at = :now WHERE id = :mid"),
                    {"err": error, "now": now, "mid": meeting_id},
                )
            elif status == "cancelled":
                existing = (
                    conn.execute(
                        text("SELECT primary_transcript_artifact_id FROM meetings WHERE id = :id"),
                        {"id": meeting_id},
                    )
                    .mappings()
                    .first()
                )
                next_status = (
                    "ready" if (existing and existing.get("primary_transcript_artifact_id")) else "pending"
                )
                conn.execute(
                    text(
                        """
                        UPDATE meetings SET status = :st, last_error = NULL, updated_at = :now
                        WHERE id = :mid AND status = 'processing'
                        """
                    ),
                    {"st": next_status, "now": now, "mid": meeting_id},
                )
    except Exception as exc:
        logger.warn(
            "Failed to record local transcription stage state",
            {"category": "transcription", "meetingId": meeting_id, "stage": stage, "error": str(exc)},
        )


def _stage_status(meeting_id: str | None, stage: str) -> str | None:
    db = _RECOVERY_DB
    if db is None or not meeting_id:
        return None
    try:
        with db.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT status FROM stage_runs WHERE meeting_id = :mid AND stage = :stage"),
                    {"mid": meeting_id, "stage": stage},
                )
                .mappings()
                .first()
            )
        return str(row["status"]) if row else None
    except Exception:
        return None


def _reset_local_stage_runs(meeting_id: str, *, job_id: str) -> None:
    """Reset stage rows and checkpoints atomically for a fresh run."""
    db = _RECOVERY_DB
    if db is None or not meeting_id:
        return
    now = _now_ms()
    try:
        with db.begin() as conn:
            conn.execute(
                text("DELETE FROM stage_checkpoints WHERE meeting_id = :mid"),
                {"mid": meeting_id},
            )
            for stage in STAGE_SEQUENCE:
                conn.execute(
                    text(
                        """
                        INSERT INTO stage_runs (
                          id, meeting_id, stage, status, provider_job_id, attempts,
                          last_error, started_at, finished_at, created_at, updated_at
                        ) VALUES (
                          :id, :mid, :stage, 'pending', NULL, 0,
                          NULL, NULL, NULL, :now, :now
                        )
                        ON CONFLICT(meeting_id, stage) DO UPDATE SET
                          status = 'pending', provider_job_id = NULL, last_error = NULL,
                          started_at = NULL, finished_at = NULL, updated_at = excluded.updated_at
                        """
                    ),
                    {"id": str(uuid.uuid4()), "mid": meeting_id, "stage": stage, "now": now},
                )
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
                    ON CONFLICT(meeting_id, stage) DO UPDATE SET
                      status = 'running', provider_job_id = excluded.provider_job_id,
                      attempts = stage_runs.attempts + 1, last_error = NULL,
                      started_at = excluded.started_at, finished_at = NULL,
                      updated_at = excluded.updated_at
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "mid": meeting_id,
                    "stage": STAGE_LOCAL_TRANSCRIBE,
                    "jid": job_id,
                    "now": now,
                },
            )
    except Exception as exc:
        logger.warn(
            "Failed to reset local transcription stage runs",
            {"category": "transcription", "meetingId": meeting_id, "error": str(exc)},
        )
        raise


def _ensure_task_context(payload: dict[str, Any]) -> bool:
    """Register a context for a step whose original caller is gone (e.g. after a restart).

    Returns True when this call created the context and the workflow must unregister it.
    """
    task_id = str(payload["taskId"])
    if task_id in _TASK_CONTEXTS:
        return False
    meeting_id = payload.get("meetingId")
    sink = _RECOVERY_PROGRESS_SINK
    on_progress = (lambda upd: sink(str(meeting_id), upd)) if sink and meeting_id else None
    wf_id = getattr(DBOS, "workflow_id", None)
    register_task_context(
        TaskExecutionContext(
            task_id=task_id,
            meeting_id=str(meeting_id) if meeting_id else None,
            on_progress=on_progress,
            db=_RECOVERY_DB,
            workflow_id=str(wf_id) if wf_id else None,
        )
    )
    return True


def _is_signal_aborted(signal: Any) -> bool:
    if signal is None:
        return False
    if hasattr(signal, "aborted"):
        return bool(signal.aborted)
    if hasattr(signal, "is_set") and callable(signal.is_set):
        return bool(signal.is_set())
    return False


def _check_aborted(payload: dict[str, Any], stage: str) -> None:
    """Raise only for an explicit abort.

    Two abort sources are honoured: the caller's abort signal (synchronous
    paths) and the stage's own ``stage_runs`` row being ``cancelled``. The row
    check is what makes a user cancel — or a failed sibling — stop long stage
    loops between segments without any cross-workflow API call.

    Server shutdown deliberately does *not* raise here: a graceful stop cancels
    the workflow task, DBOS leaves the workflow PENDING, and the next launch
    resumes it from its checkpoint. Raising a "cancelled" error on shutdown
    would record a terminal stage and destroy the run instead.
    """
    task_id = str(payload.get("taskId") or "")
    ctx = _TASK_CONTEXTS.get(task_id)
    if ctx is not None and _is_signal_aborted(ctx.signal):
        raise RuntimeError("Transcription cancelled by user")
    # Read the row from the payload rather than the (shared, per-run) context:
    # a failed sibling may have released the context while this stage still runs.
    meeting_id = payload.get("meetingId")
    if meeting_id and _stage_status(str(meeting_id), stage) == "cancelled":
        raise RuntimeError("Transcription cancelled by user")


async def _emit_progress(task_id: str, payload: dict[str, Any]) -> None:
    ctx = _TASK_CONTEXTS.get(task_id)
    if ctx is not None and callable(ctx.on_progress):
        res = ctx.on_progress(payload)
        if inspect.isawaitable(res):
            await res


def _get_pipeline(task_id: str, pipeline_options: dict[str, Any] | None) -> LocalTranscriptionPipeline:
    """Return the run's pipeline, creating (and caching) it on first use.

    One pipeline per run means one set of native engine pools per run, reused by
    every stage; the service may pre-register one (tests/alternate engines).
    """
    ctx = _TASK_CONTEXTS.get(task_id)
    if ctx is not None and ctx.pipeline is not None:
        return ctx.pipeline
    pipe = LocalTranscriptionPipeline(pipeline_options)
    if ctx is not None:
        ctx.pipeline = pipe
    return pipe


@DBOS.step(timeout_seconds=_stage_timeout_sec(STAGE_FFMPEG_DECODE), preemptible=True)
async def step_decode_audio(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload["taskId"])
    _ensure_task_context(payload)
    meeting_id = payload.get("meetingId")

    _check_aborted(payload, STAGE_FFMPEG_DECODE)

    await _emit_progress(
        task_id,
        {
            "stage": "decoding",
            "percent": 5,
            "message": "Decoding 16kHz audio samples...",
        },
    )

    audio_path = payload.get("audioPath")
    if not audio_path:
        raise ValueError("Local transcription payload has no audioPath")

    decode_t0 = time.perf_counter()
    target_sr = 16000
    decoded = await load_audio_samples({"audioPath": audio_path, "enhance": False}, target_sr)

    logger.info(
        "Decoded 16kHz mono audio samples for local transcription",
        {
            "category": "transcription",
            "meetingId": meeting_id,
            "audioDurationMs": decoded.duration_ms,
            "sampleCount": int(len(decoded.samples)),
            "elapsedMs": int(round((time.perf_counter() - decode_t0) * 1000)),
        },
    )
    doc = {
        "durationMs": decoded.duration_ms,
        "sampleRate": target_sr,
    }
    await asyncio.to_thread(_write_stage_checkpoint, payload, STAGE_FFMPEG_DECODE, doc)
    return doc


@DBOS.step(timeout_seconds=_stage_timeout_sec(STAGE_DIARIZE), preemptible=True)
async def step_diarize_turns(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload["taskId"])
    _ensure_task_context(payload)
    meeting_id = payload.get("meetingId")
    decode_info = _read_stage_checkpoint(payload, STAGE_FFMPEG_DECODE)
    duration_ms = int(decode_info["durationMs"])
    target_sr = int(decode_info.get("sampleRate") or 16000)

    _check_aborted(payload, STAGE_DIARIZE)

    await _emit_progress(
        task_id,
        {
            "stage": "diarizing",
            "percent": 15,
            "totalMs": duration_ms,
            "message": "Diarization (1/4): Initializing neural speaker segmentation...",
        },
    )

    model_id = payload.get("modelId")
    turns: list[dict[str, Any]] = []
    precomputed_centroids: dict[str, list[float]] | None = None

    if model_id in ("mock", "mock-asr"):
        # Deterministic single-turn stand-in: skip real neural diarization.
        mock_speaker = "SPEAKER_00"
        turns = [{"startMs": 0, "endMs": duration_ms, "speakerId": mock_speaker}]
        precomputed_centroids = {mock_speaker: normalize_vector([1.0] * 256)}
    else:
        samples = (await _stage_samples(payload, target_sr)).samples

        pipe = _get_pipeline(task_id, payload.get("pipelineOptions"))
        clustering_threshold = payload.get("clusteringThreshold")
        expected_speaker_count = payload.get("expectedSpeakerCount")

        ctx = _TASK_CONTEXTS.get(task_id)
        on_prog = ctx.on_progress if ctx is not None else None

        diarizer = pipe.diarizer
        seg_turns_fn = getattr(diarizer, "segment_turns", None)
        if callable(seg_turns_fn):
            seg_res: Any = seg_turns_fn(
                samples,
                target_sr,
                clustering_threshold,
                expected_speaker_count,
                on_progress=on_prog,
                meeting_id=meeting_id,
            )
            turns = await seg_res if inspect.isawaitable(seg_res) else seg_res
        else:
            diarize_fn = diarizer.diarize
            diarize_kwargs: dict[str, Any] = {}
            try:
                sig = inspect.signature(diarize_fn)
                if "on_progress" in sig.parameters:
                    diarize_kwargs["on_progress"] = on_prog
                if "meeting_id" in sig.parameters:
                    diarize_kwargs["meeting_id"] = meeting_id
            except Exception:
                pass
            speaker_segments = await diarize_fn(
                samples,
                target_sr,
                clustering_threshold,
                expected_speaker_count,
                **diarize_kwargs,
            )
            emb_dim = getattr(pipe.embedding_extractor, "dim", 256)
            cluster_map: dict[str, list[Any]] = {}
            for seg in speaker_segments:
                cid = getattr(seg, "speakerId", None) or "Speaker 1"
                s_ms = int(getattr(seg, "startMs", 0))
                e_ms = int(getattr(seg, "endMs", 0))
                turns.append({"startMs": s_ms, "endMs": e_ms, "speakerId": cid})
                cluster_map.setdefault(cid, []).append(seg)
            precomputed_centroids = {}
            for cid, segs in cluster_map.items():
                precomputed_centroids[cid] = merge_weighted_voiceprint_vectors(
                    [
                        {
                            "vector": getattr(s, "embedding", None) or ([0.0] * emb_dim),
                            "weight": max(1, int(getattr(s, "endMs", 0)) - int(getattr(s, "startMs", 0))),
                        }
                        for s in segs
                    ]
                )

        if pipe.unload_per_stage and hasattr(diarizer, "release"):
            diarizer.release()

    _check_aborted(payload, STAGE_DIARIZE)

    diar_doc: dict[str, Any] = {
        "turns": turns,
        "durationMs": duration_ms,
    }
    if precomputed_centroids is not None:
        diar_doc["precomputedCentroids"] = precomputed_centroids
    await asyncio.to_thread(_write_stage_checkpoint, payload, STAGE_DIARIZE, diar_doc)

    return {
        "turnCount": len(turns),
        "durationMs": duration_ms,
    }


@DBOS.step(timeout_seconds=_stage_timeout_sec(STAGE_SPEAKER_EMBED), preemptible=True)
async def step_embed_and_match_speakers(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload["taskId"])
    _ensure_task_context(payload)
    meeting_id = payload.get("meetingId")
    decode_info = _read_stage_checkpoint(payload, STAGE_FFMPEG_DECODE)
    diar_doc = _read_stage_checkpoint(payload, STAGE_DIARIZE)
    duration_ms = int(decode_info["durationMs"])
    target_sr = int(decode_info.get("sampleRate") or 16000)

    _check_aborted(payload, STAGE_SPEAKER_EMBED)

    turns: list[dict[str, Any]] = diar_doc.get("turns") or []
    precomputed_centroids: dict[str, VoiceprintVector] | None = diar_doc.get("precomputedCentroids")

    pipe = _get_pipeline(task_id, payload.get("pipelineOptions"))
    ctx = _TASK_CONTEXTS.get(task_id)
    on_prog = ctx.on_progress if ctx is not None else None

    if precomputed_centroids is not None:
        cluster_centroids = precomputed_centroids
    else:
        samples = (await _stage_samples(payload, target_sr)).samples
        extract_vp_fn = getattr(pipe.diarizer, "extract_cluster_voiceprints", None)
        if callable(extract_vp_fn):
            vp_res: Any = extract_vp_fn(
                samples,
                turns,
                target_sr,
                on_progress=on_prog,
                meeting_id=meeting_id,
            )
            _, cluster_centroids = await vp_res if inspect.isawaitable(vp_res) else vp_res
        else:
            cluster_centroids = {}

    if pipe.unload_per_stage and hasattr(pipe.embedding_extractor, "release"):
        pipe.embedding_extractor.release()

    _check_aborted(payload, STAGE_SPEAKER_EMBED)

    total_segments = len(turns)
    await _emit_progress(
        task_id,
        {
            "stage": "diarizing",
            "percent": 39,
            "total": total_segments,
            "totalMs": duration_ms,
            "message": f"Identified {total_segments} speech segments across clusters. Matching enrolled profiles...",
        },
    )

    all_profiles: list[dict[str, Any]] = []
    raw_candidates = payload.get("candidateSpeakers")
    normalized_candidates = (
        {str(c).strip().lower() for c in raw_candidates if str(c).strip()} if raw_candidates else None
    )

    raw_enrolled = payload.get("enrolledSpeakers") or []
    for spk_raw in raw_enrolled:
        spk = spk_raw if isinstance(spk_raw, EnrolledSpeaker) else EnrolledSpeaker.model_validate(spk_raw)
        local_ids = spk.providerIds.get("local") or []
        trusted_vectors: list[VoiceprintVector] = []
        for item in local_ids:
            try:
                parsed = json.loads(item) if isinstance(item, str) else item
                if isinstance(parsed, list) and len(parsed) > 0:
                    if isinstance(parsed[0], list):
                        for sub in parsed:
                            if isinstance(sub, list) and len(sub) > 0:
                                trusted_vectors.append(normalize_vector(sub))
                    else:
                        trusted_vectors.append(normalize_vector(parsed))
            except Exception:
                pass
        if trusted_vectors:
            all_profiles.append(
                {
                    "id": spk.id,
                    "name": spk.name,
                    "trustedVectors": trusted_vectors,
                }
            )

    candidate_profiles = (
        [
            p
            for p in all_profiles
            if p["id"].lower() in normalized_candidates or p["name"].strip().lower() in normalized_candidates
        ]
        if normalized_candidates is not None
        else all_profiles
    )

    opt_sim = payload.get("similarityThreshold")
    opt_margin = payload.get("decisionMargin")
    effective_threshold = float(opt_sim if opt_sim is not None else pipe.similarity_threshold)
    effective_margin = float(opt_margin if opt_margin is not None else pipe.decision_margin)

    cluster_to_resolved_name: dict[str, str] = {}
    discovered_map: dict[str, DiscoveredSpeakerVoiceprint] = {}
    candidate_matches: list[dict[str, Any]] = []
    cluster_diagnostics: list[dict[str, Any]] = []

    for cid, centroid in cluster_centroids.items():
        speaker_scores: list[tuple[dict[str, Any], float]] = []
        for profile in candidate_profiles:
            score = score_against_trusted_vectors(centroid, profile["trustedVectors"])
            speaker_scores.append((profile, score))
        speaker_scores.sort(key=lambda pair: pair[1], reverse=True)

        if speaker_scores:
            top1_prof, top1_score = speaker_scores[0]
            top2_prof = speaker_scores[1][0] if len(speaker_scores) > 1 else None
            top2_score = speaker_scores[1][1] if len(speaker_scores) > 1 else -1.0
            meets_threshold = top1_score >= effective_threshold
            meets_margin = (top1_score - top2_score >= effective_margin) if len(speaker_scores) > 1 else True
            cluster_diagnostics.append(
                {
                    "cluster": cid,
                    "topCandidate": top1_prof["name"],
                    "topScore": round(float(top1_score), 4),
                    "secondCandidate": top2_prof["name"] if top2_prof else None,
                    "secondScore": round(float(top2_score), 4) if top2_prof else None,
                    "threshold": effective_threshold,
                    "margin": effective_margin,
                    "meetsThreshold": meets_threshold,
                    "meetsMargin": meets_margin,
                }
            )
            if meets_threshold and meets_margin:
                candidate_matches.append(
                    {
                        "clusterId": cid,
                        "profile": top1_prof,
                        "score": top1_score,
                        "secondScore": top2_score,
                    }
                )

    candidate_matches.sort(key=lambda m: m["score"], reverse=True)
    assigned_clusters: set[str] = set()
    assigned_speaker_ids: set[str] = set()

    for match in candidate_matches:
        cid = match["clusterId"]
        prof = match["profile"]
        if cid not in assigned_clusters and prof["id"] not in assigned_speaker_ids:
            assigned_clusters.add(cid)
            assigned_speaker_ids.add(prof["id"])
            cluster_to_resolved_name[cid] = prof["name"]
            discovered_map[prof["name"]] = DiscoveredSpeakerVoiceprint(
                speakerId=prof["id"],
                name=prof["name"],
                voiceprint=cluster_centroids.get(cid) or prof["trustedVectors"][0],
                isEnrolled=True,
                similarityScore=float(match["score"]),
            )

    for cid, centroid in cluster_centroids.items():
        if cid not in cluster_to_resolved_name:
            cluster_to_resolved_name[cid] = cid
            discovered_map[cid] = DiscoveredSpeakerVoiceprint(
                speakerId=cid,
                name=cid,
                voiceprint=centroid,
                isEnrolled=False,
            )

    logger.info(
        "Resolved speaker clusters against enrolled voiceprints",
        {
            "category": "transcription",
            "meetingId": meeting_id,
            "enrolledProfiles": len(candidate_profiles),
            "resolvedClusters": cluster_to_resolved_name,
            "clusterScores": cluster_diagnostics,
            "matchedSpeakers": [
                {
                    "cluster": m["clusterId"],
                    "speaker": m["profile"]["name"],
                    "score": round(float(m["score"]), 4),
                    "secondScore": round(float(m["secondScore"]), 4),
                }
                for m in candidate_matches
            ],
        },
    )

    discovered_dicts = [d.to_dict() for d in discovered_map.values()]
    clusters_doc = {
        "clusterToResolvedName": cluster_to_resolved_name,
        "discoveredSpeakers": discovered_dicts,
        "clusterDiagnostics": cluster_diagnostics,
    }
    await asyncio.to_thread(_write_stage_checkpoint, payload, STAGE_SPEAKER_EMBED, clusters_doc)
    return {
        "clusterToResolvedName": cluster_to_resolved_name,
        "discoveredSpeakers": discovered_dicts,
    }


@DBOS.step(timeout_seconds=_stage_timeout_sec(STAGE_ASR), preemptible=True)
async def step_transcribe_asr(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload["taskId"])
    _ensure_task_context(payload)
    meeting_id = payload.get("meetingId")
    decode_info = _read_stage_checkpoint(payload, STAGE_FFMPEG_DECODE)
    diar_doc = _read_stage_checkpoint(payload, STAGE_DIARIZE)
    duration_ms = int(decode_info["durationMs"])
    target_sr = int(decode_info.get("sampleRate") or 16000)

    _check_aborted(payload, STAGE_ASR)

    turns: list[dict[str, Any]] = diar_doc.get("turns") or []
    total_segments = len(turns)

    samples = (await _stage_samples(payload, target_sr)).samples

    pipe = _get_pipeline(task_id, payload.get("pipelineOptions"))
    language = str(payload.get("language") or "en")
    model_id = payload.get("modelId")

    if model_id in ("mock", "mock-asr"):
        for _ in range(4):
            _check_aborted(payload, STAGE_ASR)
            await asyncio.sleep(0.02)

    use_whole_file_align = bool(
        payload.get("wholeFileAsr")
        and isinstance(pipe.asr_engine, CohereAsrEngine)
        and model_id not in ("mock", "mock-asr")
        and pipe.aligner is not None
        and turns
    )

    if use_whole_file_align:
        await _emit_progress(
            task_id,
            {
                "stage": "transcribing",
                "percent": 50,
                "current": 1,
                "total": total_segments,
                "totalMs": duration_ms,
                "message": "Transcribing audio with Cohere Transcribe...",
            },
        )
        full_asr = await pipe._call_asr_segment(
            samples,
            target_sr,
            {"language": language, "startMs": 0, "modelId": model_id},
        )
        if pipe.unload_per_stage and hasattr(pipe.asr_engine, "release"):
            pipe.asr_engine.release()

        asr_doc: dict[str, Any] = {
            "mode": "whole_file",
            "fullText": str(full_asr.get("text") or "").strip(),
        }
    else:
        raw_segments_out: list[dict[str, Any]] = []
        for i, turn in enumerate(turns):
            _check_aborted(payload, STAGE_ASR)
            current_num = i + 1
            pct = min(95, int(round(40 + ((i + 0.5) / max(1, total_segments)) * 55)))
            cid = str(turn.get("speakerId") or "Speaker 1")
            s_start = int(turn.get("startMs") or 0)
            s_end = int(turn.get("endMs") or 0)
            start_sample = max(0, int((s_start / 1000.0) * target_sr))
            end_sample = min(len(samples), int((s_end / 1000.0) * target_sr))
            s_samples = samples[start_sample:end_sample]

            await _emit_progress(
                task_id,
                {
                    "stage": "transcribing",
                    "percent": pct,
                    "current": current_num,
                    "total": total_segments,
                    "currentMs": s_start,
                    "totalMs": duration_ms,
                    "speaker": cid,
                    "message": f"Transcribing segment {current_num} of {total_segments} ({pct}%)",
                },
            )

            seg_t0 = time.perf_counter()
            asr_res = await pipe._call_asr_segment(
                s_samples,
                target_sr,
                {
                    "language": language,
                    "startMs": s_start,
                    "modelId": model_id,
                },
            )
            seg_elapsed_ms = int(round((time.perf_counter() - seg_t0) * 1000))
            seg_text = str(asr_res.get("text") or "")
            logger.debug(
                f"Transcribed segment {current_num}/{total_segments}",
                {
                    "category": "transcription",
                    "meetingId": meeting_id,
                    "segment": current_num,
                    "totalSegments": total_segments,
                    "speaker": cid,
                    "startMs": s_start,
                    "endMs": s_end,
                    "segmentDurationMs": max(0, s_end - s_start),
                    "chars": len(seg_text),
                    "elapsedMs": seg_elapsed_ms,
                },
            )
            raw_words = asr_res.get("words")
            serialized_words: list[dict[str, Any]] | None = None
            if isinstance(raw_words, list) and raw_words:
                serialized_words = [
                    w.model_dump()
                    if isinstance(w, TranscriptWord)
                    else TranscriptWord.model_validate(w).model_dump()
                    for w in raw_words
                ]
            raw_segments_out.append(
                {
                    "startMs": s_start,
                    "endMs": s_end,
                    "speakerId": cid,
                    "text": seg_text,
                    "words": serialized_words,
                }
            )

        if pipe.unload_per_stage and hasattr(pipe.asr_engine, "release"):
            pipe.asr_engine.release()

        asr_doc = {
            "mode": "segments",
            "segments": raw_segments_out,
        }

    await asyncio.to_thread(_write_stage_checkpoint, payload, STAGE_ASR, asr_doc)
    return {
        "mode": asr_doc["mode"],
    }


@DBOS.step(timeout_seconds=_stage_timeout_sec(STAGE_ALIGN), preemptible=True)
async def step_align_and_finalize(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload["taskId"])
    _ensure_task_context(payload)
    meeting_id = payload.get("meetingId")
    decode_info = _read_stage_checkpoint(payload, STAGE_FFMPEG_DECODE)
    diar_doc = _read_stage_checkpoint(payload, STAGE_DIARIZE)
    embed_doc = _read_stage_checkpoint(payload, STAGE_SPEAKER_EMBED)
    asr_doc = _read_stage_checkpoint(payload, STAGE_ASR)
    duration_ms = int(decode_info["durationMs"])
    target_sr = int(decode_info.get("sampleRate") or 16000)
    language = str(payload.get("language") or "en")

    _check_aborted(payload, STAGE_ALIGN)

    turns: list[dict[str, Any]] = diar_doc.get("turns") or []
    cluster_to_resolved_name: dict[str, str] = embed_doc.get("clusterToResolvedName") or {}
    discovered_speakers: list[dict[str, Any]] = embed_doc.get("discoveredSpeakers") or []
    pipe = _get_pipeline(task_id, payload.get("pipelineOptions"))
    final_segments: list[TranscriptSegment] = []

    if asr_doc.get("mode") == "whole_file":
        total_segments = len(turns)
        await _emit_progress(
            task_id,
            {
                "stage": "transcribing",
                "percent": 85,
                "current": total_segments,
                "total": total_segments,
                "totalMs": duration_ms,
                "message": "Running forced alignment for word timestamps...",
            },
        )
        decoded = await _stage_samples(payload, target_sr)
        assert pipe.aligner is not None
        aligned_words = await pipe.aligner.align(
            decoded.samples,
            target_sr,
            str(asr_doc.get("fullText") or ""),
            language=language,
            offset_ms=0,
        )
        if pipe.unload_per_stage and hasattr(pipe.aligner, "release"):
            pipe.aligner.release()

        class _TurnObj:
            def __init__(self, d: dict[str, Any]) -> None:
                self.startMs = int(d.get("startMs") or 0)
                self.endMs = int(d.get("endMs") or 0)
                self.speakerId = str(d.get("speakerId") or "Speaker 1")

        turn_objs = [_TurnObj(t) for t in turns]
        final_segments = pipe._assign_words_to_speaker_turns(
            aligned_words, turn_objs, cluster_to_resolved_name
        )
    else:
        for seg_raw in asr_doc.get("segments") or []:
            s_start = int(seg_raw.get("startMs") or 0)
            s_end = int(seg_raw.get("endMs") or 0)
            cid = str(seg_raw.get("speakerId") or "Speaker 1")
            resolved_speaker = cluster_to_resolved_name.get(cid, cid)
            seg_text = str(seg_raw.get("text") or "")
            raw_words = seg_raw.get("words")
            words: list[TranscriptWord] | None = None
            if isinstance(raw_words, list) and raw_words:
                words = [
                    TranscriptWord(
                        startMs=int(w["startMs"]),
                        endMs=int(w["endMs"]),
                        word=str(w["word"]),
                        speaker=resolved_speaker,
                        confidence=w.get("confidence"),
                    )
                    for w in raw_words
                ]
            elif seg_text.strip():
                words = proportional_word_timestamps(
                    seg_text, s_start, max(1, s_end - s_start), speaker=resolved_speaker
                )
            final_segments.append(
                TranscriptSegment(
                    startMs=s_start,
                    endMs=s_end,
                    speaker=resolved_speaker,
                    text=seg_text,
                    words=words,
                )
            )

        if pipe.unload_per_stage and pipe.aligner is not None and hasattr(pipe.aligner, "release"):
            pipe.aligner.release()

    await _emit_progress(
        task_id,
        {
            "stage": "finalizing",
            "percent": 98,
            "totalMs": duration_ms,
            "message": "Coalescing turns & formatting transcript artifacts...",
        },
    )

    coalesced = coalesce_speaker_segments(final_segments)
    transcript = Transcript(
        segments=coalesced,
        language=language,
        durationMs=duration_ms,
    )

    logger.info(
        "Local transcription pipeline complete",
        {
            "category": "transcription",
            "meetingId": meeting_id,
            "rawSegments": len(final_segments),
            "coalescedSegments": len(coalesced),
            "discoveredSpeakers": len(discovered_speakers),
            "audioDurationMs": duration_ms,
        },
    )

    return {
        "transcript": transcript.model_dump(exclude_none=True),
        "discoveredSpeakers": discovered_speakers,
    }


@DBOS.step()
async def step_claim_local_stage(meeting_id: str, task_id: str, stage: str) -> bool:
    """Claim the right to enqueue ``stage`` for this run — at most once.

    The claim is a single conditional UPDATE on ``stage_runs``: a stage is
    claimable only while its row is ``pending``, and a fan-in successor
    (``align``) additionally requires all of its dependencies to be ``done`` in
    the same statement. Two siblings finishing at the same instant therefore
    race on a row update, not on wall-clock timing, and exactly one of them
    enqueues the join.

    This is a step on purpose: DBOS journals the boolean, so a workflow that
    recovers after claiming still goes on to enqueue, while a replayed claim can
    never double-enqueue.
    """
    db = _RECOVERY_DB
    if db is None or not meeting_id:
        return True

    params: dict[str, Any] = {
        "mid": meeting_id,
        "stage": stage,
        "jid": local_stage_workflow_id(meeting_id, task_id, stage),
        "now": _now_ms(),
    }
    dep_sql = ""
    deps = _STAGE_JOIN_DEPS.get(stage, ())
    if deps:
        placeholders: list[str] = []
        for idx, dep in enumerate(deps):
            key = f"dep{idx}"
            params[key] = dep
            placeholders.append(f":{key}")
        dep_sql = (
            " AND (SELECT COUNT(*) FROM stage_runs"
            " WHERE meeting_id = :mid AND status = 'done'"
            f" AND stage IN ({', '.join(placeholders)})) = {len(deps)}"
        )
    sql = (
        "UPDATE stage_runs"
        " SET status = 'queued', provider_job_id = :jid, last_error = NULL,"
        " finished_at = NULL, updated_at = :now"
        " WHERE meeting_id = :mid AND stage = :stage AND status = 'pending'"
        f"{dep_sql}"
    )
    # Deliberately not caught: a swallowed claim error would be journaled as
    # "not claimed", the successor would never be enqueued, and the run would
    # stall silently with no error to retry. Failing the stage is honest.
    with db.begin() as conn:
        claimed = conn.execute(text(sql), params).rowcount or 0
    return claimed == 1


def _is_shutdown_interruption(exc: BaseException) -> bool:
    """True when the DBOS runtime is tearing down mid-stage.

    Teardown cancels workflow tasks, and DBOS deliberately leaves a
    task-cancelled workflow PENDING (it is not recorded as failed) so the next
    launch can recover it. Recording a terminal stage here would turn a
    restart into a dead run, so this interruption leaves every row untouched.
    """
    return isinstance(exc, asyncio.CancelledError)


def _is_local_stage_cancelled(exc: BaseException) -> bool:
    # Order matters: a missed deadline is a *failure* to retry, not a user
    # cancel, even though DBOS words it as "step was cancelled".
    if isinstance(exc, DBOSStepTimeoutError):
        return False
    if isinstance(exc, DBOSWorkflowCancelledError):
        return True
    return "cancel" in str(exc).lower()


async def _emit_local_stage_failure(
    task_id: str,
    meeting_id: str | None,
    exc: BaseException,
    *,
    cancelled: bool,
) -> None:
    message = "Transcription was cancelled." if cancelled else str(exc) or type(exc).__name__
    sink = _RECOVERY_PROGRESS_SINK
    if meeting_id and callable(sink):
        try:
            res = sink(
                meeting_id,
                {
                    "stage": "cancelled" if cancelled else "error",
                    "percent": 0,
                    "message": message,
                    "error": None if cancelled else message,
                },
            )
            if inspect.isawaitable(res):
                await res
        except Exception:
            pass
    logger.info(
        "Local transcription stage ended without a transcript",
        {
            "category": "transcription",
            "meetingId": meeting_id,
            "taskId": task_id,
            "cancelled": cancelled,
            "error": str(exc),
        },
    )


async def _enqueue_local_stage(stage: str, payload: dict[str, Any]) -> str:
    """Enqueue one stage workflow. Ids are deterministic, so replays are no-ops."""
    task_id = str(payload["taskId"])
    meeting_id = str(payload.get("meetingId") or "")
    owner = meeting_id or task_id
    wf_id = local_stage_workflow_id(owner, task_id, stage)
    queue_name = _STAGE_QUEUE_BY_STAGE[stage]
    with SetWorkflowID(wf_id), SetEnqueueOptions(deduplication_id=wf_id):
        await DBOS.enqueue_workflow_async(queue_name, run_local_stage, stage, payload)
    return wf_id


@DBOS.workflow()
async def run_local_stage(stage: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run one local transcription stage and hand off to its successors.

    The workflow does **not** await any other workflow: it runs this stage's
    step, records the stage run, claims/enqueues the next stages and returns.
    Liveness is therefore structural — a stage can only occupy the slot it was
    dequeued into — and each stage's runtime is bounded by the DBOS step timeout
    declared on its body.
    """
    body = _LOCAL_STAGE_BODIES.get(stage)
    if body is None:
        raise ValueError(f"Unknown local transcription stage: {stage}")
    task_id = str(payload["taskId"])
    meeting_id = str(payload.get("meetingId") or "") or None
    _ensure_task_context(payload)
    failed = False

    try:
        if _stage_status(meeting_id, stage) == "cancelled":
            return {}
        job_id = str(getattr(DBOS, "workflow_id", "") or "")
        if not job_id:
            job_id = local_stage_workflow_id(meeting_id or task_id, task_id, stage)

        _mark_local_stage(meeting_id, stage, "running", job_id=job_id)
        try:
            result = await body(payload)
        except BaseException as exc:
            if _is_shutdown_interruption(exc):
                # Leave all state untouched so recovery resumes this stage.
                raise
            failed = True
            if isinstance(exc, _InactiveLocalRunError):
                # Cancellation/reset already closed this run. A late thread
                # must not publish an error or cancellation over a newer run.
                return {}
            cancelled = _is_local_stage_cancelled(exc)
            message = "Cancelled by user" if cancelled else str(exc) or type(exc).__name__
            _mark_local_stage(
                meeting_id,
                stage,
                "cancelled" if cancelled else "error",
                error=message,
                job_id=job_id,
            )
            if not cancelled:
                # The run itself is over: surface it on the run row too, or the
                # meeting looks mid-transcription forever after a stage failure.
                _mark_local_stage(
                    meeting_id,
                    STAGE_LOCAL_TRANSCRIBE,
                    "error",
                    error=message,
                    job_id=local_run_entry_workflow_id(meeting_id or task_id, task_id),
                )
                # Stop the run's other in-flight stages instead of letting a
                # costly sibling burn its queue slot for hours: mark their rows
                # cancelled, which their next _check_aborted notices. No DBOS
                # API call from inside a workflow (that path can block on the
                # very workflow it is cancelling).
                _abort_local_run_siblings(meeting_id, task_id, failed_stage=stage, error=message)
            clear_local_stage_checkpoints(meeting_id, task_id=task_id)
            run_already_failed = (
                meeting_id is not None and _stage_status(meeting_id, STAGE_LOCAL_TRANSCRIBE) == "error"
            )
            if not (cancelled and run_already_failed):
                # A sibling aborted by a failed run must not replace the real
                # error with "Transcription was cancelled".
                await _emit_local_stage_failure(task_id, meeting_id, exc, cancelled=cancelled)
            raise

        if stage == STAGE_ALIGN:
            # Final stage: persist the transcript, then mark the stage done so
            # "align done" always implies a durable transcript. Nothing awaits
            # the run, so completion is owned by the pipeline, not a caller.
            if meeting_id:
                sink = _RECOVERY_COMPLETE_SINK
                if callable(sink):
                    sink_res = sink(meeting_id, result, job_id)
                    if inspect.isawaitable(sink_res):
                        await sink_res
                else:
                    logger.warn(
                        "Local transcript finished without a finalize sink; artifacts not persisted",
                        {"category": "transcription", "meetingId": meeting_id},
                    )
            clear_local_stage_checkpoints(meeting_id, task_id=task_id)
            _mark_local_stage(meeting_id, stage, "done", job_id=job_id)
            return result

        _mark_local_stage(meeting_id, stage, "done", job_id=job_id)

        try:
            for next_stage in _STAGE_SUCCESSORS.get(stage, ()):
                if await step_claim_local_stage(meeting_id or "", task_id, next_stage):
                    await _enqueue_local_stage(next_stage, payload)
                    # Announce the hop only for a stage this run actually queued:
                    # a sibling that cannot join yet must not overwrite a
                    # terminal error/progress with a "waiting" message.
                    await _emit_stage_queued_progress(meeting_id, next_stage)
        except BaseException:
            # A cancel can land between the stage finishing and the hand-off;
            # DBOS raises at the next step boundary. Treat it like any other
            # terminal outcome so the run's pipeline is released.
            failed = True
            raise
        return result
    finally:
        # Hold the run's pipeline (and its native engine pools) across all five
        # stages; release it when the run ends — final stage or a failure.
        if stage == STAGE_ALIGN or failed:
            unregister_task_context(task_id)


#: "Now queued for X" progress, keyed by the stage that was just enqueued.
_STAGE_QUEUED_PROGRESS: dict[str, tuple[str, int]] = {
    STAGE_DIARIZE: ("Audio decoded. Queued for speaker diarization...", 10),
    STAGE_SPEAKER_EMBED: ("Diarization complete. Queued for speaker matching...", 33),
    STAGE_ASR: ("Diarization complete. Queued for ASR...", 33),
    STAGE_ALIGN: ("Transcription complete. Queued for alignment & finalization...", 95),
}


async def _emit_stage_queued_progress(meeting_id: str | None, queued_stage: str) -> None:
    sink = _RECOVERY_PROGRESS_SINK
    if not meeting_id or not callable(sink):
        return
    message, percent = _STAGE_QUEUED_PROGRESS.get(queued_stage, (None, 0))
    if message is None:
        return
    try:
        res = sink(meeting_id, {"stage": "queued", "percent": percent, "message": message})
        if inspect.isawaitable(res):
            await res
    except Exception:
        pass


def local_run_entry_workflow_id(meeting_id: str, task_id: str) -> str:
    """Workflow id of the run's first stage (the id stored on the run row)."""
    return local_stage_workflow_id(meeting_id, task_id, STAGE_SEQUENCE[0])


def _abort_local_run_siblings(
    meeting_id: str | None,
    task_id: str,
    *,
    failed_stage: str,
    error: str,
) -> None:
    """Mark this run's other queued/running stages cancelled (best effort).

    The stage rows are the abort channel: a sibling stops at its next
    ``_check_aborted`` (between ASR segments, between diarizer turns) instead of
    finishing a now-pointless job and holding a concurrency-1 queue slot. This
    is a plain DB write on purpose — cancelling the sibling's workflow through
    the DBOS API from inside another workflow can wait on that sibling's own
    outcome, which is the deadlock shape this design removes.
    """
    db = _RECOVERY_DB
    if db is None or not meeting_id:
        return
    run_prefix = f"{local_run_workflow_prefix(meeting_id, task_id)}:"
    now = _now_ms()
    try:
        with db.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE stage_runs
                    SET status = 'cancelled',
                        last_error = :err,
                        finished_at = :now,
                        updated_at = :now
                    WHERE meeting_id = :mid
                      AND status IN ('queued', 'running')
                      AND stage != :failed_stage
                      AND provider_job_id LIKE :prefix
                    """
                ),
                {
                    "err": f"Aborted after {failed_stage} failed: {error}",
                    "now": now,
                    "mid": meeting_id,
                    "failed_stage": failed_stage,
                    "prefix": run_prefix + "%",
                },
            )
    except Exception as exc:
        logger.warn(
            "Failed to abort local transcription siblings",
            {"category": "transcription", "meetingId": meeting_id, "error": str(exc)},
        )


#: Serializes the "is anything running? -> reset -> enqueue" critical section.
#: Keyed by loop id because a module-level asyncio.Lock binds to the loop that
#: first uses it and tests run each case on a fresh loop.
_ENQUEUE_LOCKS: dict[int, asyncio.Lock] = {}


def _enqueue_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    key = id(loop)
    lock = _ENQUEUE_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _ENQUEUE_LOCKS[key] = lock
    return lock


async def enqueue_local_transcription(payload: dict[str, Any]) -> str:
    """Enqueue a meeting's flat local pipeline; returns the run's workflow prefix.

    Entry point for the request path: it resets the meeting's stage rows for a
    fresh run and enqueues the first stage. Everything after that is stage
    hand-off, so this coroutine never waits on transcription.
    """
    task_id = str(payload["taskId"])
    meeting_id = str(payload.get("meetingId") or "")
    if not meeting_id:
        raise ValueError("Local transcription requires a meetingId")

    await ensure_dbos_launched_async(
        config_dir=_RECOVERY_CONFIG_DIR,
        meetings_dir=_RECOVERY_MEETINGS_DIR,
    )
    async with _enqueue_lock():
        if await has_active_local_stage_workflow(meeting_id):
            # Already running (or queued from a previous process): never reset a
            # live run's stage rows just because a duplicate enqueue arrived.
            return local_run_workflow_prefix(meeting_id, task_id)
        decode_wf_id = local_run_entry_workflow_id(meeting_id, task_id)
        if _dbos_workflow_exists(decode_wf_id):
            # Same run enqueued twice: DBOS would return the existing workflow,
            # so resetting stage rows here would strand them at "pending".
            return local_run_workflow_prefix(meeting_id, task_id)
        _reset_local_stage_runs(meeting_id, job_id=decode_wf_id)
        await _enqueue_local_stage(STAGE_SEQUENCE[0], payload)
    return local_run_workflow_prefix(meeting_id, task_id)


def _dbos_workflow_exists(workflow_id: str) -> bool:
    dbos_path = get_existing_dbos_sqlite_path(
        config_dir=_RECOVERY_CONFIG_DIR,
        meetings_dir=_RECOVERY_MEETINGS_DIR,
    )
    if dbos_path is None or not dbos_path.exists():
        return False
    try:
        with sqlite3.connect(str(dbos_path), timeout=1.0) as sconn:
            row = sconn.execute(
                "SELECT 1 FROM workflow_status WHERE workflow_uuid = ? LIMIT 1",
                (workflow_id,),
            ).fetchone()
        return row is not None
    except Exception:
        return False


async def has_active_local_stage_workflow(
    meeting_id: str,
    *,
    config_dir: str | Path | None = None,
    meetings_dir: str | Path | None = None,
) -> bool:
    """True while any stage of any run for this meeting is queued or running.

    Duck-typing the durable queue rather than ``stage_runs`` is deliberate: a
    crashed process leaves rows saying ``running`` with nothing scheduled, and
    asking DBOS makes "already running?" self-healing after a restart.
    """
    if not is_dbos_launched():
        try:
            await ensure_dbos_launched_async(
                config_dir=config_dir or _RECOVERY_CONFIG_DIR,
                meetings_dir=meetings_dir or _RECOVERY_MEETINGS_DIR,
            )
        except Exception:
            return False
    dbos_path = get_existing_dbos_sqlite_path(
        config_dir=config_dir or _RECOVERY_CONFIG_DIR,
        meetings_dir=meetings_dir or _RECOVERY_MEETINGS_DIR,
    )
    if dbos_path is None or not dbos_path.exists():
        return False
    try:
        with sqlite3.connect(str(dbos_path), timeout=1.0) as sconn:
            row = sconn.execute(
                """
                SELECT 1 FROM workflow_status
                WHERE workflow_uuid LIKE :prefix
                  AND status IN ('PENDING', 'ENQUEUED', 'DELAYED')
                LIMIT 1
                """,
                {"prefix": f"stage:{meeting_id}:%"},
            ).fetchone()
        return row is not None
    except Exception:
        return False


#: Workflow function names from the nested-orchestration build. They no longer
#: exist, so DBOS recovery cannot replay them; the startup sweep cancels them
#: instead of letting recovery burn attempts on an unregistered function.
_LEGACY_LOCAL_WORKFLOW_NAMES: tuple[str, ...] = (
    "local_transcription_workflow",
    "run_transcription_driver",
)

_LEGACY_UPGRADE_ERROR = "Transcription interrupted by an upgrade; retry transcription"


async def cancel_legacy_local_transcription_workflows(
    *,
    config_dir: str | Path | None = None,
    meetings_dir: str | Path | None = None,
) -> int:
    """Cancel local transcription workflows left over from the nested design.

    The flat pipeline replaced ``run_transcription_driver`` ->
    ``local_transcription_workflow`` -> stage children. Workflows of the old
    shape may still be ENQUEUED in the DBOS database at upgrade time, and their
    functions are gone, so recovery would fail them repeatedly and leave the
    meeting stuck mid-pipeline. Cancelling them (and marking the meeting
    retryable) is the whole upgrade path — no dual code paths.
    """
    if not is_dbos_launched():
        return 0
    dbos_path = get_existing_dbos_sqlite_path(
        config_dir=config_dir or _RECOVERY_CONFIG_DIR,
        meetings_dir=meetings_dir or _RECOVERY_MEETINGS_DIR,
    )
    if dbos_path is None or not dbos_path.exists():
        return 0

    active: list[tuple[str, str, str]] = []
    try:
        with sqlite3.connect(str(dbos_path), timeout=1.0) as sconn:
            rows = sconn.execute(
                """
                SELECT workflow_uuid, name, parent_workflow_id
                FROM workflow_status
                WHERE status IN ('PENDING', 'ENQUEUED', 'DELAYED')
                """
            ).fetchall()
        active = [(str(r[0] or ""), str(r[1] or ""), str(r[2] or "")) for r in rows if r and r[0]]
    except Exception:
        return 0

    def _is_legacy(wf_id: str, name: str, parent_id: str, doomed: set[str]) -> bool:
        short_name = name.split(".")[-1]
        return (
            any(short_name.endswith(legacy) for legacy in _LEGACY_LOCAL_WORKFLOW_NAMES)
            or wf_id.startswith("local-transcribe:")
            or parent_id in doomed
        )

    doomed: set[str] = set()
    changed = True
    while changed:
        changed = False
        for wf_id, name, parent_id in active:
            if wf_id in doomed:
                continue
            if _is_legacy(wf_id, name, parent_id, doomed):
                doomed.add(wf_id)
                changed = True
    if not doomed:
        return 0

    cancelled = 0
    for wf_id in doomed:
        try:
            await DBOS.cancel_workflow_async(wf_id)
            cancelled += 1
        except Exception:
            pass

    meeting_ids: set[str] = set()
    for wf_id, _name, _parent in active:
        if wf_id not in doomed:
            continue
        for prefix in ("local-transcribe:", "transcribe:"):
            if wf_id.startswith(prefix):
                parts = wf_id.split(":")
                if len(parts) >= 2 and parts[1]:
                    meeting_ids.add(parts[1])

    db = _RECOVERY_DB
    if db is not None and meeting_ids:
        now = _now_ms()
        try:
            with db.begin() as conn:
                for meeting_id in meeting_ids:
                    conn.execute(
                        text(
                            """
                            UPDATE stage_runs
                            SET status = 'error', last_error = :err, updated_at = :now
                            WHERE meeting_id = :mid AND status IN ('pending', 'queued', 'running')
                            """
                        ),
                        {"err": _LEGACY_UPGRADE_ERROR, "now": now, "mid": meeting_id},
                    )
                    conn.execute(
                        text(
                            """
                            UPDATE meetings
                            SET status = 'pending', last_error = :err, updated_at = :now
                            WHERE id = :mid AND primary_transcript_artifact_id IS NULL
                            """
                        ),
                        {"err": _LEGACY_UPGRADE_ERROR, "now": now, "mid": meeting_id},
                    )
        except Exception:
            pass
    invalidate_dbos_progress_cache()
    return cancelled


@DBOS.step()
async def step_enhance_audio_file(source_path: str, target_path: str) -> bool:
    return await asyncio.to_thread(enhance_audio_file, source_path, target_path)


@DBOS.step()
async def step_run_registered_job(job_id: str, metadata: dict[str, Any] | None = None) -> Any:
    _ = metadata
    fn = _GENERIC_JOBS.pop(job_id, None)
    if fn is None:
        return None
    res = fn()
    if inspect.isawaitable(res):
        return await res
    return res


@DBOS.workflow()
async def run_registered_job_workflow(job_id: str, metadata: dict[str, Any] | None = None) -> Any:
    """Workflow wrapper for a callable that may itself enqueue workflows.

    Leaf background work belongs in :func:`step_run_registered_job`; use this
    for the few paths (cloud transcription) whose body fans out into child
    workflows, which DBOS forbids from inside a step. It never awaits a child:
    it only enqueues.
    """
    _ = metadata
    fn = _GENERIC_JOBS.pop(job_id, None)
    if fn is None:
        return None
    res = fn()
    if inspect.isawaitable(res):
        return await res
    return res


_LOCAL_STAGE_BODIES: dict[str, Callable[[dict[str, Any]], Any]] = {
    STAGE_FFMPEG_DECODE: step_decode_audio,
    STAGE_DIARIZE: step_diarize_turns,
    STAGE_SPEAKER_EMBED: step_embed_and_match_speakers,
    STAGE_ASR: step_transcribe_asr,
    STAGE_ALIGN: step_align_and_finalize,
}


async def enqueue_stage_job(
    queue_name: str,
    fn: Callable[[], Any],
    *,
    deduplication_id: str | None = None,
    workflow_id: str | None = None,
    label: str | None = None,
    meeting_id: str | None = None,
) -> Any:
    job_id = str(uuid.uuid4())
    _GENERIC_JOBS[job_id] = fn
    wf_id = workflow_id or f"job:{queue_name}:{job_id}"
    meta = {"label": label or queue_name, "meetingId": meeting_id}
    with SetWorkflowID(wf_id), SetEnqueueOptions(deduplication_id=deduplication_id):
        handle = await DBOS.enqueue_workflow_async(queue_name, step_run_registered_job, job_id, meta)
    return handle


async def enqueue_driver_job(
    queue_name: str,
    fn: Callable[[], Any],
    *,
    deduplication_id: str | None = None,
    workflow_id: str | None = None,
    label: str | None = None,
    meeting_id: str | None = None,
) -> Any:
    """Enqueue ``fn`` as a workflow instead of a step (cloud transcription)."""
    job_id = str(uuid.uuid4())
    _GENERIC_JOBS[job_id] = fn
    wf_id = workflow_id or f"job:{queue_name}:{job_id}"
    meta = {"label": label or queue_name, "meetingId": meeting_id}
    with SetWorkflowID(wf_id), SetEnqueueOptions(deduplication_id=deduplication_id):
        handle = await DBOS.enqueue_workflow_async(queue_name, run_registered_job_workflow, job_id, meta)
    return handle


def _extract_job_details(wf: Any) -> tuple[str, str | None]:
    raw_name = str(getattr(wf, "name", "") or "job")
    short_name = raw_name.split(".")[-1].replace("<temp>.", "")
    meeting_id: str | None = None

    inp = getattr(wf, "input", None)
    if isinstance(inp, dict):
        args = inp.get("args") or ()
        if short_name == "run_local_stage" and args and isinstance(args[0], str):
            # One generic workflow runs every stage; show which one it is.
            short_name = f"stage:{args[0]}"
        for arg in args:
            if isinstance(arg, dict):
                if not meeting_id and arg.get("meetingId"):
                    meeting_id = str(arg["meetingId"])
                if short_name in ("step_run_registered_job", "run_registered_job_workflow") and arg.get(
                    "label"
                ):
                    short_name = str(arg["label"])

    wf_id = str(getattr(wf, "workflow_id", "") or "")
    if not meeting_id:
        for prefix in ("stage:", "local-transcribe:"):
            if wf_id.startswith(prefix):
                parts = wf_id.split(":")
                if len(parts) >= 2 and parts[1]:
                    meeting_id = parts[1]
                break

    return short_name, meeting_id


async def get_debug_queues_snapshot(
    db: Engine | None = None,
    *,
    config_dir: str | Path | None = None,
    meetings_dir: str | Path | None = None,
    active_transcriptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now_ms = int(time.time() * 1000)
    dbos_url = await ensure_dbos_launched_async(config_dir=config_dir, meetings_dir=meetings_dir)

    recent_wfs, queued_wfs = await asyncio.gather(
        DBOS.list_workflows_async(limit=200, sort_desc=True, load_input=True, load_output=False),
        DBOS.list_queued_workflows_async(load_input=True, load_output=False),
    )

    by_id: dict[str, Any] = {}
    for w in recent_wfs:
        by_id[str(w.workflow_id)] = w
    for w in queued_wfs:
        by_id[str(w.workflow_id)] = w

    all_wfs = list(by_id.values())

    meeting_titles: dict[str, str] = {}
    running_stage_runs: list[dict[str, Any]] = []
    active_meeting_ids = set((active_transcriptions or {}).keys())
    for ctx in _TASK_CONTEXTS.values():
        if ctx.meeting_id:
            active_meeting_ids.add(ctx.meeting_id)
    for w in all_wfs:
        if getattr(w, "status", None) in ("PENDING", "ENQUEUED", "DELAYED"):
            _, m_id = _extract_job_details(w)
            if m_id:
                active_meeting_ids.add(m_id)

    if db is not None:

        def _query_olive_db() -> tuple[dict[str, str], list[dict[str, Any]]]:
            titles: dict[str, str] = {}
            stage_runs: list[dict[str, Any]] = []
            with db.connect() as conn:
                m_rows = conn.execute(text("SELECT id, title FROM meetings")).mappings().all()
                for r in m_rows:
                    titles[str(r["id"])] = str(r["title"])

                sr_rows = (
                    conn.execute(
                        text(
                            """
                            SELECT sr.id, sr.meeting_id, m.title AS meeting_title,
                                   sr.stage, sr.status, sr.provider_job_id,
                                   sr.attempts, sr.started_at, sr.updated_at
                            FROM stage_runs sr
                            LEFT JOIN meetings m ON m.id = sr.meeting_id
                            WHERE sr.status = 'running'
                            ORDER BY sr.updated_at DESC
                            """
                        )
                    )
                    .mappings()
                    .all()
                )
                for sr in sr_rows:
                    if str(sr["meeting_id"]) not in active_meeting_ids:
                        continue
                    started_at = sr.get("started_at")
                    stage_runs.append(
                        {
                            "id": sr["id"],
                            "meetingId": sr["meeting_id"],
                            "meetingTitle": sr.get("meeting_title") or titles.get(str(sr["meeting_id"])),
                            "stage": sr["stage"],
                            "status": sr["status"],
                            "jobId": sr.get("provider_job_id"),
                            "attempts": int(sr.get("attempts") or 1),
                            "startedAt": started_at,
                            "elapsedMs": max(0, now_ms - int(started_at)) if started_at else None,
                        }
                    )
            return titles, stage_runs

        try:
            meeting_titles, running_stage_runs = await asyncio.to_thread(_query_olive_db)
        except Exception:
            pass

    def _serialize_wf(w: Any, queue_position: int | None = None) -> dict[str, Any]:
        step_name, meeting_id = _extract_job_details(w)
        created_at = getattr(w, "created_at", None)
        dequeued_at = getattr(w, "dequeued_at", None)
        completed_at = getattr(w, "completed_at", None)
        err_obj = getattr(w, "error", None)
        ref_start = dequeued_at or created_at
        elapsed_ms = max(0, (completed_at or now_ms) - int(ref_start)) if ref_start is not None else None
        waiting_ms = max(0, (dequeued_at or now_ms) - int(created_at)) if created_at is not None else None
        return {
            "workflowId": str(w.workflow_id),
            "name": step_name,
            "rawName": str(getattr(w, "name", "") or ""),
            "status": str(getattr(w, "status", "") or ""),
            "queueName": getattr(w, "queue_name", None),
            "parentWorkflowId": getattr(w, "parent_workflow_id", None),
            "meetingId": meeting_id,
            "meetingTitle": meeting_titles.get(meeting_id) if meeting_id else None,
            "createdAt": created_at,
            "dequeuedAt": dequeued_at,
            "completedAt": completed_at,
            "elapsedMs": elapsed_ms,
            "waitingMs": waiting_ms,
            "queuePosition": queue_position,
            "deduplicationId": getattr(w, "deduplication_id", None),
            "priority": getattr(w, "priority", None),
            "recoveryAttempts": getattr(w, "recovery_attempts", None),
            "error": str(err_obj) if err_obj else None,
        }

    queues_out: list[dict[str, Any]] = []
    total_active = 0
    total_waiting = 0

    for spec in STAGE_QUEUE_SPECS:
        q_wfs = [w for w in all_wfs if getattr(w, "queue_name", None) == spec.name]
        active_raw = sorted(
            [w for w in q_wfs if getattr(w, "status", None) == "PENDING"],
            key=lambda w: int(getattr(w, "dequeued_at", None) or getattr(w, "created_at", None) or 0),
        )
        waiting_raw = sorted(
            [w for w in q_wfs if getattr(w, "status", None) in ("ENQUEUED", "DELAYED")],
            key=lambda w: (
                int(getattr(w, "priority", None) or 0),
                int(getattr(w, "created_at", None) or 0),
            ),
        )
        recent_raw = sorted(
            [
                w
                for w in q_wfs
                if getattr(w, "status", None)
                in ("SUCCESS", "ERROR", "CANCELLED", "MAX_RECOVERY_ATTEMPTS_EXCEEDED")
            ],
            key=lambda w: int(
                getattr(w, "completed_at", None)
                or getattr(w, "updated_at", None)
                or getattr(w, "created_at", None)
                or 0
            ),
            reverse=True,
        )[:8]

        active_jobs = [_serialize_wf(w) for w in active_raw]
        waiting_jobs = [_serialize_wf(w, idx + 1) for idx, w in enumerate(waiting_raw)]
        recent_jobs = [_serialize_wf(w) for w in recent_raw]

        total_active += len(active_jobs)
        total_waiting += len(waiting_jobs)

        queues_out.append(
            {
                "name": spec.name,
                "workerConcurrency": spec.worker_concurrency,
                "activeCount": len(active_jobs),
                "waitingCount": len(waiting_jobs),
                "activeJobs": active_jobs,
                "waitingJobs": waiting_jobs,
                "recentJobs": recent_jobs,
            }
        )

    # Root workflows that are not attached to a stage queue (e.g. enqueued
    # directly and not yet dequeued). Local transcription is no longer among
    # them: every stage runs on its own stage queue.
    parent_wfs = [
        _serialize_wf(w)
        for w in sorted(
            [
                w
                for w in all_wfs
                if getattr(w, "queue_name", None) is None and getattr(w, "status", None) == "PENDING"
            ],
            key=lambda w: int(getattr(w, "created_at", None) or 0),
            reverse=True,
        )
    ]

    return {
        "timestamp": now_ms,
        "dbosDatabaseUrl": dbos_url,
        "summary": {
            "totalQueues": len(queues_out),
            "totalActiveJobs": total_active,
            "totalWaitingJobs": total_waiting,
            "activeParentWorkflows": len(parent_wfs),
        },
        "queues": queues_out,
        "activeWorkflows": parent_wfs,
        "activeTranscriptions": active_transcriptions or {},
        "runningStageRuns": running_stage_runs,
    }


_DBOS_PROGRESS_CACHE: tuple[float, str, dict[str, dict[str, Any]]] | None = None


def invalidate_dbos_progress_cache() -> None:
    global _DBOS_PROGRESS_CACHE
    _DBOS_PROGRESS_CACHE = None


def get_dbos_active_progress_map(
    db: Engine | None = None,
    *,
    config_dir: str | Path | None = None,
    meetings_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    global _DBOS_PROGRESS_CACHE
    if not is_dbos_launched():
        return {}

    dbos_path = get_existing_dbos_sqlite_path(
        config_dir=config_dir or _RECOVERY_CONFIG_DIR,
        meetings_dir=meetings_dir or _RECOVERY_MEETINGS_DIR,
    )
    if dbos_path is None or not dbos_path.exists():
        return {}

    now_mono = time.monotonic()
    path_key = str(dbos_path)
    if (
        _DBOS_PROGRESS_CACHE is not None
        and _DBOS_PROGRESS_CACHE[1] == path_key
        and (now_mono - _DBOS_PROGRESS_CACHE[0]) < 0.75
    ):
        return dict(_DBOS_PROGRESS_CACHE[2])

    try:
        with sqlite3.connect(str(dbos_path), timeout=1.0) as sconn:
            sconn.row_factory = sqlite3.Row
            rows = sconn.execute(
                """
                SELECT workflow_uuid, status, name, queue_name, parent_workflow_id, created_at, priority
                FROM workflow_status
                WHERE status IN ('PENDING', 'ENQUEUED', 'DELAYED')
                ORDER BY COALESCE(priority, 0) ASC, created_at ASC
                """
            ).fetchall()
            if not rows:
                _DBOS_PROGRESS_CACHE = (now_mono, path_key, {})
                return {}

            out_map = _build_progress_map(sconn, rows, db)
    except Exception:
        return {}

    _DBOS_PROGRESS_CACHE = (now_mono, path_key, out_map)
    return dict(out_map)


_WF_MEETING_CACHE: dict[str, str | None] = {}


def _meeting_id_from_inputs(sconn: sqlite3.Connection, wf_id: str) -> str | None:
    """Read meetingId from a DBOS workflow's recorded input (cached per workflow)."""
    if wf_id in _WF_MEETING_CACHE:
        return _WF_MEETING_CACHE[wf_id]
    mid: str | None = None
    try:
        row = sconn.execute("SELECT inputs FROM workflow_input WHERE workflow_uuid = ?", (wf_id,)).fetchone()
        if row and row[0]:
            decoded = DefaultSerializer().deserialize(row[0])
            args = decoded.get("args") or ()
            # Stage workflows pass ``(stage, payload)``; other jobs pass the
            # payload first. Scan args so both shapes resolve.
            for arg in args:
                if isinstance(arg, dict) and arg.get("meetingId"):
                    mid = str(arg["meetingId"])
                    break
    except Exception:
        mid = None
    _WF_MEETING_CACHE[wf_id] = mid
    return mid


def _build_progress_map(
    sconn: sqlite3.Connection,
    rows: list[sqlite3.Row],
    db: Engine | None,
) -> dict[str, dict[str, Any]]:
    """Derive per-meeting progress from active DBOS workflows."""
    job_to_meeting: dict[str, str] = {}
    for ctx in list(_TASK_CONTEXTS.values()):
        if ctx.workflow_id and ctx.meeting_id:
            job_to_meeting[ctx.workflow_id] = ctx.meeting_id

    target_db = db or _RECOVERY_DB
    if target_db is not None:
        try:
            with target_db.connect() as conn:
                sr_rows = (
                    conn.execute(
                        text(
                            "SELECT provider_job_id, meeting_id FROM stage_runs WHERE provider_job_id IS NOT NULL"
                        )
                    )
                    .mappings()
                    .all()
                )
                for r in sr_rows:
                    jid = str(r["provider_job_id"])
                    mid = str(r["meeting_id"])
                    if jid and mid:
                        job_to_meeting[jid] = mid
        except Exception:
            pass

    def _resolve_mid(sconn: sqlite3.Connection, row: sqlite3.Row) -> str | None:
        wf_id = str(row["workflow_uuid"] or "")
        parent_id = str(row["parent_workflow_id"] or "")
        for candidate in (wf_id, parent_id):
            if not candidate:
                continue
            if candidate in job_to_meeting:
                return job_to_meeting[candidate]
            for prefix in ("stage:", "local-transcribe:"):
                if candidate.startswith(prefix):
                    parts = candidate.split(":")
                    if len(parts) >= 2 and parts[1]:
                        return parts[1]
        if wf_id.startswith("local-") and "-" in wf_id:
            base = wf_id.rsplit("-", 1)[0]
            if base in job_to_meeting:
                return job_to_meeting[base]
        return _meeting_id_from_inputs(sconn, wf_id)

    waiting_by_queue: dict[str, list[str]] = {}
    for r in rows:
        st = str(r["status"] or "")
        q_name = str(r["queue_name"] or "")
        if st in ("ENQUEUED", "DELAYED") and q_name:
            waiting_by_queue.setdefault(q_name, []).append(str(r["workflow_uuid"]))

    out: dict[str, tuple[int, dict[str, Any]]] = {}
    for r in rows:
        mid = _resolve_mid(sconn, r)
        if not mid:
            continue
        wf_id = str(r["workflow_uuid"] or "")
        st = str(r["status"] or "")
        q_name = str(r["queue_name"] or "")
        is_active = st == "PENDING"

        q_list = waiting_by_queue.get(q_name, [])
        q_total = len(q_list)
        try:
            q_pos = q_list.index(wf_id) + 1
        except ValueError:
            q_pos = 1

        rank = 0
        prog: dict[str, Any] | None = None

        if q_name == QUEUE_ALIGN:
            if is_active:
                rank = 90
                prog = {
                    "stage": "finalizing",
                    "percent": 96,
                    "message": "Aligning words & finalizing transcript...",
                }
            else:
                rank = 80
                prog = {
                    "stage": "queued",
                    "percent": 95,
                    "message": f"ASR complete. Queued for alignment (#{q_pos} of {q_total} in queue)...",
                }
        elif q_name == QUEUE_ASR:
            if is_active:
                rank = 75
                prog = {
                    "stage": "transcribing",
                    "percent": 40,
                    "message": "Transcribing speech segments with Cohere ASR...",
                }
            else:
                rank = 60
                prog = {
                    "stage": "queued",
                    "percent": 33,
                    "message": f"Diarization complete. Queued for ASR (#{q_pos} of {q_total} in queue)...",
                }
        elif q_name == QUEUE_SPEAKER_EMBED:
            if is_active:
                rank = 70
                prog = {
                    "stage": "diarizing",
                    "percent": 38,
                    "message": "Matching speaker voiceprints against enrolled profiles...",
                }
            else:
                rank = 55
                prog = {
                    "stage": "queued",
                    "percent": 33,
                    "message": f"Diarization complete. Queued for speaker matching (#{q_pos} of {q_total} in queue)...",
                }
        elif q_name == QUEUE_DIARIZE:
            if is_active:
                rank = 50
                prog = {
                    "stage": "diarizing",
                    "percent": 15,
                    "message": "Diarization (1/4): Running neural speaker segmentation & embeddings...",
                }
            else:
                rank = 40
                prog = {
                    "stage": "queued",
                    "percent": 10,
                    "message": f"Audio decoded. Queued for speaker diarization (#{q_pos} of {q_total} in queue)...",
                }
        elif q_name == QUEUE_FFMPEG:
            if is_active:
                rank = 30
                prog = {
                    "stage": "decoding",
                    "percent": 5,
                    "message": "Decoding 16kHz audio samples...",
                }
            else:
                rank = 20
                prog = {
                    "stage": "queued",
                    "percent": 2,
                    "message": f"Queued for audio decode (#{q_pos} of {q_total} in queue)...",
                }

        if prog is not None:
            prev = out.get(mid)
            if prev is None or rank >= prev[0]:
                out[mid] = (rank, prog)

    return {mid: item[1] for mid, item in out.items()}


async def cancel_dbos_workflows_for_meeting(
    meeting_id: str,
    *,
    db: Engine | None = None,
    config_dir: str | Path | None = None,
    meetings_dir: str | Path | None = None,
) -> int:
    invalidate_dbos_progress_cache()
    if not is_dbos_launched():
        # DBOS launches lazily on first enqueue, so after a restart it may
        # not be running in this process yet — but the workflows to cancel
        # are still in its database. Launch it so cancellation can proceed.
        try:
            await ensure_dbos_launched_async(
                config_dir=config_dir or _RECOVERY_CONFIG_DIR,
                meetings_dir=meetings_dir or _RECOVERY_MEETINGS_DIR,
            )
        except Exception:
            return 0
    dbos_path = get_existing_dbos_sqlite_path(
        config_dir=config_dir or _RECOVERY_CONFIG_DIR,
        meetings_dir=meetings_dir or _RECOVERY_MEETINGS_DIR,
    )
    if dbos_path is None or not dbos_path.exists():
        return 0

    parent_ids: set[str] = set()
    target_db = db or _RECOVERY_DB
    if target_db is not None:
        try:
            with target_db.connect() as conn:
                rows = (
                    conn.execute(
                        text(
                            "SELECT provider_job_id FROM stage_runs WHERE meeting_id = :mid AND provider_job_id IS NOT NULL"
                        ),
                        {"mid": meeting_id},
                    )
                    .mappings()
                    .all()
                )
                for r in rows:
                    if r["provider_job_id"]:
                        parent_ids.add(str(r["provider_job_id"]))
        except Exception:
            pass

    for ctx in list(_TASK_CONTEXTS.values()):
        if ctx.meeting_id == meeting_id and ctx.workflow_id:
            parent_ids.add(ctx.workflow_id)

    wf_ids_to_cancel: set[str] = set()
    try:
        with sqlite3.connect(str(dbos_path), timeout=1.0) as sconn:
            sconn.row_factory = sqlite3.Row
            active_rows = sconn.execute(
                """
                SELECT workflow_uuid, parent_workflow_id
                FROM workflow_status
                WHERE status IN ('PENDING', 'ENQUEUED', 'DELAYED')
                """
            ).fetchall()
            for r in active_rows:
                wid = str(r["workflow_uuid"] or "")
                pid = str(r["parent_workflow_id"] or "")
                if (
                    wid in parent_ids
                    or pid in parent_ids
                    # Flat local pipeline stages: every stage workflow id is
                    # ``stage:{meeting_id}:{run8}:{stage}``. Keep this prefix
                    # matcher in sync with ``local_stage_workflow_id`` or queued
                    # stages become un-cancellable.
                    or wid.startswith(f"stage:{meeting_id}:")
                    or pid.startswith(f"stage:{meeting_id}:")
                    # Pre-flat-pipeline ids, still cancellable during upgrade.
                    or wid.startswith(f"local-transcribe:{meeting_id}:")
                    or pid.startswith(f"local-transcribe:{meeting_id}:")
                    # Cloud transcription jobs enqueued via
                    # TranscriptionService.start_transcription use
                    # ``transcribe:{meeting_id}:{job8}`` workflow ids — keep
                    # this prefix matcher in sync with that format.
                    or wid.startswith(f"transcribe:{meeting_id}:")
                    or pid.startswith(f"transcribe:{meeting_id}:")
                    # Speaker re-identification jobs enqueued via
                    # TranscriptionService.request_speaker_reidentification use
                    # ``reidentify-speakers:{meeting_id}:{hex8}`` workflow ids — keep
                    # this prefix matcher in sync with that format.
                    or wid.startswith(f"reidentify-speakers:{meeting_id}:")
                    or pid.startswith(f"reidentify-speakers:{meeting_id}:")
                    or any(wid.startswith(f"{p}-") for p in parent_ids)
                ):
                    wf_ids_to_cancel.add(wid)
    except Exception:
        pass

    cancelled_count = 0
    for wid in wf_ids_to_cancel:
        try:
            await DBOS.cancel_workflow_async(wid)
            cancelled_count += 1
        except Exception:
            pass
    invalidate_dbos_progress_cache()
    return cancelled_count


__all__ = [
    "QUEUE_ALIGN",
    "QUEUE_ASSEMBLYAI",
    "QUEUE_ASR",
    "QUEUE_DIARIZE",
    "QUEUE_FFMPEG",
    "QUEUE_PLAUD_SYNC",
    "QUEUE_SPEAKER_EMBED",
    "QUEUE_SPEECHMATICS",
    "QUEUE_SUMMARIZE",
    "STAGE_ALIGN",
    "STAGE_ASR",
    "STAGE_DIARIZE",
    "STAGE_FFMPEG_DECODE",
    "STAGE_LOCAL_TRANSCRIBE",
    "STAGE_SEQUENCE",
    "STAGE_SPEAKER_EMBED",
    "STAGE_TIMEOUT_ENV",
    "TaskExecutionContext",
    "cancel_dbos_workflows_for_meeting",
    "cancel_legacy_local_transcription_workflows",
    "clear_local_stage_checkpoints",
    "configure_recovery_runtime",
    "enqueue_driver_job",
    "enqueue_local_transcription",
    "enqueue_stage_job",
    "get_dbos_active_progress_map",
    "get_debug_queues_snapshot",
    "has_active_local_stage_workflow",
    "invalidate_dbos_progress_cache",
    "local_run_workflow_prefix",
    "local_stage_workflow_id",
    "register_task_context",
    "run_local_stage",
    "run_registered_job_workflow",
    "step_align_and_finalize",
    "step_claim_local_stage",
    "step_decode_audio",
    "step_diarize_turns",
    "step_embed_and_match_speakers",
    "step_enhance_audio_file",
    "step_run_registered_job",
    "step_transcribe_asr",
    "unregister_task_context",
]
