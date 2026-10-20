"""DBOS staged local transcription workflow and background queue tasks."""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dbos import DBOS, SetEnqueueOptions, SetWorkflowID
from dbos._serialization import DefaultSerializer
from sqlalchemy import Engine, text

from ..logging_db import logger
from ..ml.align import proportional_word_timestamps
from ..ml.asr_cohere import CohereAsrEngine
from ..ml.audio import decode_wav, encode_wav, enhance_audio_file, load_audio_samples
from ..ml.pipeline import LocalTranscriptionPipeline
from ..ml.protocols import (
    DiscoveredSpeakerVoiceprint,
    LocalTranscriptionResult,
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
    QUEUE_INSIGHTS,
    QUEUE_PIPELINE,
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
    is_shutting_down,
)

STAGE_FFMPEG_DECODE = "ffmpeg_decode"
STAGE_DIARIZE = "diarize"
STAGE_SPEAKER_EMBED = "speaker_embed"
STAGE_ASR = "asr"
STAGE_ALIGN = "align"


@dataclass
class TaskExecutionContext:
    task_id: str
    meeting_id: str | None = None
    signal: Any = None
    on_progress: Callable[[dict[str, Any]], Any] | None = None
    pipeline: LocalTranscriptionPipeline | None = None
    pipeline_options: dict[str, Any] | None = None
    audio_bytes: bytes | None = None
    db: Engine | None = None
    now_fn: Callable[[], int] | None = None
    workflow_id: str | None = None
    has_live_caller: bool = True


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


def configure_recovery_runtime(
    db: Engine | None,
    progress_sink: Callable[[str, dict[str, Any]], Any] | None,
    complete_sink: Callable[[str, dict[str, Any], str | None], Any] | None = None,
    *,
    config_dir: str | None = None,
    meetings_dir: str | None = None,
) -> None:
    """Set process-wide hooks used by steps that DBOS resumes without a live caller."""
    global _RECOVERY_DB, _RECOVERY_PROGRESS_SINK, _RECOVERY_COMPLETE_SINK
    global _RECOVERY_CONFIG_DIR, _RECOVERY_MEETINGS_DIR
    _RECOVERY_DB = db
    _RECOVERY_PROGRESS_SINK = progress_sink
    if complete_sink is not None:
        _RECOVERY_COMPLETE_SINK = complete_sink
    if config_dir is not None:
        _RECOVERY_CONFIG_DIR = config_dir
    if meetings_dir is not None:
        _RECOVERY_MEETINGS_DIR = meetings_dir


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
            has_live_caller=False,
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


def _check_aborted(task_id: str) -> None:
    if is_shutting_down():
        raise RuntimeError("Transcription cancelled: server shutting down")
    ctx = _TASK_CONTEXTS.get(task_id)
    if ctx is not None and _is_signal_aborted(ctx.signal):
        raise RuntimeError("Transcription cancelled by user")


async def _emit_progress(task_id: str, payload: dict[str, Any]) -> None:
    ctx = _TASK_CONTEXTS.get(task_id)
    if ctx is not None and callable(ctx.on_progress):
        res = ctx.on_progress(payload)
        if inspect.isawaitable(res):
            await res


def _get_pipeline(task_id: str, pipeline_options: dict[str, Any] | None) -> LocalTranscriptionPipeline:
    ctx = _TASK_CONTEXTS.get(task_id)
    if ctx is not None and ctx.pipeline is not None:
        return ctx.pipeline
    pipe = LocalTranscriptionPipeline(pipeline_options)
    if ctx is not None:
        ctx.pipeline = pipe
    return pipe


def _update_stage_run(
    task_id: str,
    stage: str,
    status: str,
    error: str | None = None,
) -> None:
    ctx = _TASK_CONTEXTS.get(task_id)
    if ctx is None or ctx.db is None or not ctx.meeting_id:
        return
    now_ms = ctx.now_fn() if callable(ctx.now_fn) else int(time.time() * 1000)
    meeting_id = ctx.meeting_id
    job_id = ctx.workflow_id or f"dbos-{task_id[:8]}"
    try:
        with ctx.db.begin() as conn:
            if status == "running":
                conn.execute(
                    text(
                        """
                        UPDATE meetings
                        SET status = 'processing', last_error = NULL, updated_at = :now
                        WHERE id = :mid AND primary_transcript_artifact_id IS NULL AND status != 'processing'
                        """
                    ),
                    {"now": now_ms, "mid": meeting_id},
                )
                conn.execute(
                    text(
                        """
                        UPDATE stage_runs
                        SET status = 'running', last_error = NULL, finished_at = NULL, updated_at = :now
                        WHERE meeting_id = :mid AND stage = 'local_transcribe' AND status != 'running'
                        """
                    ),
                    {"now": now_ms, "mid": meeting_id},
                )
            existing = (
                conn.execute(
                    text("SELECT id, attempts FROM stage_runs WHERE meeting_id = :mid AND stage = :stage"),
                    {"mid": meeting_id, "stage": stage},
                )
                .mappings()
                .first()
            )
            if existing:
                if status == "running":
                    conn.execute(
                        text(
                            """
                            UPDATE stage_runs
                            SET status = 'running', provider_job_id = :jid,
                                attempts = :att, last_error = NULL,
                                started_at = :now, finished_at = NULL, updated_at = :now
                            WHERE id = :id
                            """
                        ),
                        {
                            "jid": job_id,
                            "att": int(existing["attempts"]) + 1,
                            "now": now_ms,
                            "id": existing["id"],
                        },
                    )
                else:
                    conn.execute(
                        text(
                            """
                            UPDATE stage_runs
                            SET status = :st, last_error = :err,
                                finished_at = :now, updated_at = :now
                            WHERE id = :id
                            """
                        ),
                        {
                            "st": status,
                            "err": error,
                            "now": now_ms,
                            "id": existing["id"],
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
                          :id, :mid, :stage, :st, :jid, 1,
                          :err, :now, :fin, :now, :now
                        )
                        """
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "mid": meeting_id,
                        "stage": stage,
                        "st": status,
                        "jid": job_id,
                        "err": error,
                        "now": now_ms,
                        "fin": None if status == "running" else now_ms,
                    },
                )
    except Exception:
        pass


@DBOS.step()
async def step_decode_audio(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload["taskId"])
    _ensure_task_context(payload)
    meeting_id = payload.get("meetingId")
    work_dir = Path(payload["workDir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    _check_aborted(task_id)
    _update_stage_run(task_id, STAGE_FFMPEG_DECODE, "running")

    try:
        await _emit_progress(
            task_id,
            {
                "stage": "decoding",
                "percent": 5,
                "message": "Decoding 16kHz audio samples...",
            },
        )

        ctx = _TASK_CONTEXTS.get(task_id)
        audio_bytes = ctx.audio_bytes if ctx is not None else None
        audio_path = payload.get("audioPath")

        decode_t0 = time.perf_counter()
        target_sr = 16000
        decoded = await load_audio_samples(
            {
                "audioPath": audio_path,
                "audioBytes": audio_bytes,
                "enhance": False,
            },
            target_sr,
        )
        wav_path = work_dir / "16k_mono.wav"
        wav_bytes = await asyncio.to_thread(encode_wav, decoded.samples, target_sr)
        await asyncio.to_thread(wav_path.write_bytes, wav_bytes)

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
        _update_stage_run(task_id, STAGE_FFMPEG_DECODE, "done")
        return {
            "wavPath": str(wav_path),
            "durationMs": decoded.duration_ms,
            "sampleRate": target_sr,
        }
    except Exception as exc:
        st = "cancelled" if "cancel" in str(exc).lower() else "error"
        _update_stage_run(task_id, STAGE_FFMPEG_DECODE, st, str(exc))
        raise


@DBOS.step()
async def step_diarize_turns(
    payload: dict[str, Any],
    decode_info: dict[str, Any],
) -> dict[str, Any]:
    task_id = str(payload["taskId"])
    _ensure_task_context(payload)
    meeting_id = payload.get("meetingId")
    work_dir = Path(payload["workDir"])
    wav_path = Path(decode_info["wavPath"])
    duration_ms = int(decode_info["durationMs"])
    target_sr = int(decode_info.get("sampleRate") or 16000)

    _check_aborted(task_id)
    _update_stage_run(task_id, STAGE_DIARIZE, "running")

    try:
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
            raw_wav = await asyncio.to_thread(wav_path.read_bytes)
            decoded = await asyncio.to_thread(decode_wav, raw_wav)
            samples = decoded.samples

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

        _check_aborted(task_id)

        diar_path = work_dir / "diarization.json"
        diar_doc: dict[str, Any] = {
            "turns": turns,
            "durationMs": duration_ms,
        }
        if precomputed_centroids is not None:
            diar_doc["precomputedCentroids"] = precomputed_centroids
        await asyncio.to_thread(diar_path.write_text, json.dumps(diar_doc), "utf-8")

        _update_stage_run(task_id, STAGE_DIARIZE, "done")
        return {
            "diarizationPath": str(diar_path),
            "turnCount": len(turns),
            "durationMs": duration_ms,
        }
    except Exception as exc:
        st = "cancelled" if "cancel" in str(exc).lower() else "error"
        _update_stage_run(task_id, STAGE_DIARIZE, st, str(exc))
        raise


@DBOS.step()
async def step_embed_and_match_speakers(
    payload: dict[str, Any],
    decode_info: dict[str, Any],
    diar_info: dict[str, Any],
) -> dict[str, Any]:
    task_id = str(payload["taskId"])
    _ensure_task_context(payload)
    meeting_id = payload.get("meetingId")
    work_dir = Path(payload["workDir"])
    wav_path = Path(decode_info["wavPath"])
    diar_path = Path(diar_info["diarizationPath"])
    duration_ms = int(decode_info["durationMs"])
    target_sr = int(decode_info.get("sampleRate") or 16000)

    _check_aborted(task_id)
    _update_stage_run(task_id, STAGE_SPEAKER_EMBED, "running")

    try:
        diar_doc = json.loads(await asyncio.to_thread(diar_path.read_text, "utf-8"))
        turns: list[dict[str, Any]] = diar_doc.get("turns") or []
        precomputed_centroids: dict[str, VoiceprintVector] | None = diar_doc.get("precomputedCentroids")

        pipe = _get_pipeline(task_id, payload.get("pipelineOptions"))
        ctx = _TASK_CONTEXTS.get(task_id)
        on_prog = ctx.on_progress if ctx is not None else None

        if precomputed_centroids is not None:
            cluster_centroids = precomputed_centroids
        else:
            raw_wav = await asyncio.to_thread(wav_path.read_bytes)
            decoded = await asyncio.to_thread(decode_wav, raw_wav)
            samples = decoded.samples
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

        _check_aborted(task_id)

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
                if p["id"].lower() in normalized_candidates
                or p["name"].strip().lower() in normalized_candidates
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
                meets_margin = (
                    (top1_score - top2_score >= effective_margin) if len(speaker_scores) > 1 else True
                )
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

        clusters_path = work_dir / "speaker_clusters.json"
        discovered_dicts = [d.to_dict() for d in discovered_map.values()]
        clusters_doc = {
            "clusterToResolvedName": cluster_to_resolved_name,
            "discoveredSpeakers": discovered_dicts,
            "clusterDiagnostics": cluster_diagnostics,
        }
        await asyncio.to_thread(clusters_path.write_text, json.dumps(clusters_doc), "utf-8")

        _update_stage_run(task_id, STAGE_SPEAKER_EMBED, "done")
        return {
            "clustersPath": str(clusters_path),
            "clusterToResolvedName": cluster_to_resolved_name,
            "discoveredSpeakers": discovered_dicts,
        }
    except Exception as exc:
        st = "cancelled" if "cancel" in str(exc).lower() else "error"
        _update_stage_run(task_id, STAGE_SPEAKER_EMBED, st, str(exc))
        raise


@DBOS.step()
async def step_transcribe_asr(
    payload: dict[str, Any],
    decode_info: dict[str, Any],
    diar_info: dict[str, Any],
) -> dict[str, Any]:
    task_id = str(payload["taskId"])
    _ensure_task_context(payload)
    meeting_id = payload.get("meetingId")
    work_dir = Path(payload["workDir"])
    wav_path = Path(decode_info["wavPath"])
    diar_path = Path(diar_info["diarizationPath"])
    duration_ms = int(decode_info["durationMs"])
    target_sr = int(decode_info.get("sampleRate") or 16000)

    _check_aborted(task_id)
    _update_stage_run(task_id, STAGE_ASR, "running")

    try:
        diar_doc = json.loads(await asyncio.to_thread(diar_path.read_text, "utf-8"))
        turns: list[dict[str, Any]] = diar_doc.get("turns") or []
        total_segments = len(turns)

        raw_wav = await asyncio.to_thread(wav_path.read_bytes)
        decoded = await asyncio.to_thread(decode_wav, raw_wav)
        samples = decoded.samples

        pipe = _get_pipeline(task_id, payload.get("pipelineOptions"))
        language = str(payload.get("language") or "en")
        model_id = payload.get("modelId")

        if model_id in ("mock", "mock-asr"):
            for _ in range(4):
                _check_aborted(task_id)
                await asyncio.sleep(0.02)

        use_whole_file_align = bool(
            payload.get("wholeFileAsr")
            and isinstance(pipe.asr_engine, CohereAsrEngine)
            and model_id not in ("mock", "mock-asr")
            and pipe.aligner is not None
            and turns
        )

        raw_asr_path = work_dir / "raw_asr.json"
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
                _check_aborted(task_id)
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

        await asyncio.to_thread(raw_asr_path.write_text, json.dumps(asr_doc), "utf-8")
        _update_stage_run(task_id, STAGE_ASR, "done")
        return {
            "rawAsrPath": str(raw_asr_path),
            "mode": asr_doc["mode"],
        }
    except Exception as exc:
        st = "cancelled" if "cancel" in str(exc).lower() else "error"
        _update_stage_run(task_id, STAGE_ASR, st, str(exc))
        raise


@DBOS.step()
async def step_align_and_finalize(
    payload: dict[str, Any],
    decode_info: dict[str, Any],
    diar_info: dict[str, Any],
    embed_info: dict[str, Any],
    asr_info: dict[str, Any],
) -> dict[str, Any]:
    task_id = str(payload["taskId"])
    _ensure_task_context(payload)
    meeting_id = payload.get("meetingId")
    wav_path = Path(decode_info["wavPath"])
    diar_path = Path(diar_info["diarizationPath"])
    raw_asr_path = Path(asr_info["rawAsrPath"])
    duration_ms = int(decode_info["durationMs"])
    target_sr = int(decode_info.get("sampleRate") or 16000)
    language = str(payload.get("language") or "en")

    _check_aborted(task_id)
    _update_stage_run(task_id, STAGE_ALIGN, "running")

    try:
        diar_doc = json.loads(await asyncio.to_thread(diar_path.read_text, "utf-8"))
        turns: list[dict[str, Any]] = diar_doc.get("turns") or []
        asr_doc = json.loads(await asyncio.to_thread(raw_asr_path.read_text, "utf-8"))
        cluster_to_resolved_name: dict[str, str] = embed_info.get("clusterToResolvedName") or {}
        discovered_speakers: list[dict[str, Any]] = embed_info.get("discoveredSpeakers") or []

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
            raw_wav = await asyncio.to_thread(wav_path.read_bytes)
            decoded = await asyncio.to_thread(decode_wav, raw_wav)
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

        if payload.get("cleanupWav", True):
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass

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

        _update_stage_run(task_id, STAGE_ALIGN, "done")
        return {
            "transcript": transcript.model_dump(exclude_none=True),
            "discoveredSpeakers": discovered_speakers,
        }
    except Exception as exc:
        st = "cancelled" if "cancel" in str(exc).lower() else "error"
        _update_stage_run(task_id, STAGE_ALIGN, st, str(exc))
        raise


@DBOS.workflow()
async def local_transcription_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload["taskId"])
    adopted = _ensure_task_context(payload)
    try:
        _check_aborted(task_id)

        await _emit_progress(
            task_id,
            {
                "stage": "queued",
                "percent": 2,
                "message": "Queued for audio decode (ffmpeg)...",
            },
        )
        decode_handle = await DBOS.enqueue_workflow_async(QUEUE_FFMPEG, step_decode_audio, payload)
        decode_info = await decode_handle.get_result(polling_interval_sec=0.05)
        _check_aborted(task_id)

        await _emit_progress(
            task_id,
            {
                "stage": "queued",
                "percent": 10,
                "totalMs": decode_info.get("durationMs"),
                "message": "Audio decoded. Queued for speaker diarization...",
            },
        )
        diar_handle = await DBOS.enqueue_workflow_async(
            QUEUE_DIARIZE, step_diarize_turns, payload, decode_info
        )
        diar_info = await diar_handle.get_result(polling_interval_sec=0.05)
        _check_aborted(task_id)

        await _emit_progress(
            task_id,
            {
                "stage": "queued",
                "percent": 33,
                "totalMs": decode_info.get("durationMs"),
                "message": "Diarization complete. Queued for speaker matching & ASR...",
            },
        )
        embed_handle = await DBOS.enqueue_workflow_async(
            QUEUE_SPEAKER_EMBED,
            step_embed_and_match_speakers,
            payload,
            decode_info,
            diar_info,
        )
        asr_handle = await DBOS.enqueue_workflow_async(
            QUEUE_ASR,
            step_transcribe_asr,
            payload,
            decode_info,
            diar_info,
        )
        embed_info, asr_info = await asyncio.gather(
            embed_handle.get_result(polling_interval_sec=0.05),
            asr_handle.get_result(polling_interval_sec=0.05),
        )
        _check_aborted(task_id)

        await _emit_progress(
            task_id,
            {
                "stage": "queued",
                "percent": 95,
                "totalMs": decode_info.get("durationMs"),
                "message": "ASR complete. Queued for alignment & finalization...",
            },
        )
        align_handle = await DBOS.enqueue_workflow_async(
            QUEUE_ALIGN,
            step_align_and_finalize,
            payload,
            decode_info,
            diar_info,
            embed_info,
            asr_info,
        )
        res_dict = await align_handle.get_result(polling_interval_sec=0.05)
        ctx = _TASK_CONTEXTS.get(task_id)
        meeting_id = payload.get("meetingId")
        if (ctx is None or not ctx.has_live_caller) and meeting_id and callable(_RECOVERY_COMPLETE_SINK):
            wf_id = getattr(DBOS, "workflow_id", None)
            comp_res = _RECOVERY_COMPLETE_SINK(
                str(meeting_id),
                res_dict,
                str(wf_id) if wf_id else None,
            )
            if inspect.isawaitable(comp_res):
                await comp_res
        return res_dict
    finally:
        if adopted:
            unregister_task_context(task_id)


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
async def run_transcription_driver(job_id: str, metadata: dict[str, Any] | None = None) -> Any:
    """Workflow wrapper for a registered driver callable.

    Unlike :func:`step_run_registered_job` (a step, for leaf work), this runs
    in *workflow* context so the driver may enqueue child workflows — DBOS
    forbids child enqueue from inside a step. Transcription drivers need
    this because the local path enqueues ``local_transcription_workflow``
    and its stage children.
    """
    _ = metadata
    fn = _GENERIC_JOBS.pop(job_id, None)
    if fn is None:
        return None
    res = fn()
    if inspect.isawaitable(res):
        return await res
    return res


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
    """Enqueues ``fn`` as a *workflow* (not a step) on ``queue_name``.

    Use this instead of :func:`enqueue_stage_job` when the callable itself
    enqueues child workflows — e.g. transcription drivers, whose local path
    fans out to ``local_transcription_workflow`` and its stage queues.
    """
    job_id = str(uuid.uuid4())
    _GENERIC_JOBS[job_id] = fn
    wf_id = workflow_id or f"job:{queue_name}:{job_id}"
    meta = {"label": label or queue_name, "meetingId": meeting_id}
    with SetWorkflowID(wf_id), SetEnqueueOptions(deduplication_id=deduplication_id):
        handle = await DBOS.enqueue_workflow_async(queue_name, run_transcription_driver, job_id, meta)
    return handle


def configure_insights_queue(insights_service: Any) -> None:
    """Register the durable ``insights`` queue hook for ``insights_service``.

    ``InsightsService.enqueue_analyze`` calls the hook; the hook enqueues
    ``analyze_meeting`` as a DBOS stage job, deduplicated per meeting.
    """
    from ..services.insights import extract as insights_extract

    async def _hook(meeting_id: str) -> bool:
        await enqueue_stage_job(
            QUEUE_INSIGHTS,
            lambda: insights_service.analyze_meeting(meeting_id),
            deduplication_id=f"insights:{meeting_id}",
            workflow_id=f"insights:{meeting_id}:{uuid.uuid4().hex[:8]}",
            label="insights:analyze",
            meeting_id=meeting_id,
        )
        return True

    insights_extract.set_enqueue_hook(_hook)


def _extract_job_details(wf: Any) -> tuple[str, str | None]:
    raw_name = str(getattr(wf, "name", "") or "job")
    short_name = raw_name.split(".")[-1].replace("<temp>.", "")
    meeting_id: str | None = None

    inp = getattr(wf, "input", None)
    if isinstance(inp, dict):
        args = inp.get("args") or ()
        for arg in args:
            if isinstance(arg, dict):
                if not meeting_id and arg.get("meetingId"):
                    meeting_id = str(arg["meetingId"])
                if short_name in ("step_run_registered_job", "run_transcription_driver") and arg.get("label"):
                    short_name = str(arg["label"])

    wf_id = str(getattr(wf, "workflow_id", "") or "")
    if not meeting_id and wf_id.startswith("local-transcribe:"):
        parts = wf_id.split(":")
        if len(parts) >= 2 and parts[1]:
            meeting_id = parts[1]

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

        if spec.name != QUEUE_PIPELINE:
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

    parent_wfs = [
        _serialize_wf(w)
        for w in sorted(
            [
                w
                for w in all_wfs
                if getattr(w, "queue_name", None) in (None, QUEUE_PIPELINE)
                and getattr(w, "status", None) == "PENDING"
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


async def run_staged_local_transcription(
    options: dict[str, Any],
    *,
    pipeline: LocalTranscriptionPipeline | None = None,
    pipeline_options: dict[str, Any] | None = None,
    db: Engine | None = None,
    now_fn: Callable[[], int] | None = None,
    workflow_id: str | None = None,
) -> LocalTranscriptionResult:
    task_id = str(uuid.uuid4())
    meeting_id = options.get("meetingId") or options.get("meeting_id")
    signal = options.get("signal")
    on_progress = options.get("onProgress") or options.get("on_progress")
    audio_path = options.get("audioPath") or options.get("audio_path")
    audio_bytes = options.get("audioBytes") or options.get("audio_bytes")

    temp_work_dir: tempfile.TemporaryDirectory[str] | None = None
    explicit_work_dir = options.get("workDir") or options.get("work_dir")
    if explicit_work_dir:
        work_dir = str(explicit_work_dir)
    elif audio_path and Path(audio_path).exists():
        meeting_folder = Path(audio_path).resolve().parent.parent
        work_dir = str(meeting_folder / "transcripts" / ".work" / task_id[:8])
    else:
        temp_work_dir = tempfile.TemporaryDirectory(prefix="olive-stage-")
        work_dir = temp_work_dir.name

    wf_id = workflow_id or f"local-transcribe:{meeting_id or task_id}:{task_id[:8]}"
    ctx = TaskExecutionContext(
        task_id=task_id,
        meeting_id=str(meeting_id) if meeting_id else None,
        signal=signal,
        on_progress=on_progress,
        pipeline=pipeline,
        pipeline_options=pipeline_options,
        audio_bytes=bytes(audio_bytes) if audio_bytes is not None else None,
        db=db,
        now_fn=now_fn,
        workflow_id=wf_id,
    )
    register_task_context(ctx)

    raw_enrolled = options.get("enrolledSpeakers") or options.get("enrolled_speakers") or []
    serialized_enrolled = [s.model_dump() if isinstance(s, EnrolledSpeaker) else s for s in raw_enrolled]

    payload: dict[str, Any] = {
        "taskId": task_id,
        "meetingId": meeting_id,
        "workDir": work_dir,
        "audioPath": str(audio_path) if audio_path else None,
        "language": options.get("language") or "en",
        "modelId": options.get("modelId") or options.get("model_id"),
        "enrolledSpeakers": serialized_enrolled,
        "candidateSpeakers": options.get("candidateSpeakers") or options.get("candidate_speakers"),
        "expectedSpeakerCount": options.get("expectedSpeakerCount") or options.get("expected_speaker_count"),
        "similarityThreshold": options.get("similarityThreshold") or options.get("similarity_threshold"),
        "decisionMargin": options.get("decisionMargin") or options.get("decision_margin"),
        "clusteringThreshold": options.get("clusteringThreshold") or options.get("clustering_threshold"),
        "wholeFileAsr": bool(options.get("wholeFileAsr") or options.get("whole_file_asr")),
        "pipelineOptions": pipeline_options or {},
        "cleanupWav": temp_work_dir is None,
    }

    abort_Event = asyncio.Event()

    def _on_abort() -> None:
        abort_Event.set()

    if signal is not None and hasattr(signal, "add_event_listener"):
        signal.add_event_listener("abort", _on_abort)

    try:
        if _is_signal_aborted(signal):
            raise RuntimeError("Transcription cancelled by user")

        await _emit_progress(
            task_id,
            {
                "stage": "queued",
                "percent": 0,
                "message": "Queued in transcription pipeline...",
            },
        )
        dedup_key = f"pipeline:{meeting_id}" if meeting_id else None
        with SetWorkflowID(wf_id), SetEnqueueOptions(deduplication_id=dedup_key):
            wf_handle = await DBOS.enqueue_workflow_async(
                QUEUE_PIPELINE,
                local_transcription_workflow,
                payload,
            )

        result_task = asyncio.create_task(wf_handle.get_result(polling_interval_sec=0.05))
        abort_wait_task = asyncio.create_task(abort_Event.wait())

        done, _ = await asyncio.wait(
            {result_task, abort_wait_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if abort_wait_task in done and not result_task.done():
            if not is_shutting_down():
                try:
                    await DBOS.cancel_workflow_async(wf_id)
                except Exception:
                    pass
                result_task.cancel()
                raise RuntimeError("Transcription cancelled by user")
            result_task.cancel()
            raise RuntimeError("Transcription cancelled: server shutting down")

        abort_wait_task.cancel()
        res_dict = await result_task
        return LocalTranscriptionResult(
            transcript=Transcript.model_validate(res_dict["transcript"]),
            discoveredSpeakers=[
                DiscoveredSpeakerVoiceprint(**d) for d in res_dict.get("discoveredSpeakers", [])
            ],
        )
    finally:
        unregister_task_context(task_id)
        if temp_work_dir is not None:
            try:
                temp_work_dir.cleanup()
            except Exception:
                pass


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
            first = args[0] if args else {}
            if isinstance(first, dict) and first.get("meetingId"):
                mid = str(first["meetingId"])
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
            if candidate.startswith("local-transcribe:"):
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
        raw_name = str(r["name"] or "")
        is_active = st == "PENDING"

        q_list = waiting_by_queue.get(q_name, [])
        q_total = len(q_list)
        try:
            q_pos = q_list.index(wf_id) + 1
        except ValueError:
            q_pos = 1

        rank = 0
        prog: dict[str, Any] | None = None

        if q_name == QUEUE_ALIGN or "step_align_and_finalize" in raw_name:
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
        elif q_name == QUEUE_ASR or "step_transcribe_asr" in raw_name:
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
        elif q_name == QUEUE_SPEAKER_EMBED or "step_embed_and_match_speakers" in raw_name:
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
        elif q_name == QUEUE_DIARIZE or "step_diarize_turns" in raw_name:
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
        elif q_name == QUEUE_FFMPEG or "step_decode_audio" in raw_name:
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
        elif "local_transcription_workflow" in raw_name:
            rank = 10
            prog = {
                "stage": "queued",
                "percent": 2,
                "message": "Queued in transcription pipeline...",
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
                    or wid.startswith(f"local-transcribe:{meeting_id}:")
                    or pid.startswith(f"local-transcribe:{meeting_id}:")
                    # Durable per-meeting transcription jobs enqueued via
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
                    # Insights analysis jobs enqueued via the hook registered by
                    # configure_insights_queue use ``insights:{meeting_id}:{hex8}``
                    # workflow ids — keep this prefix matcher in sync with that format.
                    or wid.startswith(f"insights:{meeting_id}:")
                    or pid.startswith(f"insights:{meeting_id}:")
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
    "configure_insights_queue",
    "configure_insights_queue",
    "QUEUE_ALIGN",
    "QUEUE_ASSEMBLYAI",
    "QUEUE_ASR",
    "QUEUE_DIARIZE",
    "QUEUE_FFMPEG",
    "QUEUE_INSIGHTS",
    "QUEUE_PIPELINE",
    "QUEUE_PLAUD_SYNC",
    "QUEUE_SPEAKER_EMBED",
    "QUEUE_SPEECHMATICS",
    "QUEUE_SUMMARIZE",
    "STAGE_ALIGN",
    "STAGE_ASR",
    "STAGE_DIARIZE",
    "STAGE_FFMPEG_DECODE",
    "STAGE_SPEAKER_EMBED",
    "TaskExecutionContext",
    "cancel_dbos_workflows_for_meeting",
    "configure_recovery_runtime",
    "enqueue_driver_job",
    "enqueue_stage_job",
    "get_dbos_active_progress_map",
    "get_debug_queues_snapshot",
    "invalidate_dbos_progress_cache",
    "local_transcription_workflow",
    "register_task_context",
    "run_staged_local_transcription",
    "run_transcription_driver",
    "step_align_and_finalize",
    "step_decode_audio",
    "step_diarize_turns",
    "step_embed_and_match_speakers",
    "step_enhance_audio_file",
    "step_run_registered_job",
    "step_transcribe_asr",
    "unregister_task_context",
]
