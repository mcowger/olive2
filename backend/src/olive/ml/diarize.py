"""Speaker diarization powered by pyannote.audio 4 speaker-diarization-community-1."""

from __future__ import annotations

import asyncio
import gc
import inspect
import math
import os
import threading
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from ..logging_db import logger
from ..settings import get_effective_hf_token
from .audio import resample
from .embed import WeSpeakerEmbeddingExtractor
from .protocols import LocalSpeakerSegment, SpeakerEmbedder, VoiceprintVector
from .scoring import merge_weighted_voiceprint_vectors

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

_DIARIZE_LOCK = threading.Lock()
# release() must never wait indefinitely behind another meeting's native inference
# (which holds _DIARIZE_LOCK for the whole pipeline call and may hang).
_RELEASE_LOCK_TIMEOUT_SEC = 5.0
_CACHED_PIPELINE: Any = None
MAX_VOICEPRINT_TURNS_PER_CLUSTER = 8
MAX_VOICEPRINT_TURN_SECONDS = 12.0


def _default_diarizer_threads() -> int:
    env_val = os.environ.get("OLIVE_DIARIZATION_THREADS")
    if env_val:
        try:
            parsed = int(env_val)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    cores = os.cpu_count() or 4
    return max(2, min(8, cores // 2))


def _patch_pipeline_for_fast_cpu(pipe: Any, step_sec: float = 2.5) -> None:
    """Optimizes pyannote SpeakerDiarization on CPU by skipping all-zero inactive speaker masks."""
    if getattr(pipe, "_olive_cpu_patched", False):
        return

    try:
        if hasattr(pipe, "_segmentation") and hasattr(pipe._segmentation, "step"):
            pipe._segmentation.step = step_sec
    except Exception:
        pass

    if not hasattr(pipe, "get_embeddings") or not hasattr(pipe, "_embedding"):
        return

    def _fast_get_embeddings(
        file: Any,
        binary_segmentations: Any,
        exclude_overlap: bool = False,
        hook: Callable[..., Any] | None = None,
    ) -> np.ndarray:
        from pyannote.core import SlidingWindowFeature

        duration = binary_segmentations.sliding_window.duration
        num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

        if exclude_overlap:
            min_num_samples = pipe._embedding.min_num_samples
            num_samples = duration * pipe._embedding.sample_rate
            min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)
            clean_frames = 1.0 * (np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2)
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data * clean_frames,
                binary_segmentations.sliding_window,
            )
        else:
            min_num_frames = -1
            clean_segmentations = binary_segmentations

        active_items: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
        for c_idx, ((chunk, masks), (_, clean_masks)) in enumerate(
            zip(binary_segmentations, clean_segmentations, strict=False)
        ):
            masks_np = np.nan_to_num(masks, nan=0.0).astype(np.float32)
            clean_np = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)
            waveform: torch.Tensor | None = None
            for s_idx, (mask, clean_mask) in enumerate(zip(masks_np.T, clean_np.T, strict=False)):
                # Skip speaker slots that have 0 active frames in this window
                if float(np.sum(mask)) <= 0.0:
                    continue
                used_mask = clean_mask if float(np.sum(clean_mask)) > min_num_frames else mask
                if waveform is None:
                    waveform, _ = pipe._audio.crop(file, chunk, mode="pad")
                assert waveform is not None
                active_items.append((c_idx, s_idx, waveform[None], torch.from_numpy(used_mask)[None]))

        dim = int(pipe._embedding.dimension)
        embeddings = np.full((num_chunks, num_speakers, dim), np.nan, dtype=np.float32)
        if not active_items:
            if hook is not None:
                hook("embeddings", embeddings, total=1, completed=1)
            return embeddings

        bs = max(1, int(getattr(pipe, "embedding_batch_size", 32)))
        batch_count = max(1, math.ceil(len(active_items) / bs))
        if hook is not None:
            hook(
                "embeddings",
                None,
                total=batch_count,
                completed=0,
            )

        for b_idx in range(0, len(active_items), bs):
            batch = active_items[b_idx : b_idx + bs]
            w_batch = torch.vstack([it[2] for it in batch])
            m_batch = torch.vstack([it[3] for it in batch])
            emb_batch: np.ndarray = pipe._embedding(w_batch, masks=m_batch)
            for (c_idx, s_idx, _, _), emb_vec in zip(batch, emb_batch, strict=False):
                embeddings[c_idx, s_idx] = emb_vec
            if hook is not None:
                hook(
                    "embeddings",
                    emb_batch,
                    total=batch_count,
                    completed=(b_idx // bs) + 1,
                )

        return embeddings

    pipe.get_embeddings = _fast_get_embeddings
    pipe._olive_cpu_patched = True


class PyannoteSpeakerDiarizer:
    """Neural speaker diarizer using pyannote speaker-diarization-community-1 + WeSpeaker embeddings."""

    def __init__(
        self,
        embedding_extractor: SpeakerEmbedder | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        cfg = {**(config or {}), **kwargs}
        self.embedding_extractor = embedding_extractor or WeSpeakerEmbeddingExtractor(
            {"numThreads": cfg.get("numThreads") or _default_diarizer_threads()}
        )
        self.clustering_threshold = float(cfg.get("clusteringThreshold") or 0.50)
        self.num_threads = int(cfg.get("numThreads") or _default_diarizer_threads())
        self.min_duration_on = float(cfg.get("minDurationOn") or 0.3)
        self.min_duration_off = float(cfg.get("minDurationOff") or 0.5)
        self.segmentation_step_sec = float(
            os.environ.get("OLIVE_DIARIZATION_STEP_SEC") or cfg.get("segmentationStepSec") or 2.5
        )
        self._pipeline: Any = None

    def _ensure_pipeline_sync(self) -> Any:
        global _CACHED_PIPELINE
        if self._pipeline is not None:
            return self._pipeline

        with _DIARIZE_LOCK:
            if self._pipeline is not None:
                return self._pipeline
            if _CACHED_PIPELINE is not None:
                self._pipeline = _CACHED_PIPELINE
                return self._pipeline

            t0 = time.perf_counter()
            torch.set_num_threads(self.num_threads)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    from pyannote.audio.telemetry import set_telemetry_metrics

                    set_telemetry_metrics(False)
                except Exception:
                    pass
                from pyannote.audio import Pipeline

                hf_token = get_effective_hf_token()
                candidates = (
                    [
                        ("pyannote/speaker-diarization-community-1", hf_token),
                        ("pyannote-community/speaker-diarization-community-1", hf_token),
                        ("pyannote-community/speaker-diarization-community-1", None),
                    ]
                    if hf_token
                    else [
                        ("pyannote-community/speaker-diarization-community-1", None),
                        ("pyannote/speaker-diarization-community-1", None),
                    ]
                )
                last_err: Exception | None = None
                loaded_repo: str | None = None
                for repo_id, tok in candidates:
                    try:
                        kwargs: dict[str, Any] = {}
                        if tok:
                            kwargs["token"] = tok
                        pipe: Any = Pipeline.from_pretrained(repo_id, **kwargs)
                        if pipe is not None:
                            pipe.to(torch.device("cpu"))
                            _patch_pipeline_for_fast_cpu(pipe, step_sec=self.segmentation_step_sec)
                        _CACHED_PIPELINE = pipe
                        self._pipeline = pipe
                        loaded_repo = repo_id
                        break
                    except Exception as exc:
                        last_err = exc
                if self._pipeline is None and last_err is not None:
                    raise last_err

                logger.info(
                    "Loaded pyannote speaker diarization pipeline on CPU",
                    {
                        "category": "transcription",
                        "repo": loaded_repo,
                        "threads": self.num_threads,
                        "segmentationStepSec": self.segmentation_step_sec,
                        "embeddingBatchSize": getattr(self._pipeline, "embedding_batch_size", None),
                        "elapsedMs": int(round((time.perf_counter() - t0) * 1000)),
                    },
                )
            return self._pipeline

    def _run_pyannote_sync(
        self,
        samples_16k: np.ndarray,
        expected_speaker_count: int | None = None,
        progress_cb: Callable[[dict[str, Any]], Any] | None = None,
        meeting_id: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> list[tuple[float, float, int]]:
        pipe = self._ensure_pipeline_sync()
        torch.set_num_threads(self.num_threads)
        duration_ms = int(round((len(samples_16k) / 16000.0) * 1000))
        waveform = torch.from_numpy(samples_16k).unsqueeze(0)
        kwargs: dict[str, Any] = {}
        if expected_speaker_count and expected_speaker_count > 0:
            kwargs["num_speakers"] = int(expected_speaker_count)

        step_timers: dict[str, float] = {}
        last_batch_time = time.perf_counter()

        def _emit_sync(payload: dict[str, Any]) -> None:
            if not callable(progress_cb):
                return
            try:
                res = progress_cb(payload)
                if inspect.iscoroutine(res):
                    if loop is not None and not loop.is_closed():
                        asyncio.run_coroutine_threadsafe(res, loop)
                    else:
                        res.close()
            except Exception:
                pass

        def _pyannote_hook(
            step_name: str,
            step_artifact: Any,
            file: Any = None,
            total: int | None = None,
            completed: int | None = None,
        ) -> None:
            nonlocal last_batch_time
            from ..workflows.runtime import is_shutting_down

            if is_shutting_down():
                raise RuntimeError("Transcription cancelled: server shutting down")
            now = time.perf_counter()
            if step_name not in step_timers:
                step_timers[step_name] = now
                last_batch_time = now

            step_elapsed_ms = int(round((now - step_timers[step_name]) * 1000))
            batch_elapsed_ms = int(round((now - last_batch_time) * 1000))

            if step_name == "segmentation":
                if completed is not None and total is not None:
                    last_batch_time = now
                    pct = 15 + int(round(4.0 * min(1.0, completed / max(1, total))))
                    _emit_sync(
                        {
                            "stage": "diarizing",
                            "percent": pct,
                            "current": min(completed, total),
                            "total": total,
                            "totalMs": duration_ms,
                            "message": f"Diarization (1/4): Running neural speech segmentation (window {min(completed, total)}/{total})...",
                        }
                    )
                else:
                    num_chunks = int(step_artifact.data.shape[0]) if hasattr(step_artifact, "data") else None
                    logger.info(
                        "Diarization (1/4): Neural speech segmentation complete",
                        {
                            "category": "transcription",
                            "meetingId": meeting_id,
                            "windows": num_chunks,
                            "audioDurationMs": duration_ms,
                            "elapsedMs": step_elapsed_ms,
                        },
                    )
            elif step_name == "speaker_counting":
                max_spk = (
                    int(np.nanmax(step_artifact.data))
                    if hasattr(step_artifact, "data") and step_artifact.data.size > 0
                    else 0
                )
                logger.info(
                    "Diarization (2/4): Frame-level speaker counting complete",
                    {
                        "category": "transcription",
                        "meetingId": meeting_id,
                        "maxConcurrentSpeakers": max_spk,
                        "elapsedMs": step_elapsed_ms,
                    },
                )
                _emit_sync(
                    {
                        "stage": "diarizing",
                        "percent": 20,
                        "totalMs": duration_ms,
                        "message": f"Diarization (2/4): Detected up to {max_spk} concurrent speakers. Starting neural speaker embeddings...",
                    }
                )
            elif step_name == "embeddings":
                if completed is not None and total is not None:
                    last_batch_time = now
                    pct = 21 + int(round(10.0 * min(1.0, completed / max(1, total))))
                    if completed == 0:
                        logger.info(
                            "Diarization (3/4): Starting neural speaker embedding batches",
                            {
                                "category": "transcription",
                                "meetingId": meeting_id,
                                "totalBatches": total,
                                "batchSize": getattr(pipe, "embedding_batch_size", 32),
                            },
                        )
                    else:
                        logger.info(
                            f"Diarization (3/4): Speaker embedding batch {completed}/{total} complete",
                            {
                                "category": "transcription",
                                "meetingId": meeting_id,
                                "batch": completed,
                                "totalBatches": total,
                                "batchElapsedMs": batch_elapsed_ms,
                                "stepElapsedMs": step_elapsed_ms,
                            },
                        )
                    _emit_sync(
                        {
                            "stage": "diarizing",
                            "percent": pct,
                            "current": completed,
                            "total": total,
                            "totalMs": duration_ms,
                            "message": f"Diarization (3/4): Extracting speaker embeddings (batch {completed}/{total})...",
                        }
                    )
            elif step_name == "discrete_diarization":
                logger.info(
                    "Diarization (4/4): VBx speaker clustering & turn reconstruction complete",
                    {
                        "category": "transcription",
                        "meetingId": meeting_id,
                        "elapsedMs": step_elapsed_ms,
                    },
                )
                _emit_sync(
                    {
                        "stage": "diarizing",
                        "percent": 32,
                        "totalMs": duration_ms,
                        "message": "Diarization (4/4): Speaker clustering complete. Extracting 256-d voiceprints...",
                    }
                )

        with _DIARIZE_LOCK, warnings.catch_warnings(), torch.inference_mode():
            warnings.simplefilter("ignore")
            out = pipe(
                {"waveform": waveform, "sample_rate": 16000, "uri": meeting_id or "audio"},
                hook=_pyannote_hook,
                **kwargs,
            )

        diar = (
            getattr(out, "exclusive_speaker_diarization", None)
            or getattr(out, "speaker_diarization", None)
            or out
        )

        label_map: dict[str, int] = {}
        all_turns: list[tuple[float, float, int]] = []
        min_dur = max(0.05, float(self.min_duration_on))
        raw_segments: list[tuple[float, float, int]] = []
        for turn, _, label in diar.itertracks(yield_label=True):
            lbl_str = str(label)
            if lbl_str not in label_map:
                label_map[lbl_str] = len(label_map)
            spk_idx = label_map[lbl_str]
            start_s = float(turn.start)
            end_s = float(turn.end)
            dur_s = end_s - start_s
            if dur_s >= 0.05:
                all_turns.append((start_s, end_s, spk_idx))
            if dur_s >= min_dur:
                raw_segments.append((start_s, end_s, spk_idx))
        return raw_segments if raw_segments else all_turns

    async def segment_turns(
        self,
        samples: np.ndarray | list[float],
        sample_rate: int = 16000,
        clustering_threshold_override: float | None = None,
        expected_speaker_count: int | None = None,
        on_progress: Callable[[dict[str, Any]], Any] | None = None,
        meeting_id: str | None = None,
    ) -> list[dict[str, Any]]:
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1).astype(np.float32)
        if arr.size == 0:
            return []

        if sample_rate != 16000:
            arr = resample(arr, sample_rate, 16000)
            sample_rate = 16000

        total_duration_ms = int(round((len(arr) / float(sample_rate)) * 1000))

        logger.info(
            "Starting neural speaker diarization",
            {
                "category": "transcription",
                "meetingId": meeting_id,
                "audioDurationMs": total_duration_ms,
                "expectedSpeakerCount": expected_speaker_count,
                "threads": self.num_threads,
            },
        )

        raw_segments: list[tuple[float, float, int]] = []
        running_loop = asyncio.get_running_loop()
        try:
            self._ensure_pipeline_sync()
            raw_segments = await asyncio.to_thread(
                self._run_pyannote_sync,
                arr,
                expected_speaker_count,
                on_progress,
                meeting_id,
                running_loop,
            )
        except Exception as exc:
            logger.warn(
                "pyannote diarization encountered fallback to single speaker",
                {
                    "category": "transcription",
                    "meetingId": meeting_id,
                    "error": str(exc),
                },
            )
            raw_segments = []

        if not raw_segments:
            return [{"startMs": 0, "endMs": total_duration_ms, "speakerId": "Speaker 1"}]

        raw_segments.sort(key=lambda t: t[0])
        coalesced_raw: list[tuple[float, float, int]] = []
        cur_s, cur_e, cur_spk = raw_segments[0]
        for nxt_s, nxt_e, nxt_spk in raw_segments[1:]:
            gap_ms = (nxt_s - cur_e) * 1000.0
            if nxt_spk == cur_spk and 0.0 <= gap_ms <= 1500.0:
                cur_e = max(cur_e, nxt_e)
            else:
                coalesced_raw.append((cur_s, cur_e, cur_spk))
                cur_s, cur_e, cur_spk = nxt_s, nxt_e, nxt_spk
        coalesced_raw.append((cur_s, cur_e, cur_spk))

        turns: list[dict[str, Any]] = []
        speaker_clusters: set[str] = set()
        for start_s, end_s, spk_idx in coalesced_raw:
            start_ms = int(round(start_s * 1000))
            end_ms = int(round(end_s * 1000))
            start_sample = max(0, int(start_s * sample_rate))
            end_sample = min(len(arr), int(np.ceil(end_s * sample_rate)))
            if end_sample <= start_sample:
                continue
            speaker_id = f"Speaker {spk_idx + 1}"
            speaker_clusters.add(speaker_id)
            turns.append(
                {
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "speakerId": speaker_id,
                }
            )

        if not turns:
            return [{"startMs": 0, "endMs": total_duration_ms, "speakerId": "Speaker 1"}]

        logger.info(
            "Coalesced diarization turns across speaker clusters",
            {
                "category": "transcription",
                "meetingId": meeting_id,
                "rawTurns": len(raw_segments),
                "coalescedTurns": len(turns),
                "speakerClusters": len(speaker_clusters),
            },
        )
        return turns

    async def extract_cluster_voiceprints(
        self,
        samples: np.ndarray | list[float],
        turns: list[dict[str, Any]],
        sample_rate: int = 16000,
        on_progress: Callable[[dict[str, Any]], Any] | None = None,
        meeting_id: str | None = None,
    ) -> tuple[list[LocalSpeakerSegment], dict[str, VoiceprintVector]]:
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1).astype(np.float32)
        if arr.size == 0 or not turns:
            return [], {}

        if sample_rate != 16000:
            arr = resample(arr, sample_rate, 16000)
            sample_rate = 16000

        total_duration_ms = int(round((len(arr) / float(sample_rate)) * 1000))
        candidate_segments: list[LocalSpeakerSegment] = []
        turns_by_cluster: dict[str, list[int]] = {}

        for turn in turns:
            start_ms = int(turn["startMs"])
            end_ms = int(turn["endMs"])
            speaker_id = str(turn.get("speakerId") or "Speaker 1")
            start_sample = max(0, int((start_ms / 1000.0) * sample_rate))
            end_sample = min(len(arr), int(np.ceil((end_ms / 1000.0) * sample_rate)))
            if end_sample <= start_sample:
                continue
            seg_samples = arr[start_sample:end_sample]
            seg_idx = len(candidate_segments)
            candidate_segments.append(
                LocalSpeakerSegment(
                    startMs=start_ms,
                    endMs=end_ms,
                    speakerId=speaker_id,
                    samples=seg_samples,
                    embedding=None,
                )
            )
            turns_by_cluster.setdefault(speaker_id, []).append(seg_idx)

        if not candidate_segments:
            emb = await self.embedding_extractor.extract(arr, sample_rate)
            seg = LocalSpeakerSegment(
                startMs=0,
                endMs=total_duration_ms,
                speakerId="Speaker 1",
                samples=arr,
                embedding=emb,
            )
            return [seg], {"Speaker 1": emb}

        vp_t0 = time.perf_counter()
        max_clip_samples = int(MAX_VOICEPRINT_TURN_SECONDS * sample_rate)
        num_clusters = len(turns_by_cluster)
        cluster_centroids: dict[str, VoiceprintVector] = {}

        for cluster_i, (speaker_id, seg_indices) in enumerate(turns_by_cluster.items(), 1):
            ranked_indices = sorted(
                seg_indices,
                key=lambda idx: candidate_segments[idx].endMs - candidate_segments[idx].startMs,
                reverse=True,
            )
            selected_indices = ranked_indices[:MAX_VOICEPRINT_TURNS_PER_CLUSTER]
            weighted_vectors: list[dict[str, Any]] = []

            if callable(on_progress):
                pct = 33 + int(round(5.0 * (cluster_i / max(1, num_clusters))))
                res = on_progress(
                    {
                        "stage": "diarizing",
                        "percent": pct,
                        "current": cluster_i,
                        "total": num_clusters,
                        "totalMs": total_duration_ms,
                        "message": f"Diarization: Extracting 256-d WeSpeaker voiceprints for {speaker_id} ({len(selected_indices)} representative turns)...",
                    }
                )
                if inspect.isawaitable(res):
                    await res

            from ..workflows.runtime import is_shutting_down

            for idx in selected_indices:
                if is_shutting_down():
                    raise RuntimeError("Transcription cancelled: server shutting down")
                seg = candidate_segments[idx]
                clip = seg.samples
                if len(clip) > max_clip_samples:
                    mid = len(clip) // 2
                    half = max_clip_samples // 2
                    clip = clip[mid - half : mid - half + max_clip_samples]
                emb = await self.embedding_extractor.extract(clip, sample_rate)
                seg.embedding = emb
                weighted_vectors.append(
                    {
                        "vector": emb,
                        "weight": max(1, seg.endMs - seg.startMs),
                    }
                )

            cluster_centroid: VoiceprintVector = merge_weighted_voiceprint_vectors(weighted_vectors)
            cluster_centroids[speaker_id] = cluster_centroid
            for idx in seg_indices:
                if candidate_segments[idx].embedding is None:
                    candidate_segments[idx].embedding = cluster_centroid

        logger.info(
            "Completed WeSpeaker voiceprint extraction",
            {
                "category": "transcription",
                "meetingId": meeting_id,
                "speakerClusters": num_clusters,
                "coalescedTurns": len(candidate_segments),
                "voiceprintElapsedMs": int(round((time.perf_counter() - vp_t0) * 1000)),
            },
        )
        return candidate_segments, cluster_centroids

    async def diarize(
        self,
        samples: np.ndarray | list[float],
        sample_rate: int = 16000,
        clustering_threshold_override: float | None = None,
        expected_speaker_count: int | None = None,
        on_progress: Callable[[dict[str, Any]], Any] | None = None,
        meeting_id: str | None = None,
    ) -> list[LocalSpeakerSegment]:
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1).astype(np.float32)
        if arr.size == 0:
            return []

        if sample_rate != 16000:
            arr = resample(arr, sample_rate, 16000)
            sample_rate = 16000

        turns = await self.segment_turns(
            arr,
            sample_rate,
            clustering_threshold_override=clustering_threshold_override,
            expected_speaker_count=expected_speaker_count,
            on_progress=on_progress,
            meeting_id=meeting_id,
        )
        segments, _ = await self.extract_cluster_voiceprints(
            arr,
            turns,
            sample_rate,
            on_progress=on_progress,
            meeting_id=meeting_id,
        )
        return segments

    def release(self, clear_cache: bool = False) -> None:
        global _CACHED_PIPELINE
        if not _DIARIZE_LOCK.acquire(timeout=_RELEASE_LOCK_TIMEOUT_SEC):
            # Another pipeline is inside native inference. Drop only our own reference
            # (an atomic attribute store) and leave the shared cache for a later
            # release rather than blocking teardown behind a possibly hung call.
            self._pipeline = None
            logger.warn(
                "Diarizer release skipped shared-cache clear: diarize lock busy",
                {"category": "transcription", "timeoutSec": _RELEASE_LOCK_TIMEOUT_SEC},
            )
            return
        try:
            self._pipeline = None
            if clear_cache:
                _CACHED_PIPELINE = None
                gc.collect()
        finally:
            _DIARIZE_LOCK.release()


SherpaSpeakerDiarizer = PyannoteSpeakerDiarizer
LocalSpeakerDiarizer = PyannoteSpeakerDiarizer
