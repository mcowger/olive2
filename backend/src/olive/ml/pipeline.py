"""Local transcription, diarization, forced alignment, and voiceprint pipeline (port of pipeline.ts)."""

from __future__ import annotations

import inspect
import json
import time
from typing import Any

from ..logging_db import logger
from ..schemas import (
    EnrolledSpeaker,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    coalesce_speaker_segments,
)
from .align import CtcForcedAligner, proportional_word_timestamps
from .asr_cohere import CohereAsrEngine
from .audio import load_audio_samples
from .diarize import PyannoteSpeakerDiarizer
from .embed import WeSpeakerEmbeddingExtractor
from .protocols import (
    Aligner,
    AsrBackend,
    Diarizer,
    DiscoveredSpeakerVoiceprint,
    LocalTranscriptionResult,
    SpeakerEmbedder,
    VoiceprintVector,
)
from .scoring import (
    merge_weighted_voiceprint_vectors,
    normalize_vector,
    score_against_trusted_vectors,
)


def _is_aborted(signal: Any) -> bool:
    if signal is None:
        return False
    if hasattr(signal, "aborted"):
        return bool(signal.aborted)
    if hasattr(signal, "is_set") and callable(signal.is_set):
        return bool(signal.is_set())
    return False


async def _emit(cb: Any, payload: dict[str, Any]) -> None:
    if not callable(cb):
        return
    res = cb(payload)
    if inspect.isawaitable(res):
        await res


class LocalTranscriptionPipeline:
    """Orchestrates decoding, diarization, voiceprint matching, ASR, and forced alignment."""

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        custom_diarizer: Diarizer | Any | None = None,
        custom_extractor: SpeakerEmbedder | Any | None = None,
        custom_asr_engine: AsrBackend | Any | None = None,
        custom_aligner: Aligner | Any | None = None,
    ) -> None:
        opts = options or {}
        diar_cfg = opts.get("diarizerConfig") or opts.get("diarizer_config") or {}
        asr_cfg = opts.get("asrConfig") or opts.get("asr_config") or {}
        vp_cfg = opts.get("voiceprintConfig") or opts.get("voiceprint_config") or {}

        self.embedding_extractor: SpeakerEmbedder = (
            custom_extractor
            or WeSpeakerEmbeddingExtractor({"numThreads": diar_cfg.get("numThreads")})
        )
        self.diarizer: Diarizer = (
            custom_diarizer
            or PyannoteSpeakerDiarizer(self.embedding_extractor, diar_cfg)
        )
        self.asr_engine: AsrBackend = custom_asr_engine or CohereAsrEngine(asr_cfg)
        self.aligner: Aligner | None = custom_aligner or CtcForcedAligner()

        vp_sim = vp_cfg.get("similarityThreshold")
        vp_margin = vp_cfg.get("decisionMargin")
        self.similarity_threshold: float = float(vp_sim if vp_sim is not None else 0.45)
        self.decision_margin: float = float(vp_margin if vp_margin is not None else 0.06)
        self.unload_per_stage: bool = bool(opts.get("unloadPerStage", True))

    def close(self) -> None:
        """Release every engine's ONNX/torch session (and its intra-op thread pool) now.

        Teardown must not wait for GC. Each engine (ASR, aligner, diarizer, embedder)
        holds an onnxruntime/torch session whose thread pool stays alive until the
        session object is finalized. Under DBOS recovery a wedged workflow can mint
        fresh pipelines, and relying on a GC pass that may never run under load let
        those pools accumulate (observed: 20k+ threads in one container). Releasing
        here frees them deterministically. Idempotent and never raises — teardown on
        an error path must not mask the original failure.
        """
        for attr in ("asr_engine", "aligner", "diarizer", "embedding_extractor"):
            engine = getattr(self, attr, None)
            release = getattr(engine, "release", None)
            if not callable(release):
                continue
            kwargs: dict[str, Any] = {}
            try:
                if "clear_cache" in inspect.signature(release).parameters:
                    kwargs["clear_cache"] = True
            except (TypeError, ValueError):
                kwargs = {}
            try:
                release(**kwargs)
            except Exception as exc:
                logger.warn(
                    "Engine release during pipeline.close() failed",
                    {"category": "transcription", "engine": attr, "error": str(exc)},
                )

    async def transcribe(self, options: dict[str, Any]) -> LocalTranscriptionResult:
        pipeline_t0 = time.perf_counter()
        signal = options.get("signal")
        on_progress = options.get("onProgress") or options.get("on_progress")
        meeting_id = options.get("meetingId") or options.get("meeting_id")

        if _is_aborted(signal):
            raise RuntimeError("Transcription cancelled by user")

        await _emit(
            on_progress,
            {
                "stage": "decoding",
                "percent": 5,
                "message": "Decoding 16kHz audio samples...",
            },
        )

        # 1. Decode audio to 16kHz mono float32
        decode_t0 = time.perf_counter()
        target_sr = 16000
        decoded = await load_audio_samples(
            {
                "audioPath": options.get("audioPath") or options.get("audio_path"),
                "audioBytes": options.get("audioBytes") or options.get("audio_bytes"),
                "enhance": False,
            },
            target_sr,
        )
        samples = decoded.samples
        duration_ms = decoded.duration_ms
        logger.info(
            "Decoded 16kHz mono audio samples for local transcription",
            {
                "category": "transcription",
                "meetingId": meeting_id,
                "audioDurationMs": duration_ms,
                "sampleCount": int(len(samples)),
                "elapsedMs": int(round((time.perf_counter() - decode_t0) * 1000)),
            },
        )

        if _is_aborted(signal):
            raise RuntimeError("Transcription cancelled by user")

        await _emit(
            on_progress,
            {
                "stage": "diarizing",
                "percent": 15,
                "totalMs": duration_ms,
                "message": "Diarization (1/4): Initializing neural speaker segmentation...",
            },
        )

        # 2. Perform Speaker Diarization (and extract turn embeddings)
        clustering_threshold = options.get("clusteringThreshold")
        expected_speaker_count = options.get("expectedSpeakerCount")
        diarize_fn = self.diarizer.diarize
        diarize_kwargs: dict[str, Any] = {}
        try:
            sig = inspect.signature(diarize_fn)
            if "on_progress" in sig.parameters:
                diarize_kwargs["on_progress"] = on_progress
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

        if self.unload_per_stage:
            if hasattr(self.diarizer, "release"):
                self.diarizer.release()
            if hasattr(self.embedding_extractor, "release"):
                self.embedding_extractor.release()

        if _is_aborted(signal):
            raise RuntimeError("Transcription cancelled by user")

        total_segments = len(speaker_segments)
        await _emit(
            on_progress,
            {
                "stage": "diarizing",
                "percent": 39,
                "total": total_segments,
                "totalMs": duration_ms,
                "message": f"Identified {total_segments} speech segments across clusters. Matching enrolled profiles...",
            },
        )

        # 3. Prepare enrolled speaker profiles with multiple trusted vectors
        all_profiles: list[dict[str, Any]] = []
        raw_candidates = options.get("candidateSpeakers") or options.get("candidate_speakers")
        normalized_candidates = (
            {str(c).strip().lower() for c in raw_candidates if str(c).strip()}
            if raw_candidates
            else None
        )

        raw_enrolled = options.get("enrolledSpeakers") or options.get("enrolled_speakers") or []
        for spk_raw in raw_enrolled:
            spk = (
                spk_raw
                if isinstance(spk_raw, EnrolledSpeaker)
                else EnrolledSpeaker.model_validate(spk_raw)
            )
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

        # 4. Group segments by cluster & compute duration-weighted centroid per cluster
        cluster_map: dict[str, list[Any]] = {}
        for seg in speaker_segments:
            cid = getattr(seg, "speakerId", None) or "Speaker 1"
            cluster_map.setdefault(cid, []).append(seg)

        emb_dim = getattr(self.embedding_extractor, "dim", 256)
        cluster_centroids: dict[str, VoiceprintVector] = {}
        for cid, segs in cluster_map.items():
            centroid = merge_weighted_voiceprint_vectors(
                [
                    {
                        "vector": getattr(s, "embedding", None) or ([0.0] * emb_dim),
                        "weight": max(1, int(getattr(s, "endMs", 0)) - int(getattr(s, "startMs", 0))),
                    }
                    for s in segs
                ]
            )
            cluster_centroids[cid] = centroid

        # 5. Global Cluster-to-Speaker Assignment with decision margin & threshold
        opt_sim = options.get("similarityThreshold")
        opt_margin = options.get("decisionMargin")
        effective_threshold = float(
            opt_sim if opt_sim is not None else self.similarity_threshold
        )
        effective_margin = float(
            opt_margin if opt_margin is not None else self.decision_margin
        )

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

        # 6. Transcribe audio & align word timestamps
        language = str(options.get("language") or "en")
        model_id = options.get("modelId") or options.get("model_id")
        final_segments: list[TranscriptSegment] = []

        use_whole_file_align = bool(
            options.get("wholeFileAsr")
            and isinstance(self.asr_engine, CohereAsrEngine)
            and model_id not in ("mock", "mock-asr")
            and self.aligner is not None
        )

        if use_whole_file_align and speaker_segments:
            await _emit(
                on_progress,
                {
                    "stage": "transcribing",
                    "percent": 50,
                    "current": 1,
                    "total": total_segments,
                    "totalMs": duration_ms,
                    "message": "Transcribing audio with Cohere Transcribe...",
                },
            )
            full_asr = await self._call_asr_segment(
                samples,
                target_sr,
                {"language": language, "startMs": 0, "modelId": model_id},
            )
            if self.unload_per_stage and hasattr(self.asr_engine, "release"):
                self.asr_engine.release()

            full_text = str(full_asr.get("text") or "").strip()
            await _emit(
                on_progress,
                {
                    "stage": "transcribing",
                    "percent": 85,
                    "current": total_segments,
                    "total": total_segments,
                    "totalMs": duration_ms,
                    "message": "Running forced alignment for word timestamps...",
                },
            )
            assert self.aligner is not None
            aligned_words = await self.aligner.align(
                samples, target_sr, full_text, language=language, offset_ms=0
            )
            if self.unload_per_stage and hasattr(self.aligner, "release"):
                self.aligner.release()

            final_segments = self._assign_words_to_speaker_turns(
                aligned_words, speaker_segments, cluster_to_resolved_name
            )
        else:
            for i, seg in enumerate(speaker_segments):
                if _is_aborted(signal):
                    raise RuntimeError("Transcription cancelled by user")

                current_num = i + 1
                pct = min(95, int(round(40 + ((i + 0.5) / max(1, total_segments)) * 55)))
                cid = getattr(seg, "speakerId", "Speaker 1")
                resolved_speaker = cluster_to_resolved_name.get(cid, cid)
                s_start = int(getattr(seg, "startMs", 0))
                s_end = int(getattr(seg, "endMs", 0))
                s_samples = seg.samples

                await _emit(
                    on_progress,
                    {
                        "stage": "transcribing",
                        "percent": pct,
                        "current": current_num,
                        "total": total_segments,
                        "currentMs": s_start,
                        "totalMs": duration_ms,
                        "speaker": resolved_speaker,
                        "message": f"Transcribing segment {current_num} of {total_segments} ({pct}%)",
                    },
                )

                seg_t0 = time.perf_counter()
                asr_res = await self._call_asr_segment(
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
                        "speaker": resolved_speaker,
                        "startMs": s_start,
                        "endMs": s_end,
                        "segmentDurationMs": max(0, s_end - s_start),
                        "chars": len(seg_text),
                        "elapsedMs": seg_elapsed_ms,
                    },
                )
                raw_words = asr_res.get("words")
                words: list[TranscriptWord] | None = None
                if isinstance(raw_words, list) and raw_words:
                    words = [
                        w if isinstance(w, TranscriptWord) else TranscriptWord.model_validate(w)
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

            if self.unload_per_stage:
                if hasattr(self.asr_engine, "release"):
                    self.asr_engine.release()
                if self.aligner is not None and hasattr(self.aligner, "release"):
                    self.aligner.release()

        await _emit(
            on_progress,
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
                "discoveredSpeakers": len(discovered_map),
                "audioDurationMs": duration_ms,
                "totalElapsedMs": int(round((time.perf_counter() - pipeline_t0) * 1000)),
            },
        )
        return LocalTranscriptionResult(
            transcript=transcript,
            discoveredSpeakers=list(discovered_map.values()),
        )

    async def _call_asr_segment(
        self,
        samples: Any,
        sample_rate: int,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        fn = getattr(self.asr_engine, "transcribe_segment", None) or getattr(
            self.asr_engine, "transcribeSegment", None
        )
        if not callable(fn):
            raise AttributeError("asr_engine missing transcribe_segment/transcribeSegment")
        res: Any = fn(samples, sample_rate, options)
        if inspect.isawaitable(res):
            res = await res
        return res  # type: ignore[return-value]

    def _assign_words_to_speaker_turns(
        self,
        words: list[TranscriptWord],
        speaker_segments: list[Any],
        cluster_to_resolved_name: dict[str, str],
    ) -> list[TranscriptSegment]:
        if not words:
            return []
        turns = [
            (
                int(getattr(s, "startMs", 0)),
                int(getattr(s, "endMs", 0)),
                cluster_to_resolved_name.get(
                    getattr(s, "speakerId", "Speaker 1"),
                    getattr(s, "speakerId", "Speaker 1"),
                ),
            )
            for s in speaker_segments
        ]
        segments: list[TranscriptSegment] = []
        for w in words:
            w_mid = 0.5 * (w.startMs + w.endMs)
            best_spk = min(
                turns,
                key=lambda t: 0.0
                if (t[0] <= w_mid <= t[1])
                else min(abs(w_mid - t[0]), abs(w_mid - t[1])),
            )[2]
            w_with_spk = TranscriptWord(
                word=w.word,
                startMs=w.startMs,
                endMs=w.endMs,
                confidence=w.confidence,
                speaker=best_spk,
            )
            if segments and segments[-1].speaker == best_spk:
                prev = segments[-1]
                prev.endMs = max(prev.endMs, w.endMs)
                prev.text = f"{prev.text} {w.word}".strip()
                prev.words = [*(prev.words or []), w_with_spk]
            else:
                segments.append(
                    TranscriptSegment(
                        startMs=w.startMs,
                        endMs=w.endMs,
                        speaker=best_spk,
                        text=w.word,
                        words=[w_with_spk],
                    )
                )
        return segments
