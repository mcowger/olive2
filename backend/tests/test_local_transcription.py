"""Local ASR, diarization, and WeSpeaker voiceprint pipeline tests (port of local-transcription.test.ts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from local_pipeline_helpers import wait_for_local_run
from olive.app import create_app
from olive.db import create_db
from olive.layout import meeting_paths
from olive.ml import (
    AbortController,
    LocalSpeakerSegment,
    LocalTranscriptionPipeline,
    MockLocalAsrEngine,
    PyannoteSpeakerDiarizer,
    WeSpeakerEmbeddingExtractor,
    cosine_similarity,
    decode_wav,
    encode_wav,
    load_audio_samples,
    merge_weighted_voiceprint_vectors,
    proportional_word_timestamps,
    resample,
    score_against_trusted_vectors,
)
from olive.services.transcription import TranscriptionService
from olive.workflows.transcription_workflow import STAGE_LOCAL_TRANSCRIBE, STAGE_SEQUENCE

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio"
FIXTURE_MULTI_SPEAKER_PATH = FIXTURES_DIR / "sample-multi-speaker.wav"
FIXTURE_SPEECHMATICS_PATH = FIXTURES_DIR / "speechmatics-sample.wav"


def create_single_voice_wav(
    freq_hz: float = 220.0, duration_sec: float = 4.0, sample_rate: int = 16000
) -> bytes:
    num_samples = int(sample_rate * duration_sec)
    t = np.arange(num_samples, dtype=np.float32) / float(sample_rate)
    samples = 0.5 * np.sin(2.0 * np.pi * freq_hz * t) + 0.2 * np.sin(2.0 * np.pi * (freq_hz * 2.0) * t)
    return encode_wav(samples, sample_rate)


class _FakeEmbeddingExtractor:
    """Deterministic stand-in for WeSpeakerEmbeddingExtractor (no model download)."""

    dim = 256

    async def extract(self, samples: np.ndarray, sample_rate: int = 16000) -> list[float]:
        rng = np.random.default_rng(1234)
        vec = rng.standard_normal(self.dim)
        norm = float(np.linalg.norm(vec)) or 1.0
        return [float(v) / norm for v in vec]

    def release(self) -> None:
        pass


class _FakeDiarizer:
    """Single-turn stand-in for PyannoteSpeakerDiarizer (no pipeline download)."""

    def __init__(self, extractor: Any | None = None) -> None:
        self._extractor = extractor or _FakeEmbeddingExtractor()

    async def diarize(
        self,
        samples: np.ndarray,
        sample_rate: int = 16000,
        clustering_threshold_override: float | None = None,
        expected_speaker_count: int | None = None,
        **kwargs: Any,
    ) -> list[LocalSpeakerSegment]:
        arr = np.asarray(samples, dtype=np.float32)
        duration_ms = int(round(len(arr) / float(sample_rate) * 1000))
        embedding = await self._extractor.extract(arr, sample_rate)
        return [
            LocalSpeakerSegment(
                startMs=0,
                endMs=duration_ms,
                speakerId="SPEAKER_00",
                samples=arr,
                embedding=embedding,
            )
        ]

    def release(self) -> None:
        pass


class _FakeAligner:
    """Proportional-timestamp stand-in for CtcForcedAligner (no model download)."""

    async def align(
        self,
        samples: np.ndarray,
        sample_rate: int,
        text: str,
        language: str = "en",
        offset_ms: int = 0,
    ) -> Any:
        arr = np.asarray(samples, dtype=np.float32)
        duration_ms = int(round(len(arr) / float(sample_rate) * 1000))
        return proportional_word_timestamps(text, offset_ms, max(1, duration_ms))

    def release(self) -> None:
        pass


def _make_fake_pipeline(
    options: dict[str, Any] | None = None, asr_engine: Any | None = None
) -> LocalTranscriptionPipeline:
    """Build a LocalTranscriptionPipeline with no real ML models."""
    extractor = _FakeEmbeddingExtractor()
    return LocalTranscriptionPipeline(
        options or {},
        _FakeDiarizer(extractor),
        extractor,
        asr_engine or MockLocalAsrEngine(),
        _FakeAligner(),
    )


def test_wav_codec_roundtrip_and_resampling() -> None:
    original = np.array([0.0, 0.5, 1.0, -0.5, -1.0, 0.25], dtype=np.float32)
    encoded = encode_wav(original, 16000)
    decoded = decode_wav(encoded)

    assert decoded.sample_rate == 16000
    assert decoded.channels == 1
    assert len(decoded.samples) == len(original)
    assert abs(float(decoded.samples[1]) - 0.5) < 0.01

    resampled = resample(decoded.samples, 16000, 8000)
    assert len(resampled) == round(len(original) / 2)


@pytest.mark.live
async def test_wespeaker_neural_embedding_extractor_and_separation_gate() -> None:
    extractor = WeSpeakerEmbeddingExtractor()
    dec_sm = decode_wav(FIXTURE_SPEECHMATICS_PATH.read_bytes())
    dec_multi = decode_wav(FIXTURE_MULTI_SPEAKER_PATH.read_bytes())
    sr = 16000

    spk1_a = dec_sm.samples[: sr * 5]
    spk1_b = dec_sm.samples[sr * 5 : sr * 10]
    spk2 = dec_multi.samples[int(21.8 * sr) : int(28.0 * sr)]

    emb1_a = await extractor.extract(spk1_a, sr)
    emb1_b = await extractor.extract(spk1_b, sr)
    emb2 = await extractor.extract(spk2, sr)

    assert len(emb1_a) == 256
    assert len(emb1_b) == 256
    assert len(emb2) == 256

    within_sim = cosine_similarity(emb1_a, emb1_b)
    assert within_sim > 0.70

    between_sim = cosine_similarity(emb1_a, emb2)
    assert between_sim < 0.40

    separation = within_sim - between_sim
    assert separation > 0.25

    trusted_vectors = [emb1_a, emb1_b]
    match_score = score_against_trusted_vectors(emb1_b, trusted_vectors)
    assert match_score >= within_sim

    weighted_centroid = merge_weighted_voiceprint_vectors(
        [
            {"vector": emb1_a, "weight": 5000},
            {"vector": emb1_b, "weight": 5000},
        ]
    )
    assert len(weighted_centroid) == 256
    assert cosine_similarity(weighted_centroid, emb1_a) > 0.85


@pytest.mark.live
async def test_pyannote_neural_diarizer_splits_multi_speaker_audio() -> None:
    extractor = WeSpeakerEmbeddingExtractor()
    diarizer = PyannoteSpeakerDiarizer(extractor, {"clusteringThreshold": 0.50})

    decoded = decode_wav(FIXTURE_MULTI_SPEAKER_PATH.read_bytes())
    segments = await diarizer.diarize(decoded.samples, 16000, 0.50, 2)

    assert len(segments) > 1
    assert segments[0].startMs < segments[0].endMs
    assert segments[0].embedding is not None
    assert len(segments[0].embedding) == 256

    distinct_speakers = {s.speakerId for s in segments}
    assert len(distinct_speakers) >= 2


async def test_local_pipeline_transcribes_with_candidate_roster_and_decision_margin() -> None:
    extractor = _FakeEmbeddingExtractor()
    asr_engine = MockLocalAsrEngine(["Welcome to the team sync meeting.", "Thanks, happy to be here."])

    pipeline = _make_fake_pipeline({"voiceprintConfig": {"similarityThreshold": 0.45}}, asr_engine)

    audio_bytes = create_single_voice_wav(220.0, 6.0)
    decoded = await load_audio_samples({"audioBytes": audio_bytes, "enhance": False}, 16000)
    spk1_embedding = await extractor.extract(decoded.samples[: 16000 * 6], 16000)

    enrolled_alice = {
        "id": "spk-alice-uuid",
        "name": "Alice Smith",
        "providerIds": {"local": [json.dumps(spk1_embedding)]},
    }

    result = await pipeline.transcribe(
        {
            "audioBytes": audio_bytes,
            "language": "en",
            "enrolledSpeakers": [enrolled_alice],
            "candidateSpeakers": ["Alice Smith"],
            "expectedSpeakerCount": 1,
            "similarityThreshold": 0.45,
            "decisionMargin": 0.05,
        }
    )

    assert len(result.transcript.segments) > 0
    assert result.transcript.segments[0].speaker == "Alice Smith"

    alice_disc = next((s for s in result.discoveredSpeakers if s.name == "Alice Smith"), None)
    assert alice_disc is not None
    assert alice_disc.isEnrolled is True
    assert (alice_disc.similarityScore or 0.0) > 0.45


async def test_end_to_end_meeting_transcription_using_local_provider(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    now = 1_700_000_000_000
    meeting_id = "m-local-test-1"

    paths = meeting_paths(meetings_dir, now, "Architecture Discussion", meeting_id)
    Path(paths.audio_dir).mkdir(parents=True, exist_ok=True)
    audio_bytes = FIXTURE_SPEECHMATICS_PATH.read_bytes()
    (Path(paths.folder) / "audio/meeting.wav").write_bytes(audio_bytes)

    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, 'Architecture Discussion', :now, :end, 'upload', 'ready', '[]',
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {"id": meeting_id, "now": now, "end": now + 12000},
        )
        conn.execute(
            text(
                """
                INSERT INTO recordings (
                  id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at
                ) VALUES (
                  'rec-local-1', :mid, 'audio/meeting.wav', 'audio/wav', 12000, :sz,
                  'fake-sha-local-wav', 'upload', NULL, :now
                )
                """
            ),
            {"mid": meeting_id, "sz": len(audio_bytes), "now": now},
        )

    mock_asr = MockLocalAsrEngine(
        ["Let's review the local ASR engine.", "The neural diarization works seamlessly."]
    )
    local_pipeline = _make_fake_pipeline({}, mock_asr)
    service = TranscriptionService(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "localPipeline": local_pipeline,
            "defaultProvider": "local",
            "now": lambda: now + 1000,
        }
    )

    started = await service.transcribe_meeting(meeting_id, {"provider": "local"})
    assert started["status"] == "running"
    assert started["jobId"].startswith(f"stage:{meeting_id}:")

    # Fire-and-forget: the pipeline finishes on its own stage chain.
    stages = await wait_for_local_run(handle.engine, meeting_id)
    assert stages[STAGE_LOCAL_TRANSCRIBE] == "done"
    assert [stages[s] for s in STAGE_SEQUENCE] == ["done"] * len(STAGE_SEQUENCE)

    with handle.engine.connect() as conn:
        meeting = (
            conn.execute(
                text("SELECT status, primary_transcript_artifact_id FROM meetings WHERE id = :id"),
                {"id": meeting_id},
            )
            .mappings()
            .one()
        )
    assert meeting["status"] == "ready"
    assert meeting["primary_transcript_artifact_id"]

    handle.engine.dispose()


async def test_cancel_inflight_local_transcription_via_service_and_api(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    config_dir = str(tmp_path / "config")
    meetings_dir = str(tmp_path / "meetings")
    meeting_id = "m-cancel-test"
    now = 1_700_000_000_000

    paths = meeting_paths(meetings_dir, now, "Cancel Test", meeting_id)
    Path(paths.audio_dir).mkdir(parents=True, exist_ok=True)
    audio_bytes = create_single_voice_wav(440.0, 5.0)
    (Path(paths.audio_dir) / "rec.wav").write_bytes(audio_bytes)

    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, 'Cancel Test', :now, :end, 'upload', 'pending', '[]',
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {"id": meeting_id, "now": now, "end": now + 5000},
        )
        conn.execute(
            text(
                """
                INSERT INTO recordings (
                  id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at
                ) VALUES (
                  'rec-cancel-1', :mid, 'audio/rec.wav', 'audio/wav', 5000, :sz,
                  'fake-sha-cancel', 'upload', NULL, :now
                )
                """
            ),
            {"mid": meeting_id, "sz": len(audio_bytes), "now": now},
        )

    local_pipeline = _make_fake_pipeline({})
    service = TranscriptionService(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "localPipeline": local_pipeline,
            "defaultProvider": "local",
            "now": lambda: now + 50,
        }
    )

    abort_controller = AbortController()
    abort_controller.abort()

    result = await service.transcribe_meeting(
        meeting_id, {"provider": "local", "signal": abort_controller.signal}
    )
    assert result["status"] == "cancelled"

    with handle.engine.connect() as conn:
        stage_run = (
            conn.execute(
                text("SELECT * FROM stage_runs WHERE meeting_id = :mid"),
                {"mid": meeting_id},
            )
            .mappings()
            .first()
        )
        assert stage_run is not None
        assert stage_run["status"] == "cancelled"

    app = create_app(
        {
            "db": handle.engine,
            "configDir": config_dir,
            "meetingsDir": meetings_dir,
            "transcriptionService": service,
        }
    )
    client = TestClient(app)
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO stage_checkpoints (meeting_id, stage, run_id, doc, updated_at) "
                "VALUES (:mid, 'diarize', 'run-cancel', '{}', :now)"
            ),
            {"mid": meeting_id, "now": now},
        )
    cancel_res = client.post(f"/api/meetings/{meeting_id}/transcribe/cancel")
    assert cancel_res.status_code == 200
    assert cancel_res.json()["success"] is True
    with handle.engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM stage_checkpoints")).scalar() == 0
    handle.engine.dispose()
