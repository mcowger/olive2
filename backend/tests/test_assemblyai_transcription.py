"""AssemblyAI provider flow through TranscriptionService (polling, no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import text

from olive.db import create_db
from olive.layout import meeting_paths
from olive.ml.identify import match_cluster_centroids, rename_transcript_speakers, turns_from_transcript
from olive.ml.protocols import DiscoveredSpeakerVoiceprint
from olive.providers.assemblyai import AssemblyAIClient
from olive.schemas import EnrolledSpeaker, Transcript, TranscriptSegment, TranscriptWord
from olive.services.transcription import STAGE_ASSEMBLYAI_TRANSCRIBE, TranscriptionService
from olive.settings import clear_app_config_cache, resolve_paths, save_app_config

RAW_TRANSCRIPT: dict[str, Any] = {
    "id": "aai-job-123",
    "status": "completed",
    "language_code": "en",
    "audio_duration": 5,
    "utterances": [
        {
            "text": "Hello everyone.",
            "start": 100,
            "end": 1000,
            "confidence": 0.99,
            "speaker": "A",
            "words": [
                {"text": "Hello", "start": 100, "end": 500, "confidence": 0.99, "speaker": "A"},
                {"text": "everyone.", "start": 600, "end": 1000, "confidence": 0.98, "speaker": "A"},
            ],
        },
        {
            "text": "Hi there.",
            "start": 2000,
            "end": 3000,
            "confidence": 0.97,
            "speaker": "B",
            "words": [
                {"text": "Hi", "start": 2000, "end": 2400, "confidence": 0.97, "speaker": "B"},
                {"text": "there.", "start": 2500, "end": 3000, "confidence": 0.97, "speaker": "B"},
            ],
        },
    ],
}


class _FakeTranscriber:
    def __init__(self) -> None:
        self.polls = 0
        self.configs: list[dict[str, Any]] = []

    async def submit(self, _audio: Any, config: dict[str, Any]) -> dict[str, Any]:
        self.configs.append(config)
        return {"id": "aai-job-123", "status": "queued"}

    async def get_transcript(self, _job_id: str) -> dict[str, Any]:
        self.polls += 1
        if self.polls < 2:
            return {"id": "aai-job-123", "status": "processing"}
        return RAW_TRANSCRIPT


def _seed_meeting(tmp_path: Path) -> tuple[Any, str, Any]:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    now = 1_700_000_000_000
    meeting_id = "m-aai-1"
    paths = meeting_paths(meetings_dir, now, "AAI Meeting", meeting_id)
    Path(paths.audio_dir).mkdir(parents=True, exist_ok=True)
    (Path(paths.folder) / "audio/test.m4a").write_bytes(b"fake audio bytes")
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id, last_error, created_at, updated_at)
                VALUES (:id, 'AAI Meeting', :now, :end, 'upload', 'ready', '[]', NULL, NULL, NULL, :now, :now)
                """
            ),
            {"id": meeting_id, "now": now, "end": now + 5000},
        )
        conn.execute(
            text(
                """
                INSERT INTO recordings (id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at)
                VALUES ('rec-aai', :mid, 'audio/test.m4a', 'audio/mp4', 5000, 16, 'sha-aai', 'upload', NULL, :now)
                """
            ),
            {"mid": meeting_id, "now": now},
        )
    return handle, meeting_id, paths


def _service(handle: Any, meetings_dir: str, **extra: Any) -> TranscriptionService:
    client = AssemblyAIClient({"apiKey": "test-key", "transcriber": _FakeTranscriber()})
    return TranscriptionService(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "configDir": str(Path(meetings_dir).parent / "config"),
            "assemblyaiClient": client,
            "now": lambda: 1_700_000_001_000,
            **extra,
        }
    )


async def test_model_selection_follows_persisted_settings_without_restart(tmp_path: Path) -> None:
    handle, meeting_id, _ = _seed_meeting(tmp_path)
    service = _service(handle, str(tmp_path / "meetings"))
    fake = service.assemblyai_client.raw_client
    paths = resolve_paths({"OLIVE_CONFIG_DIR": str(tmp_path / "config")})
    messages: list[str] = []
    options = {
        "provider": "assemblyai",
        "pollIntervalMs": 10,
        "force": True,
        "onProgress": lambda update: messages.append(update.message),
    }

    result = await service.transcribe_meeting(meeting_id, options)
    assert result["status"] == "done"
    assert fake.configs[-1]["speech_models"] == ["universal-3-5-pro"]
    assert "Universal-3.5 Pro" in messages[0]

    save_app_config({"assemblyaiSpeechModel": "assemblyai-stt-3-7"}, paths)
    messages.clear()
    result = await service.transcribe_meeting(meeting_id, options)
    assert result["status"] == "done"
    assert fake.configs[-1]["speech_models"] == ["assemblyai-stt-3-7"]
    assert "AssemblyAI STT 3.7" in messages[0]

    clear_app_config_cache()
    restarted = _service(handle, str(tmp_path / "meetings"))
    messages.clear()
    result = await restarted.transcribe_meeting(meeting_id, options)
    assert result["status"] == "done"
    assert restarted.assemblyai_client.raw_client.configs[-1]["speech_models"] == ["assemblyai-stt-3-7"]
    assert "AssemblyAI STT 3.7" in messages[0]
    handle.engine.dispose()
    clear_app_config_cache()


async def test_assemblyai_polling_flow_persists_artifacts_and_cloud_labels(tmp_path: Path) -> None:
    handle, meeting_id, paths = _seed_meeting(tmp_path)
    service = _service(handle, str(tmp_path / "meetings"))

    progress_updates: list[tuple[str, float, str]] = []
    result = await service.transcribe_meeting(
        meeting_id,
        {
            "provider": "assemblyai",
            "pollIntervalMs": 10,
            "onProgress": lambda u: progress_updates.append((u.stage, u.percent, u.message)),
        },
    )

    assert result["status"] == "done"
    stages = [s for s, _p, _m in progress_updates]
    assert stages[0] == "uploading"
    assert "transcribing" in stages
    assert any("poll #" in m for _s, _p, m in progress_updates)
    assert all(p == round(p) for _s, p, _m in progress_updates)
    assert stages[-1] == "done"
    assert result["jobId"] == "aai-job-123"
    with handle.engine.connect() as conn:
        stage_run = (
            conn.execute(
                text("SELECT * FROM stage_runs WHERE meeting_id = :mid AND stage = :st"),
                {"mid": meeting_id, "st": STAGE_ASSEMBLYAI_TRANSCRIBE},
            )
            .mappings()
            .one()
        )
        assert stage_run["status"] == "done"
        assert stage_run["provider_job_id"] == "aai-job-123"
        artifacts = (
            conn.execute(
                text("SELECT provider, format FROM artifacts WHERE meeting_id = :mid ORDER BY format"),
                {"mid": meeting_id},
            )
            .mappings()
            .all()
        )
        assert [(a["provider"], a["format"]) for a in artifacts] == [
            ("assemblyai", "json"),
            ("assemblyai", "txt"),
        ]
        speakers = conn.execute(text("SELECT name FROM speakers ORDER BY name")).mappings().all()
        # Fake audio cannot be decoded/embedded -> cloud labels are the
        # fallback in the transcript, but unidentified "Speaker N" clusters
        # are not people: no speaker rows / meeting links are created.
        assert speakers == []
        assert conn.execute(text("SELECT * FROM meeting_speakers")).mappings().all() == []

    saved = json.loads((Path(paths.folder) / "transcripts/assemblyai.json").read_text())
    assert saved["segments"][0]["speaker"] == "Speaker 1"
    assert "Speaker 2: Hi there." in (Path(paths.folder) / "transcripts/assemblyai.txt").read_text()
    handle.engine.dispose()


async def test_assemblyai_flow_renames_speakers_from_local_voiceprints(tmp_path: Path) -> None:
    handle, meeting_id, paths = _seed_meeting(tmp_path)

    async def fake_identifier(_audio_path: str, transcript: Transcript, enrolled: list[EnrolledSpeaker]):
        renamed = rename_transcript_speakers(transcript, {"Speaker 1": "Alice", "Speaker 2": "Speaker 2"})
        return renamed, [
            DiscoveredSpeakerVoiceprint(
                speakerId="spk-alice",
                name="Alice",
                voiceprint=[1.0, 0.0],
                isEnrolled=True,
                similarityScore=0.9,
            )
        ]

    service = _service(handle, str(tmp_path / "meetings"), cloudSpeakerIdentifier=fake_identifier)
    result = await service.transcribe_meeting(meeting_id, {"provider": "assemblyai", "pollIntervalMs": 10})
    assert result["status"] == "done"

    saved = json.loads((Path(paths.folder) / "transcripts/assemblyai.json").read_text())
    assert saved["segments"][0]["speaker"] == "Alice"
    assert saved["segments"][0]["words"][0]["speaker"] == "Alice"
    with handle.engine.connect() as conn:
        speakers = conn.execute(text("SELECT id, name FROM speakers ORDER BY name")).mappings().all()
        # Alice matched by voiceprint; Speaker 2 (unmatched) stays only as a
        # transcript label, not a person.
        assert [(s["id"], s["name"]) for s in speakers] == [("spk-alice", "Alice")]
        links = conn.execute(text("SELECT speaker_id FROM meeting_speakers")).mappings().all()
        assert [link["speaker_id"] for link in links] == ["spk-alice"]
    handle.engine.dispose()


def test_turns_and_centroid_matching() -> None:
    transcript = Transcript(
        segments=[
            TranscriptSegment(
                startMs=0,
                endMs=2000,
                speaker="Speaker 1",
                text="hi",
                words=[TranscriptWord(startMs=0, endMs=2000, word="hi", speaker="Speaker 1")],
            ),
            TranscriptSegment(startMs=3000, endMs=4000, speaker="Speaker 2", text="yo"),
        ]
    )
    assert turns_from_transcript(transcript) == [
        {"startMs": 0, "endMs": 2000, "speakerId": "Speaker 1"},
        {"startMs": 3000, "endMs": 4000, "speakerId": "Speaker 2"},
    ]

    enrolled = [
        EnrolledSpeaker(id="spk-alice", name="Alice", providerIds={"local": [json.dumps([1.0, 0.0])]})
    ]
    mapping, discovered = match_cluster_centroids(
        {"Speaker 1": [0.99, 0.01], "Speaker 2": [0.0, 1.0]}, enrolled
    )
    assert mapping == {"Speaker 1": "Alice", "Speaker 2": "Speaker 2"}
    by_name = {d.name: d for d in discovered}
    assert by_name["Alice"].isEnrolled is True
    assert by_name["Speaker 2"].isEnrolled is False


async def test_identify_speakers_emits_progress() -> None:
    import numpy as np

    from olive.ml.identify import identify_speakers_from_turns

    class _FakeExtractor:
        dim = 2

        async def extract(self, samples: Any, sample_rate: int = 16000) -> list[float]:
            return [1.0, 0.0]

    transcript = Transcript(
        segments=[
            TranscriptSegment(startMs=0, endMs=5000, speaker="Speaker 1", text="hello world"),
            TranscriptSegment(startMs=6000, endMs=11000, speaker="Speaker 2", text="hi there"),
        ]
    )
    enrolled = [
        EnrolledSpeaker(id="spk-alice", name="Alice", providerIds={"local": [json.dumps([1.0, 0.0])]})
    ]
    updates: list[dict[str, Any]] = []
    result = await identify_speakers_from_turns(
        np.zeros(16000 * 12, dtype=np.float32),
        16000,
        transcript,
        enrolled,
        embedding_extractor=_FakeExtractor(),
        on_progress=updates.append,
    )
    assert result is not None
    phases = [str(u.get("phase")) for u in updates]
    assert "embedding" in phases
    assert "matching" in phases
    assert any(u.get("total") == 2 for u in updates)
