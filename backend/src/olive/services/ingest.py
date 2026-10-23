"""Audio ingestion and SHA-256 deduplication service (port of ingest/service.ts)."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import io
import math
import re
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..layout import ensure_meeting_folder, meeting_paths
from ..logging_db import Logger
from ..logging_db import logger as default_logger
from ..ml.opus import is_raw_plaud_opus, wrap_raw_opus_bytes
from .meetings import to_meeting

MIME_TO_EXT: dict[str, str] = {
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "audio/aac": "aac",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/ogg": "ogg",
    "audio/vorbis": "ogg",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/webm": "webm",
}

EXT_TO_MIME: dict[str, str] = {
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "webm": "audio/webm",
}


def parse_duration_ms_value(value: Any) -> int | None:
    """Parse an explicit millisecond duration field (``durationMs``/``duration_ms``)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return int(round(parsed))


def parse_duration_seconds_or_ms(value: Any) -> int | None:
    """Parse a bare ``duration`` field that may be seconds or milliseconds.

    Values below 1000 are treated as seconds (Plaud Cloud reports durations
    in seconds); larger values are treated as milliseconds already.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    if parsed < 1000:
        return int(round(parsed * 1000))
    return int(round(parsed))


def probe_audio_duration_ms_sync(audio_bytes: bytes, extension: str = "m4a") -> int | None:
    """Best-effort server-side audio duration probe. Never raises; returns None on failure."""
    if not audio_bytes:
        return None
    # Fast path for WAV: read the header without spawning a subprocess.
    try:
        if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
            import soundfile as sf

            info = sf.info(io.BytesIO(audio_bytes))
            if info.samplerate and info.frames:
                duration_ms = int(round(info.frames / float(info.samplerate) * 1000))
                return duration_ms if duration_ms > 0 else None
    except Exception:
        pass
    suffix = f".{(extension or 'm4a').lower().lstrip('.') or 'm4a'}"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    tmp.name,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        if proc.returncode != 0:
            return None
        seconds = float((proc.stdout or "").strip())
        if not math.isfinite(seconds) or seconds <= 0:
            return None
        duration_ms = int(round(seconds * 1000))
        return duration_ms if duration_ms > 0 else None
    except Exception:
        return None


def resolve_extension_and_mime(
    mime_header: str | None = None,
    filename: str | None = None,
) -> dict[str, str]:
    ext = Path(filename).suffix.lstrip(".").lower() if filename else ""
    normalized_mime = (mime_header or "").split(";", 1)[0].strip().lower()

    if not ext and normalized_mime in MIME_TO_EXT:
        ext = MIME_TO_EXT[normalized_mime]
    elif not ext:
        ext = "m4a"

    final_mime = (
        normalized_mime
        if (normalized_mime.startswith("audio/") and normalized_mime != "audio/x-m4a")
        else EXT_TO_MIME.get(ext, "audio/mp4")
    )
    return {"extension": ext, "mime": final_mime}


class IngestService:
    def __init__(
        self,
        *,
        db: Engine,
        meetings_dir: str,
        transcription_service: Any = None,
        default_transcription_provider: Literal["speechmatics", "assemblyai", "muse", "local"] = "local",
        logger: Logger | None = None,
        now: Callable[[], int] | None = None,
    ) -> None:
        self.db = db
        self.meetings_dir = meetings_dir
        self.transcription_service = transcription_service
        self.default_transcription_provider: Literal["speechmatics", "assemblyai", "muse", "local"] = (
            default_transcription_provider
        )
        self.logger = logger or default_logger
        self.now = now or (lambda: int(time.time() * 1000))

    def set_default_transcription_provider(
        self, provider: Literal["speechmatics", "assemblyai", "muse", "local"]
    ) -> None:
        self.default_transcription_provider = provider

    async def ingest_audio(self, options: dict[str, Any]) -> dict[str, Any]:
        current_time = self.now()
        start_time = int(options.get("startTime") or options.get("start_time") or current_time)
        audio_bytes = bytes(options.get("audioBytes") or options.get("audio_bytes") or b"")
        # Canonicalize headerless raw Plaud Opus on disk as Ogg Opus.
        if is_raw_plaud_opus(audio_bytes):
            audio_bytes = wrap_raw_opus_bytes(audio_bytes)
        filename: str | None = options.get("filename")
        mime_in: str | None = options.get("mime")
        sha256 = hashlib.sha256(audio_bytes).hexdigest()

        self.logger.debug(
            "Ingesting audio payload",
            {
                "category": "ingest",
                "sizeBytes": len(audio_bytes),
                "mime": mime_in,
                "filename": filename,
                "sha256": sha256,
            },
        )

        # 1. SHA-256 Deduplication check
        existing_meeting = None
        with self.db.connect() as conn:
            existing_rec = (
                conn.execute(
                    text("SELECT * FROM recordings WHERE sha256 = :sha256"),
                    {"sha256": sha256},
                )
                .mappings()
                .first()
            )

            if existing_rec:
                existing_meeting = (
                    conn.execute(
                        text("SELECT * FROM meetings WHERE id = :id"),
                        {"id": existing_rec["meeting_id"]},
                    )
                    .mappings()
                    .first()
                )
        if existing_rec and existing_meeting:
            # Logging can check out its own DB connection; release this one first.
            self.logger.info(
                "Audio upload deduplicated via SHA-256",
                {
                    "category": "ingest",
                    "sha256": sha256,
                    "meetingId": existing_meeting["id"],
                    "recordingId": existing_rec["id"],
                },
            )
            return {
                "meetingId": existing_meeting["id"],
                "recordingId": existing_rec["id"],
                "deduped": True,
                "meeting": to_meeting(existing_meeting).model_dump(),
                "audioPath": existing_rec["path"],
            }

        # 2. Resolve filename, extension, and mime
        resolved = resolve_extension_and_mime(mime_in, filename)
        extension = resolved["extension"]
        mime = resolved["mime"]
        meeting_id = str(uuid.uuid4())
        recording_id = str(uuid.uuid4())
        source = options.get("source") or "upload"
        duration_ms: int | None = None
        for key in ("durationMs", "duration_ms"):
            duration_ms = parse_duration_ms_value(options.get(key))
            if duration_ms is not None:
                break
        if duration_ms is None:
            duration_ms = parse_duration_seconds_or_ms(options.get("duration"))
        if duration_ms is None:
            # The client doesn't always know the duration (iOS Shortcut,
            # curl, web upload), so probe the audio server-side instead of
            # storing a zero-length meeting.
            try:
                duration_ms = await asyncio.to_thread(probe_audio_duration_ms_sync, audio_bytes, extension)
            except Exception as exc:
                self.logger.warn(
                    "Audio duration probe failed; storing meeting without duration",
                    {"category": "ingest", "error": str(exc)},
                )
                duration_ms = None
        end_time = start_time + int(duration_ms or 0)

        raw_title = (options.get("title") or "").strip()
        if raw_title:
            title = raw_title
        elif filename:
            stem = re.sub(r"\.[^/.]+$", "", filename)
            title = re.sub(r"[_-]", " ", stem)
        else:
            title = f"Audio Recording {datetime.fromtimestamp(start_time / 1000.0).strftime('%Y-%m-%d')}"

        # 3. Write audio bytes to disk
        paths = ensure_meeting_folder(meeting_paths(self.meetings_dir, start_time, title, meeting_id))
        relative_path = f"audio/{recording_id}.{extension}"
        dest_path = Path(paths.folder) / relative_path

        self.logger.debug(
            "Writing ingested audio file to disk",
            {
                "category": "ingest",
                "meetingId": meeting_id,
                "recordingId": recording_id,
                "destinationAbsolutePath": str(dest_path),
                "extension": extension,
                "mime": mime,
            },
        )
        dest_path.write_bytes(audio_bytes)

        # 4. Create Meeting and Recording records in database
        with self.db.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO meetings (
                      id, title, start_time, end_time, source, status, tags,
                      primary_transcript_artifact_id, primary_summary_artifact_id,
                      last_error, created_at, updated_at
                    ) VALUES (
                      :id, :title, :start_time, :end_time, :source, 'ready', '[]',
                      NULL, NULL, NULL, :created_at, :updated_at
                    )
                    """
                ),
                {
                    "id": meeting_id,
                    "title": title,
                    "start_time": start_time,
                    "end_time": end_time,
                    "source": source,
                    "created_at": current_time,
                    "updated_at": current_time,
                },
            )
            conn.execute(
                text(
                    """
                    INSERT INTO recordings (
                      id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                      provider, provider_recording_id, created_at
                    ) VALUES (
                      :id, :meeting_id, :path, :mime, :duration_ms, :size_bytes, :sha256,
                      :provider, NULL, :created_at
                    )
                    """
                ),
                {
                    "id": recording_id,
                    "meeting_id": meeting_id,
                    "path": relative_path,
                    "mime": mime,
                    "duration_ms": int(duration_ms) if duration_ms is not None else None,
                    "size_bytes": len(audio_bytes),
                    "sha256": sha256,
                    "provider": source,
                    "created_at": current_time,
                },
            )
            created_meeting = (
                conn.execute(
                    text("SELECT * FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .one()
            )

        self.logger.info(
            "Audio ingested successfully",
            {
                "category": "ingest",
                "meetingId": meeting_id,
                "recordingId": recording_id,
                "source": source,
                "sizeBytes": len(audio_bytes),
                "sha256": sha256,
            },
        )

        # 5. Trigger automatic transcription if requested
        auto_transcribe = bool(options.get("autoTranscribe") or options.get("auto_transcribe"))
        if auto_transcribe and self.transcription_service is not None:
            provider = (
                options.get("transcriptionProvider")
                or options.get("transcription_provider")
                or self.default_transcription_provider
            )
            model_id = options.get("modelId") or options.get("model_id")
            self.logger.debug(
                "Triggering auto-transcription after audio ingest",
                {
                    "category": "transcription",
                    "meetingId": meeting_id,
                    "provider": provider,
                },
            )

            # Durable work must go through the DBOS job queue (see AGENTS.md).
            # start_transcription enqueues onto the provider's queue and
            # returns immediately, so awaiting it here only waits for the
            # enqueue — never run transcribe_meeting in a bare asyncio task.
            start_fn = getattr(self.transcription_service, "start_transcription", None) or getattr(
                self.transcription_service, "startTranscription", None
            )
            if callable(start_fn):
                try:
                    res = start_fn(
                        meeting_id,
                        {"provider": provider, "modelId": model_id},
                    )
                    if inspect.isawaitable(res):
                        await res
                except Exception as exc:
                    self.logger.error(
                        "Auto-transcription after ingest failed",
                        {
                            "category": "transcription",
                            "meetingId": meeting_id,
                            "error": str(exc),
                        },
                    )

        return {
            "meetingId": meeting_id,
            "recordingId": recording_id,
            "deduped": False,
            "meeting": to_meeting(created_meeting).model_dump(),
            "audioPath": relative_path,
        }
