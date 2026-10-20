"""Transcript import: turn an existing Zoom/SRT/text transcript into a no-audio meeting.

Insights must work on meetings that never had audio in Olive. :class:`TranscriptImportService`
parses the file with :mod:`olive.services.transcript_parsers`, creates a ``source='import'``
meeting with **no** ``recordings`` row, writes the transcript artifacts directly, and hands the
meeting to the insights ``analyze_signals`` step. The ASR/diarization pipeline is skipped
entirely.

Imported transcripts carry display *names*, not voiceprints, so speakers are linked by name
(``identity_source='name'``) using the same rules as speaker-name confirmation: case-insensitive
match against ``speakers`` and ``speaker_aliases``, the user's own ``meAliases`` resolving to the
single "me" speaker, generic placeholder labels never linked. A name that matches more than one
person is reported in ``unmappedNames`` rather than guessed.

The meeting start time is never defaulted to "now": a wrong date silently corrupts month/week
trends, so :class:`MissingStartTimeError` is raised when none is supplied or derivable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

from ..db import DbHandle, get_db
from ..layout import ensure_meeting_folder, meeting_paths
from ..logging_db import logger
from ..settings import resolve_paths
from ..speaker_labels import is_generic_speaker_label
from .insights.identity import normalize_alias
from .insights.naming import SpeakerNamingService
from .insights.prompts import load_insights_config
from .transcript_parsers import ImportFormat, parse_transcript

TRANSCRIPT_JSON_REL = "transcripts/import.json"
TRANSCRIPT_TXT_REL = "transcripts/import.txt"
DEFAULT_TITLE = "Imported transcript"


class MissingStartTimeError(ValueError):
    """No start time was supplied and none could be derived from the filename or content."""


def _normalize_content(content: str) -> str:
    """Strip a UTF-8 BOM and normalise line endings so equal text hashes equal."""
    if content.startswith("﻿"):
        content = content[1:]
    return content.replace("\r\n", "\n").replace("\r", "\n")


def _decode(content: bytes | str) -> str:
    if isinstance(content, str):
        return content
    return bytes(content).decode("utf-8-sig")


def _parse_start_time(value: Any) -> int | None:
    """Coerce a user-supplied start (epoch ms or ISO-8601; naive means UTC) to epoch ms."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"Invalid startTime: {value!r}")
    if isinstance(value, int | float):
        try:
            return int(value)
        except (OverflowError, ValueError) as exc:  # inf / nan
            raise ValueError(f"Invalid startTime: {value!r}") from exc
    raw = str(value).strip()
    if re.fullmatch(r"\d{1,30}(?:\.\d{1,30})?", raw):
        try:
            return int(float(raw))
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"Invalid startTime: {value!r}") from exc
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid startTime: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def _title_from(filename: str | None) -> str:
    if not filename:
        return DEFAULT_TITLE
    stem = re.sub(r"\.[^/.]+$", "", Path(filename).name)
    return re.sub(r"[_-]", " ", stem).strip() or DEFAULT_TITLE


def _render_text(segments: Sequence[Any]) -> str:
    return "\n\n".join(f"{seg.speaker}: {seg.text}" for seg in segments)


class TranscriptImportService:
    """Create no-audio meetings from existing transcript files."""

    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        db: Engine | DbHandle | None = None,
        meetings_dir: str | None = None,
        config_dir: str | None = None,
        insights_service: Any | None = None,
        now: Callable[[], int] | None = None,
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
        self.insights_service = (
            insights_service if insights_service is not None else opts.get("insightsService")
        )
        self.now = now or (lambda: int(time.time() * 1000))

    # -- public API --------------------------------------------------------

    async def import_transcript(self, opts: Mapping[str, Any]) -> dict[str, Any]:
        """Import one transcript file.

        ``opts`` keys: ``content`` (bytes or str), ``filename``, ``format`` (``vtt``/``srt``/
        ``text``; auto-detected when omitted), ``title``, ``startTime`` (epoch ms or ISO-8601),
        ``subtype`` (e.g. ``zoom``; detected when omitted) and ``meAliases``.

        Raises:
            MissingStartTimeError: no start time supplied or derivable.
            ValueError: undecodable/empty content or an invalid ``startTime``/``format``.
        """
        content = _normalize_content(_decode(opts.get("content") or ""))
        filename: str | None = opts.get("filename")
        content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()

        existing = self._find_import(content_sha)
        if existing is not None:
            logger.info(
                "Transcript import deduplicated via content hash",
                {"category": "import", "sha256": content_sha, "meetingId": existing},
            )
            return self._deduped_result(existing)

        fmt: ImportFormat | None = opts.get("format")
        if fmt is not None and fmt not in ("vtt", "srt", "text"):
            raise ValueError(f"Unsupported transcript format: {fmt!r}")
        # Regex-heavy and O(size): keep it off the event loop so a slow parse cannot stall the server.
        parsed = await asyncio.to_thread(parse_transcript, content, filename, fmt)

        start_time = _parse_start_time(opts.get("startTime"))
        if start_time is None:
            start_time = parsed.startedAtMs
        if start_time is None:
            raise MissingStartTimeError(
                "Could not determine the meeting start time from the filename or content; "
                "supply startTime explicitly"
            )

        segments = parsed.transcript.segments
        subtype = str(opts.get("subtype") or parsed.subtype or "") or None
        title = " ".join(str(opts.get("title") or "").split()) or _title_from(filename)
        end_time = start_time + max((s.endMs for s in segments), default=0)
        meeting_id = str(uuid.uuid4())
        json_art_id = str(uuid.uuid4())
        txt_art_id = str(uuid.uuid4())
        provider = f"import:{subtype or parsed.format}"
        tags = ["import", subtype] if subtype else ["import"]
        now = self.now()

        paths = ensure_meeting_folder(meeting_paths(self.meetings_dir, start_time, title, meeting_id))
        folder = Path(paths.folder)
        try:
            (folder / TRANSCRIPT_JSON_REL).write_text(
                f"{json.dumps(parsed.transcript.model_dump(exclude_none=True), indent=2)}\n",
                encoding="utf-8",
            )
            (folder / TRANSCRIPT_TXT_REL).write_text(f"{_render_text(segments)}\n", encoding="utf-8")

            with self.db.begin() as conn:
                self._insert_meeting(conn, meeting_id, title, start_time, end_time, tags, now)
                for art_id, fmt_name, rel in (
                    (json_art_id, "json", TRANSCRIPT_JSON_REL),
                    (txt_art_id, "txt", TRANSCRIPT_TXT_REL),
                ):
                    conn.execute(
                        text(
                            """
                            INSERT INTO artifacts (id, meeting_id, recording_id, kind, provider, format, path, created_at)
                            VALUES (:id, :mid, NULL, 'transcript', :provider, :format, :path, :now)
                            """
                        ),
                        {
                            "id": art_id,
                            "mid": meeting_id,
                            "provider": provider,
                            "format": fmt_name,
                            "path": rel,
                            "now": now,
                        },
                    )
                conn.execute(
                    text("UPDATE meetings SET primary_transcript_artifact_id = :aid WHERE id = :mid"),
                    {"aid": json_art_id, "mid": meeting_id},
                )
                conn.execute(
                    text(
                        """
                        INSERT INTO transcript_imports (id, content_sha256, meeting_id, source_format, subtype, created_at)
                        VALUES (:id, :sha, :mid, :fmt, :subtype, :now)
                        """
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "sha": content_sha,
                        "mid": meeting_id,
                        "fmt": parsed.format,
                        "subtype": subtype,
                        "now": now,
                    },
                )
                conn.execute(
                    text(
                        """
                        INSERT INTO stage_runs (
                          id, meeting_id, stage, status, attempts, started_at, finished_at, created_at, updated_at
                        ) VALUES (:id, :mid, 'import', 'done', 0, :now, :now, :now, :now)
                        """
                    ),
                    {"id": str(uuid.uuid4()), "mid": meeting_id, "now": now},
                )
                labels = list(dict.fromkeys(s.speaker for s in segments))
                speakers, unmapped = self._link_speakers(
                    conn, meeting_id, labels, self._me_aliases(opts.get("meAliases")), json_art_id, now
                )
        except IntegrityError:
            # Lost a race with a concurrent import of identical content.
            shutil.rmtree(folder, ignore_errors=True)
            raced = self._find_import(content_sha)
            if raced is None:
                raise
            return self._deduped_result(raced)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise

        warnings = list(parsed.warnings)
        logger.info(
            "Transcript imported",
            {
                "category": "import",
                "meetingId": meeting_id,
                "format": parsed.format,
                "subtype": subtype,
                "segments": len(segments),
            },
        )
        if self.insights_service is not None:
            try:
                await self.insights_service.enqueue_analyze(meeting_id)
            except Exception as exc:
                logger.warn(
                    "Insights enqueue after transcript import failed",
                    {"category": "import", "meetingId": meeting_id, "error": str(exc)},
                )
                warnings.append(f"Insights analysis was not queued: {exc}")

        return {
            "meetingId": meeting_id,
            "deduped": False,
            "format": parsed.format,
            "subtype": subtype,
            "segmentCount": len(segments),
            "startTime": start_time,
            "speakers": speakers,
            "unmappedNames": unmapped,
            "warnings": warnings,
            "timingSynthetic": parsed.timingSynthetic,
        }

    # -- internals ---------------------------------------------------------

    def _find_import(self, content_sha: str) -> str | None:
        with self.db.connect() as conn:
            row = conn.execute(
                text("SELECT meeting_id FROM transcript_imports WHERE content_sha256 = :h"),
                {"h": content_sha},
            ).first()
        return str(row[0]) if row else None

    def _deduped_result(self, meeting_id: str) -> dict[str, Any]:
        """The result for content already imported, in the same shape as a fresh import.

        Callers (the route, the UI) read ``warnings``/``speakers``/``unmappedNames`` without
        checking ``deduped`` first, so none of those keys may be missing. Values come from the
        existing meeting; what was not persisted (``timingSynthetic``) is ``None``.
        """
        with self.db.connect() as conn:
            meeting = (
                conn.execute(
                    text("SELECT start_time, title FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            imp = (
                conn.execute(
                    text("SELECT source_format, subtype FROM transcript_imports WHERE meeting_id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            linked = (
                conn.execute(
                    text(
                        """
                        SELECT s.name AS name, s.id AS speaker_id, ms.identity_source AS identity_source
                        FROM meeting_speakers ms INNER JOIN speakers s ON s.id = ms.speaker_id
                        WHERE ms.meeting_id = :id ORDER BY s.name, s.id
                        """
                    ),
                    {"id": meeting_id},
                )
                .mappings()
                .all()
            )
        segment_count: int | None = None
        if meeting is not None:
            try:
                folder = meeting_paths(
                    self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting_id
                ).folder
                payload = json.loads((Path(folder) / TRANSCRIPT_JSON_REL).read_text(encoding="utf-8"))
                segment_count = len(payload.get("segments") or [])
            except (OSError, ValueError, AttributeError):
                segment_count = None
        return {
            "meetingId": meeting_id,
            "deduped": True,
            "format": imp["source_format"] if imp else None,
            "subtype": imp["subtype"] if imp else None,
            "segmentCount": segment_count,
            "startTime": int(meeting["start_time"]) if meeting else None,
            "speakers": [
                {
                    "label": r["name"],
                    "speakerId": r["speaker_id"],
                    "identitySource": r["identity_source"] or "name",
                }
                for r in linked
            ],
            "unmappedNames": [],
            "warnings": [],
            "timingSynthetic": None,
        }

    def _me_aliases(self, supplied: Sequence[str] | None) -> set[str]:
        configured = load_insights_config(self.config_dir).get("meAliases")
        raw = [str(a) for a in configured] if isinstance(configured, list) else []
        raw += [str(a) for a in (supplied or [])]
        return {normalize_alias(a) for a in raw if normalize_alias(a)}

    @staticmethod
    def _insert_meeting(
        conn: Connection, meeting_id: str, title: str, start: int, end: int, tags: list[str], now: int
    ) -> None:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, :title, :start, :end, 'import', 'ready', :tags,
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {
                "id": meeting_id,
                "title": title,
                "start": start,
                "end": end,
                "tags": json.dumps(tags),
                "now": now,
            },
        )

    @staticmethod
    def _link_speakers(
        conn: Connection,
        meeting_id: str,
        labels: Sequence[str],
        me_aliases: set[str],
        evidence_artifact_id: str,
        now: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Link each non-generic label to a speaker by name; report ambiguous names."""
        speakers = [
            dict(r)
            for r in conn.execute(
                text("SELECT id, name, enrolled_at FROM speakers ORDER BY created_at, id")
            ).mappings()
        ]
        by_name: dict[str, set[str]] = {}
        for sp in speakers:
            by_name.setdefault(normalize_alias(sp["name"]), set()).add(sp["id"])
        by_alias: dict[str, set[str]] = {}
        for r in conn.execute(
            text("SELECT alias_norm, speaker_id FROM speaker_aliases WHERE speaker_id IS NOT NULL")
        ).mappings():
            by_alias.setdefault(r["alias_norm"], set()).add(r["speaker_id"])
        me_id: str | None = SpeakerNamingService._find_me(  # pyright: ignore[reportPrivateUsage]
            speakers, {a: next(iter(ids)) for a, ids in by_alias.items() if len(ids) == 1}, me_aliases
        )

        result: list[dict[str, Any]] = []
        unmapped: list[str] = []
        for label in labels:
            if is_generic_speaker_label(label):
                result.append({"label": label, "speakerId": None, "identitySource": None})
                continue
            display = " ".join(label.split())
            norm = normalize_alias(display)
            speaker_id: str | None
            if norm in me_aliases and me_id is not None:
                speaker_id = me_id
            else:
                candidates = by_name.get(norm, set()) | by_alias.get(norm, set())
                if len(candidates) > 1:
                    unmapped.append(label)
                    result.append({"label": label, "speakerId": None, "identitySource": None})
                    continue
                if candidates:
                    speaker_id = next(iter(candidates))
                else:
                    speaker_id = str(uuid.uuid4())
                    conn.execute(
                        text("INSERT INTO speakers (id, name, created_at) VALUES (:id, :name, :now)"),
                        {"id": speaker_id, "name": display, "now": now},
                    )
                    by_name[norm] = {speaker_id}
                    if norm in me_aliases:
                        me_id = speaker_id
            conn.execute(
                text(
                    """
                    INSERT INTO meeting_speakers (meeting_id, speaker_id, evidence_artifact_id, identity_source)
                    VALUES (:mid, :sid, :aid, 'name')
                    ON CONFLICT(meeting_id, speaker_id) DO NOTHING
                    """
                ),
                {"mid": meeting_id, "sid": speaker_id, "aid": evidence_artifact_id},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO speaker_aliases (id, alias_norm, speaker_id, kind, confirmed, created_at)
                    VALUES (:id, :alias, :sid, 'name', 0, :now)
                    ON CONFLICT(alias_norm, speaker_id) DO NOTHING
                    """
                ),
                {"id": str(uuid.uuid4()), "alias": norm, "sid": speaker_id, "now": now},
            )
            result.append({"label": label, "speakerId": speaker_id, "identitySource": "name"})
        return result, unmapped


__all__ = ["MissingStartTimeError", "TranscriptImportService"]
