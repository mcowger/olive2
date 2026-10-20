"""Read-only census of stored meeting transcripts, for sizing Olive Insights.

Reads the meetings table and each meeting's primary transcript JSON (falling
back to the latest transcript JSON, as ``TranscriptionService`` does) and
reports corpus statistics: transcript coverage and shapes, segment and word
totals with an estimated token count, meeting-duration buckets, the share of
generic (unidentified) speaker labels, distinct labels per meeting, meetings
with no identified speakers, and month-by-month coverage. Nothing is written:
no migrations, no WAL switch, no audio or ML imports.

Transcript parsing mirrors ``services.transcription._parse_stored_transcript``
line for line. It is duplicated rather than imported because that module pulls
in ``ml.identify`` and therefore the pyannote/torch stack. Keep the two in sync.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ..db import DbHandle
from ..layout import meeting_paths
from ..plaud.poller import plaud_segments_to_transcript
from ..schemas import Transcript
from ..speaker_labels import is_generic_speaker_label

MINUTE_MS = 60_000
TOKENS_PER_WORD = 1.3

# (label, lower bound in ms inclusive, upper bound in ms exclusive or None)
DURATION_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("<5m", 0, 5 * MINUTE_MS),
    ("5-30m", 5 * MINUTE_MS, 30 * MINUTE_MS),
    ("30-90m", 30 * MINUTE_MS, 90 * MINUTE_MS),
    ("90m-3h", 90 * MINUTE_MS, 180 * MINUTE_MS),
    (">3h", 180 * MINUTE_MS, None),
)
UNKNOWN_DURATION = "unknown"


def open_readonly_engine(database_path: str) -> Engine:
    """Return an engine that can only read the SQLite file at ``database_path``.

    Opens with ``mode=ro`` so no migrations run and the journal mode is left
    untouched, which makes it safe to point at a live database.
    """
    path = Path(database_path).resolve()
    return create_engine(f"sqlite:///file:{path}?mode=ro&uri=true", connect_args={"timeout": 30.0})


def compute_census(db: Engine | DbHandle, meetings_dir: str | Path) -> dict[str, Any]:
    """Compute corpus statistics over every meeting's stored transcript.

    ``db`` is an engine or handle; only SELECT statements are issued.
    ``meetings_dir`` is the root that meeting folders live under.
    """
    engine = db.engine if isinstance(db, DbHandle) else db
    with engine.connect() as conn:
        meetings = [dict(row) for row in conn.execute(text(_MEETINGS_SQL)).mappings().all()]
        for meeting in meetings:
            meeting["has_identified_speaker"] = bool(
                conn.execute(
                    text("SELECT 1 FROM meeting_speakers WHERE meeting_id = :mid LIMIT 1"),
                    {"mid": meeting["id"]},
                ).first()
            )
            meeting["transcript"] = resolve_transcript_row(conn, meeting)
            meeting["duration_ms"] = recording_duration(conn, meeting["id"])

    root = Path(meetings_dir)
    shapes: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    segment_counts: list[int] = []
    word_counts: list[int] = []
    label_counts: list[int] = []
    generic_segments = 0
    named_segments = 0
    loaded = 0
    duration_counts: Counter[str] = Counter()
    month_counts: Counter[str] = Counter()
    with_primary = 0
    fallback_only = 0
    no_transcript = 0

    for meeting in meetings:
        month_counts[_month_of(meeting["start_time"])] += 1
        primary_id = meeting["primary_transcript_artifact_id"]
        artifact = meeting["transcript"]
        if artifact is None:
            no_transcript += 1
        elif artifact["id"] == primary_id:
            with_primary += 1
        else:
            fallback_only += 1

        transcript: Transcript | None = None
        if artifact is not None:
            folder = meeting_paths(
                str(root), int(meeting["start_time"]), meeting["title"], meeting["id"]
            ).folder
            loaded_result = load_transcript(Path(folder) / artifact["path"], meeting["duration_ms"])
            if isinstance(loaded_result, str):
                failures[loaded_result] += 1
            else:
                shape, transcript = loaded_result
                shapes[shape] += 1
                loaded += 1
                segment_counts.append(len(transcript.segments))
                word_counts.append(sum(len(seg.text.split()) for seg in transcript.segments))
                labels = {seg.speaker.strip() for seg in transcript.segments}
                label_counts.append(len(labels))
                for seg in transcript.segments:
                    if is_generic_speaker_label(seg.speaker):
                        generic_segments += 1
                    else:
                        named_segments += 1

        duration_ms = _meeting_duration_ms(meeting, transcript)
        duration_counts[_duration_bucket(duration_ms)] += 1

    total_segments = sum(segment_counts)
    total_words = sum(word_counts)
    total_speaker_segments = generic_segments + named_segments
    return {
        "meetings": {
            "total": len(meetings),
            "with_primary_transcript": with_primary,
            "without_primary_transcript": len(meetings) - with_primary,
            "fallback_transcript_only": fallback_only,
            "no_transcript": no_transcript,
        },
        "transcripts": {
            "loaded": loaded,
            "shapes": dict(sorted(shapes.items())),
            "load_failures": dict(sorted(failures.items())),
            "segments": {"total": total_segments, "per_meeting": _distribution(segment_counts)},
            "words": {
                "total": total_words,
                "per_meeting": _distribution(word_counts),
                "estimated_tokens": int(round(total_words * TOKENS_PER_WORD)),
            },
        },
        "speakers": {
            "generic_segments": generic_segments,
            "named_segments": named_segments,
            "generic_share": (generic_segments / total_speaker_segments if total_speaker_segments else None),
            "distinct_labels_per_meeting": _distribution(label_counts),
            "meetings_with_no_identified_speakers": sum(
                1 for meeting in meetings if not meeting["has_identified_speaker"]
            ),
        },
        "duration_histogram": {
            label: duration_counts.get(label, 0)
            for label in [bucket[0] for bucket in DURATION_BUCKETS] + [UNKNOWN_DURATION]
        },
        "months": dict(sorted(month_counts.items())),
    }


def format_census_report(stats: dict[str, Any]) -> str:
    """Render the output of :func:`compute_census` as a plain-text report."""
    meetings = stats["meetings"]
    transcripts = stats["transcripts"]
    speakers = stats["speakers"]
    lines = [
        "Olive Insights corpus census (read-only)",
        "",
        "Meetings",
        f"  {meetings['total']} total",
        f"  {meetings['with_primary_transcript']} with a primary transcript",
        f"  {meetings['fallback_transcript_only']} with a fallback transcript only",
        f"  {meetings['no_transcript']} with no transcript",
        "",
        "Transcripts",
        f"  {transcripts['loaded']} loaded",
        f"  shapes: {_format_counts(transcripts['shapes'])}",
        f"  load failures: {_format_counts(transcripts['load_failures'])}",
        f"  {transcripts['segments']['total']} segments"
        f" (per meeting: {_format_distribution(transcripts['segments']['per_meeting'])})",
        f"  {transcripts['words']['total']} words"
        f" (per meeting: {_format_distribution(transcripts['words']['per_meeting'])})",
        f"  ~{transcripts['words']['estimated_tokens']} estimated tokens (words x {TOKENS_PER_WORD})",
        "",
        "Speakers",
        f"  generic segments: {speakers['generic_segments']}",
        f"  named segments: {speakers['named_segments']}",
        f"  generic share: {_format_share(speakers['generic_share'])}",
        f"  distinct labels per meeting: {_format_distribution(speakers['distinct_labels_per_meeting'])}",
        f"  meetings with no identified speakers: {speakers['meetings_with_no_identified_speakers']}",
        "",
        "Duration",
    ]
    lines += [f"  {label}: {count}" for label, count in stats["duration_histogram"].items()]
    lines += ["", "Months"]
    lines += [f"  {month}: {count}" for month, count in stats["months"].items()] or ["  none"]
    return "\n".join(lines)


_MEETINGS_SQL = """
SELECT id, title, start_time, end_time, primary_transcript_artifact_id
FROM meetings
ORDER BY start_time
"""


def resolve_transcript_row(conn: Any, meeting: dict[str, Any]) -> dict[str, Any] | None:
    """Primary transcript JSON artifact, else the latest transcript JSON artifact."""
    primary_id = meeting["primary_transcript_artifact_id"]
    if primary_id:
        row = (
            conn.execute(
                text(
                    "SELECT id, path FROM artifacts "
                    "WHERE id = :id AND kind = 'transcript' AND format = 'json'"
                ),
                {"id": primary_id},
            )
            .mappings()
            .first()
        )
        if row:
            return dict(row)
    row = (
        conn.execute(
            text(
                "SELECT id, path FROM artifacts WHERE meeting_id = :mid "
                "AND kind = 'transcript' AND format = 'json' ORDER BY created_at DESC LIMIT 1"
            ),
            {"mid": meeting["id"]},
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


def recording_duration(conn: Any, meeting_id: str) -> int | None:
    value = conn.execute(
        text("SELECT duration_ms FROM recordings WHERE meeting_id = :mid ORDER BY created_at ASC LIMIT 1"),
        {"mid": meeting_id},
    ).scalar()
    return int(value) if value else None


def load_transcript(path: Path, duration_ms: int | None) -> tuple[str, Transcript] | str:
    """Return ``(shape, transcript)`` or a failure reason for one transcript file."""
    if not path.is_file():
        return "missing_file"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "invalid_json"
    try:
        return _parse_transcript(raw, duration_ms)
    except ValueError:
        return "unrecognized_shape"


def _parse_transcript(raw: Any, duration_ms: int | float | None) -> tuple[str, Transcript]:
    """Mirror of ``services.transcription._parse_stored_transcript``, also reporting the shape."""
    if isinstance(raw, dict):
        return "segments_object", Transcript.model_validate(raw)
    if isinstance(raw, list) and raw:
        first = next((item for item in raw if isinstance(item, dict)), None)
        if first is not None and ("start_time" in first or "end_time" in first or "content" in first):
            converted = plaud_segments_to_transcript(raw, duration_ms)
            if converted is not None:
                return "raw_plaud", converted
        segments = []
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
                {
                    "startMs": start,
                    "endMs": end,
                    "speaker": str(item.get("speaker") or "Speaker"),
                    "text": text_value,
                }
            )
        if segments:
            return "bare_list", Transcript.model_validate({"segments": segments})
    raise ValueError("unrecognized transcript shape")


def _meeting_duration_ms(meeting: dict[str, Any], transcript: Transcript | None) -> int | None:
    """Meeting end minus start; falls back to the transcript's last segment end."""
    span = int(meeting["end_time"]) - int(meeting["start_time"])
    if span > 0:
        return span
    if transcript is not None and transcript.segments:
        last_end = max(seg.endMs for seg in transcript.segments)
        if last_end > 0:
            return last_end
    return None


def _duration_bucket(duration_ms: int | None) -> str:
    if duration_ms is None:
        return UNKNOWN_DURATION
    for label, lower, upper in DURATION_BUCKETS:
        if duration_ms >= lower and (upper is None or duration_ms < upper):
            return label
    return UNKNOWN_DURATION


def _month_of(start_time_ms: int) -> str:
    return datetime.fromtimestamp(int(start_time_ms) / 1000.0, tz=UTC).strftime("%Y-%m")


def _distribution(values: list[int]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def _format_distribution(dist: dict[str, float | int] | None) -> str:
    if dist is None:
        return "n/a"
    return f"min {dist['min']}, median {dist['median']:g}, max {dist['max']}"


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key} {value}" for key, value in counts.items())


def _format_share(share: float | None) -> str:
    return "n/a" if share is None else f"{share:.1%}"


# Backward-compatible private names for the transcript-read helpers made public above.
_resolve_transcript_row = resolve_transcript_row
_recording_duration = recording_duration
_load_transcript = load_transcript

__all__ = [
    "DURATION_BUCKETS",
    "compute_census",
    "format_census_report",
    "load_transcript",
    "open_readonly_engine",
    "recording_duration",
    "resolve_transcript_row",
]
