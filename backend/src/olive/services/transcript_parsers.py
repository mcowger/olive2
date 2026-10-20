"""Transcript import parsers (Zoom/WebVTT, SRT, speaker-labeled text).

Pure functions (stdlib only; no DB, network or ML) that turn an existing
transcript file into the same internal :class:`~olive.schemas.Transcript` shape
the ASR pipeline produces: speaker-labelled segments with millisecond times.
Imported transcripts carry display *names*, not voiceprints.

Entry points: :func:`parse_transcript` (dispatching), :func:`detect_format`,
:func:`infer_start_time`, plus the per-format :func:`parse_vtt`,
:func:`parse_srt` and :func:`parse_speaker_text`.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from ..schemas import Transcript, TranscriptSegment

ImportFormat = Literal["vtt", "srt", "text"]

UNKNOWN_SPEAKER = "Unknown"
NO_SEGMENTS_MESSAGE = "No speech segments found"

# Synthetic pacing when a source has no timestamps: ~150 words per minute.
SYNTHETIC_MS_PER_WORD = 400
# Segment duration used for the final timed segment, which has no successor.
_MIN_SYNTHETIC_MS = SYNTHETIC_MS_PER_WORD
_MAX_SPEAKER_NAME_CHARS = 60

_TS = r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?"
_CUE_RE = re.compile(rf"(?P<start>{_TS})\s*-->\s*(?P<end>{_TS})")
_SRT_TIMING_RE = re.compile(r"\d{1,2}:\d{2}:\d{2},\d{1,3}\s*-->")
_VTT_TIMING_RE = re.compile(r"(?:\d{1,2}:)?\d{1,2}:\d{2}\.\d{1,3}\s*-->")
_VOICE_RE = re.compile(r"<v(?:\.[^\s>]*)*\s+([^>]+)>", re.IGNORECASE)
_TAG_RE = re.compile(r"</?[^>\n]+>")
_SPEAKER_PREFIX_RE = re.compile(r"^(?P<name>[^\W\d_][^:\n]*?)\s*:\s+(?P<text>\S.*)$")

_TEXT_TS = r"\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?"
# "[HH:MM:SS] Name: text"
_BRACKET_TS_RE = re.compile(rf"^\[(?P<ts>{_TEXT_TS})\]\s*(?P<name>[^:\n]+?)\s*:\s*(?P<text>.*)$")
# "Name (HH:MM:SS): text"
_PAREN_TS_RE = re.compile(rf"^(?P<name>[^()\n:]+?)\s*\((?P<ts>{_TEXT_TS})\)\s*:\s*(?P<text>.*)$")
# Zoom .txt: "[Name] HH:MM:SS" with the text on following lines
_ZOOM_HEADER_RE = re.compile(rf"^\[(?P<name>[^\]\n]+)\]\s+(?P<ts>{_TEXT_TS})\s*$")
# "Name: text"
_PLAIN_RE = re.compile(r"^(?P<name>[^\W\d_][^:\n]*?)\s*:\s*(?P<text>\S.*)$")

_ZOOM_FILENAME_RE = re.compile(r"zoom|GMT\d{8}-\d{6}", re.IGNORECASE)
_ZOOM_GMT_RE = re.compile(r"GMT(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})")
_ISO_RE = re.compile(
    r"(?<!\d)(\d{4})-(\d{2})-(\d{2})"
    r"(?:[T _](\d{2})[:.](\d{2})(?:[:.](\d{2}))?)?"
)


@dataclass
class ParsedImport:
    """Result of parsing an imported transcript.

    Attributes:
        transcript: Parsed transcript in the internal ``Transcript`` shape.
        format: Source format that was parsed.
        subtype: Source flavour, e.g. ``"zoom"``; ``None`` when unknown.
        startedAtMs: Meeting start (UTC epoch ms) inferred from the filename or
            header; ``None`` when absent (callers decide the fallback).
        timingSynthetic: True when segment times were generated rather than read.
        warnings: Non-fatal issues found while parsing.
    """

    transcript: Transcript
    format: ImportFormat
    subtype: str | None = None
    startedAtMs: int | None = None
    timingSynthetic: bool = False
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(content: str) -> str:
    """Strip a UTF-8 BOM and normalise line endings to ``\\n``."""
    if content.startswith("﻿"):
        content = content[1:]
    return content.replace("\r\n", "\n").replace("\r", "\n")


def _ts_to_ms(value: str) -> int:
    """Convert ``[HH:]MM:SS[.,mmm]`` to milliseconds."""
    value = value.replace(",", ".")
    clock, _, frac = value.partition(".")
    parts = [int(p) for p in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    millis = int(frac.ljust(3, "0")[:3]) if frac else 0
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def _word_count(text: str) -> int:
    return len(text.split())


def _synthetic_duration(text: str) -> int:
    return max(_word_count(text), 1) * SYNTHETIC_MS_PER_WORD


def _split_speaker(text: str) -> tuple[str, str]:
    """Split ``"Name: text"`` into ``(name, text)``; unlabeled text is ``Unknown``."""
    m = _SPEAKER_PREFIX_RE.match(text)
    if m and len(m.group("name")) <= _MAX_SPEAKER_NAME_CHARS and "//" not in m.group("name"):
        return m.group("name").strip(), m.group("text").strip()
    return UNKNOWN_SPEAKER, text.strip()


def _merge(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Merge consecutive same-speaker segments (text joined with a space)."""
    merged: list[TranscriptSegment] = []
    for seg in segments:
        if merged and merged[-1].speaker == seg.speaker:
            prev = merged[-1]
            prev.endMs = max(prev.endMs, seg.endMs)
            prev.text = f"{prev.text} {seg.text}"
        else:
            merged.append(seg)
    return merged


def _build(segments: list[TranscriptSegment]) -> Transcript:
    return Transcript(
        segments=segments,
        durationMs=max((s.endMs for s in segments), default=None),
    )


def _is_zoom_cue_style(has_numeric_ids: bool, labeled_cues: int) -> bool:
    return has_numeric_ids and labeled_cues > 0


# ---------------------------------------------------------------------------
# Cue formats (VTT / SRT)
# ---------------------------------------------------------------------------


def _parse_cues(content: str) -> tuple[list[TranscriptSegment], bool, int, list[str]]:
    """Parse WebVTT/SRT cue blocks.

    Returns ``(segments, saw_numeric_ids, labeled_cue_count, warnings)`` where the
    segments are already merged by consecutive speaker.
    """
    blocks = re.split(r"\n[ \t]*\n", _normalize(content).strip("\n"))
    raw: list[TranscriptSegment] = []
    warnings: list[str] = []
    numeric_ids = False
    labeled = 0
    skipped = 0
    for block in blocks:
        lines = [ln.rstrip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        head = lines[0].strip()
        if head.startswith("WEBVTT") or re.match(r"(NOTE|STYLE|REGION)(\s|$)", head):
            continue
        timing_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if timing_idx is None:
            skipped += 1
            continue
        m = _CUE_RE.search(lines[timing_idx])
        if not m:
            skipped += 1
            continue
        if timing_idx > 0 and lines[0].strip().isdigit():
            numeric_ids = True
        start, end = _ts_to_ms(m.group("start")), _ts_to_ms(m.group("end"))
        body = lines[timing_idx + 1 :]
        voice = next((vm for ln in body if (vm := _VOICE_RE.search(ln))), None)
        text = " ".join(ln.strip() for ln in body)
        text = html.unescape(_TAG_RE.sub("", text)).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        if voice:
            speaker = html.unescape(voice.group(1)).strip() or UNKNOWN_SPEAKER
            labeled += 1
        else:
            speaker, text = _split_speaker(text)
            if speaker != UNKNOWN_SPEAKER:
                labeled += 1
        raw.append(TranscriptSegment(startMs=start, endMs=max(end, start), speaker=speaker, text=text))
    if skipped:
        warnings.append(f"Skipped {skipped} block(s) without a valid cue timing line")
    return _merge(raw), numeric_ids, labeled, warnings


def parse_vtt(content: str, filename: str | None = None) -> ParsedImport:
    """Parse a WebVTT file (Zoom ``Name: text`` cues or ``<v Name>`` voice tags).

    Hours are optional (``MM:SS.mmm``), multi-line cues are joined with a space,
    ``NOTE``/``STYLE``/``REGION`` blocks are ignored, cues with no speaker become
    ``"Unknown"`` and consecutive same-speaker cues are merged.
    """
    segments, numeric_ids, labeled, warnings = _parse_cues(content)
    zoom = bool(filename and _ZOOM_FILENAME_RE.search(filename)) or (_is_zoom_cue_style(numeric_ids, labeled))
    return ParsedImport(
        transcript=_build(segments),
        format="vtt",
        subtype="zoom" if zoom else None,
        warnings=warnings,
    )


def parse_srt(content: str, filename: str | None = None) -> ParsedImport:
    """Parse an SRT file (comma milliseconds, optional ``Name:`` prefix)."""
    segments, _numeric, _labeled, warnings = _parse_cues(content)
    zoom = bool(filename and _ZOOM_FILENAME_RE.search(filename))
    return ParsedImport(
        transcript=_build(segments),
        format="srt",
        subtype="zoom" if zoom else None,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Speaker-labeled text
# ---------------------------------------------------------------------------


def _match_header(line: str) -> tuple[str, int | None, str, bool] | None:
    """Match a speaker header line.

    Returns ``(speaker, start_ms | None, inline_text, is_zoom_header)`` or ``None``.
    """
    stripped = line.strip()
    m = _ZOOM_HEADER_RE.match(stripped)
    if m:
        return m.group("name").strip(), _ts_to_ms(m.group("ts")), "", True
    m = _BRACKET_TS_RE.match(stripped)
    if m:
        return m.group("name").strip(), _ts_to_ms(m.group("ts")), m.group("text").strip(), False
    m = _PAREN_TS_RE.match(stripped)
    if m:
        return m.group("name").strip(), _ts_to_ms(m.group("ts")), m.group("text").strip(), False
    m = _PLAIN_RE.match(stripped)
    if m and len(m.group("name")) <= _MAX_SPEAKER_NAME_CHARS and "//" not in m.group("name"):
        return m.group("name").strip(), None, m.group("text").strip(), False
    return None


def parse_speaker_text(content: str, filename: str | None = None) -> ParsedImport:
    """Parse speaker-labeled plain text.

    Supported line forms: ``Name: text``, ``[HH:MM:SS] Name: text``,
    ``Name (HH:MM:SS): text`` and Zoom's ``[Name] HH:MM:SS`` header followed by
    text lines. Un-headed lines continue the previous turn. When no timestamps
    exist, sequential offsets are synthesised from word counts
    (``timingSynthetic=True`` plus a warning).

    Raises:
        ValueError: If no speaker-labeled line is found.
    """
    turns: list[dict] = []  # speaker, start (ms|None), parts
    zoom = bool(filename and _ZOOM_FILENAME_RE.search(filename))
    for line in _normalize(content).split("\n"):
        if not line.strip():
            continue
        header = _match_header(line)
        if header:
            speaker, start, inline, is_zoom = header
            zoom = zoom or is_zoom
            turns.append({"speaker": speaker, "start": start, "parts": [inline] if inline else []})
        elif turns:
            turns[-1]["parts"].append(line.strip())
    turns = [t for t in turns if t["parts"]]
    if not turns:
        raise ValueError(NO_SEGMENTS_MESSAGE)

    texts = [" ".join(t["parts"]) for t in turns]
    timed = [t["start"] is not None for t in turns]
    warnings: list[str] = []
    synthetic = not all(timed)
    if not any(timed):
        warnings.append("No timestamps found; segment times are synthetic (estimated from word counts)")
    elif synthetic:
        warnings.append(
            "Some turns have no timestamp; their times are synthetic (estimated from word counts)"
        )

    starts: list[int] = []
    cursor = 0
    for turn, text in zip(turns, texts, strict=False):
        start = turn["start"] if turn["start"] is not None else cursor
        starts.append(start)
        cursor = start + _synthetic_duration(text)

    segments: list[TranscriptSegment] = []
    for i, (turn, text) in enumerate(zip(turns, texts, strict=False)):
        start = starts[i]
        if i + 1 < len(turns) and starts[i + 1] >= start:
            end = starts[i + 1]
        else:
            end = start + _synthetic_duration(text)
        segments.append(
            TranscriptSegment(startMs=start, endMs=max(end, start), speaker=turn["speaker"], text=text)
        )
    return ParsedImport(
        transcript=_build(_merge(segments)),
        format="text",
        subtype="zoom" if zoom else None,
        timingSynthetic=synthetic,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Detection, start-time inference, dispatch
# ---------------------------------------------------------------------------


def detect_format(content: str, filename: str | None = None) -> ImportFormat:
    """Detect ``"vtt"``, ``"srt"`` or ``"text"`` from content, then filename."""
    body = _normalize(content).lstrip()
    if body.startswith("WEBVTT"):
        return "vtt"
    head = body[:4000]
    if _SRT_TIMING_RE.search(head):
        return "srt"
    name = (filename or "").lower()
    if name.endswith(".vtt"):
        return "vtt"
    if name.endswith(".srt"):
        return "srt"
    if _VTT_TIMING_RE.search(head) and "-->" in head:
        return "vtt"
    return "text"


def _to_epoch_ms(y: int, mo: int, d: int, h: int = 0, mi: int = 0, s: int = 0) -> int | None:
    try:
        return int(datetime(y, mo, d, h, mi, s, tzinfo=UTC).timestamp() * 1000)
    except ValueError:
        return None


def _iso_to_ms(text: str) -> int | None:
    m = _ISO_RE.search(text)
    if not m:
        return None
    parts = [int(g) if g else 0 for g in m.groups()]
    return _to_epoch_ms(*parts)


def infer_start_time(filename: str | None, content: str) -> int | None:
    """Infer the meeting start as UTC epoch ms, or ``None`` if not derivable.

    Checks, in order: a Zoom ``GMT20240315-150000`` filename, an ISO date in the
    filename, then an ISO date in the first lines of the content. Never defaults
    to the current time; naive ISO values are treated as UTC.
    """
    if filename:
        m = _ZOOM_GMT_RE.search(filename)
        if m:
            ms = _to_epoch_ms(*(int(g) for g in m.groups()))
            if ms is not None:
                return ms
        ms = _iso_to_ms(filename)
        if ms is not None:
            return ms
        if _ISO_RE.search(filename):
            return None
    header = "\n".join(_normalize(content).split("\n")[:20])
    return _iso_to_ms(header)


def parse_transcript(
    content: str,
    filename: str | None = None,
    fmt: ImportFormat | None = None,
) -> ParsedImport:
    """Parse an imported transcript into a :class:`ParsedImport`.

    Args:
        content: Raw file text (a UTF-8 BOM and CRLF endings are tolerated).
        filename: Original filename; used for format, subtype and start hints.
        fmt: Force a format instead of auto-detecting.

    Raises:
        ValueError: ``"No speech segments found"`` when nothing parseable exists.
    """
    content = _normalize(content)
    chosen = fmt or detect_format(content, filename)
    if chosen == "vtt":
        parsed = parse_vtt(content, filename)
    elif chosen == "srt":
        parsed = parse_srt(content, filename)
    else:
        parsed = parse_speaker_text(content, filename)
    if not parsed.transcript.segments:
        raise ValueError(NO_SEGMENTS_MESSAGE)
    parsed.startedAtMs = infer_start_time(filename, content)
    return parsed
