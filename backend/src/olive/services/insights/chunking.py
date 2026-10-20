"""Deterministic transcript chunking for long meetings.

Chunks are contiguous ranges of whole segments (a segment is never split). Turn numbers are
global indices into the full transcript so a moment's ``turnStartIdx`` means the same thing
regardless of which chunk produced it.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

from ...schemas import TranscriptSegment
from .validate import ExtractedMoment


@dataclass(frozen=True)
class Chunk:
    """Half-open range ``[start, end)`` of segments sent in one LLM call.

    ``core_start`` is the first segment not already covered by the previous chunk; segments in
    ``[start, core_start)`` are overlap context and do not count toward talk-time weighting.
    """

    index: int
    start: int
    end: int
    core_start: int


def format_turn(index: int, segment: TranscriptSegment) -> str:
    """Render one segment as the ``[turn] Speaker: words`` line the prompt cites."""
    return f"[{index}] {segment.speaker or 'Unknown'}: {segment.text}"


def _cost(index: int, segment: TranscriptSegment) -> int:
    return len(format_turn(index, segment)) + 1  # trailing newline


def chunk_segments(
    segments: Sequence[TranscriptSegment], chunk_chars: int, overlap_turns: int = 0
) -> list[Chunk]:
    """Split ``segments`` into budgeted chunks on turn boundaries with a small overlap.

    A chunk always contains at least one new segment, so an oversize segment becomes its own
    chunk rather than being split. The overlap is trimmed when it would push a chunk over budget.
    """
    n = len(segments)
    if n == 0:
        return []
    costs = [_cost(i, s) for i, s in enumerate(segments)]

    chunks: list[Chunk] = []
    start = 0
    core_start = 0
    while True:
        end = start
        used = 0
        while end < n:
            if end > core_start and used + costs[end] > chunk_chars:
                break
            used += costs[end]
            end += 1
        chunks.append(Chunk(index=len(chunks), start=start, end=end, core_start=core_start))
        if end >= n:
            return chunks

        nxt = max(start + 1, end - max(0, overlap_turns))
        # Budget for the next chunk is: overlap + its first new segment.
        while nxt < end and sum(costs[nxt : end + 1]) > chunk_chars:
            nxt += 1
        start = nxt
        core_start = end


def _norm_label(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def dedup_moments[M: ExtractedMoment](moments: Sequence[M]) -> list[M]:
    """Drop moments repeated across chunk overlaps, keeping the first occurrence.

    Identity is (speaker, turn-index range, event type), all normalized; input order is
    preserved, so the result is deterministic.
    """
    seen: set[Hashable] = set()
    kept: list[M] = []
    for m in moments:
        start = int(m.turnStartIdx)
        end = getattr(m, "turnEndIdx", None)
        key = (
            _norm_label(m.speaker),
            start,
            start if end is None else int(end),
            _norm_label(m.eventType),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(m)
    return kept


__all__ = ["Chunk", "chunk_segments", "dedup_moments", "format_turn"]
