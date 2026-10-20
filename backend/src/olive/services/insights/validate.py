"""Parsing and validation of LLM output for Olive Insights.

``LlmService.generate_text`` returns plain text only (no JSON mode), so replies may carry
reasoning blocks, code fences, or chatty prose around the JSON object. :func:`parse_llm_json`
recovers the object; :class:`ChunkExtraction` validates its shape; :func:`parse_with_repair`
allows exactly one corrective re-ask before giving up.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import ValidationError, field_validator

from ...schemas import CamelModel

_THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*")

_REPAIR_ECHO_LIMIT = 1500


class InsightsValidationError(ValueError):
    """The LLM reply could not be parsed into the expected extraction shape."""


class ExtractedSpeaker(CamelModel):
    """One speaker's behavioral read for a chunk; dimension values are ordinal words."""

    speaker: str
    dimensions: dict[str, str] = {}
    stance: str | None = None

    @field_validator("dimensions", mode="before")
    @classmethod
    def _stringify_dimension_values(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {str(k): (v if isinstance(v, str) else str(v)) for k, v in value.items()}
        return value


class ExtractedMoment(CamelModel):
    """A citable interpersonal event. ``turnStartIdx`` is the only position the model supplies."""

    speaker: str
    toward: str | None = None
    turnStartIdx: int
    turnEndIdx: int | None = None
    eventType: str
    valence: Literal["good", "bad", "neutral", "ambiguous"]
    framework: str | None = None
    dimension: str | None = None
    severity: str | float
    quote: str
    rationale: str = ""
    guidance: str | None = None

    @field_validator("valence", mode="before")
    @classmethod
    def _normalize_valence(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class ChunkExtraction(CamelModel):
    """The validated reply for one transcript chunk."""

    speakers: list[ExtractedSpeaker] = []
    moments: list[ExtractedMoment] = []


def _balanced_object_end(text: str, start: int) -> int | None:
    """Index just past the ``{...}`` that opens at ``start``, honoring JSON strings; else None."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def parse_llm_json(text: str) -> dict[str, Any]:
    """Extract the first parseable top-level JSON object from an LLM reply.

    Strips ``<think>`` blocks and code-fence markers, then takes the first balanced ``{...}``
    that parses as a JSON object, skipping stray brace runs in surrounding prose.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text or "")
    # An unclosed opener is dropped with the model's reasoning; a lone closer ends it.
    closers = list(_THINK_CLOSE_RE.finditer(cleaned))
    if closers:
        cleaned = cleaned[closers[-1].end() :]
    cleaned = _FENCE_RE.sub("", cleaned)

    pos = cleaned.find("{")
    while pos != -1:
        end = _balanced_object_end(cleaned, pos)
        if end is not None:
            try:
                parsed = json.loads(cleaned[pos:end])
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        pos = cleaned.find("{", pos + 1)
    raise InsightsValidationError("No JSON object found in the model reply")


def validate_chunk(text: str) -> ChunkExtraction:
    """Parse ``text`` and validate it against :class:`ChunkExtraction`."""
    data = parse_llm_json(text)
    try:
        return ChunkExtraction.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8])
        raise InsightsValidationError(f"JSON does not match the expected shape ({problems})") from exc


async def parse_with_repair(ask: Callable[[str], Awaitable[str]], prompt: str) -> tuple[ChunkExtraction, str]:
    """Ask once; on a parse/validation failure re-ask exactly once with the error appended.

    Returns the validated extraction and the raw reply it came from. A second failure raises
    :class:`InsightsValidationError`; the caller fails the stage run. There is no retry loop.
    """
    first = await ask(prompt)
    try:
        return validate_chunk(first), first
    except InsightsValidationError as exc:
        error = str(exc)

    repair_prompt = (
        f"{prompt}\n\n"
        "Your previous reply was rejected and could not be used.\n"
        f"Problem: {error}\n"
        f"Previous reply (truncated):\n{first[:_REPAIR_ECHO_LIMIT]}\n\n"
        "Reply again with ONLY the corrected JSON object, with no commentary."
    )
    second = await ask(repair_prompt)
    try:
        return validate_chunk(second), second
    except InsightsValidationError as exc:
        raise InsightsValidationError(f"{exc} (after one repair attempt; first error: {error})") from exc


__all__ = [
    "ChunkExtraction",
    "ExtractedMoment",
    "ExtractedSpeaker",
    "InsightsValidationError",
    "parse_llm_json",
    "parse_with_repair",
    "validate_chunk",
]
