"""Meeting auto-title generation service.

Generates a concise meeting title from the meeting transcript using the
configured default LLM (provider/model from ``LlmSettings``).
"""

from __future__ import annotations

import inspect
import json
import re
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..db import DbHandle, get_db
from ..layout import meeting_paths
from ..logging_db import logger
from ..schemas import LlmGenerateResult
from ..settings import resolve_paths
from .llm import LlmService
from .meetings import update_meeting_title

# Transcript characters sent to the LLM (head-truncated to bound cost/latency).
AUTO_TITLE_MAX_INPUT_CHARS = 12_000
# Default title format limits: single line, at most 80 chars, at least 3.
AUTO_TITLE_MAX_LENGTH = 80
AUTO_TITLE_MIN_LENGTH = 3

AUTO_TITLE_SYSTEM_PROMPT = (
    "You generate concise meeting titles. Reply with ONLY the title text: "
    "3-8 words in Title Case, plain text, no quotes, no emoji, no trailing period, "
    f"at most {AUTO_TITLE_MAX_LENGTH} characters. Name the topic or decision, not the speakers."
)

AUTO_TITLE_USER_PROMPT_TEMPLATE = (
    "Meeting date: {date}\n"
    "Transcript excerpt:\n{transcript}\n\n"
    f"Write a meeting title (3-8 words, Title Case, max {AUTO_TITLE_MAX_LENGTH} characters)."
)


def sanitize_generated_title(raw: str) -> str:
    """Normalize raw LLM output into a safe single-line meeting title."""
    text_out = (raw or "").strip()
    if not text_out:
        return ""
    # Keep the first non-empty line only.
    for line in text_out.splitlines():
        stripped = line.strip()
        if stripped:
            text_out = stripped
            break
    # Strip wrapping quotes/brackets/markdown emphasis the model may add.
    text_out = re.sub(r"^[\s>\"'`*#\-–—:;()\[\]{}]+", "", text_out)
    text_out = re.sub(r"[\s\"'`*#.。!?:;()\[\]{}]+$", "", text_out)
    text_out = re.sub(r"\s+", " ", text_out).strip()
    if len(text_out) > AUTO_TITLE_MAX_LENGTH:
        cut = text_out[:AUTO_TITLE_MAX_LENGTH].rstrip()
        # Prefer a word boundary when one exists near the cutoff.
        space = cut.rfind(" ")
        if space >= AUTO_TITLE_MIN_LENGTH:
            cut = cut[:space]
        text_out = cut.rstrip()
    return text_out


def _format_transcript(raw_transcript: str, transcript_format: str | None) -> str:
    if transcript_format == "json":
        try:
            parsed = json.loads(raw_transcript)
            segs = parsed.get("segments") if isinstance(parsed, dict) else None
            if isinstance(segs, list):
                return "\n".join(
                    f"{seg.get('speaker') or 'Unknown'}: {seg.get('text', '')}"
                    for seg in segs
                    if isinstance(seg, dict) and str(seg.get("text", "")).strip()
                )
        except Exception:
            pass
    return raw_transcript


async def _call_llm_generate(llm_service: Any, options: dict[str, Any]) -> LlmGenerateResult:
    inst_dict = getattr(llm_service, "__dict__", {})
    fn = (
        inst_dict.get("generate_text")
        or inst_dict.get("generateText")
        or getattr(llm_service, "generate_text", None)
        or getattr(llm_service, "generateText", None)
    )
    if not callable(fn):
        raise AttributeError("llm_service has no generate_text/generateText method")
    res = fn(options)
    if inspect.isawaitable(res):
        res = await res
    return res if isinstance(res, LlmGenerateResult) else LlmGenerateResult.model_validate(res)


class TitleService:
    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        db: Engine | DbHandle | None = None,
        meetings_dir: str | None = None,
        llm_service: LlmService | Any | None = None,
    ) -> None:
        opts = options or {}
        raw_db = db or opts.get("db")
        if isinstance(raw_db, DbHandle):
            self.db: Engine = raw_db.engine
        elif isinstance(raw_db, Engine):
            self.db = raw_db
        else:
            self.db = get_db()

        self.meetings_dir = str(
            meetings_dir
            or opts.get("meetingsDir")
            or opts.get("meetings_dir")
            or resolve_paths().meetings_dir
        )
        self.llm_service = (
            llm_service or opts.get("llmService") or opts.get("llm_service") or LlmService(db=self.db)
        )

    def _load_transcript(self, meeting_id: str) -> tuple[Mapping[Any, Any], str]:
        with self.db.connect() as conn:
            meeting = (
                conn.execute(
                    text("SELECT * FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            artifacts = (
                conn.execute(
                    text("SELECT * FROM artifacts WHERE meeting_id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .all()
            )

        folder = meeting_paths(
            self.meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]
        ).folder

        primary_tid = meeting.get("primary_transcript_artifact_id")
        transcript_artifact = next((a for a in artifacts if a["id"] == primary_tid), None)
        if not transcript_artifact:
            transcript_artifact = next((a for a in artifacts if a["kind"] == "transcript"), None)
        if not transcript_artifact:
            raise ValueError("No transcript found for meeting. Please transcribe the recording first.")

        full_path = Path(folder) / transcript_artifact["path"]
        if not full_path.exists():
            raise ValueError(f"Transcript artifact not found on disk at: {transcript_artifact['path']}")

        raw = full_path.read_text(encoding="utf-8")
        formatted = _format_transcript(raw, transcript_artifact.get("format"))
        if not formatted.strip():
            raise ValueError("Transcript is empty; cannot generate a title.")
        return meeting, formatted

    async def generate_title(
        self,
        meeting_id: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate (but do not persist) a title with the default LLM.

        ``provider``/``model`` override the configured defaults when given;
        ``None`` falls through to the default LLM in ``LlmSettings``.
        """
        meeting, formatted = self._load_transcript(meeting_id)
        excerpt = formatted[:AUTO_TITLE_MAX_INPUT_CHARS]
        date_formatted = datetime.fromtimestamp(int(meeting["start_time"]) / 1000.0).strftime("%b %d, %Y")
        user_prompt = AUTO_TITLE_USER_PROMPT_TEMPLATE.format(date=date_formatted, transcript=excerpt)

        llm_result = await _call_llm_generate(
            self.llm_service,
            {
                "provider": provider,
                "model": model,
                "systemPrompt": AUTO_TITLE_SYSTEM_PROMPT,
                "prompt": user_prompt,
            },
        )
        title = sanitize_generated_title(llm_result.text)
        if len(title) < AUTO_TITLE_MIN_LENGTH:
            raise RuntimeError("LLM returned an unusable title; please try again or rename manually.")
        return {
            "title": title,
            "provider": llm_result.provider,
            "model": llm_result.model,
            "durationMs": llm_result.durationMs,
            "usage": llm_result.usage.model_dump(exclude_none=True) if llm_result.usage else {},
        }

    generateTitle = generate_title

    async def generate_and_apply(
        self,
        meeting_id: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate a title with the default LLM and persist it as the meeting title."""
        generated = await self.generate_title(meeting_id, provider=provider, model=model)
        updated = update_meeting_title(self.db, meeting_id, generated["title"], self.meetings_dir)
        if not updated:
            raise ValueError(f"Meeting not found: {meeting_id}")
        now = int(time.time() * 1000)
        logger.info(
            "Auto-generated meeting title",
            {
                "meetingId": meeting_id,
                "title": generated["title"],
                "provider": generated["provider"],
                "model": generated["model"],
            },
        )
        try:
            from .usage import record_llm_usage

            record_llm_usage(
                self.db,
                meeting_id=meeting_id,
                kind="title",
                provider=str(generated["provider"]),
                model=str(generated["model"]),
                input_tokens=int((generated["usage"] or {}).get("inputTokens") or 0),
                output_tokens=int((generated["usage"] or {}).get("outputTokens") or 0),
                total_tokens=int((generated["usage"] or {}).get("totalTokens") or 0),
                created_at=now,
                source=f"title:{meeting_id}",
                detail={"title": generated["title"]},
            )
        except Exception:
            pass
        return {
            "meeting": updated.model_dump(),
            "title": generated["title"],
            "provider": generated["provider"],
            "model": generated["model"],
        }

    generateAndApply = generate_and_apply
