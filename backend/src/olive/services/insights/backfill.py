"""Backfill: enqueue Insights analysis for already-transcribed meetings."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

from sqlalchemy import text

from ...layout import meeting_paths
from ...logging_db import logger
from . import extract
from .prompts import compute_prompt_version, load_insights_prompts


def _is_current(db: Any, meeting: Any, folder: str, prompt_version: str) -> bool:
    """True when a signals row already exists for this transcript + prompt version."""
    with db.connect() as conn:
        art = (
            conn.execute(
                text("SELECT path FROM artifacts WHERE id = :id"),
                {"id": meeting["primary_transcript_artifact_id"]},
            )
            .mappings()
            .first()
        )
        if not art:
            return False
        try:
            raw = (Path(folder) / art["path"]).read_text(encoding="utf-8")
        except OSError:
            return False
        transcript_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        row = conn.execute(
            text(
                "SELECT 1 FROM meeting_speaker_signals "
                "WHERE meeting_id = :mid AND transcript_hash = :h AND prompt_version = :v LIMIT 1"
            ),
            {"mid": meeting["id"], "h": transcript_hash, "v": prompt_version},
        ).first()
    return row is not None


async def backfill(
    db: Any,
    meetings_dir: str | Path,
    config_dir: str | Path,
    *,
    frm: int | None = None,
    to: int | None = None,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """Enqueue analysis for meetings with a primary transcript.

    Uses the registered queue hook (DBOS ``insights`` queue, deduplicated per meeting), so it is
    a no-op returning ``queued=0`` when no hook is registered. Meetings whose current transcript
    already has signals for the current prompt version are skipped unless ``force``. ``frm``/``to``
    bound ``start_time`` (ms, inclusive); ``limit`` caps how many are queued.
    """
    hook = extract.get_enqueue_hook()
    if hook is None:
        logger.info("Insights backfill skipped: no queue hook registered", {"category": "insights"})
        return {"queued": 0, "skipped": 0}

    clauses = ["primary_transcript_artifact_id IS NOT NULL"]
    params: dict[str, Any] = {}
    if frm is not None:
        clauses.append("start_time >= :frm")
        params["frm"] = frm
    if to is not None:
        clauses.append("start_time <= :to")
        params["to"] = to
    with db.connect() as conn:
        meetings = (
            conn.execute(
                text(f"SELECT * FROM meetings WHERE {' AND '.join(clauses)} ORDER BY start_time ASC"),
                params,
            )
            .mappings()
            .all()
        )

    prompt_version = compute_prompt_version(load_insights_prompts(str(config_dir)))
    queued = skipped = 0
    for meeting in meetings:
        if limit is not None and queued >= limit:
            break
        folder = meeting_paths(
            meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]
        ).folder
        if not force and _is_current(db, meeting, str(folder), prompt_version):
            skipped += 1
            continue
        res = hook(meeting["id"])
        if inspect.isawaitable(res):
            res = await res
        if res:
            queued += 1
        else:
            skipped += 1
    return {"queued": queued, "skipped": skipped}
