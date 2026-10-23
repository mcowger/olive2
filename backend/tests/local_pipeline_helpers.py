"""Shared helpers for exercising the flat local transcription pipeline in tests."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sqlalchemy import text

TERMINAL = ("done", "error", "cancelled")


async def wait_for_local_run(
    db: Any,
    meeting_id: str,
    *,
    timeout: float = 45.0,
    interval: float = 0.05,
) -> dict[str, str]:
    """Wait until the meeting's local run reaches a terminal stage status.

    Local stage workflows are fire-and-forget — nothing awaits the pipeline — so
    tests poll ``stage_runs`` instead of awaiting a result.
    """
    deadline = time.monotonic() + timeout
    last: dict[str, str] = {}
    while time.monotonic() < deadline:
        with db.connect() as conn:
            rows = (
                conn.execute(
                    text("SELECT stage, status FROM stage_runs WHERE meeting_id = :mid"),
                    {"mid": meeting_id},
                )
                .mappings()
                .all()
            )
        last = {str(r["stage"]): str(r["status"]) for r in rows}
        if last.get("local_transcribe") in TERMINAL:
            return last
        await asyncio.sleep(interval)
    raise AssertionError(f"local transcription for {meeting_id} did not finish; stage_runs={last}")


async def wait_for_stage(
    db: Any,
    meeting_id: str,
    stage: str,
    *,
    timeout: float = 45.0,
    interval: float = 0.05,
) -> str:
    """Wait until one stage row reaches a terminal status and return it."""
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        with db.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT status FROM stage_runs WHERE meeting_id = :mid AND stage = :stage"),
                    {"mid": meeting_id, "stage": stage},
                )
                .mappings()
                .first()
            )
        last = str(row["status"]) if row else ""
        if last in TERMINAL:
            return last
        await asyncio.sleep(interval)
    raise AssertionError(f"stage {stage} for {meeting_id} did not finish; status={last!r}")


async def wait_for_meeting_status(db: Any, meeting_id: str, status: str, *, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    last = ""
    error = ""
    while time.monotonic() < deadline:
        with db.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT status, last_error FROM meetings WHERE id = :id"), {"id": meeting_id}
                )
                .mappings()
                .first()
            )
        last = str(row["status"]) if row else ""
        error = str(row["last_error"] or "") if row else ""
        if last == status:
            return
        await asyncio.sleep(0.05)
    with db.connect() as conn:
        stages = (
            conn.execute(
                text("SELECT stage, status, last_error FROM stage_runs WHERE meeting_id = :mid"),
                {"mid": meeting_id},
            )
            .mappings()
            .all()
        )
    detail = {str(r["stage"]): (str(r["status"]), r["last_error"]) for r in stages}
    raise AssertionError(
        f"meeting {meeting_id} never reached {status!r}; last={last!r} error={error!r} stages={detail}"
    )
