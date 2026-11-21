"""Durable Plaud PCS follow-up poll (named DBOS workflow on the ``plaud_sync`` queue).

When a newly discovered recording has no Plaud transcript/summary yet, the
poller waits ``plaudPcsWaitMinutes`` before importing it audio-only. The
APScheduler date job that fires at that deadline is scheduling glue only: it
calls :func:`enqueue_plaud_pcs_followup`, and the sync itself runs here as a
named workflow whose inputs are persisted by DBOS, so a restart between
enqueue and completion recovers it instead of losing it (unlike
``enqueue_driver_job``, whose callable lives in process-local ``_GENERIC_JOBS``).

The poll itself runs as a DBOS step, so its outcome is recorded once. Child
jobs the poll would queue (transcription hand-off, Plaud speaker
identification) can't be enqueued from inside a step, so the step returns
them and the workflow queues them afterwards, in the recorded order. A replay
after a restart then reuses the recorded step result and issues the same
child enqueues in the same order, which DBOS requires (it fails a replay
whose enqueues differ with ``DBOSUnexpectedStepError``).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

from dbos import DBOS, SetEnqueueOptions, SetWorkflowID
from dbos._error import DBOSQueueDeduplicatedError

from .queues import QUEUE_PLAUD_SYNC

#: At most one follow-up poll is queued or running at a time.
PCS_FOLLOWUP_DEDUP_ID = "plaud-pcs-followup"
#: How long a recovered workflow waits for the live poller to register at boot
#: (DBOS launches, and can resume queued work, before the poller starts).
POLLER_REGISTRATION_WAIT_SEC = 30.0


class PcsFollowupPoller(Protocol):
    async def run_pcs_followup_poll(self) -> dict[str, Any]: ...

    async def run_deferred_handoffs(self, handoffs: list[dict[str, str]]) -> int: ...


_PLAUD_POLLER: PcsFollowupPoller | None = None


def register_plaud_poller(poller: PcsFollowupPoller) -> None:
    """Make the live (scheduled) poller available to recovered follow-ups."""
    global _PLAUD_POLLER
    _PLAUD_POLLER = poller


def unregister_plaud_poller(poller: PcsFollowupPoller) -> None:
    global _PLAUD_POLLER
    if _PLAUD_POLLER is poller:
        _PLAUD_POLLER = None


def pcs_followup_workflow_id(expiry_ms: int) -> str:
    """Deterministic per deadline, so re-firing the same timer is a no-op."""
    return f"plaud-pcs-followup:{int(expiry_ms)}"


async def enqueue_plaud_pcs_followup(expiry_ms: int) -> str | None:
    """Enqueue the follow-up poll for ``expiry_ms``.

    Returns the workflow id, or ``None`` when another follow-up is already
    queued or running (DBOS deduplication on :data:`PCS_FOLLOWUP_DEDUP_ID`).
    """
    wf_id = pcs_followup_workflow_id(expiry_ms)
    try:
        with SetWorkflowID(wf_id), SetEnqueueOptions(deduplication_id=PCS_FOLLOWUP_DEDUP_ID):
            await DBOS.enqueue_workflow_async(QUEUE_PLAUD_SYNC, run_plaud_pcs_followup, int(expiry_ms))
    except DBOSQueueDeduplicatedError:
        return None
    return wf_id


async def _wait_for_registered_poller(timeout_sec: float) -> PcsFollowupPoller | None:
    deadline = time.monotonic() + max(0.0, timeout_sec)
    poller = _PLAUD_POLLER
    while poller is None and time.monotonic() < deadline:
        await asyncio.sleep(0.5)
        poller = _PLAUD_POLLER
    return poller


@DBOS.step()
async def step_plaud_pcs_followup_poll() -> dict[str, Any]:
    """One poll cycle; returns its result plus the child jobs it deferred."""
    poller = _PLAUD_POLLER
    if poller is None:
        return {"skipped": True, "handoffs": []}
    return await poller.run_pcs_followup_poll()


@DBOS.workflow()
async def run_plaud_pcs_followup(expiry_ms: int) -> dict[str, Any]:
    """Run one Plaud sync for recordings whose PCS wait expired at ``expiry_ms``."""
    poller = await _wait_for_registered_poller(POLLER_REGISTRATION_WAIT_SEC)
    if poller is None:
        # The regular interval poll (and the boot poll) still covers it.
        return {"status": "skipped", "reason": "Plaud poller is not running", "expiryMs": int(expiry_ms)}
    result = await step_plaud_pcs_followup_poll()
    if result.get("skipped"):
        return {"status": "skipped", "reason": "Plaud poller is not running", "expiryMs": int(expiry_ms)}
    handoffs = list(result.get("handoffs") or [])
    queued = await poller.run_deferred_handoffs(handoffs) if handoffs else 0
    return {
        "status": "done",
        "expiryMs": int(expiry_ms),
        "discovered": int(result.get("discovered") or 0),
        "resolved": int(result.get("resolved") or 0),
        "pcsPending": int(result.get("pcsPending") or 0),
        "error": result.get("error"),
        "handoffs": len(handoffs),
        "handoffsQueued": queued,
    }


__all__ = [
    "PCS_FOLLOWUP_DEDUP_ID",
    "POLLER_REGISTRATION_WAIT_SEC",
    "PcsFollowupPoller",
    "enqueue_plaud_pcs_followup",
    "pcs_followup_workflow_id",
    "register_plaud_poller",
    "run_plaud_pcs_followup",
    "step_plaud_pcs_followup_poll",
    "unregister_plaud_poller",
]
