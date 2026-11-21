"""PCS follow-up poll runs as a durable, deduplicated DBOS workflow on ``plaud_sync`` (#34).

The APScheduler date job is scheduling glue only; the sync itself must be the
named workflow ``run_plaud_pcs_followup`` so it is bounded by the queue,
deduplicated, and recovered after a restart.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dbos import DBOS
from sqlalchemy import text

import olive.plaud.poller as poller_module
import olive.workflows.plaud_workflow as pw
import olive.workflows.runtime as rt
from olive.db import create_db
from olive.plaud import PlaudPoller
from olive.providers.assemblyai import AssemblyAIClient
from olive.services.transcription import TranscriptionService
from olive.workflows import QUEUE_PLAUD_SYNC, QUEUE_SUMMARIZE, enqueue_stage_job
from test_assemblyai_transcription import RAW_TRANSCRIPT
from test_plaud import (
    _fake_fetch,
    _FakePlaudClient,
    _FakeResponse,
    _pending_detail,
    _ready_detail,
)


class _Env:
    def __init__(self, tmp_path: Path) -> None:
        self.config_dir = str(tmp_path / "config")
        self.meetings_dir = str(tmp_path / "meetings")
        Path(self.config_dir).mkdir(parents=True, exist_ok=True)
        Path(self.meetings_dir).mkdir(parents=True, exist_ok=True)

    async def launch(self) -> None:
        await rt.ensure_dbos_launched_async(config_dir=self.config_dir, meetings_dir=self.meetings_dir)

    def _rows(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        with sqlite3.connect(str(Path(self.config_dir) / "dbos.sqlite"), timeout=1.0) as sconn:
            sconn.row_factory = sqlite3.Row
            return [dict(r) for r in sconn.execute(sql, params).fetchall()]

    def workflow_row(self, workflow_id: str) -> dict[str, Any] | None:
        rows = self._rows(
            "SELECT workflow_uuid, status, queue_name, name FROM workflow_status WHERE workflow_uuid = ?",
            (workflow_id,),
        )
        return rows[0] if rows else None

    def children_of(self, workflow_id: str) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT workflow_uuid, queue_name FROM workflow_status WHERE parent_workflow_id = ?",
            (workflow_id,),
        )

    async def wait_status(self, workflow_id: str, statuses: tuple[str, ...], timeout: float = 20.0) -> str:
        deadline = time.monotonic() + timeout
        last: str | None = None
        while time.monotonic() < deadline:
            row = self.workflow_row(workflow_id)
            last = row["status"] if row else None
            if last in statuses:
                return last
            await asyncio.sleep(0.05)
        raise AssertionError(f"{workflow_id} never reached {statuses}; last={last}")

    async def result_of(self, workflow_id: str) -> dict[str, Any]:
        assert await self.wait_status(workflow_id, ("SUCCESS", "ERROR")) == "SUCCESS"
        return await (await DBOS.retrieve_workflow_async(workflow_id)).get_result()


@pytest.fixture
async def env(tmp_path: Path) -> AsyncIterator[_Env]:
    """A real DBOS runtime on a tmp_path database."""
    e = _Env(tmp_path)
    await e.launch()
    try:
        yield e
    finally:
        pw._PLAUD_POLLER = None
        await asyncio.to_thread(rt.shutdown_dbos)


class _BlockingPoller:
    """Stand-in poller whose follow-up poll blocks until released."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.calls = 0

    async def run_pcs_followup_poll(self) -> dict[str, Any]:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return {"discovered": 0, "resolved": 1, "pcsPending": 0, "error": None, "handoffs": []}

    async def run_deferred_handoffs(self, handoffs: list[dict[str, str]]) -> int:
        return len(handoffs)


class _FakeTranscription:
    """Records hand-offs; optionally enqueues a real DBOS job and/or blocks on the first call."""

    def __init__(self, *, enqueue_job: bool = False, block_first: bool = False) -> None:
        self.calls: list[str] = []
        self.errors: list[str] = []
        self.enqueue_job = enqueue_job
        self.block_first = block_first
        self.first_started = asyncio.Event()
        self.hold = asyncio.Event()

    async def start_transcription(self, meeting_id: str) -> dict[str, Any]:
        self.calls.append(meeting_id)
        if self.enqueue_job:

            async def _leaf() -> str:
                return meeting_id

            try:
                await enqueue_stage_job(
                    QUEUE_SUMMARIZE,
                    _leaf,
                    deduplication_id=f"test-transcribe:{meeting_id}",
                    label="test-transcribe",
                    meeting_id=meeting_id,
                )
            except BaseException as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
                raise
        if self.block_first and len(self.calls) == 1:
            self.first_started.set()
            await self.hold.wait()
        return {"status": "running"}


def _make_poller(
    env: _Env,
    clock: dict[str, int],
    transcription: Any = None,
    *,
    detail: dict[str, Any] | None = None,
    pcs_wait_ms: int = 300_000,
) -> tuple[Any, PlaudPoller, _FakePlaudClient]:
    handle = create_db(":memory:")
    client = _FakePlaudClient(detail or _pending_detail())
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": env.meetings_dir,
            "configDir": env.config_dir,
            "client": client,
            "fetchImpl": _fake_fetch(),
            "now": lambda: clock["now"],
            "pcsWaitMs": pcs_wait_ms,
            "transcriptionService": transcription,
        }
    )
    return handle, poller, client


def _only_meeting_id(handle: Any) -> str:
    with handle.engine.connect() as conn:
        return str(conn.execute(text("SELECT id FROM meetings")).scalar_one())


async def test_timer_enqueues_named_workflow_on_plaud_sync_and_it_ingests(env: _Env) -> None:
    clock = {"now": 1_000_000}
    handle, poller, client = _make_poller(env, clock)
    pw.register_plaud_poller(poller)

    first = await poller.trigger()
    assert first["pcsPending"] == 1
    expiry = 1_000_000 + 300_000

    clock["now"] = expiry + 2_000
    workflow_id = await poller._enqueue_pcs_followup(expiry)
    assert workflow_id is not None
    assert workflow_id == pw.pcs_followup_workflow_id(expiry)

    result = await env.result_of(workflow_id)
    assert result["status"] == "done"
    assert result["resolved"] == 1
    assert result["pcsPending"] == 0

    row = env.workflow_row(workflow_id)
    assert row is not None
    assert row["queue_name"] == QUEUE_PLAUD_SYNC
    assert row["name"].endswith("run_plaud_pcs_followup")
    assert row["status"] == "SUCCESS"

    # Re-firing the same deadline is a no-op (deterministic workflow id).
    assert await poller._enqueue_pcs_followup(expiry) == workflow_id
    await asyncio.sleep(0.3)
    assert client.get_file_calls == 2  # discovery-pass check + the follow-up only
    handle.engine.dispose()


async def test_followup_is_deduplicated_while_one_is_queued_or_running(env: _Env) -> None:
    clock = {"now": 1_000_000}
    handle, poller, _client = _make_poller(env, clock)
    blocking = _BlockingPoller()
    pw.register_plaud_poller(blocking)

    first_id = await poller._enqueue_pcs_followup(2_000_000)
    assert first_id is not None
    await asyncio.wait_for(blocking.started.wait(), 20)

    # A second deadline while the first follow-up is running is deduplicated.
    assert await poller._enqueue_pcs_followup(2_100_000) is None
    assert env.workflow_row(pw.pcs_followup_workflow_id(2_100_000)) is None

    blocking.release.set()
    await env.result_of(first_id)
    # Once it has finished, a new deadline enqueues normally.
    second_id = await poller._enqueue_pcs_followup(2_200_000)
    assert second_id == pw.pcs_followup_workflow_id(2_200_000)
    await env.result_of(second_id)
    assert blocking.calls == 2
    handle.engine.dispose()


async def test_followup_is_recovered_after_restart(env: _Env) -> None:
    stuck = _BlockingPoller()  # never released: the process "dies" mid-poll
    pw.register_plaud_poller(stuck)
    workflow_id = await pw.enqueue_plaud_pcs_followup(3_000_000)
    assert workflow_id is not None
    await asyncio.wait_for(stuck.started.wait(), 20)

    await asyncio.to_thread(rt.shutdown_dbos)
    pw._PLAUD_POLLER = None
    healthy = _BlockingPoller()
    healthy.release.set()
    await env.launch()
    # Recovery may resume before the poller registers at boot; it must wait.
    await asyncio.sleep(0.3)
    pw.register_plaud_poller(healthy)

    result = await env.result_of(workflow_id)
    assert result["status"] == "done"
    assert healthy.calls == 1


async def test_followup_skips_when_no_poller_registers(env: _Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pw, "POLLER_REGISTRATION_WAIT_SEC", 0.2)
    workflow_id = await pw.enqueue_plaud_pcs_followup(4_000_000)
    assert workflow_id is not None
    result = await env.result_of(workflow_id)
    assert result["status"] == "skipped"


async def test_enqueue_failure_leaves_deadline_to_interval_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"now": 1_000_000}
    handle, poller, client = _make_poller(_Env(tmp_path), clock)

    async def _boom(*_a: Any, **_k: Any) -> str:
        raise RuntimeError("dbos down")

    monkeypatch.setattr("olive.workflows.ensure_dbos_launched_async", _boom)
    assert await poller._enqueue_pcs_followup(5_000_000) is None
    assert client.get_file_calls == 0  # nothing ran outside the queue
    handle.engine.dispose()


async def test_interval_trigger_joins_followup_poll_in_flight(tmp_path: Path) -> None:
    clock = {"now": 1_000_000}
    handle, poller, client = _make_poller(_Env(tmp_path), clock)
    gate = asyncio.Event()
    original = client.getFile

    async def _slow_get_file(fid: str) -> dict[str, Any]:
        await gate.wait()
        return await original(fid)

    client.getFile = _slow_get_file  # type: ignore[method-assign]

    followup = asyncio.ensure_future(poller.run_pcs_followup_poll())
    await asyncio.sleep(0.05)
    assert poller._in_flight is not None and not poller._in_flight.done()
    assert poller.trigger_background() is False
    joined = asyncio.ensure_future(poller.trigger())
    await asyncio.sleep(0.05)
    gate.set()
    res_followup, res_joined = await asyncio.gather(followup, joined)
    assert res_followup["completedAt"] == res_joined["completedAt"]  # one shared cycle
    assert client.get_file_calls == 1
    assert poller._in_flight is None
    handle.engine.dispose()


async def test_followup_poll_returns_transcription_handoff_instead_of_queueing(tmp_path: Path) -> None:
    clock = {"now": 1_000_000}
    transcription = _FakeTranscription()
    handle, poller, _client = _make_poller(_Env(tmp_path), clock, transcription)
    await poller.trigger()  # discovery; still pending
    mid = _only_meeting_id(handle)

    clock["now"] += 302_000
    result = await poller.run_pcs_followup_poll()
    assert result["resolved"] == 1
    assert result["handoffs"] == [
        {"kind": "transcribe", "meetingId": mid, "context": "audio-only Plaud meeting"}
    ]
    await asyncio.sleep(0.05)
    assert transcription.calls == []  # nothing was queued from inside the poll

    assert await poller.run_deferred_handoffs(result["handoffs"]) == 1
    assert transcription.calls == [mid]
    handle.engine.dispose()


async def test_followup_poll_returns_identify_handoff_instead_of_queueing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recording whose Plaud transcript is ready defers speaker identification too."""
    handle, poller, _client = _make_poller(_Env(tmp_path), {"now": 1_000_000}, detail=_ready_detail())
    queued: list[str] = []

    async def _record_enqueue(meeting_id: str) -> bool:
        queued.append(meeting_id)
        return True

    monkeypatch.setattr(poller, "_enqueue_plaud_speaker_identification", _record_enqueue)

    result = await poller.run_pcs_followup_poll()
    mid = _only_meeting_id(handle)
    assert result["resolved"] == 1
    assert result["handoffs"] == [{"kind": "identify", "meetingId": mid}]
    assert queued == []

    assert await poller.run_deferred_handoffs(result["handoffs"]) == 1
    assert queued == [mid]
    handle.engine.dispose()


async def test_interval_poll_still_hands_off_inline(tmp_path: Path) -> None:
    """Deferral is opt-in per poll: interval and manual polls queue hand-offs as before."""
    clock = {"now": 1_000_000}
    transcription = _FakeTranscription()
    handle, poller, _client = _make_poller(_Env(tmp_path), clock, transcription)
    await poller.trigger()
    mid = _only_meeting_id(handle)

    clock["now"] += 302_000
    result = await poller.trigger()
    assert result["resolved"] == 1
    assert "handoffs" not in result
    await asyncio.sleep(0.05)
    assert transcription.calls == [mid]
    handle.engine.dispose()


async def test_workflow_queues_deferred_handoff_as_its_dbos_child(env: _Env) -> None:
    clock = {"now": 1_000_000}
    transcription = _FakeTranscription(enqueue_job=True)
    handle, poller, _client = _make_poller(env, clock, transcription)
    pw.register_plaud_poller(poller)
    await poller.trigger()
    mid = _only_meeting_id(handle)

    clock["now"] += 302_000
    workflow_id = await poller._enqueue_pcs_followup(1_300_000)
    assert workflow_id is not None
    result = await env.result_of(workflow_id)
    assert result["status"] == "done"
    assert result["handoffs"] == 1
    assert result["handoffsQueued"] == 1
    assert transcription.calls == [mid]
    assert transcription.errors == []
    assert [c["queue_name"] for c in env.children_of(workflow_id)] == [QUEUE_SUMMARIZE]
    handle.engine.dispose()


async def test_replay_after_restart_mid_handoff_reuses_recorded_poll(env: _Env) -> None:
    """Crash after the poll step completed but while hand-offs are being queued.

    The replay must reuse the recorded poll result instead of polling again
    (a second poll would no longer see the meeting as pending, so its hand-off
    would be lost), issue the same hand-off again, and finish.
    """
    clock = {"now": 1_000_000}
    transcription = _FakeTranscription(enqueue_job=True, block_first=True)
    handle, poller, client = _make_poller(env, clock, transcription)
    pw.register_plaud_poller(poller)
    await poller.trigger()
    mid = _only_meeting_id(handle)

    clock["now"] += 302_000
    workflow_id = await poller._enqueue_pcs_followup(1_300_000)
    assert workflow_id is not None
    await asyncio.wait_for(transcription.first_started.wait(), 20)
    polls_before = client.get_file_calls

    await asyncio.to_thread(rt.shutdown_dbos)  # simulated crash mid hand-off
    await env.launch()

    result = await env.result_of(workflow_id)
    assert result["status"] == "done"
    assert result["handoffsQueued"] == 1
    assert client.get_file_calls == polls_before  # the recorded poll step was reused
    assert transcription.calls == [mid, mid]  # the hand-off was issued again on replay
    assert transcription.errors == []
    assert len(env.children_of(workflow_id)) == 1  # ...and resolved to the same child job
    handle.engine.dispose()


async def test_chained_followups_with_real_scheduler_do_not_leak_job_context(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end with a real AsyncIOScheduler and a real DBOS runtime.

    Two recordings wait with staggered deadlines, so the first follow-up
    re-arms the timer from inside its DBOS step. asyncio timers capture the
    context they are created in; if that step's context leaked into the
    scheduler's timer chain, the second follow-up could not be enqueued and
    later scheduler jobs (interval polls) could not hand off.
    """
    monkeypatch.setattr(poller_module, "PCS_FOLLOWUP_SLACK_MS", 0)
    monkeypatch.setattr(poller_module, "PCS_FOLLOWUP_MIN_DELAY_MS", 0)
    clock = {"now": 1_000_000}
    transcription = _FakeTranscription(enqueue_job=True)
    handle, poller, _client = _make_poller(env, clock, transcription, pcs_wait_ms=300)
    downloads = 0

    async def _distinct_audio(_url: str) -> _FakeResponse:
        nonlocal downloads
        downloads += 1
        return _FakeResponse(content=f"audio-{downloads}".encode())  # avoid sha256 de-duplication

    poller.fetch_impl = _distinct_audio
    pw.register_plaud_poller(poller)

    real_enqueue = poller._enqueue_pcs_followup

    async def _enqueue_once_deadline_passed(expiry_ms: int) -> str | None:
        # In production the timer fires on the wall clock just after the deadline.
        clock["now"] = max(clock["now"], expiry_ms + 1)
        return await real_enqueue(expiry_ms)

    monkeypatch.setattr(poller, "_enqueue_pcs_followup", _enqueue_once_deadline_passed)

    late_job_ran = asyncio.Event()

    async def _later_scheduler_job() -> None:
        # What an interval poll does after ingesting a meeting.
        poller._hand_off_transcription("meeting-from-interval-poll", "test")
        late_job_ran.set()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(lambda: None, "interval", minutes=15, id="plaud_poll")
    scheduler.add_job(_later_scheduler_job, "date", run_date=datetime.now(UTC) + timedelta(seconds=1.5))
    scheduler.start()
    poller._scheduler = scheduler
    try:
        await poller.trigger()  # discovers recording A; its wait expires at 1_000_300
        mid_a = _only_meeting_id(handle)
        # Recording B was first seen 200 ms later, so it is still waiting when A's follow-up runs.
        with handle.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, "
                    "created_at, updated_at) VALUES ('m-b', 'B', 1, 2, 'plaud', 'processing', '[]', 1, 1)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO plaud_ingest_state (meeting_id, plaud_file_id, first_seen_at, "
                    "pcs_deadline_at, pcs_resolved) VALUES ('m-b', 'plaud-file-b', 1000200, 1000500, 0)"
                )
            )

        first = await env.wait_status(pw.pcs_followup_workflow_id(1_000_300), ("SUCCESS", "ERROR"))
        # B's timer is armed from inside the first follow-up's step.
        second = await env.wait_status(pw.pcs_followup_workflow_id(1_000_500), ("SUCCESS", "ERROR"))
        assert (first, second) == ("SUCCESS", "SUCCESS")

        await asyncio.wait_for(late_job_ran.wait(), 20)
        deadline = time.monotonic() + 20
        while len(transcription.calls) < 3 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert sorted(transcription.calls) == sorted([mid_a, "m-b", "meeting-from-interval-poll"])
        assert transcription.errors == []
    finally:
        scheduler.shutdown(wait=False)
        handle.engine.dispose()


async def test_followup_hands_off_to_real_transcription_service(env: _Env) -> None:
    """The deferred hand-off works with the real TranscriptionService (cloud provider path).

    ``start_transcription`` runs in the follow-up workflow's context and
    enqueues the provider job as its child; that job then transcribes and
    persists as usual.
    """

    class _InstantTranscriber:
        async def submit(self, _audio: Any, _config: dict[str, Any]) -> dict[str, Any]:
            return {"id": "aai-job-123", "status": "queued"}

        async def get_transcript(self, _job_id: str) -> dict[str, Any]:
            return RAW_TRANSCRIPT

    clock = {"now": 1_000_000}
    handle = create_db(":memory:")
    service = TranscriptionService(
        {
            "db": handle.engine,
            "meetingsDir": env.meetings_dir,
            "configDir": env.config_dir,
            "assemblyaiClient": AssemblyAIClient(
                {"apiKey": "test-key", "transcriber": _InstantTranscriber()}
            ),
            "defaultProvider": "assemblyai",
            "now": lambda: clock["now"],
        }
    )
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": env.meetings_dir,
            "configDir": env.config_dir,
            "client": _FakePlaudClient(_pending_detail()),
            "fetchImpl": _fake_fetch(),
            "now": lambda: clock["now"],
            "pcsWaitMs": 300_000,
            "transcriptionService": service,
        }
    )
    pw.register_plaud_poller(poller)
    await poller.trigger()
    mid = _only_meeting_id(handle)

    clock["now"] += 302_000
    workflow_id = await poller._enqueue_pcs_followup(1_300_000)
    assert workflow_id is not None
    result = await env.result_of(workflow_id)
    assert (result["handoffs"], result["handoffsQueued"]) == (1, 1)

    children = env.children_of(workflow_id)
    assert len(children) == 1
    assert children[0]["workflow_uuid"].startswith(f"transcribe:{mid}:")
    assert children[0]["queue_name"] == "assemblyai"
    assert await env.wait_status(children[0]["workflow_uuid"], ("SUCCESS", "ERROR")) == "SUCCESS"
    with handle.engine.connect() as conn:
        meeting = conn.execute(text("SELECT * FROM meetings WHERE id = :id"), {"id": mid}).mappings().one()
    assert meeting["primary_transcript_artifact_id"] is not None
    handle.engine.dispose()
