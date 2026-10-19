"""Driver robustness: completion-only engine teardown + progress-stall guard.

Regression cover for a local-transcription backlog that wedged for ~14h with
20k+ threads: a stuck pipeline driver kept re-creating ONNX/torch sessions under
DBOS recovery and had no stall timeout, so one hung stage pinned its worker slot
and blocked the whole queue. These tests pin the two fixes:

* ``LocalTranscriptionPipeline.close()`` + ``_close_owned_pipeline`` releasing an
  owned pipeline only after the workflow COMPLETED (never on the forced/stall path,
  where a wedged native call could still be using the session — a use-after-free).
* ``_ProgressStallWatch`` + the ``run_staged`` wiring: a job merely waiting in the
  queue (``"queued"`` markers) must never be cancelled; only an active stage that
  goes silent past the timeout is.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from types import SimpleNamespace

import pytest

from olive.ml.pipeline import LocalTranscriptionPipeline
from olive.workflows import transcription_workflow as tw


class _SpyEngine:
    """Fake ML engine recording release() calls; release() takes clear_cache."""

    def __init__(self, *, raises: bool = False) -> None:
        self.release_calls = 0
        self._raises = raises

    def release(self, clear_cache: bool = False) -> None:
        self.release_calls += 1
        if self._raises:
            raise RuntimeError("release boom")


class _NoArgEngine:
    """Fake engine whose release() takes no args (like the ASR engine)."""

    def __init__(self) -> None:
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


def _spy_pipeline(*, asr_raises: bool = False) -> tuple[LocalTranscriptionPipeline, dict[str, _SpyEngine]]:
    engines = {
        "asr": _SpyEngine(raises=asr_raises),
        "aligner": _SpyEngine(),
        "diarizer": _SpyEngine(),
        "extractor": _SpyEngine(),
    }
    pipe = LocalTranscriptionPipeline(
        {},
        custom_diarizer=engines["diarizer"],
        custom_extractor=engines["extractor"],
        custom_asr_engine=engines["asr"],
        custom_aligner=engines["aligner"],
    )
    return pipe, engines


# --- Fix A: deterministic, completion-only teardown --------------------------------


def test_pipeline_close_releases_all_engines_and_is_idempotent() -> None:
    pipe, engines = _spy_pipeline()
    pipe.close()
    assert all(e.release_calls == 1 for e in engines.values())
    pipe.close()  # idempotent — must not raise
    assert all(e.release_calls == 2 for e in engines.values())


def test_pipeline_close_swallows_engine_errors() -> None:
    pipe, engines = _spy_pipeline(asr_raises=True)
    pipe.close()  # one engine's release raises; close() must still finish
    assert all(e.release_calls == 1 for e in engines.values())


def test_pipeline_close_handles_release_without_clear_cache_arg() -> None:
    noarg = _NoArgEngine()
    pipe = LocalTranscriptionPipeline(
        {},
        custom_diarizer=_SpyEngine(),
        custom_extractor=_SpyEngine(),
        custom_asr_engine=noarg,
        custom_aligner=_SpyEngine(),
    )
    pipe.close()
    assert noarg.release_calls == 1  # called exactly once, no double-release


def test_close_owned_pipeline_releases_engines() -> None:
    pipe, engines = _spy_pipeline()
    ctx = tw.TaskExecutionContext(task_id="rob-owned", owns_pipeline=True, pipeline=pipe)
    tw.register_task_context(ctx)
    try:
        tw._close_owned_pipeline("rob-owned")
        assert all(e.release_calls == 1 for e in engines.values())
        assert ctx.pipeline is None  # cleared so later stages can't reuse a closed pipe
    finally:
        tw.unregister_task_context("rob-owned")


def test_close_owned_pipeline_skips_caller_owned() -> None:
    pipe, engines = _spy_pipeline()
    ctx = tw.TaskExecutionContext(task_id="rob-unowned", owns_pipeline=False, pipeline=pipe)
    tw.register_task_context(ctx)
    try:
        tw._close_owned_pipeline("rob-unowned")
        assert all(e.release_calls == 0 for e in engines.values())
    finally:
        tw.unregister_task_context("rob-unowned")


def test_unregister_does_not_close_pipeline() -> None:
    # UAF-safety: unregister runs in a finally that fires on forced cancel too, so it
    # must NOT release sessions a wedged native call may still be inside.
    pipe, engines = _spy_pipeline()
    ctx = tw.TaskExecutionContext(task_id="rob-finally", owns_pipeline=True, pipeline=pipe)
    tw.register_task_context(ctx)
    tw.unregister_task_context("rob-finally")
    assert tw.get_task_context("rob-finally") is None
    assert all(e.release_calls == 0 for e in engines.values())


# --- Fix B: progress-stall watchdog ------------------------------------------------


async def test_stall_watch_trips_when_active_stage_goes_silent() -> None:
    w = tw._ProgressStallWatch(0.2)
    w.start()
    w.progress()  # a stage started, then goes silent
    await asyncio.sleep(0.6)
    assert w.event.is_set()
    w.cancel()


async def test_stall_watch_does_not_trip_while_only_waiting() -> None:
    # A job parked in the queue (only "queued" markers) must never be cancelled.
    w = tw._ProgressStallWatch(0.2)
    w.start()
    w.waiting()
    await asyncio.sleep(0.6)
    assert not w.event.is_set()
    w.cancel()


async def test_stall_watch_pauses_when_requeued_after_progress() -> None:
    # Progress, then back to a sub-queue wait → enforcement pauses, no false trip.
    w = tw._ProgressStallWatch(0.2)
    w.start()
    w.progress()
    w.waiting()
    await asyncio.sleep(0.6)
    assert not w.event.is_set()
    w.cancel()


async def test_stall_watch_quiet_with_steady_progress() -> None:
    w = tw._ProgressStallWatch(0.5)
    w.start()
    for _ in range(16):
        w.progress()
        await asyncio.sleep(0.05)
    assert not w.event.is_set()
    w.cancel()


def _patch_dbos(monkeypatch: pytest.MonkeyPatch, cancelled: dict[str, str]) -> None:
    async def _fake_cancel(wf_id: str) -> None:
        cancelled["wf"] = wf_id

    async def _noop_enqueue(queue: object, fn: object, *a: object, **k: object) -> object:  # pragma: no cover
        raise AssertionError("enqueue should be overridden per test")

    monkeypatch.setattr(
        tw, "DBOS", SimpleNamespace(enqueue_workflow_async=_noop_enqueue, cancel_workflow_async=_fake_cancel)
    )
    monkeypatch.setattr(tw, "SetWorkflowID", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(tw, "SetEnqueueOptions", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(tw, "is_shutting_down", lambda: False)


class _EmitThenHang:
    """Fake workflow handle: emit one progress event of `stage`, then never finish."""

    def __init__(self, stage: str) -> None:
        self.stage = stage

    async def get_result(self, polling_interval_sec: float = 0.05) -> dict[str, object]:
        await asyncio.sleep(0.02)  # let run_staged install the stamped progress hook
        for ctx in list(tw._TASK_CONTEXTS.values()):
            if callable(ctx.on_progress):
                res = ctx.on_progress({"stage": self.stage, "percent": 50})
                if inspect.isawaitable(res):
                    await res
        await asyncio.sleep(30)
        return {}


async def test_run_staged_cancels_on_active_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    cancelled: dict[str, str] = {}
    _patch_dbos(monkeypatch, cancelled)

    async def _enqueue(queue: object, fn: object, *a: object, **k: object) -> _EmitThenHang:
        return _EmitThenHang("transcribing")  # a real stage started, then hangs

    monkeypatch.setattr(tw.DBOS, "enqueue_workflow_async", _enqueue)

    with pytest.raises(RuntimeError, match="stalled"):
        await tw.run_staged_local_transcription(
            {"meetingId": "rob-active", "stallTimeoutSec": 0.3, "onProgress": lambda u: None},
        )
    assert cancelled.get("wf")  # wedged workflow cancelled → slot freed
    assert all(c.meeting_id != "rob-active" for c in tw._TASK_CONTEXTS.values())


async def test_run_staged_does_not_cancel_while_queued(monkeypatch: pytest.MonkeyPatch) -> None:
    # F1 regression: a job emitting only "queued" (waiting its turn) past the timeout
    # must NOT be stall-cancelled.
    cancelled: dict[str, str] = {}
    _patch_dbos(monkeypatch, cancelled)

    async def _enqueue(queue: object, fn: object, *a: object, **k: object) -> _EmitThenHang:
        return _EmitThenHang("queued")

    monkeypatch.setattr(tw.DBOS, "enqueue_workflow_async", _enqueue)

    task = asyncio.create_task(
        tw.run_staged_local_transcription(
            {"meetingId": "rob-queued", "stallTimeoutSec": 0.3, "onProgress": lambda u: None},
        )
    )
    await asyncio.sleep(0.6)  # well past the timeout
    assert not cancelled  # still queued → never cancelled
    assert not task.done()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# --- Primary fix: the local driver must not share the pipeline queue (deadlock) -----


def test_local_driver_queue_is_not_the_pipeline_queue() -> None:
    # The local-transcription DRIVER (run_transcription_driver) awaits its child
    # local_transcription_workflow, which runs on QUEUE_PIPELINE. If the driver also
    # ran on QUEUE_PIPELINE, two concurrent drivers would hold both of its 2 slots
    # while their children sat un-schedulable — the deadlock that stalled the backlog.
    from olive.services.transcription import PROVIDER_LOCAL, TRANSCRIBE_QUEUE_BY_PROVIDER
    from olive.workflows.queues import QUEUE_PIPELINE

    assert TRANSCRIBE_QUEUE_BY_PROVIDER[PROVIDER_LOCAL] != QUEUE_PIPELINE


def test_driver_queue_registered_and_outsizes_pipeline() -> None:
    from olive.workflows.queues import QUEUE_DRIVER, QUEUE_PIPELINE, STAGE_QUEUE_SPECS

    specs = {s.name: s.worker_concurrency for s in STAGE_QUEUE_SPECS}
    assert QUEUE_DRIVER in specs, "driver queue must be registered so DBOS creates it"
    # driver concurrency must exceed the pipeline's so the pipeline is never starved
    assert specs[QUEUE_DRIVER] > specs[QUEUE_PIPELINE]


# --- Primary fix: ML engines are built once and reused (bounded threads) ------------


def test_get_pipeline_builds_shared_pipeline_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # A fresh pipeline per meeting leaked onnx/torch threads that teardown never reclaimed;
    # the pipeline must be a process-wide singleton, kept loaded, reused across meetings.
    tw.reset_shared_pipeline()
    built: list[dict[str, object]] = []

    class _DummyPipeline:
        def __init__(self, opts: dict[str, object]) -> None:
            built.append(opts)
            self.unload_per_stage = bool(opts.get("unloadPerStage", True))

    monkeypatch.setattr(tw, "LocalTranscriptionPipeline", _DummyPipeline)
    c1 = tw.TaskExecutionContext(task_id="shared-1")
    c2 = tw.TaskExecutionContext(task_id="shared-2")
    tw.register_task_context(c1)
    tw.register_task_context(c2)
    try:
        p1 = tw._get_pipeline("shared-1", {"unloadPerStage": True})
        p2 = tw._get_pipeline("shared-2", {})
        assert p1 is p2  # same instance → not rebuilt per meeting
        assert len(built) == 1
        assert built[0]["unloadPerStage"] is False  # kept loaded so sessions are reused
    finally:
        tw.unregister_task_context("shared-1")
        tw.unregister_task_context("shared-2")
        tw.reset_shared_pipeline()


def test_dbos_config_caps_executor_threads() -> None:
    # DBOS defaults its workflow executor to sys.maxsize (unbounded). The transcription
    # driver blocks awaiting its child (which blocks awaiting stages) + frequent polling,
    # so a backlog spawned ~950 "dbos-executor" threads and OOM-crashed the container
    # (measured via py-spy). The config must cap the executor to a bounded value.
    from olive.workflows.runtime import _build_dbos_config

    cfg = _build_dbos_config("sqlite:///tmp/olive-test.sqlite")
    mx = cfg.get("max_executor_threads")
    assert isinstance(mx, int)
    assert 0 < mx <= 256  # bounded, not the unbounded default


def test_get_pipeline_honors_injected_pipeline() -> None:
    tw.reset_shared_pipeline()
    pipe, _ = _spy_pipeline()
    ctx = tw.TaskExecutionContext(task_id="shared-inj", pipeline=pipe)
    tw.register_task_context(ctx)
    try:
        assert tw._get_pipeline("shared-inj", {}) is pipe  # injection wins over singleton
    finally:
        tw.unregister_task_context("shared-inj")
        tw.reset_shared_pipeline()
