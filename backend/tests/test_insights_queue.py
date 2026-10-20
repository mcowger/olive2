"""Olive Insights Phase 1b: DBOS ``insights`` queue wiring, auto-enqueue on finalize, and backfill."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

# Import order matters: services first avoids the workflows<->ml cycle.
import olive.services.transcription  # noqa: F401
from olive import workflows
from olive.db import create_db
from olive.layout import meeting_paths
from olive.providers.assemblyai import AssemblyAIClient
from olive.services.insights import extract as insights_extract
from olive.services.insights.backfill import backfill
from olive.services.insights.prompts import compute_prompt_version, load_insights_prompts
from olive.services.transcription import TranscriptionService
from olive.workflows import queues
from olive.workflows import transcription_workflow as wf
from test_assemblyai_transcription import _FakeTranscriber, _seed_meeting  # noqa: E402  (shared fixtures)


@pytest.fixture(autouse=True)
def _reset_hook() -> Iterator[None]:
    insights_extract.set_enqueue_hook(None)
    yield
    insights_extract.set_enqueue_hook(None)


# -- queue registration -------------------------------------------------------


def test_insights_queue_registered_with_single_worker() -> None:
    assert queues.QUEUE_INSIGHTS == "insights"
    specs = {s.name: s for s in queues.STAGE_QUEUE_SPECS}
    assert specs["insights"].worker_concurrency == 1
    assert "QUEUE_INSIGHTS" in queues.__all__
    assert workflows.QUEUE_INSIGHTS == "insights"
    assert wf.QUEUE_INSIGHTS == "insights"


# -- hook -> enqueue_stage_job ------------------------------------------------


class _StubInsights:
    def __init__(self) -> None:
        self.analyzed: list[str] = []
        self.forced: list[bool] = []

    async def analyze_meeting(self, meeting_id: str, *, force: bool = False) -> dict[str, Any]:
        self.analyzed.append(meeting_id)
        self.forced.append(force)
        return {"status": "done"}


class _FakeDbosEnqueue:
    """Stands in for ``DBOS.enqueue_workflow_async`` and records what would be queued."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return object()


@pytest.fixture
def dbos_calls(monkeypatch: pytest.MonkeyPatch) -> _FakeDbosEnqueue:
    fake = _FakeDbosEnqueue()
    monkeypatch.setattr(wf.DBOS, "enqueue_workflow_async", fake)

    async def launched(**_kw: Any) -> str:
        return "sqlite://"

    monkeypatch.setattr(wf, "ensure_dbos_launched_async", launched)
    return fake


async def test_hook_enqueues_registered_function_with_meeting_id_and_force(
    dbos_calls: _FakeDbosEnqueue,
) -> None:
    wf.configure_insights_queue(_StubInsights())
    hook = insights_extract.get_enqueue_hook()
    assert hook is not None
    assert await hook("m-1") is True
    assert await hook("m-2", force=True) is True

    (first_args, _), (second_args, _) = dbos_calls.calls
    # DBOS recovers a workflow by its registered function name, so a closure/lambda would be lost on
    # restart: the queued callable is a module-level function and the payload is plain data.
    assert first_args[0] == "insights"
    assert first_args[1] is wf.step_analyze_insights_meeting
    assert first_args[1].__name__ != "<lambda>"
    assert first_args[2:] == ("m-1", False)
    assert second_args[2:] == ("m-2", True)


async def test_hook_uses_dedup_and_workflow_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str | None, str | None]] = []

    class _Ctx:
        def __init__(self, tag: str, value: str | None) -> None:
            self.tag, self.value = tag, value

        def __enter__(self) -> None:
            seen.append((self.tag, self.value))

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(wf, "SetWorkflowID", lambda wid: _Ctx("wid", wid))
    monkeypatch.setattr(
        wf, "SetEnqueueOptions", lambda deduplication_id=None: _Ctx("dedup", deduplication_id)
    )
    monkeypatch.setattr(wf.DBOS, "enqueue_workflow_async", _FakeDbosEnqueue())

    async def launched(**_kw: Any) -> str:
        return "sqlite://"

    monkeypatch.setattr(wf, "ensure_dbos_launched_async", launched)
    wf.configure_insights_queue(_StubInsights())
    hook = insights_extract.get_enqueue_hook()
    assert hook is not None
    assert await hook("m-1") is True

    values = dict(seen)
    assert values["dedup"] == "insights:m-1"
    assert values["wid"] is not None and values["wid"].startswith("insights:m-1:")
    assert len(values["wid"].split(":")[2]) == 8


async def test_recovered_job_rebuilds_service_and_forwards_force() -> None:
    stub = _StubInsights()
    wf.configure_insights_queue(stub)
    res = await wf.step_analyze_insights_meeting("m-9", True)
    assert res == {"status": "done"}
    assert stub.analyzed == ["m-9"]
    assert stub.forced == [True]

    # After a restart no live service is registered: the factory rebuilds it.
    wf._INSIGHTS_SERVICE = None
    rebuilt = _StubInsights()
    wf.configure_insights_service_factory(lambda: rebuilt)
    try:
        await wf.step_analyze_insights_meeting("m-10")
    finally:
        wf.configure_insights_service_factory(None)
    assert rebuilt.analyzed == ["m-10"] and rebuilt.forced == [False]


async def test_enqueue_launches_dbos_first(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []

    async def launched(**_kw: Any) -> str:
        order.append("launch")
        return "sqlite://"

    async def enqueue(*_a: Any, **_k: Any) -> Any:
        order.append("enqueue")
        return object()

    monkeypatch.setattr(wf, "ensure_dbos_launched_async", launched)
    monkeypatch.setattr(wf.DBOS, "enqueue_workflow_async", enqueue)
    wf.configure_insights_queue(_StubInsights())
    hook = insights_extract.get_enqueue_hook()
    assert hook is not None
    assert await hook("m-1") is True
    assert order == ["launch", "enqueue"]


@pytest.mark.parametrize("failing", ["launch", "enqueue"])
async def test_enqueue_failures_are_handled_not_raised(monkeypatch: pytest.MonkeyPatch, failing: str) -> None:
    async def launched(**_kw: Any) -> str:
        if failing == "launch":
            raise RuntimeError("DBOS failed to launch")
        return "sqlite://"

    async def enqueue(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("No DBOS was created yet")

    monkeypatch.setattr(wf, "ensure_dbos_launched_async", launched)
    monkeypatch.setattr(wf.DBOS, "enqueue_workflow_async", enqueue)
    wf.configure_insights_queue(_StubInsights())
    hook = insights_extract.get_enqueue_hook()
    assert hook is not None
    assert await hook("m-1") is False  # reported, never an unhandled exception


async def test_enqueue_analyze_threads_force_to_the_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olive.services.insights.extract import InsightsService

    monkeypatch.setenv("OLIVE_INSIGHTS_ENABLED", "1")
    got: list[tuple[str, bool]] = []

    async def hook(meeting_id: str, force: bool = False) -> bool:
        got.append((meeting_id, force))
        return True

    insights_extract.set_enqueue_hook(hook)
    handle = create_db(":memory:")
    svc = InsightsService(
        db=handle.engine, meetings_dir=str(tmp_path), config_dir=str(tmp_path), llm_service=None
    )
    assert await svc.enqueue_analyze("a") is True
    assert await svc.enqueue_analyze("b", force=True) is True
    assert got == [("a", False), ("b", True)]
    handle.engine.dispose()


# -- auto-enqueue on finalize -------------------------------------------------


class _RecordingInsights:
    def __init__(self, *, fail: bool = False) -> None:
        self.enqueued: list[str] = []
        self.fail = fail

    async def enqueue_analyze(self, meeting_id: str) -> bool:
        if self.fail:
            raise RuntimeError("boom")
        self.enqueued.append(meeting_id)
        return True


def _tx_service(handle: Any, meetings_dir: str, **extra: Any) -> TranscriptionService:
    client = AssemblyAIClient({"apiKey": "test-key", "transcriber": _FakeTranscriber()})
    return TranscriptionService(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "assemblyaiClient": client,
            "now": lambda: 1_700_000_001_000,
            **extra,
        }
    )


async def test_cloud_finalize_enqueues_insights(tmp_path: Path) -> None:
    handle, meeting_id, _ = _seed_meeting(tmp_path)
    svc = _RecordingInsights()
    service = _tx_service(handle, str(tmp_path / "meetings"))
    service.set_insights_service(svc)

    result = await service.transcribe_meeting(meeting_id, {"provider": "assemblyai", "pollIntervalMs": 10})
    assert result["status"] == "done"
    await asyncio.sleep(0.05)  # let the fire-and-forget enqueue task run
    assert svc.enqueued == [meeting_id]
    handle.engine.dispose()


async def test_enqueue_insights_noop_without_service() -> None:
    service = TranscriptionService({"db": create_db(":memory:").engine, "meetingsDir": "/nonexistent"})
    assert service.insights_service is None
    await service._enqueue_insights("m-1")  # must not raise


async def test_enqueue_insights_swallows_service_errors() -> None:
    service = TranscriptionService({"db": create_db(":memory:").engine, "meetingsDir": "/nonexistent"})
    service.set_insights_service(_RecordingInsights(fail=True))
    await service._enqueue_insights("m-1")  # must not raise


async def test_disabled_insights_service_queues_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olive.services.insights.extract import InsightsService

    hooked: list[str] = []
    insights_extract.set_enqueue_hook(lambda mid: hooked.append(mid) or True)
    monkeypatch.setenv("OLIVE_INSIGHTS_ENABLED", "0")
    handle = create_db(":memory:")
    insights = InsightsService(
        db=handle.engine, meetings_dir=str(tmp_path), config_dir=str(tmp_path), llm_service=None
    )
    service = TranscriptionService({"db": handle.engine, "meetingsDir": str(tmp_path)})
    service.set_insights_service(insights)
    await service._enqueue_insights("m-1")
    assert hooked == []
    handle.engine.dispose()


def test_both_finalize_sites_schedule_insights_next_to_auto_summarize() -> None:
    src = inspect.getsource(TranscriptionService)
    sites = src.count("asyncio.create_task(self._enqueue_auto_summarize(")
    assert sites == 2
    assert src.count("asyncio.create_task(self._enqueue_insights(") == sites


# -- backfill -----------------------------------------------------------------


def _seed_transcribed(
    handle: Any, meetings_dir: str, meeting_id: str, start: int, *, transcript: bool = True
) -> str:
    paths = meeting_paths(meetings_dir, start, "Sync", meeting_id)
    Path(paths.transcripts_dir).mkdir(parents=True, exist_ok=True)
    raw = json.dumps({"segments": [{"startMs": 0, "endMs": 10, "speaker": "A", "text": meeting_id}]})
    (Path(paths.folder) / "transcripts/t.json").write_text(raw, encoding="utf-8")
    aid = f"art-{meeting_id}" if transcript else None
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, "
                "primary_transcript_artifact_id, primary_summary_artifact_id, last_error, created_at, updated_at) "
                "VALUES (:id, 'Sync', :s, :e, 'upload', 'ready', '[]', NULL, NULL, NULL, :s, :s)"
            ),
            {"id": meeting_id, "s": start, "e": start + 1000},
        )
        if transcript:
            conn.execute(
                text(
                    "INSERT INTO artifacts (id, meeting_id, recording_id, kind, provider, format, path, created_at) "
                    "VALUES (:a, :m, NULL, 'transcript', 'local', 'json', 'transcripts/t.json', :s)"
                ),
                {"a": aid, "m": meeting_id, "s": start},
            )
            conn.execute(
                text("UPDATE meetings SET primary_transcript_artifact_id = :a WHERE id = :m"),
                {"a": aid, "m": meeting_id},
            )
    return raw


def _mark_analyzed(handle: Any, config_dir: str, meeting_id: str, raw: str) -> None:
    pv = compute_prompt_version(load_insights_prompts(config_dir))
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO meeting_speaker_signals (id, meeting_id, speaker_label, dimensions, talk_ms, "
                "turn_count, interrupt_count, transcript_hash, prompt_version, created_at) "
                "VALUES (:id, :m, 'A', '{}', 1, 1, 0, :h, :pv, 0)"
            ),
            {
                "id": f"sig-{meeting_id}",
                "m": meeting_id,
                "h": hashlib.sha256(raw.encode()).hexdigest(),
                "pv": pv,
            },
        )


class _Queued(list[str]):
    """Meeting ids the backfill hook received, plus the ``force`` flag it got for each."""

    def __init__(self) -> None:
        super().__init__()
        self.forced: list[bool] = []


@pytest.fixture
def backfill_env(tmp_path: Path) -> tuple[Any, str, str, _Queued]:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    config_dir = str(tmp_path / "config")
    Path(meetings_dir).mkdir()
    Path(config_dir).mkdir()
    queued = _Queued()

    async def hook(meeting_id: str, force: bool = False) -> bool:
        queued.append(meeting_id)
        queued.forced.append(force)
        return True

    insights_extract.set_enqueue_hook(hook)
    return handle, meetings_dir, config_dir, queued


async def test_backfill_queues_transcribed_meetings_only(backfill_env: Any) -> None:
    handle, meetings_dir, config_dir, queued = backfill_env
    _seed_transcribed(handle, meetings_dir, "m-a", 1_000)
    _seed_transcribed(handle, meetings_dir, "m-b", 2_000)
    _seed_transcribed(handle, meetings_dir, "m-none", 3_000, transcript=False)
    res = await backfill(handle.engine, meetings_dir, config_dir)
    assert res == {"queued": 2, "skipped": 0}
    assert queued == ["m-a", "m-b"]
    handle.engine.dispose()


async def test_backfill_skips_analyzed_unless_force(backfill_env: Any) -> None:
    handle, meetings_dir, config_dir, queued = backfill_env
    raw = _seed_transcribed(handle, meetings_dir, "m-a", 1_000)
    _seed_transcribed(handle, meetings_dir, "m-b", 2_000)
    _mark_analyzed(handle, config_dir, "m-a", raw)

    res = await backfill(handle.engine, meetings_dir, config_dir)
    assert res == {"queued": 1, "skipped": 1}
    assert queued == ["m-b"]

    assert queued.forced == [False]

    queued.clear()
    queued.forced.clear()
    res = await backfill(handle.engine, meetings_dir, config_dir, force=True)
    assert res == {"queued": 2, "skipped": 0}
    assert queued == ["m-a", "m-b"]
    assert queued.forced == [True, True]
    handle.engine.dispose()


async def test_backfill_honors_date_range_and_limit(backfill_env: Any) -> None:
    handle, meetings_dir, config_dir, queued = backfill_env
    for i, mid in enumerate(["m-1", "m-2", "m-3", "m-4"], start=1):
        _seed_transcribed(handle, meetings_dir, mid, i * 1_000)

    res = await backfill(handle.engine, meetings_dir, config_dir, frm=2_000, to=3_000)
    assert queued == ["m-2", "m-3"]
    assert res["queued"] == 2

    queued.clear()
    res = await backfill(handle.engine, meetings_dir, config_dir, limit=3)
    assert queued == ["m-1", "m-2", "m-3"]
    assert res["queued"] == 3
    handle.engine.dispose()


async def test_backfill_without_hook_queues_nothing(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    _seed_transcribed(handle, str(tmp_path), "m-a", 1_000)
    insights_extract.set_enqueue_hook(None)
    assert await backfill(handle.engine, str(tmp_path), str(tmp_path)) == {"queued": 0, "skipped": 0}
    handle.engine.dispose()


async def test_forced_backfill_carries_force_and_reanalyzes(tmp_path: Path) -> None:
    from test_insights_extract import _main_response, _make_service, _seed

    handle, meetings_dir, config_dir, llm, service = _make_service(tmp_path, lambda n, o: _main_response())
    meeting_id = _seed(handle.engine, meetings_dir)
    await service.analyze_meeting(meeting_id)
    assert len(llm.calls) == 1

    async def hook(mid: str, force: bool = False) -> bool:
        await service.analyze_meeting(mid, force=force)
        return True

    insights_extract.set_enqueue_hook(hook)
    res = await backfill(handle.engine, meetings_dir, config_dir)
    assert res == {"queued": 0, "skipped": 1} and len(llm.calls) == 1  # already analyzed

    res = await backfill(handle.engine, meetings_dir, config_dir, force=True)
    assert res == {"queued": 1, "skipped": 0}
    assert len(llm.calls) == 2  # the forced job re-analyzed instead of skipping
    handle.engine.dispose()
