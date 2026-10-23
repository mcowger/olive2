"""Acceptance tests for the flat stage hand-off local transcription pipeline.

The design under test: each stage of local transcription is its own DBOS
workflow on its own queue, a stage hands off to its successors and returns, and
**no workflow ever awaits another workflow**. Liveness therefore comes from
queue concurrency alone, stuck processing is bounded by per-stage DBOS step
timeouts, and persistence is owned by the final stage.

Stage bodies are swapped for cheap fakes so these tests exercise the real DBOS
queues, workflows, fan-in claims and cancellation without any ML.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from dbos import DBOS
from sqlalchemy import event, text

import olive.app  # noqa: F401  (import order avoids a workflows <-> ml circular import)
from olive.db import create_db
from olive.workflows import queues as queues_mod
from olive.workflows import runtime as runtime_mod
from olive.workflows import transcription_workflow as tw
from olive.workflows.transcription_workflow import (
    STAGE_ALIGN,
    STAGE_ASR,
    STAGE_DIARIZE,
    STAGE_FFMPEG_DECODE,
    STAGE_SEQUENCE,
    STAGE_SPEAKER_EMBED,
    enqueue_local_transcription,
    local_run_workflow_prefix,
    local_stage_workflow_id,
)

# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


class StageRecorder:
    """Records stage-body calls and peak concurrency per stage."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.active: dict[str, int] = {}
        self.peak: dict[str, int] = {}
        self.finalized: list[dict[str, Any]] = []
        self.order: list[str] = []

    def enter(self, stage: str, meeting_id: str) -> None:
        self.calls.append((stage, meeting_id))
        self.order.append(stage)
        self.active[stage] = self.active.get(stage, 0) + 1
        self.peak[stage] = max(self.peak.get(stage, 0), self.active[stage])

    def exit(self, stage: str) -> None:
        self.active[stage] = self.active.get(stage, 1) - 1


@pytest.fixture
def pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fresh DBOS runtime + in-memory Olive DB wired to the stage workflows."""

    class Harness:
        def __init__(self) -> None:
            self.db = create_db(":memory:")
            self.engine = self.db.engine
            self.config_dir = str(tmp_path / "config")
            self.meetings_dir = str(tmp_path / "meetings")
            Path(self.meetings_dir).mkdir(parents=True, exist_ok=True)
            self.recorder = StageRecorder()
            self.progress: list[tuple[str, dict[str, Any]]] = []

        def seed_meeting(self, meeting_id: str) -> None:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO meetings (
                          id, title, start_time, end_time, source, status, tags,
                          primary_transcript_artifact_id, primary_summary_artifact_id,
                          last_error, created_at, updated_at
                        ) VALUES (
                          :id, 'Stage Meeting', 1, 2, 'upload', 'pending', '[]',
                          NULL, NULL, NULL, 1, 1
                        )
                        """
                    ),
                    {"id": meeting_id},
                )

        def payload(self, meeting_id: str, *, audio: str | None = None) -> dict[str, Any]:
            task_id = uuid.uuid4().hex
            return {
                "taskId": task_id,
                "meetingId": meeting_id,
                "audioPath": audio or str(tmp_path / "audio.wav"),
                "language": "en",
                "modelId": "mock",
                "enrolledSpeakers": [],
                "candidateSpeakers": None,
                "expectedSpeakerCount": None,
                "similarityThreshold": None,
                "clusteringThreshold": None,
                "wholeFileAsr": False,
                "pipelineOptions": {},
                "cleanupWav": False,
            }

        def stage_status(self, meeting_id: str) -> dict[str, str]:
            with self.engine.connect() as conn:
                rows = (
                    conn.execute(
                        text("SELECT stage, status FROM stage_runs WHERE meeting_id = :mid"),
                        {"mid": meeting_id},
                    )
                    .mappings()
                    .all()
                )
            return {str(r["stage"]): str(r["status"]) for r in rows}

        def workflow_rows(self, meeting_id: str) -> list[tuple[str, str]]:
            path = Path(self.config_dir) / "dbos.sqlite"
            if not path.exists():
                return []
            with sqlite3.connect(str(path), timeout=1.0) as sconn:
                rows = sconn.execute(
                    """
                    SELECT workflow_uuid, status FROM workflow_status
                    WHERE workflow_uuid LIKE :prefix
                    """,
                    {"prefix": f"stage:{meeting_id}:%"},
                ).fetchall()
            return [(str(r[0]), str(r[1])) for r in rows]

        async def wait_for_quiescent(self, meeting_id: str, *, timeout: float = 20.0) -> None:
            """Wait until none of the meeting's stage workflows are still active.

            Fire-and-forget stages can outlive the assertions; letting them write
            to a disposed in-memory DB only produces noise (and hidden leaks).
            """
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                active = [
                    row
                    for row in self.workflow_rows(meeting_id)
                    if row[1] in ("PENDING", "ENQUEUED", "DELAYED")
                ]
                if not active:
                    return
                await asyncio.sleep(0.05)
            raise AssertionError(
                f"workflows for {meeting_id} never settled: {self.workflow_rows(meeting_id)}"
            )

        async def wait_for(
            self,
            meeting_id: str,
            stage: str,
            *,
            timeout: float = 30.0,
            statuses: tuple[str, ...] = ("done", "error", "cancelled"),
        ) -> str:
            deadline = time.monotonic() + timeout
            last: str | None = None
            while time.monotonic() < deadline:
                last = self.stage_status(meeting_id).get(stage)
                if last in statuses:
                    return last
                await asyncio.sleep(0.02)
            raise AssertionError(
                f"stage {stage} for {meeting_id} never reached {statuses}; "
                f"stage_runs={self.stage_status(meeting_id)} workflows={self.workflow_rows(meeting_id)}"
            )

    harness = Harness()

    async def _progress_sink(meeting_id: str, update: dict[str, Any]) -> None:
        harness.progress.append((meeting_id, update))

    async def _complete_sink(meeting_id: str, result: dict[str, Any], wf_id: str | None) -> None:
        harness.recorder.finalized.append({"meetingId": meeting_id, "result": result, "workflowId": wf_id})
        tw._mark_local_stage(meeting_id, "local_transcribe", "done")

    tw.configure_recovery_runtime(
        db=harness.engine,
        progress_sink=_progress_sink,
        complete_sink=_complete_sink,
        config_dir=harness.config_dir,
        meetings_dir=harness.meetings_dir,
    )
    tw._DBOS_PROGRESS_CACHE = None
    tw._WF_MEETING_CACHE.clear()
    tw._TASK_CONTEXTS.clear()

    try:
        yield harness
    finally:
        tw._TASK_CONTEXTS.clear()
        harness.engine.dispose()


def _install_fake_bodies(
    monkeypatch: pytest.MonkeyPatch, harness: StageRecorder, bodies: dict[str, Any]
) -> None:
    monkeypatch.setattr(tw, "_LOCAL_STAGE_BODIES", bodies)


def _fake_transcript() -> dict[str, Any]:
    return {
        "transcript": {"segments": [], "language": "en", "durationMs": 1000},
        "discoveredSpeakers": [],
    }


def _make_bodies(
    recorder: StageRecorder,
    *,
    sleep: dict[str, float] | None = None,
    fail: dict[str, BaseException] | None = None,
) -> dict[str, Any]:
    """Cheap stand-ins for the stage steps, recording order and concurrency."""
    delays = sleep or {}
    failures = fail or {}
    bodies: dict[str, Any] = {}
    for stage in STAGE_SEQUENCE:

        async def _body(
            payload: dict[str, Any],
            _stage: str = stage,
        ) -> dict[str, Any]:
            meeting_id = str(payload.get("meetingId") or "")
            recorder.enter(_stage, meeting_id)
            try:
                await asyncio.sleep(delays.get(_stage, 0.01))
                if _stage in failures:
                    raise failures[_stage]
                if _stage == STAGE_ALIGN:
                    return _fake_transcript()
                return {"stage": _stage}
            finally:
                recorder.exit(_stage)

        bodies[stage] = _body
    return bodies


async def _enqueue(harness: Any, meeting_id: str, payload: dict[str, Any]) -> str:
    return await enqueue_local_transcription(payload)


# --------------------------------------------------------------------------
# criterion 1/2: no nesting, no pipeline queue
# --------------------------------------------------------------------------


def _workflow_function_nodes(module_path: Path) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            src = ast.unparse(dec)
            if "DBOS.workflow" in src:
                out.append(node)
    return out


def test_no_workflow_awaits_another_workflow() -> None:
    module_path = Path(inspect.getsourcefile(tw) or "")
    workflow_nodes = _workflow_function_nodes(module_path)
    assert workflow_nodes, "no DBOS workflow functions found"

    offenders: list[str] = []
    for node in workflow_nodes:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                if inner.func.attr in ("get_result", "get_result_async"):
                    offenders.append(f"{node.name}:{inner.lineno}")
    assert not offenders, f"workflow bodies must not await other workflows: {offenders}"


def test_pipeline_queue_is_gone() -> None:
    assert not hasattr(queues_mod, "QUEUE_PIPELINE")
    assert "pipeline" not in {spec.name for spec in queues_mod.STAGE_QUEUE_SPECS}


def test_local_provider_has_no_single_queue_route() -> None:
    from olive.services.transcription import PROVIDER_LOCAL, TRANSCRIBE_QUEUE_BY_PROVIDER

    assert PROVIDER_LOCAL not in TRANSCRIBE_QUEUE_BY_PROVIDER
    assert set(TRANSCRIBE_QUEUE_BY_PROVIDER) == {"speechmatics", "assemblyai", "muse"}


# --------------------------------------------------------------------------
# criterion 3: deadlock regression
# --------------------------------------------------------------------------


async def test_concurrent_meetings_all_finish_with_serial_diarize(
    pipeline: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four meetings at once must not deadlock, and stage concurrency still holds.

    Under the old design two drivers held both ``pipeline`` slots while awaiting
    children that needed the same slots. Flat stages have no slot holder waiting,
    so all four runs complete and the 1-slot diarize queue stays serial.
    """
    harness = pipeline
    bodies = _make_bodies(harness.recorder, sleep={STAGE_DIARIZE: 0.05, STAGE_ASR: 0.05})
    _install_fake_bodies(monkeypatch, harness.recorder, bodies)

    meeting_ids = [f"m-deadlock-{i}" for i in range(4)]
    payloads = {}
    for meeting_id in meeting_ids:
        harness.seed_meeting(meeting_id)
        payloads[meeting_id] = harness.payload(meeting_id)
        await _enqueue(harness, meeting_id, payloads[meeting_id])

    for meeting_id in meeting_ids:
        assert await harness.wait_for(meeting_id, STAGE_ALIGN) == "done"

    statuses = harness.stage_status(meeting_ids[0])
    for stage in STAGE_SEQUENCE:
        assert statuses[stage] == "done"
    assert harness.recorder.peak[STAGE_DIARIZE] == 1
    assert harness.recorder.peak[STAGE_ASR] == 1
    assert len(harness.recorder.finalized) == len(meeting_ids)


# --------------------------------------------------------------------------
# criterion 4: a stuck stage is bounded and isolated
# --------------------------------------------------------------------------


@DBOS.step(timeout_seconds=0.25, preemptible=True)
async def _slow_step(payload: dict[str, Any]) -> dict[str, Any]:
    await asyncio.sleep(float(payload.get("__sleepSec", 30)))
    return {"slept": True}


async def test_stage_timeout_fails_one_meeting_and_spares_the_others(
    pipeline: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage that blows its DBOS deadline fails that run — other runs finish."""
    harness = pipeline
    bodies = _make_bodies(harness.recorder)
    healthy_asr = bodies[STAGE_ASR]
    stuck_id = "m-timeout-stuck"

    async def _asr(payload: dict[str, Any]) -> dict[str, Any]:
        # Only this meeting's ASR hangs; the queue is shared, so the other
        # meeting also proves the slot is released on timeout.
        if payload.get("meetingId") == stuck_id:
            return await _slow_step({**payload, "__sleepSec": 30})
        return await healthy_asr(payload)

    bodies[STAGE_ASR] = _asr
    _install_fake_bodies(monkeypatch, harness.recorder, bodies)

    healthy_id = "m-timeout-healthy"
    harness.seed_meeting(stuck_id)
    harness.seed_meeting(healthy_id)

    await _enqueue(harness, stuck_id, harness.payload(stuck_id))
    await _enqueue(harness, healthy_id, harness.payload(healthy_id))

    assert await harness.wait_for(stuck_id, STAGE_ASR) == "error"
    # The deadline is reported to the user as a retryable failure.
    with harness.engine.connect() as conn:
        row = (
            conn.execute(
                text("SELECT last_error FROM stage_runs WHERE meeting_id = :mid AND stage = :stage"),
                {"mid": stuck_id, "stage": STAGE_ASR},
            )
            .mappings()
            .one()
        )
        meeting = (
            conn.execute(text("SELECT last_error FROM meetings WHERE id = :id"), {"id": stuck_id})
            .mappings()
            .one()
        )
    assert row["last_error"]
    assert "timeout" in str(row["last_error"]).lower()
    assert meeting["last_error"] == row["last_error"]
    # A failed stage never enqueues its successor.
    assert harness.stage_status(stuck_id)[STAGE_ALIGN] == "pending"
    assert not [row for row in harness.workflow_rows(stuck_id) if row[0].endswith(f":{STAGE_ALIGN}")]

    # The other meeting is unaffected by the stuck one (it shares the 1-slot ASR
    # queue, which is released when the stuck coroutine is cancelled).
    assert await harness.wait_for(healthy_id, STAGE_ALIGN) == "done"


# --------------------------------------------------------------------------
# criterion 5: cancel mid-stage
# --------------------------------------------------------------------------


@DBOS.step(preemptible=True)
async def _long_step(payload: dict[str, Any]) -> dict[str, Any]:
    await asyncio.sleep(float(payload.get("__sleepSec", 30)))
    return {"slept": True}


async def test_cancel_mid_stage_cancels_run_and_enqueues_no_successor(
    pipeline: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = pipeline
    bodies = _make_bodies(harness.recorder)

    async def _blocking_asr(payload: dict[str, Any]) -> dict[str, Any]:
        return await _long_step({**payload, "__sleepSec": 30})

    bodies[STAGE_ASR] = _blocking_asr
    _install_fake_bodies(monkeypatch, harness.recorder, bodies)

    meeting_id = "m-cancel-stage"
    harness.seed_meeting(meeting_id)
    await _enqueue(harness, meeting_id, harness.payload(meeting_id))

    assert await harness.wait_for(meeting_id, STAGE_ASR, statuses=("running",)) == "running"

    cancelled = await tw.cancel_dbos_workflows_for_meeting(
        meeting_id,
        db=harness.engine,
        config_dir=harness.config_dir,
        meetings_dir=harness.meetings_dir,
    )
    assert cancelled >= 1

    assert await harness.wait_for(meeting_id, STAGE_ASR) == "cancelled"
    # Pre-created rows stay "pending": the join stage must never be enqueued.
    assert harness.stage_status(meeting_id)[STAGE_ALIGN] == "pending"
    assert not [row for row in harness.workflow_rows(meeting_id) if row[0].endswith(f":{STAGE_ALIGN}")]
    # No orphan queued/running workflows are left behind for this meeting.
    leftover = [
        row for row in harness.workflow_rows(meeting_id) if row[1] in ("ENQUEUED", "PENDING", "DELAYED")
    ]
    assert leftover == [], leftover


# --------------------------------------------------------------------------
# criterion 6: fan-in enqueues align exactly once, in either order
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slow_stage",
    [STAGE_SPEAKER_EMBED, STAGE_ASR],
)
async def test_align_is_enqueued_once_regardless_of_finish_order(
    pipeline: Any, monkeypatch: pytest.MonkeyPatch, slow_stage: str
) -> None:
    harness = pipeline
    bodies = _make_bodies(harness.recorder, sleep={slow_stage: 0.12})
    _install_fake_bodies(monkeypatch, harness.recorder, bodies)

    meeting_id = f"m-fanin-{slow_stage}"
    harness.seed_meeting(meeting_id)
    await _enqueue(harness, meeting_id, harness.payload(meeting_id))

    assert await harness.wait_for(meeting_id, STAGE_ALIGN) == "done"
    align_rows = [row for row in harness.workflow_rows(meeting_id) if row[0].endswith(f":{STAGE_ALIGN}")]
    assert len(align_rows) == 1, align_rows
    assert sum(1 for stage, _ in harness.recorder.calls if stage == STAGE_ALIGN) == 1


async def test_claim_step_is_single_winner(pipeline: Any) -> None:
    """The claim is atomic: only one caller may claim a pending stage."""
    harness = pipeline
    meeting_id = "m-claim"
    harness.seed_meeting(meeting_id)
    task_id = uuid.uuid4().hex
    tw._reset_local_stage_runs(meeting_id, job_id="decodewf")

    first = await tw.step_claim_local_stage(meeting_id, task_id, STAGE_DIARIZE)
    second = await tw.step_claim_local_stage(meeting_id, task_id, STAGE_DIARIZE)
    assert first is True
    assert second is False
    assert harness.stage_status(meeting_id)[STAGE_DIARIZE] == "queued"

    # A join stage cannot be claimed before all of its dependencies are done.
    assert await tw.step_claim_local_stage(meeting_id, task_id, STAGE_ALIGN) is False
    tw._mark_local_stage(meeting_id, STAGE_SPEAKER_EMBED, "done")
    assert await tw.step_claim_local_stage(meeting_id, task_id, STAGE_ALIGN) is False
    tw._mark_local_stage(meeting_id, STAGE_ASR, "done")
    assert await tw.step_claim_local_stage(meeting_id, task_id, STAGE_ALIGN) is True


# --------------------------------------------------------------------------
# criterion 7: durability — stages rebuild inputs from database checkpoints
# --------------------------------------------------------------------------


async def test_stages_resume_from_database_checkpoints_without_in_memory_state(
    pipeline: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each stage rebuilds its inputs from SQLite, so a resumed stage works.

    A restart never replays in-memory results: DBOS re-runs the interrupted
    stage in a fresh process. Running the real stage steps directly, one after
    another with a payload that carries only paths, is exactly that situation.
    """
    harness = pipeline
    meeting_id = "m-durable"
    harness.seed_meeting(meeting_id)

    audio = tmp_path / "tone.wav"
    import numpy as np

    from olive.ml.audio import encode_wav

    samples = np.zeros(16000, dtype=np.float32)
    audio.write_bytes(encode_wav(samples, 16000))

    payload = harness.payload(meeting_id, audio=str(audio))
    tw._reset_local_stage_runs(
        meeting_id, job_id=tw.local_run_entry_workflow_id(meeting_id, payload["taskId"])
    )

    # decode runs as a plain function outside DBOS (no step journal).
    await tw.step_decode_audio(payload)
    decode_doc = tw._read_stage_checkpoint(payload, STAGE_FFMPEG_DECODE)
    assert decode_doc["durationMs"] == 1000
    assert decode_doc["sampleRate"] == 16000

    # Drop all per-run Python state before each resumed stage.
    tw._TASK_CONTEXTS.clear()
    resumed = json.loads(json.dumps(payload))
    await tw.step_diarize_turns(resumed)
    assert tw._read_stage_checkpoint(resumed, STAGE_DIARIZE)["turns"]

    tw._TASK_CONTEXTS.clear()
    await tw.step_embed_and_match_speakers(resumed)
    tw._TASK_CONTEXTS.clear()
    await tw.step_transcribe_asr(resumed)
    tw._TASK_CONTEXTS.clear()
    result = await tw.step_align_and_finalize(resumed)
    assert "transcript" in result
    for stage in (STAGE_DIARIZE, STAGE_SPEAKER_EMBED, STAGE_ASR):
        assert tw._read_stage_checkpoint(resumed, stage), stage


async def test_duplicate_enqueue_does_not_reset_a_live_run(
    pipeline: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = pipeline
    bodies = _make_bodies(harness.recorder, sleep={STAGE_ASR: 0.15})
    _install_fake_bodies(monkeypatch, harness.recorder, bodies)

    meeting_id = "m-duplicate"
    harness.seed_meeting(meeting_id)
    payload = harness.payload(meeting_id)
    await _enqueue(harness, meeting_id, payload)
    assert await harness.wait_for(meeting_id, STAGE_DIARIZE, statuses=("done",)) == "done"

    before = harness.stage_status(meeting_id)
    await enqueue_local_transcription(payload)  # second enqueue of the same meeting
    after = harness.stage_status(meeting_id)
    # The live run's progress is untouched (no reset back to pending).
    assert after[STAGE_DIARIZE] == before[STAGE_DIARIZE]
    assert await harness.wait_for(meeting_id, STAGE_ALIGN) == "done"
    align_rows = [row for row in harness.workflow_rows(meeting_id) if row[0].endswith(f":{STAGE_ALIGN}")]
    assert len(align_rows) == 1


# --------------------------------------------------------------------------
# criterion 8: payloads never carry audio bytes
# --------------------------------------------------------------------------


def test_stage_payload_carries_paths_not_audio_bytes() -> None:
    source = inspect.getsource(tw.step_decode_audio)
    assert '"audioPath"' in source
    assert "audioBytes" not in source


def test_run_workflow_ids_are_run_scoped_and_match_cancel_prefix() -> None:
    meeting_id = "m-ids"
    first = local_run_workflow_prefix(meeting_id, "a" * 32)
    second = local_run_workflow_prefix(meeting_id, "b" * 32)
    assert first != second, "a forced re-run must not reuse completed workflow ids"
    assert first.startswith(f"stage:{meeting_id}:")
    assert local_stage_workflow_id(meeting_id, "a" * 32, STAGE_ASR) == f"{first}:{STAGE_ASR}"


# --------------------------------------------------------------------------
# criterion 9: every workflow-id format is cancellable
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "workflow_id",
    [
        "stage:{mid}:abcd1234:ffmpeg_decode",
        "stage:{mid}:abcd1234:align",
        "transcribe:{mid}:deadbeef",
        "reidentify-speakers:{mid}:deadbeef",
        "local-transcribe:{mid}:deadbeef",
    ],
)
async def test_cancel_matcher_covers_every_workflow_id_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow_id: str
) -> None:
    meeting_id = "m-matcher"
    dbos_path = tmp_path / "dbos.sqlite"
    with sqlite3.connect(str(dbos_path)) as sconn:
        sconn.execute(
            "CREATE TABLE workflow_status (workflow_uuid TEXT, parent_workflow_id TEXT, status TEXT)"
        )
        sconn.execute(
            "INSERT INTO workflow_status VALUES (?, ?, 'ENQUEUED')",
            (workflow_id.format(mid=meeting_id), None),
        )

    cancelled: list[str] = []

    async def fake_cancel(wid: str) -> None:
        cancelled.append(wid)

    async def fake_ensure(**_kwargs: Any) -> str:
        return f"sqlite:///{dbos_path}"

    monkeypatch.setattr(tw, "is_dbos_launched", lambda: True)
    monkeypatch.setattr(tw, "get_existing_dbos_sqlite_path", lambda **_k: dbos_path)
    monkeypatch.setattr(
        tw, "DBOS", type("FakeDBOS", (), {"cancel_workflow_async": staticmethod(fake_cancel)})
    )

    count = await tw.cancel_dbos_workflows_for_meeting(
        meeting_id,
        db=None,
        config_dir=str(tmp_path),
        meetings_dir=str(tmp_path),
    )
    assert count == 1
    assert cancelled == [workflow_id.format(mid=meeting_id)]


# --------------------------------------------------------------------------
# criterion 10: the executor cap is a net, not a limiter
# --------------------------------------------------------------------------


def test_executor_cap_is_sized_from_queue_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLIVE_DBOS_MAX_EXECUTOR_THREADS", raising=False)
    queue_slots = sum(spec.worker_concurrency for spec in queues_mod.STAGE_QUEUE_SPECS)
    cap = runtime_mod.resolve_max_executor_threads()
    assert cap == queue_slots + 16
    assert cap > queue_slots, "the cap must never serialize healthy stage work"
    assert runtime_mod._build_dbos_config("sqlite:///x.sqlite").get("max_executor_threads") == cap


def test_executor_cap_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLIVE_DBOS_MAX_EXECUTOR_THREADS", "7")
    assert runtime_mod.resolve_max_executor_threads() == 7
    monkeypatch.setenv("OLIVE_DBOS_MAX_EXECUTOR_THREADS", "not-a-number")
    assert runtime_mod.resolve_max_executor_threads() > 7


# --------------------------------------------------------------------------
# criterion 7 (cont.): a killed runtime resumes the durable run on relaunch
# --------------------------------------------------------------------------


async def test_run_resumes_after_runtime_restart(pipeline: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kill the DBOS runtime mid-stage and relaunch: the run still finishes.

    This is the real restart path — the runtime is destroyed while a stage
    workflow is in flight and launched again against the same system database,
    so DBOS recovery (not a test fixture) has to pick the run back up.
    """
    import olive.workflows.runtime as rt

    harness = pipeline
    hold = asyncio.Event()  # never set: the first decode is killed mid-flight
    held_calls: list[str] = []

    async def _held_decode(payload: dict[str, Any]) -> dict[str, Any]:
        held_calls.append(str(payload.get("meetingId")))
        try:
            await asyncio.wait_for(hold.wait(), timeout=30)
        except TimeoutError:
            pass
        return {"stage": STAGE_FFMPEG_DECODE}

    bodies = _make_bodies(harness.recorder)
    bodies[STAGE_FFMPEG_DECODE] = _held_decode
    _install_fake_bodies(monkeypatch, harness.recorder, bodies)

    meeting_id = "m-restart"
    harness.seed_meeting(meeting_id)
    payload = harness.payload(meeting_id)
    await _enqueue(harness, meeting_id, payload)
    assert await harness.wait_for(meeting_id, STAGE_FFMPEG_DECODE, statuses=("running",)) == "running"

    # Simulated crash: destroy the runtime without letting the stage finish.
    await asyncio.to_thread(rt.shutdown_dbos)
    # Relaunch against the same DBOS database with fast bodies; recovery must
    # resume the durable run rather than waiting for a new request.
    _install_fake_bodies(monkeypatch, harness.recorder, _make_bodies(harness.recorder))
    await rt.ensure_dbos_launched_async(config_dir=harness.config_dir, meetings_dir=harness.meetings_dir)

    assert await harness.wait_for(meeting_id, STAGE_ALIGN) == "done"
    align_rows = [row for row in harness.workflow_rows(meeting_id) if row[0].endswith(f":{STAGE_ALIGN}")]
    assert len(align_rows) == 1, align_rows
    # The killed attempt ran decode and never finished it; recovery re-ran the
    # stage from its durable checkpoint and every later stage ran exactly once.
    assert held_calls == [meeting_id], held_calls
    assert sum(1 for stage, _ in harness.recorder.calls if stage == STAGE_FFMPEG_DECODE) == 1
    for stage in STAGE_SEQUENCE[1:]:
        assert sum(1 for s, _ in harness.recorder.calls if s == stage) == 1, harness.recorder.calls


# --------------------------------------------------------------------------
# upgrade path: pre-flat-pipeline workflows are cancelled, meetings retryable
# --------------------------------------------------------------------------


async def test_legacy_local_workflows_are_cancelled_on_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old nested-orchestration workflows cannot be recovered — cancel them."""
    handle = create_db(":memory:")
    meeting_id = "m-legacy"
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :mid, 'Legacy', 1, 2, 'upload', 'processing', '[]',
                  NULL, NULL, NULL, 1, 1
                )
                """
            ),
            {"mid": meeting_id},
        )
        conn.execute(
            text(
                """
                INSERT INTO stage_runs (
                  id, meeting_id, stage, status, provider_job_id, attempts,
                  last_error, started_at, finished_at, created_at, updated_at
                ) VALUES (
                  'sr-legacy', :mid, 'local_transcribe', 'running', NULL, 1,
                  NULL, 1, NULL, 1, 1
                )
                """
            ),
            {"mid": meeting_id},
        )

    dbos_path = tmp_path / "dbos.sqlite"
    with sqlite3.connect(str(dbos_path)) as sconn:
        sconn.execute(
            "CREATE TABLE workflow_status (workflow_uuid TEXT, name TEXT, parent_workflow_id TEXT, status TEXT)"
        )
        sconn.executemany(
            "INSERT INTO workflow_status VALUES (?, ?, ?, ?)",
            [
                (
                    f"local-transcribe:{meeting_id}:deadbeef",
                    "<temp>.local_transcription_workflow",
                    None,
                    "PENDING",
                ),
                (
                    "<temp>.abc123",
                    "<temp>.step_diarize_turns",
                    f"local-transcribe:{meeting_id}:deadbeef",
                    "PENDING",
                ),
                (f"transcribe:{meeting_id}:cafe0001", "<temp>.run_transcription_driver", None, "ENQUEUED"),
                ("stage:other-meeting:abcd1234:asr", "<temp>.run_local_stage", None, "PENDING"),
            ],
        )

    cancelled: list[str] = []

    async def fake_cancel(wid: str) -> None:
        cancelled.append(wid)

    monkeypatch.setattr(tw, "is_dbos_launched", lambda: True)
    monkeypatch.setattr(tw, "get_existing_dbos_sqlite_path", lambda **_k: dbos_path)
    monkeypatch.setattr(
        tw, "DBOS", type("FakeDBOS", (), {"cancel_workflow_async": staticmethod(fake_cancel)})
    )
    tw.configure_recovery_runtime(db=handle.engine, progress_sink=None, complete_sink=None)

    count = await tw.cancel_legacy_local_transcription_workflows(
        config_dir=str(tmp_path), meetings_dir=str(tmp_path)
    )

    # Legacy parent, its child, and the old driver are cancelled; the new
    # stage workflow for another meeting is left alone.
    assert count == 3
    assert sorted(cancelled) == [
        "<temp>.abc123",
        f"local-transcribe:{meeting_id}:deadbeef",
        f"transcribe:{meeting_id}:cafe0001",
    ]

    with handle.engine.connect() as conn:
        stage = (
            conn.execute(text("SELECT status, last_error FROM stage_runs WHERE id = 'sr-legacy'"))
            .mappings()
            .one()
        )
        meeting = (
            conn.execute(text("SELECT status, last_error FROM meetings WHERE id = :id"), {"id": meeting_id})
            .mappings()
            .one()
        )
    assert stage["status"] == "error"
    assert "upgrade" in str(stage["last_error"]).lower()
    assert meeting["status"] == "pending"
    assert "upgrade" in str(meeting["last_error"]).lower()
    handle.engine.dispose()


# --------------------------------------------------------------------------
# shutdown must not burn a run; failures must be visible and contained
# --------------------------------------------------------------------------


async def test_shutdown_leaves_the_stage_resumable(pipeline: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A graceful stop (task cancellation) must not mark the stage terminal."""
    harness = pipeline
    bodies = _make_bodies(harness.recorder)

    async def _interrupted(payload: dict[str, Any]) -> dict[str, Any]:
        raise asyncio.CancelledError()

    bodies[STAGE_ASR] = _interrupted
    _install_fake_bodies(monkeypatch, harness.recorder, bodies)

    meeting_id = "m-shutdown"
    harness.seed_meeting(meeting_id)
    await _enqueue(harness, meeting_id, harness.payload(meeting_id))

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if harness.stage_status(meeting_id).get(STAGE_ASR) == "running":
            break
        await asyncio.sleep(0.02)
    # Give the failure path time to write (it must not).
    await asyncio.sleep(0.3)

    status = harness.stage_status(meeting_id)
    assert status.get(STAGE_ASR) == "running", status
    with harness.engine.connect() as conn:
        meeting = (
            conn.execute(text("SELECT status, last_error FROM meetings WHERE id = :id"), {"id": meeting_id})
            .mappings()
            .one()
        )
    assert meeting["status"] == "processing"
    assert meeting["last_error"] is None
    # DBOS kept the workflow PENDING, which is what recovery picks up.
    rows = dict(harness.workflow_rows(meeting_id))
    assert rows[f"stage:{meeting_id}:{_run8(harness, meeting_id)}:{STAGE_ASR}"] == "PENDING"


def _run8(harness: Any, meeting_id: str) -> str:
    rows = [wid for wid, _ in harness.workflow_rows(meeting_id) if wid.endswith(f":{STAGE_ASR}")]
    return rows[-1].split(":")[2]


async def test_stage_failure_marks_the_run_row_and_meeting_error(
    pipeline: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed stage must be visible on the run row, not only the stage row."""
    harness = pipeline
    bodies = _make_bodies(harness.recorder, fail={STAGE_DIARIZE: RuntimeError("diarizer exploded")})
    _install_fake_bodies(monkeypatch, harness.recorder, bodies)

    meeting_id = "m-failure"
    harness.seed_meeting(meeting_id)
    await _enqueue(harness, meeting_id, harness.payload(meeting_id))

    assert await harness.wait_for(meeting_id, STAGE_DIARIZE) == "error"
    status = harness.stage_status(meeting_id)
    assert status["local_transcribe"] == "error", status
    with harness.engine.connect() as conn:
        meeting = (
            conn.execute(text("SELECT last_error FROM meetings WHERE id = :id"), {"id": meeting_id})
            .mappings()
            .one()
        )
    assert "diarizer exploded" in str(meeting["last_error"])


async def test_stage_failure_cancels_the_parallel_sibling(
    pipeline: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If one of embed/asr fails, the other must not keep burning a slot for hours."""
    harness = pipeline

    async def _failing_embed(payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("voiceprint extractor died")

    async def _blocking_asr(payload: dict[str, Any]) -> dict[str, Any]:
        # Mirrors the real ASR loop: cooperative abort checks between segments.
        for _ in range(600):
            tw._check_aborted(payload, STAGE_ASR)
            await asyncio.sleep(0.05)
        return {"stage": STAGE_ASR}

    bodies = _make_bodies(harness.recorder)
    bodies[STAGE_SPEAKER_EMBED] = _failing_embed
    bodies[STAGE_ASR] = _blocking_asr
    _install_fake_bodies(monkeypatch, harness.recorder, bodies)

    meeting_id = "m-sibling-cancel"
    harness.seed_meeting(meeting_id)
    await _enqueue(harness, meeting_id, harness.payload(meeting_id))

    assert await harness.wait_for(meeting_id, STAGE_SPEAKER_EMBED) == "error"
    assert await harness.wait_for(meeting_id, STAGE_ASR) == "cancelled"
    assert harness.stage_status(meeting_id)[STAGE_ALIGN] == "pending"
    # The visible outcome stays the real failure, not a sibling's abort.
    terminal = [upd["stage"] for _mid, upd in harness.progress if upd.get("stage") in ("error", "cancelled")]
    assert terminal[0] == "error"
    assert "cancelled" not in terminal[1:]
    await harness.wait_for_quiescent(meeting_id)


def test_stage_row_writes_are_scoped_to_the_run(pipeline: Any) -> None:
    """A superseded run must not overwrite the rows a newer run just reset."""
    harness = pipeline
    meeting_id = "m-run-scope"
    harness.seed_meeting(meeting_id)
    run_a, run_b = "a" * 32, "b" * 32
    tw._reset_local_stage_runs(meeting_id, job_id=f"stage:{meeting_id}:{run_a[:8]}:ffmpeg_decode")

    # Run A starts diarize, then run B resets the meeting's rows.
    tw._mark_local_stage(
        meeting_id, STAGE_DIARIZE, "running", job_id=f"stage:{meeting_id}:{run_a[:8]}:diarize"
    )
    tw._reset_local_stage_runs(meeting_id, job_id=f"stage:{meeting_id}:{run_b[:8]}:ffmpeg_decode")
    # A late write from run A must be ignored.
    tw._mark_local_stage(
        meeting_id, STAGE_DIARIZE, "cancelled", job_id=f"stage:{meeting_id}:{run_a[:8]}:diarize"
    )

    assert harness.stage_status(meeting_id)[STAGE_DIARIZE] == "pending"
    # A fresh "running" from run B is accepted and stamps its own id.
    tw._mark_local_stage(
        meeting_id, STAGE_DIARIZE, "running", job_id=f"stage:{meeting_id}:{run_b[:8]}:diarize"
    )
    with harness.engine.connect() as conn:
        row = (
            conn.execute(
                text(
                    "SELECT status, provider_job_id FROM stage_runs WHERE meeting_id = :mid AND stage = :stage"
                ),
                {"mid": meeting_id, "stage": STAGE_DIARIZE},
            )
            .mappings()
            .one()
        )
    assert row["status"] == "running"
    assert row["provider_job_id"] == f"stage:{meeting_id}:{run_b[:8]}:diarize"


def test_checkpoints_are_run_scoped_and_cleared(pipeline: Any) -> None:
    """Stage outputs are keyed to the run and dropped when the run ends."""
    harness = pipeline
    meeting_id = "m-checkpoints"
    harness.seed_meeting(meeting_id)
    payload = harness.payload(meeting_id)
    other = {**payload, "taskId": "0" * 32}
    tw._reset_local_stage_runs(
        meeting_id, job_id=tw.local_run_entry_workflow_id(meeting_id, payload["taskId"])
    )

    tw._write_stage_checkpoint(payload, STAGE_DIARIZE, {"turns": [{"startMs": 0, "endMs": 10}]})
    assert tw._read_stage_checkpoint(payload, STAGE_DIARIZE)["turns"]
    with pytest.raises(RuntimeError, match="superseded run"):
        tw._read_stage_checkpoint(other, STAGE_DIARIZE)

    tw.clear_local_stage_checkpoints(meeting_id, task_id=other["taskId"])
    assert tw._read_stage_checkpoint(payload, STAGE_DIARIZE)["turns"]
    tw.clear_local_stage_checkpoints(meeting_id, task_id=payload["taskId"])
    with pytest.raises(RuntimeError, match="Missing"):
        tw._read_stage_checkpoint(payload, STAGE_DIARIZE)


def test_superseded_run_cannot_replace_or_clear_new_checkpoints(pipeline: Any) -> None:
    harness = pipeline
    meeting_id = "m-checkpoint-retry"
    harness.seed_meeting(meeting_id)
    old, new = harness.payload(meeting_id), harness.payload(meeting_id)
    tw._reset_local_stage_runs(meeting_id, job_id=tw.local_run_entry_workflow_id(meeting_id, old["taskId"]))
    tw._write_stage_checkpoint(old, STAGE_SPEAKER_EMBED, {"old": True})
    tw._reset_local_stage_runs(meeting_id, job_id=tw.local_run_entry_workflow_id(meeting_id, new["taskId"]))
    with pytest.raises(RuntimeError, match="Missing"):
        tw._read_stage_checkpoint(old, STAGE_SPEAKER_EMBED)
    tw._write_stage_checkpoint(new, STAGE_SPEAKER_EMBED, {"new": True})

    with pytest.raises(RuntimeError, match="superseded"):
        tw._write_stage_checkpoint(old, STAGE_SPEAKER_EMBED, {"old": True})
    tw.clear_local_stage_checkpoints(meeting_id, task_id=old["taskId"])
    assert tw._read_stage_checkpoint(new, STAGE_SPEAKER_EMBED) == {"new": True}


@pytest.mark.parametrize("status", ["done", "error", "cancelled"])
def test_terminal_run_cannot_recreate_checkpoints(pipeline: Any, status: str) -> None:
    harness = pipeline
    meeting_id = "m-checkpoint-terminal"
    harness.seed_meeting(meeting_id)
    payload = harness.payload(meeting_id)
    entry_id = tw.local_run_entry_workflow_id(meeting_id, payload["taskId"])
    tw._reset_local_stage_runs(meeting_id, job_id=entry_id)
    tw._write_stage_checkpoint(payload, STAGE_ASR, {"segments": []})
    tw._mark_local_stage(meeting_id, "local_transcribe", status, job_id=entry_id)
    tw.clear_local_stage_checkpoints(meeting_id, task_id=payload["taskId"])

    with pytest.raises(RuntimeError, match="cancelled"):
        tw._write_stage_checkpoint(payload, STAGE_ASR, {"late": True})
    with pytest.raises(RuntimeError, match="Missing"):
        tw._read_stage_checkpoint(payload, STAGE_ASR)


async def test_failed_checkpoint_reset_rolls_back_and_does_not_enqueue(pipeline: Any) -> None:
    harness = pipeline
    meeting_id = "m-reset-failed"
    harness.seed_meeting(meeting_id)
    old, new = harness.payload(meeting_id), harness.payload(meeting_id)
    tw._reset_local_stage_runs(meeting_id, job_id=tw.local_run_entry_workflow_id(meeting_id, old["taskId"]))
    tw._write_stage_checkpoint(old, STAGE_DIARIZE, {"turns": []})

    def fail_reset(_conn: Any, _cursor: Any, statement: str, *_args: Any) -> None:
        if "INSERT INTO stage_runs" in statement:
            raise RuntimeError("simulated database failure")

    event.listen(harness.engine, "before_cursor_execute", fail_reset)
    try:
        with pytest.raises(RuntimeError, match="simulated database failure"):
            await enqueue_local_transcription(new)
    finally:
        event.remove(harness.engine, "before_cursor_execute", fail_reset)
    assert harness.workflow_rows(meeting_id) == []
    assert tw._read_stage_checkpoint(old, STAGE_DIARIZE) == {"turns": []}


@pytest.mark.parametrize("operation", ["read", "write"])
async def test_superseded_stage_does_not_publish_failure_over_new_run(
    pipeline: Any, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    harness = pipeline
    meeting_id = "m-stale-progress"
    harness.seed_meeting(meeting_id)
    old, new = harness.payload(meeting_id), harness.payload(meeting_id)

    async def superseded_decode(payload: dict[str, Any]) -> dict[str, Any]:
        tw._reset_local_stage_runs(
            meeting_id, job_id=tw.local_run_entry_workflow_id(meeting_id, new["taskId"])
        )
        tw._write_stage_checkpoint(new, STAGE_FFMPEG_DECODE, {"new": True})
        if operation == "read":
            tw._read_stage_checkpoint(payload, STAGE_FFMPEG_DECODE)
        else:
            tw._write_stage_checkpoint(payload, STAGE_FFMPEG_DECODE, {"old": True})
        raise AssertionError("inactive run should have been rejected")

    bodies = _make_bodies(harness.recorder)
    bodies[STAGE_FFMPEG_DECODE] = superseded_decode
    _install_fake_bodies(monkeypatch, harness.recorder, bodies)
    await _enqueue(harness, meeting_id, old)
    await harness.wait_for_quiescent(meeting_id)
    assert tw._read_stage_checkpoint(new, STAGE_FFMPEG_DECODE) == {"new": True}
    assert not [upd for _mid, upd in harness.progress if upd.get("stage") in ("cancelled", "error")]
    assert harness.stage_status(meeting_id)["local_transcribe"] == "running"
