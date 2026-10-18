"""Named DBOS stage queues for Olive background pipelines."""

from __future__ import annotations

from dataclasses import dataclass

# The local-transcription driver (run_transcription_driver) runs here, NOT on
# QUEUE_PIPELINE. The driver enqueues its child local_transcription_workflow onto
# QUEUE_PIPELINE and then awaits it; if the driver shared that 2-slot queue, two
# concurrent drivers would hold both slots while their children sat un-schedulable —
# a deadlock that stalled the whole backlog. Keeping the driver on its own queue lets
# QUEUE_PIPELINE bound the actual concurrent-meeting work (2) without being starved by
# waiting drivers.
QUEUE_DRIVER = "transcribe_driver"
QUEUE_PIPELINE = "pipeline"
QUEUE_PLAUD_SYNC = "plaud_sync"
QUEUE_FFMPEG = "ffmpeg"
QUEUE_DIARIZE = "diarize"
QUEUE_SPEAKER_EMBED = "speaker_embed"
QUEUE_ASR = "asr"
QUEUE_ALIGN = "align"
QUEUE_SPEECHMATICS = "speechmatics"
QUEUE_ASSEMBLYAI = "assemblyai"
QUEUE_MUSE = "muse"
QUEUE_SUMMARIZE = "summarize"


@dataclass(frozen=True)
class StageQueueSpec:
    name: str
    worker_concurrency: int
    polling_interval_sec: float = 0.05


STAGE_QUEUE_SPECS: tuple[StageQueueSpec, ...] = (
    # Driver concurrency only needs to exceed QUEUE_PIPELINE's so the pipeline is kept
    # fed; drivers mostly just await their child, so this is cheap.
    StageQueueSpec(name=QUEUE_DRIVER, worker_concurrency=8),
    StageQueueSpec(name=QUEUE_PIPELINE, worker_concurrency=2),
    StageQueueSpec(name=QUEUE_PLAUD_SYNC, worker_concurrency=1),
    StageQueueSpec(name=QUEUE_FFMPEG, worker_concurrency=2),
    StageQueueSpec(name=QUEUE_DIARIZE, worker_concurrency=1),
    StageQueueSpec(name=QUEUE_SPEAKER_EMBED, worker_concurrency=1),
    StageQueueSpec(name=QUEUE_ASR, worker_concurrency=1),
    StageQueueSpec(name=QUEUE_ALIGN, worker_concurrency=1),
    StageQueueSpec(name=QUEUE_SPEECHMATICS, worker_concurrency=2),
    StageQueueSpec(name=QUEUE_ASSEMBLYAI, worker_concurrency=2),
    StageQueueSpec(name=QUEUE_MUSE, worker_concurrency=2),
    StageQueueSpec(name=QUEUE_SUMMARIZE, worker_concurrency=2),
)

__all__ = [
    "QUEUE_ALIGN",
    "QUEUE_ASSEMBLYAI",
    "QUEUE_MUSE",
    "QUEUE_ASR",
    "QUEUE_DRIVER",
    "QUEUE_DIARIZE",
    "QUEUE_FFMPEG",
    "QUEUE_PIPELINE",
    "QUEUE_PLAUD_SYNC",
    "QUEUE_SPEAKER_EMBED",
    "QUEUE_SPEECHMATICS",
    "QUEUE_SUMMARIZE",
    "STAGE_QUEUE_SPECS",
    "StageQueueSpec",
]
