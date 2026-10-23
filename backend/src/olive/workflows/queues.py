"""Named DBOS stage queues for Olive background pipelines."""

from __future__ import annotations

from dataclasses import dataclass

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
    "QUEUE_DIARIZE",
    "QUEUE_FFMPEG",
    "QUEUE_PLAUD_SYNC",
    "QUEUE_SPEAKER_EMBED",
    "QUEUE_SPEECHMATICS",
    "QUEUE_SUMMARIZE",
    "STAGE_QUEUE_SPECS",
    "StageQueueSpec",
]
