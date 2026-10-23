"""SQLAlchemy 2.0 table & ORM definitions matching today's SQLite schema."""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Index, Integer, MetaData, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    metadata = MetaData()


class MeetingModel(Base):
    __tablename__ = "meetings"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    start_time: Mapped[int] = mapped_column(Integer, nullable=False)
    end_time: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'pending'"))
    tags: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'[]'"))
    primary_transcript_artifact_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("artifacts.id"), nullable=True
    )
    primary_summary_artifact_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("artifacts.id"), nullable=True
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)


class RecordingModel(Base):
    __tablename__ = "recordings"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    meeting_id: Mapped[str] = mapped_column(Text, ForeignKey("meetings.id"), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    mime: Mapped[str] = mapped_column(Text, nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    provider_recording_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class ArtifactModel(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    meeting_id: Mapped[str] = mapped_column(Text, ForeignKey("meetings.id"), nullable=False)
    recording_id: Mapped[str | None] = mapped_column(Text, ForeignKey("recordings.id"), nullable=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    format: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class SpeakerModel(Base):
    __tablename__ = "speakers"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    provider_ids: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'{}'"))
    enrolled_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    enrollment_clip_paths: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'[]'"))
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class MeetingSpeakerModel(Base):
    __tablename__ = "meeting_speakers"

    meeting_id: Mapped[str] = mapped_column(Text, ForeignKey("meetings.id"), primary_key=True)
    speaker_id: Mapped[str] = mapped_column(Text, ForeignKey("speakers.id"), primary_key=True)
    evidence_artifact_id: Mapped[str | None] = mapped_column(Text, ForeignKey("artifacts.id"), nullable=True)


class StageRunModel(Base):
    __tablename__ = "stage_runs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    meeting_id: Mapped[str] = mapped_column(Text, ForeignKey("meetings.id"), nullable=False)
    stage: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'pending'"))
    provider_job_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finished_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)


class StageCheckpointModel(Base):
    """Intermediate output of one local transcription stage, keyed per meeting.

    Each JSON document carries its run id. Writes require an active run, and
    terminal cleanup deletes only that run's rows so late writes or cleanup
    from a cancelled run cannot replace a newer run's inputs.
    """

    __tablename__ = "stage_checkpoints"

    meeting_id: Mapped[str] = mapped_column(Text, ForeignKey("meetings.id"), primary_key=True)
    stage: Mapped[str] = mapped_column(Text, primary_key=True)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    doc: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)


class PlaudIngestStateModel(Base):
    __tablename__ = "plaud_ingest_state"

    meeting_id: Mapped[str] = mapped_column(Text, ForeignKey("meetings.id"), primary_key=True)
    plaud_file_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    first_seen_at: Mapped[int] = mapped_column(Integer, nullable=False)
    pcs_deadline_at: Mapped[int] = mapped_column(Integer, nullable=False)
    pcs_resolved: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    duplicate_of_meeting_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("meetings.id"), nullable=True
    )


class PlaudDeletedFileModel(Base):
    __tablename__ = "plaud_deleted_files"

    plaud_file_id: Mapped[str] = mapped_column(Text, primary_key=True)
    deleted_at: Mapped[int] = mapped_column(Integer, nullable=False)


class SyncStateModel(Base):
    __tablename__ = "sync_state"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class TemplateModel(Base):
    __tablename__ = "templates"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    user_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    is_default: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    is_builtin: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)


class LogModel(Base):
    __tablename__ = "logs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    level: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    meeting_id: Mapped[str | None] = mapped_column(Text, ForeignKey("meetings.id"), nullable=True)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    meeting_id: Mapped[str] = mapped_column(Text, ForeignKey("meetings.id"), nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    usage: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class UsageEventModel(Base):
    __tablename__ = "usage_events"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    meeting_id: Mapped[str | None] = mapped_column(Text, ForeignKey("meetings.id"), nullable=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    audio_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


Index("idx_meetings_start_time", MeetingModel.start_time.desc())
Index("idx_recordings_meeting", RecordingModel.meeting_id)
Index(
    "idx_recordings_provider_id",
    RecordingModel.provider,
    RecordingModel.provider_recording_id,
    unique=True,
    sqlite_where=text("provider_recording_id IS NOT NULL"),
)
Index("idx_artifacts_meeting", ArtifactModel.meeting_id)
Index("idx_stage_runs_meeting_stage", StageRunModel.meeting_id, StageRunModel.stage, unique=True)
Index("idx_templates_name", TemplateModel.name)
Index("idx_logs_created_at", LogModel.created_at.desc())
Index("idx_logs_level", LogModel.level)
Index("idx_logs_category", LogModel.category)
Index("idx_logs_meeting_id", LogModel.meeting_id)
Index("idx_chat_messages_meeting_created", ChatMessageModel.meeting_id, ChatMessageModel.created_at)
Index("idx_usage_events_created", UsageEventModel.created_at.desc())
Index("idx_usage_events_meeting", UsageEventModel.meeting_id)

__all__ = [
    "ArtifactModel",
    "Base",
    "ChatMessageModel",
    "LogModel",
    "MeetingModel",
    "MeetingSpeakerModel",
    "PlaudIngestStateModel",
    "PlaudDeletedFileModel",
    "RecordingModel",
    "SpeakerModel",
    "StageCheckpointModel",
    "StageRunModel",
    "SyncStateModel",
    "TemplateModel",
    "UsageEventModel",
]
