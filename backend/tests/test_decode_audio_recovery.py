"""Recovery self-heal for the audio-decode step.

Regression cover for a local transcription that died terminally after an OOM
restart with:

    ValueError: Either audio_path or audio_bytes must be provided

Cause: a bytes-only caller (an in-flight upload whose audio lived only in
``TaskExecutionContext.audio_bytes``, in memory) persisted ``audioPath=None`` in
the durable DBOS payload. When the container was OOM-killed mid-transcription,
DBOS recovered the workflow in a fresh process where the in-memory bytes were
gone and the payload carried no path, so ``step_decode_audio`` had nothing to
decode and raised. The driver re-enqueued the meeting so no recording was lost,
but that attempt burned terminally on every restart.

Fix: ``step_decode_audio`` re-resolves the recording's on-disk path from the DB
by ``meetingId`` whenever the in-memory bytes are gone and the payload path is
missing/absent, so recovery proceeds from the file that is always on disk. These
tests pin the two seams that carry that logic:

* ``_resolve_recording_audio_path`` — meetingId -> on-disk audio path via the DB.
* ``_effective_decode_source`` — the source-selection the decode step uses, which
  must fall back to the DB path on recovery and otherwise prefer live inputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from olive.db import create_db
from olive.layout import ensure_meeting_folder, meeting_paths
from olive.workflows import transcription_workflow as tw


def _seed_meeting_with_recording(
    engine: object,
    meetings_dir: Path,
    *,
    meeting_id: str = "mtg-recov",
    start_time_ms: int = 1_700_000_000_000,
    title: str = "Recovery Test",
    rec_filename: str = "audio.m4a",
    write_file: bool = True,
) -> str:
    """Insert a meeting + recording and (optionally) create the audio file on disk.

    Returns the absolute path the resolver is expected to produce.
    """
    from sqlalchemy import text

    paths = meeting_paths(meetings_dir, start_time_ms, title, meeting_id)
    ensure_meeting_folder(paths)
    audio_abs = Path(paths.folder) / rec_filename
    if write_file:
        audio_abs.write_bytes(b"\x00\x00fake-audio-container")

    with engine.begin() as conn:  # type: ignore[attr-defined]
        conn.execute(
            text(
                "INSERT INTO meetings (id, title, start_time, end_time, source, status, created_at, updated_at) "
                "VALUES (:id, :title, :st, :et, 'upload', 'ready', :now, :now)"
            ),
            {"id": meeting_id, "title": title, "st": start_time_ms, "et": start_time_ms, "now": start_time_ms},
        )
        conn.execute(
            text(
                "INSERT INTO recordings "
                "(id, meeting_id, path, mime, size_bytes, sha256, provider, created_at) "
                "VALUES (:rid, :mid, :path, :mime, :size, :sha, 'upload', :now)"
            ),
            {
                "rid": f"rec-{meeting_id}",
                "mid": meeting_id,
                "path": rec_filename,
                "mime": "audio/mp4",
                "size": 21,
                "sha": f"sha-{meeting_id}",
                "now": start_time_ms,
            },
        )
    return str(audio_abs)


# --- _resolve_recording_audio_path --------------------------------------------------


def test_resolve_recording_audio_path_returns_ondisk_path(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = tmp_path / "meetings"
    expected = _seed_meeting_with_recording(handle.engine, meetings_dir)

    got = tw._resolve_recording_audio_path("mtg-recov", handle.engine, str(meetings_dir))

    assert got == expected
    assert Path(got).exists()


def test_resolve_recording_audio_path_none_when_file_missing(tmp_path: Path) -> None:
    # Row exists but the audio file is not on disk → no usable path (do not invent one).
    handle = create_db(":memory:")
    meetings_dir = tmp_path / "meetings"
    _seed_meeting_with_recording(handle.engine, meetings_dir, write_file=False)

    assert tw._resolve_recording_audio_path("mtg-recov", handle.engine, str(meetings_dir)) is None


def test_resolve_recording_audio_path_none_when_no_recording(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    assert tw._resolve_recording_audio_path("nope", handle.engine, str(tmp_path)) is None


def test_resolve_recording_audio_path_none_on_missing_inputs(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    assert tw._resolve_recording_audio_path(None, handle.engine, str(tmp_path)) is None
    assert tw._resolve_recording_audio_path("x", None, str(tmp_path)) is None
    assert tw._resolve_recording_audio_path("x", handle.engine, None) is None


# --- _effective_decode_source (the decode step's source selection) ------------------


def test_effective_source_recovers_db_path_when_bytes_gone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # THE REGRESSION: recovery context (no audio_bytes) + payload audioPath=None.
    # Before the fix this left the decode step with neither source -> ValueError.
    handle = create_db(":memory:")
    meetings_dir = tmp_path / "meetings"
    expected = _seed_meeting_with_recording(handle.engine, meetings_dir)

    # Simulate the recovery runtime wiring (configure_recovery_runtime sets these).
    monkeypatch.setattr(tw, "_RECOVERY_DB", handle.engine)
    monkeypatch.setattr(tw, "_RECOVERY_MEETINGS_DIR", str(meetings_dir))

    ctx = tw.TaskExecutionContext(task_id="t1", meeting_id="mtg-recov", audio_bytes=None, has_live_caller=False)
    payload = {"taskId": "t1", "meetingId": "mtg-recov", "audioPath": None}

    audio_path, audio_bytes = tw._effective_decode_source(payload, ctx)

    assert audio_bytes is None
    assert audio_path == expected  # resolved from the DB so recovery can decode


def test_effective_source_prefers_live_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A live caller still holding bytes must be used as-is; no DB lookup needed.
    handle = create_db(":memory:")
    monkeypatch.setattr(tw, "_RECOVERY_DB", handle.engine)
    monkeypatch.setattr(tw, "_RECOVERY_MEETINGS_DIR", str(tmp_path))

    ctx = tw.TaskExecutionContext(task_id="t2", meeting_id="mtg-recov", audio_bytes=b"live-bytes")
    payload = {"taskId": "t2", "meetingId": "mtg-recov", "audioPath": None}

    audio_path, audio_bytes = tw._effective_decode_source(payload, ctx)

    assert audio_bytes == b"live-bytes"
    assert audio_path is None  # bytes present → path left untouched


def test_effective_source_keeps_existing_payload_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A persisted, on-disk payload path is used directly without re-resolving.
    handle = create_db(":memory:")
    monkeypatch.setattr(tw, "_RECOVERY_DB", handle.engine)
    monkeypatch.setattr(tw, "_RECOVERY_MEETINGS_DIR", str(tmp_path))
    real = tmp_path / "already.wav"
    real.write_bytes(b"x")

    ctx = tw.TaskExecutionContext(task_id="t3", meeting_id="mtg-recov", audio_bytes=None)
    payload = {"taskId": "t3", "meetingId": "mtg-recov", "audioPath": str(real)}

    audio_path, audio_bytes = tw._effective_decode_source(payload, ctx)

    assert audio_path == str(real)
    assert audio_bytes is None


def test_effective_source_returns_nothing_when_unresolvable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Truly no audio anywhere → return (None, None); the clear ValueError in
    # load_audio_samples is the right terminal signal, not a silent wrong decode.
    handle = create_db(":memory:")
    monkeypatch.setattr(tw, "_RECOVERY_DB", handle.engine)
    monkeypatch.setattr(tw, "_RECOVERY_MEETINGS_DIR", str(tmp_path))

    ctx = tw.TaskExecutionContext(task_id="t4", meeting_id="ghost", audio_bytes=None)
    payload = {"taskId": "t4", "meetingId": "ghost", "audioPath": None}

    audio_path, audio_bytes = tw._effective_decode_source(payload, ctx)

    assert audio_path is None
    assert audio_bytes is None
