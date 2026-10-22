"""Meeting merge service: concatenate several meetings into the earliest one.

Inverse of :mod:`olive.services.split`. A session split across multiple clips
(one file per recorder restart, or several Plaud files) can be stitched back
together:

* audio is concatenated with ffmpeg's ``concat`` filter, re-encoding so mixed
  codecs/sample rates are fine;
* transcript segments are shifted by each meeting's offset and joined in
  chronological order (no re-transcription);
* summaries are dropped (they describe a single clip and would be stale);
* tags and speaker links are unioned;
* the source meetings are deleted (folder + DB row), which also tombstones
  Plaud file ids so the poller does not resurrect them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..layout import meeting_paths, sanitize_filename
from .meetings import delete_meeting, to_meeting
from .split import (
    _codec_args,
    _delete_stale_enhanced_files,
    _probe_duration_ms,
    _transcript_to_text,
)


def _shift_timed_items(items: list[Any], offset_ms: int) -> list[Any]:
    """Return copies of timed dicts (segments or words) shifted by offset_ms."""
    shifted_items: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        shifted = dict(item)
        try:
            start = int(item.get("startMs", item.get("start_time", 0)))
            end = int(item.get("endMs", item.get("end_time", start)))
        except (TypeError, ValueError):
            start = 0
            end = 0
        shifted["startMs"] = start + offset_ms
        shifted["endMs"] = end + offset_ms
        shifted["start_time"] = shifted["startMs"]
        shifted["end_time"] = shifted["endMs"]
        shifted_items.append(shifted)
    return shifted_items


def shift_segments(segments: list[Any], offset_ms: int) -> list[dict[str, Any]]:
    """Shift a transcript's segments (and their words) by offset_ms."""
    out: list[dict[str, Any]] = []
    for seg in _shift_timed_items(segments, offset_ms):
        words = seg.get("words")
        if isinstance(words, list):
            seg["words"] = _shift_timed_items(words, offset_ms)
        out.append(seg)
    return out


def merge_transcript_payloads(entries: list[tuple[Any, int]]) -> Any:
    """Join ``(payload, offset_ms)`` pairs into one transcript payload.

    Preserves the dict shape when any payload is a ``{"segments": [...]}``
    object; otherwise returns a bare segment list. Unknown shapes are skipped.
    """
    base_dict: dict[str, Any] | None = None
    dict_payloads: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    saw_list = False
    for payload, offset_ms in entries:
        if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
            dict_payloads.append(payload)
            if base_dict is None:
                base_dict = payload
            segments = payload["segments"]
        elif isinstance(payload, list):
            saw_list = True
            segments = payload
        else:
            continue
        merged.extend(shift_segments(segments, offset_ms))

    merged.sort(key=lambda s: (int(s.get("startMs", 0)), int(s.get("endMs", 0))))
    if base_dict is not None:
        out = dict(base_dict)
        # Fill metadata (language, model, …) from later payloads when missing.
        for payload in dict_payloads[1:]:
            for key, value in payload.items():
                if key != "segments" and key not in out:
                    out[key] = value
        out["segments"] = merged
        out["durationMs"] = max((int(s.get("endMs", 0)) for s in merged), default=0)
        return out
    if not saw_list:
        raise ValueError("unrecognized transcript shape")
    return merged


def _concat_audio(sources: list[Path], dst: Path, ext: str) -> None:
    """Concatenate audio files into dst, re-encoding to keep mixed inputs valid."""
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for src in sources:
        args += ["-i", str(src)]
    graph = "".join(f"[{i}:a]" for i in range(len(sources)))
    graph += f"concat=n={len(sources)}:v=0:a=1[out]"
    args += ["-filter_complex", graph, "-map", "[out]", *_codec_args(ext), str(dst)]
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        err = (proc.stderr or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"ffmpeg merge failed ({err})")


def _ordered_ids(meeting_ids: list[str]) -> list[str]:
    ids: list[str] = []
    for raw in meeting_ids:
        mid = str(raw).strip()
        if mid and mid not in ids:
            ids.append(mid)
    return ids


def merge_meetings(db: Engine, meetings_dir: str, meeting_ids: list[str]) -> dict[str, Any]:
    """Merge meetings into the earliest one and delete the rest.

    Returns ``{"meeting": <merged meeting>, "deletedMeetingIds": [...]}``.
    Raises ValueError on bad input, RuntimeError on ffmpeg failure.
    """
    ids = _ordered_ids(meeting_ids)
    if len(ids) < 2:
        raise ValueError("Select at least two meetings to merge")

    with db.connect() as conn:
        rows = []
        for mid in ids:
            row = conn.execute(text("SELECT * FROM meetings WHERE id = :id"), {"id": mid}).mappings().first()
            if not row:
                raise ValueError("Meeting not found")
            rows.append(dict(row))

        rows.sort(key=lambda r: (int(r["start_time"]), int(r["created_at"])))

        recordings: list[dict[str, Any]] = []
        transcript_artifacts: list[list[dict[str, Any]]] = []
        speaker_links: list[list[dict[str, Any]]] = []
        for row in rows:
            rec = (
                conn.execute(
                    text("SELECT * FROM recordings WHERE meeting_id = :id ORDER BY created_at ASC LIMIT 1"),
                    {"id": row["id"]},
                )
                .mappings()
                .first()
            )
            if not rec:
                raise ValueError(f"Meeting “{row['title']}” has no recording to merge")
            recordings.append(dict(rec))
            artifacts = (
                conn.execute(
                    text(
                        "SELECT * FROM artifacts WHERE meeting_id = :id AND kind = 'transcript' "
                        "AND format = 'json' ORDER BY created_at ASC"
                    ),
                    {"id": row["id"]},
                )
                .mappings()
                .all()
            )
            transcript_artifacts.append([dict(a) for a in artifacts])
            links = (
                conn.execute(
                    text("SELECT * FROM meeting_speakers WHERE meeting_id = :id"),
                    {"id": row["id"]},
                )
                .mappings()
                .all()
            )
            speaker_links.append([dict(link) for link in links])

    if any(str(row["status"]) == "processing" for row in rows):
        raise ValueError("Cannot merge while a transcription is running")

    target = rows[0]
    target_recording = recordings[0]
    folders = [
        Path(meeting_paths(meetings_dir, int(row["start_time"]), row["title"], row["id"]).folder)
        for row in rows
    ]

    # --- Resolve each meeting's audio path and duration (probe wins).
    audio_paths: list[Path] = []
    durations: list[int] = []
    for row, rec, folder in zip(rows, recordings, folders, strict=True):
        src = folder / rec["path"]
        if not src.exists():
            raise ValueError(f"Audio file for “{row['title']}” is missing on disk")
        audio_paths.append(src)
        duration = _probe_duration_ms(src)
        if duration is None and rec.get("duration_ms") is not None:
            duration = int(rec["duration_ms"])
        if duration is None or duration <= 0:
            raise ValueError(f"Audio duration for “{row['title']}” is unknown")
        durations.append(duration)

    offsets: list[int] = []
    cursor = 0
    for duration in durations:
        offsets.append(cursor)
        cursor += duration
    total_ms = cursor

    # --- Read every transcript payload before mutating disk.
    payload_entries: list[tuple[Any, int]] = []
    for row, artifacts, folder, offset in zip(rows, transcript_artifacts, folders, offsets, strict=True):
        artifact = None
        primary_id = row.get("primary_transcript_artifact_id")
        for candidate in artifacts:
            if candidate["id"] == primary_id:
                artifact = candidate
                break
        if artifact is None and artifacts:
            artifact = artifacts[0]
        if artifact is None:
            continue
        try:
            payload = json.loads((folder / artifact["path"]).read_text(encoding="utf-8"))
        except Exception:
            continue
        payload_entries.append((payload, offset))

    target_folder = folders[0]
    src_audio = audio_paths[0]
    ext = src_audio.suffix.lstrip(".").lower() or "m4a"
    target_artifacts = transcript_artifacts[0]

    # Destination artifact: keep the target's primary transcript if it has one,
    # else the first transcript it has, else create a fresh one.
    dest_artifact: dict[str, Any] | None = None
    primary_id = target.get("primary_transcript_artifact_id")
    for candidate in target_artifacts:
        if candidate["id"] == primary_id:
            dest_artifact = candidate
            break
    if dest_artifact is None and target_artifacts:
        dest_artifact = target_artifacts[0]

    new_artifact_id: str | None = None
    if dest_artifact is not None:
        dest_artifact_id = str(dest_artifact["id"])
        dest_rel = str(dest_artifact["path"])
        dest_provider = str(dest_artifact["provider"])
    else:
        dest_artifact_id = str(uuid.uuid4())
        new_artifact_id = dest_artifact_id
        first_provider = next(
            (str(art["provider"]) for artifacts in transcript_artifacts for art in artifacts),
            "merged",
        )
        dest_provider = first_provider
        stem = sanitize_filename(first_provider) or "merged"
        dest_rel = f"transcripts/{stem}.json"

    merged_payload: Any | None = None
    if payload_entries:
        try:
            merged_payload = merge_transcript_payloads(payload_entries)
        except ValueError:
            merged_payload = None

    # --- Concatenate audio to a temp file first (ffmpeg may fail).
    tmp = target_folder / f".merge_{uuid.uuid4().hex[:8]}.{ext}"
    try:
        _concat_audio(audio_paths, tmp, ext)
        merged_bytes = tmp.read_bytes()
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    if not merged_bytes:
        raise RuntimeError("ffmpeg merge produced empty audio")

    src_audio.write_bytes(merged_bytes)
    _delete_stale_enhanced_files(target_folder, str(target_recording["id"]))
    sha = hashlib.sha256(merged_bytes).hexdigest()

    if merged_payload is not None:
        dest = target_folder / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(merged_payload, indent=2) + "\n", encoding="utf-8")
        try:
            (target_folder / str(Path(dest_rel).with_suffix(".txt"))).write_text(
                _transcript_to_text(merged_payload) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

    # --- Union tags (target order first, then new ones from sources).
    tags: list[str] = []
    for row in rows:
        try:
            raw_tags = json.loads(row.get("tags") or "[]")
        except Exception:
            raw_tags = []
        if not isinstance(raw_tags, list):
            continue
        for tag in raw_tags:
            if isinstance(tag, str) and tag and tag not in tags:
                tags.append(tag)

    now = int(time.time() * 1000)
    new_start = int(target["start_time"])
    new_end = new_start + total_ms
    deleted_ids = [row["id"] for row in rows[1:]]

    with db.begin() as conn:
        # Stale summaries: they described individual clips.
        summary_rows = (
            conn.execute(
                text("SELECT * FROM artifacts WHERE meeting_id = :id AND kind = 'summary'"),
                {"id": target["id"]},
            )
            .mappings()
            .all()
        )
        for summary in summary_rows:
            try:
                (target_folder / summary["path"]).unlink(missing_ok=True)
            except Exception:
                pass

        conn.execute(
            text(
                "UPDATE meetings SET primary_transcript_artifact_id = NULL, "
                "primary_summary_artifact_id = NULL WHERE id = :id"
            ),
            {"id": target["id"]},
        )
        conn.execute(
            text("DELETE FROM artifacts WHERE meeting_id = :id AND kind = 'summary'"),
            {"id": target["id"]},
        )

        # Drop the target's other transcript artifacts; the merged payload
        # supersedes them (otherwise the UI would show one clip's segments).
        for artifact in target_artifacts:
            if merged_payload is not None and str(artifact["id"]) == dest_artifact_id:
                continue
            conn.execute(text("DELETE FROM artifacts WHERE id = :id"), {"id": artifact["id"]})
            try:
                (target_folder / artifact["path"]).unlink(missing_ok=True)
            except Exception:
                pass

        if new_artifact_id is not None and merged_payload is not None:
            conn.execute(
                text(
                    """
                    INSERT INTO artifacts (
                      id, meeting_id, recording_id, kind, provider, format, path, created_at
                    ) VALUES (:id, :mid, :rid, 'transcript', :prov, 'json', :path, :now)
                    """
                ),
                {
                    "id": new_artifact_id,
                    "mid": target["id"],
                    "rid": target_recording["id"],
                    "prov": dest_provider,
                    "path": dest_rel,
                    "now": now,
                },
            )

        conn.execute(
            text(
                "UPDATE meetings SET end_time = :end, tags = :tags, status = 'ready', "
                "last_error = NULL, primary_transcript_artifact_id = :aid, updated_at = :now "
                "WHERE id = :id"
            ),
            {
                "end": new_end,
                "tags": json.dumps(tags),
                "aid": dest_artifact_id if merged_payload is not None else None,
                "now": now,
                "id": target["id"],
            },
        )
        conn.execute(
            text(
                "UPDATE recordings SET size_bytes = :size, sha256 = :sha, duration_ms = :dur WHERE id = :id"
            ),
            {
                "size": len(merged_bytes),
                "sha": sha,
                "dur": total_ms,
                "id": target_recording["id"],
            },
        )

        # Union speaker links from every meeting. Source evidence artifacts are
        # about to be deleted, so keep the speaker without evidence.
        for index, links in enumerate(speaker_links):
            for link in links:
                if index == 0:
                    continue
                conn.execute(
                    text(
                        "INSERT OR IGNORE INTO meeting_speakers "
                        "(meeting_id, speaker_id, evidence_artifact_id) "
                        "VALUES (:mid, :sid, NULL)"
                    ),
                    {"mid": target["id"], "sid": link["speaker_id"]},
                )

        updated = (
            conn.execute(text("SELECT * FROM meetings WHERE id = :id"), {"id": target["id"]})
            .mappings()
            .first()
        )

    # --- Delete the source meetings (DB rows + folders, plus Plaud tombstones).
    for mid in deleted_ids:
        delete_meeting(db, mid, meetings_dir)

    assert updated is not None
    return {"meeting": to_meeting(updated).model_dump(), "deletedMeetingIds": deleted_ids}
