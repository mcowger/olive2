"""Meeting split service: cut one meeting into two at a timestamp.

Covers the "forgot to stop recording" case (30 min meeting + 2 h commute):
splits the audio file with ffmpeg, splits every JSON transcript artifact on
the same boundary (re-anchoring Part B to zero, no re-transcription), drops
summaries on both sides (they describe the whole recording and would be
stale), copies speaker links, and creates the second half as a new meeting.
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

from ..layout import ensure_meeting_folder, meeting_paths
from .meetings import to_meeting

MIN_EDGE_MARGIN_MS = 5_000

_EXT_TO_CODEC: dict[str, list[str]] = {
    "m4a": ["-c:a", "aac"],
    "aac": ["-c:a", "aac"],
    "mp3": ["-c:a", "libmp3lame"],
    "wav": ["-c:a", "pcm_s16le"],
    "ogg": ["-c:a", "libopus"],
    "opus": ["-c:a", "libopus"],
    "flac": ["-c:a", "flac"],
    "webm": ["-c:a", "libopus"],
}


def _codec_args(ext: str) -> list[str]:
    return _EXT_TO_CODEC.get(ext.lower(), ["-c:a", "aac"])


def _delete_stale_enhanced_files(folder: Path, recording_id: str) -> None:
    """Remove cached voice-enhanced derivatives for a recording.

    The audio route serves ``audio/{recording_id}.enhanced.mp3`` whenever it
    exists (see ``app.get_meeting_audio_route``), and the enhancer never
    rebuilds an existing file. Any caller that rewrites the source audio
    in place — e.g. this split service overwriting Part A — must delete
    the stale derivative or playback keeps serving the pre-split audio.
    """
    try:
        candidates = [folder / "audio" / f"{recording_id}.enhanced.mp3"]
        candidates += sorted((folder / "audio").glob(f"{recording_id}.enhanced.*"))
        seen: set[Path] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    except Exception:
        pass


def _probe_duration_ms(path: Path) -> int | None:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-hide_banner",
                "-loglevel",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            secs = float(proc.stdout.strip())
            return int(round(secs * 1000))
    except Exception:
        pass
    return None


def _ffmpeg_cut(src: Path, dst: Path, start_ms: int, end_ms: int | None) -> None:
    """Cut src[start_ms:end_ms] into dst, re-encoding audio for sample accuracy.

    Stream-copy (`-c copy`) is fast but only cuts on packet boundaries, which
    can be seconds off for compressed audio. Meetings are short enough that a
    re-encode is cheap and keeps the split point exact.
    """
    ext = src.suffix.lstrip(".").lower() or "m4a"
    start_s = start_ms / 1000.0
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(src),
    ]
    if end_ms is not None:
        dur_s = (end_ms - start_ms) / 1000.0
        args += ["-t", f"{dur_s:.3f}"]
    args += _codec_args(ext) + ["-avoid_negative_ts", "make_zero", str(dst)]
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        err = (proc.stderr or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"ffmpeg split failed ({err})")


def _shift_words(words: list[dict[str, Any]] | None, split_ms: int, side: str) -> list[Any] | None:
    if not words:
        return words
    kept: list[Any] = []
    for w in words:
        if not isinstance(w, dict):
            continue
        try:
            ws = int(w.get("startMs", w.get("start_time", 0)))
            we = int(w.get("endMs", w.get("end_time", ws)))
        except (TypeError, ValueError):
            continue
        if side == "a" and ws < split_ms:
            kept.append({**w, "endMs": min(we, split_ms), "end_time": min(we, split_ms)})
        elif side == "b" and we > split_ms:
            shifted = {**w}
            shifted["startMs"] = max(0, ws - split_ms)
            shifted["endMs"] = max(0, we - split_ms)
            shifted["start_time"] = shifted["startMs"]
            shifted["end_time"] = shifted["endMs"]
            kept.append(shifted)
    return kept


def _split_segment_list(
    segments: list[dict[str, Any]], split_ms: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split segment dicts on split_ms; Part B re-anchored to zero.

    Segments fully before the cut stay in A (end clipped); fully after move
    to B (shifted). A segment straddling the cut is divided by word
    timestamps when available, otherwise assigned by midpoint with timestamps
    clipped/shifted.
    """
    part_a: list[dict[str, Any]] = []
    part_b: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        try:
            s = int(seg.get("startMs", seg.get("start_time", 0)))
            e = int(seg.get("endMs", seg.get("end_time", s)))
        except (TypeError, ValueError):
            continue
        words = seg.get("words") if isinstance(seg.get("words"), list) else None
        if e <= split_ms:
            out = dict(seg)
            out["startMs"] = s
            out["endMs"] = min(e, split_ms)
            out["start_time"] = out["startMs"]
            out["end_time"] = out["endMs"]
            out["words"] = _shift_words(words, split_ms, "a")
            part_a.append(out)
        elif s >= split_ms:
            out = dict(seg)
            out["startMs"] = s - split_ms
            out["endMs"] = e - split_ms
            out["start_time"] = out["startMs"]
            out["end_time"] = out["endMs"]
            out["words"] = _shift_words(words, split_ms, "b")
            part_b.append(out)
        else:
            # Straddling segment.
            if words:
                wa = _shift_words(words, split_ms, "a") or []
                wb = _shift_words(words, split_ms, "b") or []
                text_a = " ".join(
                    str(w.get("word", "")) for w in wa if str(w.get("word", "")).strip()
                ).strip()
                text_b = " ".join(
                    str(w.get("word", "")) for w in wb if str(w.get("word", "")).strip()
                ).strip()
                base_text = str(seg.get("text", seg.get("content", "")))
                if text_a:
                    out_a = dict(seg)
                    out_a["startMs"] = s
                    out_a["endMs"] = split_ms
                    out_a["start_time"] = s
                    out_a["end_time"] = split_ms
                    out_a["text"] = text_a
                    out_a["content"] = text_a
                    out_a["words"] = wa
                    part_a.append(out_a)
                if text_b:
                    out_b = dict(seg)
                    out_b["startMs"] = 0
                    out_b["endMs"] = e - split_ms
                    out_b["start_time"] = 0
                    out_b["end_time"] = e - split_ms
                    out_b["text"] = text_b or base_text
                    out_b["content"] = text_b or base_text
                    out_b["words"] = wb
                    part_b.append(out_b)
                elif base_text.strip():
                    # All words fell on one side; keep full text there.
                    if (wa and not wb) or ((s + e) / 2 <= split_ms):
                        out_a = dict(seg)
                        out_a["startMs"] = s
                        out_a["endMs"] = split_ms
                        out_a["start_time"] = s
                        out_a["end_time"] = split_ms
                        out_a["words"] = wa
                        part_a.append(out_a)
                    else:
                        out_b = dict(seg)
                        out_b["startMs"] = 0
                        out_b["endMs"] = e - split_ms
                        out_b["start_time"] = 0
                        out_b["end_time"] = e - split_ms
                        out_b["words"] = wb
                        part_b.append(out_b)
            elif ((s + e) / 2) <= split_ms:
                out = dict(seg)
                out["startMs"] = s
                out["endMs"] = split_ms
                out["start_time"] = s
                out["end_time"] = split_ms
                part_a.append(out)
            else:
                out = dict(seg)
                out["startMs"] = 0
                out["endMs"] = e - split_ms
                out["start_time"] = 0
                out["end_time"] = e - split_ms
                part_b.append(out)
    return part_a, part_b


def split_transcript_payload(raw: Any, split_ms: int) -> tuple[Any, Any]:
    """Split a parsed transcript JSON payload, preserving its shape."""
    if isinstance(raw, dict) and isinstance(raw.get("segments"), list):
        segs_a, segs_b = _split_segment_list(raw["segments"], split_ms)
        out_a = dict(raw)
        out_b = dict(raw)
        out_a["segments"] = segs_a
        out_b["segments"] = segs_b
        for out, segs, base in ((out_a, segs_a, 0), (out_b, segs_b, split_ms)):
            if segs:
                out["durationMs"] = max(
                    int(s.get("endMs", s.get("end_time", 0))) for s in segs if isinstance(s, dict)
                )
            else:
                out["durationMs"] = 0
            _ = base
        return out_a, out_b
    if isinstance(raw, list):
        segs_a, segs_b = _split_segment_list(raw, split_ms)
        return segs_a, segs_b
    raise ValueError("unrecognized transcript shape")


def _transcript_to_text(payload: Any) -> str:
    if isinstance(payload, dict):
        segs = payload.get("segments") or []
    elif isinstance(payload, list):
        segs = payload
    else:
        return ""
    lines: list[str] = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        txt = str(s.get("text", s.get("content", ""))).strip()
        if not txt:
            continue
        lines.append(f"{s.get('speaker', 'Speaker')}: {txt}")
    return "\n\n".join(lines)


def split_meeting(db: Engine, meetings_dir: str, meeting_id: str, split_ms: int) -> dict[str, Any]:
    """Split a meeting's audio + transcripts at split_ms; drop summaries.

    Returns {"meeting": <updated Part 1>, "newMeeting": <Part 2>}.
    Raises ValueError on bad input, RuntimeError on ffmpeg failure.
    """
    if not isinstance(split_ms, bool):
        try:
            split_ms = int(split_ms)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("splitMs must be epoch milliseconds offset") from exc
    if split_ms <= 0:
        raise ValueError("splitMs must be positive")

    with db.connect() as conn:
        meeting = (
            conn.execute(text("SELECT * FROM meetings WHERE id = :id"), {"id": meeting_id}).mappings().first()
        )
        if not meeting:
            raise ValueError("Meeting not found")
        if str(meeting["status"]) == "processing":
            raise ValueError("Cannot split while transcription is running")
        recording = (
            conn.execute(
                text("SELECT * FROM recordings WHERE meeting_id = :id ORDER BY created_at ASC LIMIT 1"),
                {"id": meeting_id},
            )
            .mappings()
            .first()
        )
        if not recording:
            raise ValueError("Meeting has no recording to split")
        transcript_artifacts = (
            conn.execute(
                text(
                    "SELECT * FROM artifacts WHERE meeting_id = :id AND kind = 'transcript' "
                    "AND format = 'json' ORDER BY created_at ASC"
                ),
                {"id": meeting_id},
            )
            .mappings()
            .all()
        )
        speaker_links = (
            conn.execute(
                text("SELECT * FROM meeting_speakers WHERE meeting_id = :id"),
                {"id": meeting_id},
            )
            .mappings()
            .all()
        )

    meeting = dict(meeting)
    recording = dict(recording)
    folder = Path(
        meeting_paths(meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]).folder
    )
    src_audio = folder / recording["path"]
    if not src_audio.exists():
        raise ValueError("Audio file not found on disk")

    total_ms = recording.get("duration_ms")
    total_ms = int(total_ms) if total_ms is not None else None
    probed = _probe_duration_ms(src_audio)
    if probed:
        total_ms = probed
    if total_ms is None:
        raise ValueError(
            "Audio duration unknown; transcribe the meeting first so a split point can be validated"
        )

    if split_ms < MIN_EDGE_MARGIN_MS or split_ms > total_ms - MIN_EDGE_MARGIN_MS:
        raise ValueError(
            f"splitMs must be between {MIN_EDGE_MARGIN_MS}ms and {total_ms - MIN_EDGE_MARGIN_MS}ms "
            f"(audio is {total_ms}ms)"
        )

    # --- Cut audio into temp files first (ffmpeg may fail; keep originals intact).
    tmp_a = src_audio.parent / f".split_a_{uuid.uuid4().hex[:8]}{src_audio.suffix}"
    tmp_b = src_audio.parent / f".split_b_{uuid.uuid4().hex[:8]}{src_audio.suffix}"
    try:
        _ffmpeg_cut(src_audio, tmp_a, 0, split_ms)
        _ffmpeg_cut(src_audio, tmp_b, split_ms, None)
        bytes_a = tmp_a.read_bytes()
        bytes_b = tmp_b.read_bytes()
    finally:
        for p in (tmp_a, tmp_b):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
    if not bytes_a or not bytes_b:
        raise RuntimeError("ffmpeg split produced empty audio")

    now = int(time.time() * 1000)
    orig_start = int(meeting["start_time"])
    orig_end = int(meeting["end_time"])
    part_b_start = orig_start + split_ms
    # Preserve a zero-duration meeting's end; otherwise Part 1 ends at the cut.
    part_a_end = orig_start + split_ms if orig_end > orig_start else orig_end
    part_b_end = orig_end if orig_end >= part_b_start else part_b_start
    new_meeting_id = str(uuid.uuid4())
    new_title = f"{meeting['title']} (Part 2)"
    new_recording_id = str(uuid.uuid4())
    ext = src_audio.suffix.lstrip(".").lower() or "m4a"

    # --- Split transcript payloads in memory before touching disk/DB.
    split_payloads: dict[str, tuple[Any, Any, str]] = {}  # artifact_id -> (partA, partB, rel_path)
    for art in transcript_artifacts:
        art = dict(art)
        try:
            raw = json.loads((folder / art["path"]).read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            payload_a, payload_b = split_transcript_payload(raw, split_ms)
        except ValueError:
            continue
        split_payloads[art["id"]] = (payload_a, payload_b, art["path"])

    # --- Write Part A audio + transcripts.
    new_folder = Path(meeting_paths(meetings_dir, part_b_start, new_title, new_meeting_id).folder)
    ensure_meeting_folder(meeting_paths(meetings_dir, part_b_start, new_title, new_meeting_id))
    part_b_rel = f"audio/{new_recording_id}.{ext}"
    (new_folder / part_b_rel).write_bytes(bytes_b)
    src_audio.write_bytes(bytes_a)
    # The source audio was rewritten in place: drop the cached voice-enhanced
    # derivative or the audio route keeps serving the pre-split full recording.
    _delete_stale_enhanced_files(folder, str(recording["id"]))
    sha_a = hashlib.sha256(bytes_a).hexdigest()
    sha_b = hashlib.sha256(bytes_b).hexdigest()

    for _art_id, (payload_a, _payload_b, rel_path) in split_payloads.items():
        (folder / rel_path).write_text(json.dumps(payload_a, indent=2) + "\n", encoding="utf-8")
        # Refresh the plain-text twin alongside the canonical JSON.
        txt_twin = Path(rel_path).with_suffix(".txt")
        twin_full = folder / str(txt_twin)
        if twin_full.exists() or (folder / rel_path).name.endswith(".json"):
            try:
                twin_full.write_text(_transcript_to_text(payload_a) + "\n", encoding="utf-8")
            except Exception:
                pass

    # --- Write Part B transcripts.
    new_artifact_ids: dict[str, str] = {}
    for art in transcript_artifacts:
        art = dict(art)
        entry = split_payloads.get(art["id"])
        if entry is None:
            continue
        _payload_a, payload_b, rel_path = entry
        dest = new_folder / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload_b, indent=2) + "\n", encoding="utf-8")
        txt_twin = Path(rel_path).with_suffix(".txt")
        try:
            (new_folder / str(txt_twin)).write_text(_transcript_to_text(payload_b) + "\n", encoding="utf-8")
        except Exception:
            pass

    # --- DB updates.
    with db.begin() as conn:
        # Drop summaries on the original (stale: they describe the whole).
        summary_rows = (
            conn.execute(
                text("SELECT * FROM artifacts WHERE meeting_id = :id AND kind = 'summary'"),
                {"id": meeting_id},
            )
            .mappings()
            .all()
        )
        for s in summary_rows:
            try:
                (folder / s["path"]).unlink(missing_ok=True)
            except Exception:
                pass
        conn.execute(
            text("DELETE FROM artifacts WHERE meeting_id = :id AND kind = 'summary'"), {"id": meeting_id}
        )
        # Insights signals and moments describe the whole meeting; they are stale
        # after the cut and are recomputed per part.
        for tbl in ("meeting_speaker_signals", "meeting_moments"):
            conn.execute(text(f"DELETE FROM {tbl} WHERE meeting_id = :id"), {"id": meeting_id})
        conn.execute(
            text(
                "UPDATE meetings SET end_time = :end, primary_summary_artifact_id = NULL, "
                "updated_at = :now WHERE id = :id"
            ),
            {"end": part_a_end, "now": now, "id": meeting_id},
        )
        conn.execute(
            text(
                "UPDATE recordings SET size_bytes = :size, sha256 = :sha, duration_ms = :dur WHERE id = :id"
            ),
            {"size": len(bytes_a), "sha": sha_a, "dur": split_ms, "id": recording["id"]},
        )

        # Retarget .txt twins already on disk (rows may not exist for all).
        for art in transcript_artifacts:
            art = dict(art)
            entry = split_payloads.get(art["id"])
            if entry is None:
                continue
            _pa, _pb, rel_path = entry
            txt_rel = str(Path(rel_path).with_suffix(".txt"))
            conn.execute(
                text("UPDATE artifacts SET recording_id = :rid WHERE id = :id"),
                {"rid": recording["id"], "id": art["id"]},
            )
            _ = txt_rel

        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, :title, :start, :end, :source, 'ready', :tags,
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {
                "id": new_meeting_id,
                "title": new_title,
                "start": part_b_start,
                "end": part_b_end,
                "source": meeting["source"],
                "tags": meeting.get("tags") or "[]",
                "now": now,
            },
        )
        conn.execute(
            text(
                """
                INSERT INTO recordings (
                  id, meeting_id, path, mime, duration_ms, size_bytes, sha256,
                  provider, provider_recording_id, created_at
                ) VALUES (
                  :id, :mid, :path, :mime, :dur, :size, :sha, :prov, NULL, :now
                )
                """
            ),
            {
                "id": new_recording_id,
                "mid": new_meeting_id,
                "path": part_b_rel,
                "mime": recording["mime"],
                "dur": total_ms - split_ms,
                "size": len(bytes_b),
                "sha": sha_b,
                "prov": recording["provider"],
                "now": now,
            },
        )
        primary_b: str | None = None
        for art in transcript_artifacts:
            art = dict(art)
            entry = split_payloads.get(art["id"])
            if entry is None:
                continue
            _pa, _pb, rel_path = entry
            new_art_id = str(uuid.uuid4())
            new_artifact_ids[art["id"]] = new_art_id
            conn.execute(
                text(
                    """
                    INSERT INTO artifacts (
                      id, meeting_id, recording_id, kind, provider, format, path, created_at
                    ) VALUES (:id, :mid, :rid, 'transcript', :prov, 'json', :path, :now)
                    """
                ),
                {
                    "id": new_art_id,
                    "mid": new_meeting_id,
                    "rid": new_recording_id,
                    "prov": art["provider"],
                    "path": rel_path,
                    "now": now,
                },
            )
            if meeting.get("primary_transcript_artifact_id") == art["id"] and primary_b is None:
                primary_b = new_art_id
        if primary_b:
            conn.execute(
                text("UPDATE meetings SET primary_transcript_artifact_id = :aid WHERE id = :id"),
                {"aid": primary_b, "id": new_meeting_id},
            )
        for link in speaker_links:
            link = dict(link)
            new_evidence = new_artifact_ids.get(str(link.get("evidence_artifact_id") or ""))
            conn.execute(
                text(
                    "INSERT INTO meeting_speakers (meeting_id, speaker_id, evidence_artifact_id) "
                    "VALUES (:mid, :sid, :eid)"
                ),
                {"mid": new_meeting_id, "sid": link["speaker_id"], "eid": new_evidence},
            )

        updated_a = (
            conn.execute(text("SELECT * FROM meetings WHERE id = :id"), {"id": meeting_id}).mappings().first()
        )
        created_b = (
            conn.execute(text("SELECT * FROM meetings WHERE id = :id"), {"id": new_meeting_id})
            .mappings()
            .first()
        )

    assert updated_a and created_b
    return {"meeting": to_meeting(updated_a).model_dump(), "newMeeting": to_meeting(created_b).model_dump()}
