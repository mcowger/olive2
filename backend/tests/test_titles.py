"""Auto-title generation service and API tests."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import text

from olive.app import create_app
from olive.db import create_db
from olive.layout import meeting_paths
from olive.services.llm import LlmService
from olive.services.titles import (
    AUTO_TITLE_MAX_LENGTH,
    TitleService,
    sanitize_generated_title,
)


def _create_test_context(tmp_path: Path, llm_text: str = '"Quarterly Planning Review"'):
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    Path(meetings_dir).mkdir(parents=True, exist_ok=True)

    llm_service = LlmService(db=handle.engine, config_dir=str(tmp_path / "config"))
    captured: dict[str, Any] = {}

    async def fake_generate_text(options: dict[str, Any]) -> dict[str, Any]:
        captured.update(options)
        return {
            "text": llm_text,
            "provider": options.get("provider") or "google",
            "model": options.get("model") or "gemini-2.5-flash",
            "durationMs": 120,
            "usage": {"inputTokens": 400, "outputTokens": 8, "totalTokens": 408},
        }

    llm_service.generate_text = fake_generate_text  # type: ignore[method-assign]
    llm_service.generateText = fake_generate_text  # type: ignore[assignment]

    title_service = TitleService(
        db=handle.engine,
        meetings_dir=meetings_dir,
        llm_service=llm_service,
    )
    return handle, meetings_dir, title_service, captured


def _seed_meeting_with_transcript(engine, meetings_dir: str, title: str = "Untitled Meeting") -> str:
    now = int(time.time() * 1000)
    meeting_id = "meeting-autotitle-1"

    paths = meeting_paths(meetings_dir, now, title, meeting_id)
    Path(paths.transcripts_dir).mkdir(parents=True, exist_ok=True)

    transcript = {
        "segments": [
            {
                "startMs": 0,
                "endMs": 5000,
                "speaker": "Alice",
                "text": "Welcome everyone. Let's finalize the quarterly plan and hiring goals.",
            },
            {
                "startMs": 5200,
                "endMs": 12000,
                "speaker": "Bob",
                "text": "We agreed to open two backend roles and freeze the roadmap in June.",
            },
        ]
    }
    transcript_rel = "transcripts/transcript.json"
    (Path(paths.folder) / transcript_rel).write_text(json.dumps(transcript), encoding="utf-8")
    transcript_art_id = "artifact-transcript-autotitle-1"

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, :title, :now, :end, 'upload', 'ready', '[]',
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {"id": meeting_id, "title": title, "now": now, "end": now + 600_000},
        )
        conn.execute(
            text(
                """
                INSERT INTO artifacts (
                  id, meeting_id, recording_id, kind, provider, format, path, created_at
                ) VALUES (
                  :aid, :mid, NULL, 'transcript', 'speechmatics', 'json', :path, :now
                )
                """
            ),
            {"aid": transcript_art_id, "mid": meeting_id, "path": transcript_rel, "now": now},
        )
        conn.execute(
            text("UPDATE meetings SET primary_transcript_artifact_id = :aid WHERE id = :mid"),
            {"aid": transcript_art_id, "mid": meeting_id},
        )

    return meeting_id


def test_sanitize_generated_title_strips_quotes_and_truncates() -> None:
    assert sanitize_generated_title('  "Quarterly Planning Review"  ') == "Quarterly Planning Review"
    assert sanitize_generated_title("**Sprint Retro**\nSecond line here") == "Sprint Retro"
    assert sanitize_generated_title("") == ""
    long_title = sanitize_generated_title("Word " * 50)
    assert len(long_title) <= AUTO_TITLE_MAX_LENGTH
    assert "\n" not in long_title


async def test_generate_title_uses_default_llm_and_sanitizes(tmp_path: Path) -> None:
    handle, meetings_dir, title_service, captured = _create_test_context(tmp_path)
    meeting_id = _seed_meeting_with_transcript(handle.engine, meetings_dir)

    res = await title_service.generate_title(meeting_id)

    # None provider/model falls through to the configured default LLM.
    assert captured.get("provider") is None
    assert captured.get("model") is None
    assert captured.get("maxTokens") is None
    assert captured.get("temperature") is None
    assert res["title"] == "Quarterly Planning Review"
    assert res["provider"] == "google"


async def test_generate_and_apply_persists_title_and_renames_folder(tmp_path: Path) -> None:
    handle, meetings_dir, title_service, _ = _create_test_context(tmp_path, llm_text="Hiring Plan Finalized")
    meeting_id = _seed_meeting_with_transcript(handle.engine, meetings_dir)
    old_folder = meeting_paths(meetings_dir, int(time.time() * 1000), "Untitled Meeting", meeting_id)

    res = await title_service.generate_and_apply(meeting_id)

    assert res["title"] == "Hiring Plan Finalized"
    assert res["meeting"]["title"] == "Hiring Plan Finalized"
    with handle.engine.connect() as conn:
        row = (
            conn.execute(text("SELECT title FROM meetings WHERE id = :id"), {"id": meeting_id})
            .mappings()
            .first()
        )
    assert row and row["title"] == "Hiring Plan Finalized"
    assert not Path(old_folder.folder).exists()


async def test_generate_title_without_transcript_raises(tmp_path: Path) -> None:
    handle, meetings_dir, title_service, _ = _create_test_context(tmp_path)
    now = int(time.time() * 1000)
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  'meeting-no-transcript', 'Empty', :now, :end, 'upload', 'ready', '[]',
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {"now": now, "end": now + 600_000},
        )

    try:
        await title_service.generate_title("meeting-no-transcript")
    except ValueError as exc:
        assert "transcript" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for missing transcript")


def test_auto_title_route_uses_injected_service(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    Path(meetings_dir).mkdir(parents=True, exist_ok=True)

    class StubTitleService:
        async def generate_and_apply(self, meeting_id: str, **kwargs: Any) -> dict[str, Any]:
            return {
                "meeting": {"id": meeting_id, "title": "Stubbed Title"},
                "title": "Stubbed Title",
                "provider": "google",
                "model": "gemini-2.5-flash",
            }

        async def generateAndApply(self, meeting_id: str, **kwargs: Any) -> dict[str, Any]:
            return await self.generate_and_apply(meeting_id, **kwargs)

    app = create_app({"db": handle.engine, "meetingsDir": meetings_dir, "titleService": StubTitleService()})
    client = TestClient(app)
    res = client.post("/api/meetings/abc123/auto-title", json={})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["title"] == "Stubbed Title"
    assert body["meeting"]["title"] == "Stubbed Title"
