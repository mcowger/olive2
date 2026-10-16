"""Model Context Protocol (MCP) server and tools (port of packages/server/src/mcp/tools.ts)."""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from mcp.server.mcpserver import MCPServer
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .layout import meeting_paths
from .services.meetings import (
    delete_meeting,
    get_meeting,
    list_meetings,
    list_tags,
    update_meeting_start_time,
    update_meeting_tags,
    update_meeting_title,
)

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}

_WRITE_TRUTHY = {"1", "true", "yes", "on", "y"}


def is_mcp_write_enabled(explicit: bool | None = None) -> bool:
    """Whether MCP write tools are enabled. Off by default; opt in with OLIVE_MCP_WRITE=1."""
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("OLIVE_MCP_WRITE", "").strip().lower() in _WRITE_TRUTHY


def _write_disabled_error() -> dict[str, Any]:
    return _error_result(
        "Write tools are disabled. Set OLIVE_MCP_WRITE=1 on the server to enable "
        "meeting and speaker edits, tags, titles, transcription, and summary tools."
    )


def _normalize_tag_list_for_mcp(value: Any) -> list[str] | None:
    if not isinstance(value, (list, tuple)):
        return None
    seen: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        s = item.strip()
        if s and s not in seen:
            seen.append(s)
    return seen


def _text_result(value: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(value, indent=2)}]}


def _error_result(error: Any) -> dict[str, Any]:
    return {
        "isError": True,
        "content": [{"type": "text", "text": str(error)}],
    }


def _parse_summary_action_items(summary: str | None) -> list[dict[str, Any]]:
    if not summary:
        return []
    out: list[dict[str, Any]] = []
    for line in summary.splitlines():
        m = re.match(r"^\s*[-*]\s+\[([ xX])\]\s+(.+?)\s*$", line)
        if m:
            out.append(
                {
                    "text": m.group(2),
                    "completed": m.group(1).lower() == "x",
                }
            )
    return out


def _load_transcript_text(db: Engine, meetings_dir: str, meeting: dict[str, Any]) -> str:
    with db.connect() as conn:
        artifacts = (
            conn.execute(
                text(
                    """
                SELECT path, format FROM artifacts
                WHERE meeting_id = :mid AND kind = 'transcript'
                ORDER BY created_at DESC
                """
                ),
                {"mid": meeting["id"]},
            )
            .mappings()
            .all()
        )

    art = next((a for a in artifacts if a["format"] == "txt"), None) or (artifacts[0] if artifacts else None)
    if not art:
        return ""
    try:
        folder = Path(
            meeting_paths(meetings_dir, int(meeting["startTime"]), meeting["title"], meeting["id"]).folder
        )
        return (folder / art["path"]).read_text(encoding="utf-8")
    except Exception:
        return ""


def _transcript_snippet(text_content: str, query: str) -> str | None:
    idx = text_content.lower().find(query.lower())
    if idx < 0:
        return None
    start = max(0, idx - 160)
    end = min(len(text_content), idx + len(query) + 240)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text_content) else ""
    return f"{prefix}{text_content[start:end].strip()}{suffix}"


def _list_all_meetings(db: Engine, meetings_dir: str, query: dict[str, Any]) -> list[dict[str, Any]]:
    meetings: list[dict[str, Any]] = []
    offset = 0
    while True:
        resp = list_meetings(db, {**query, "limit": 100, "offset": offset}, meetings_dir)
        batch = resp["meetings"]
        meetings.extend(batch)
        total = int(resp["pagination"]["total"])
        offset += len(batch)
        if offset >= total or not batch:
            break
    return meetings


_MEETING_FILTER_PROPERTIES: dict[str, Any] = {
    "search": {"type": "string", "description": "Search meeting titles, tags, transcripts, and summaries"},
    "tags": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Exact tag match (case-insensitive, AND-combined). Unlike search, never matches transcript text.",
    },
    "source": {"type": "string", "enum": ["plaud", "upload", "ios-shortcut"]},
    "status": {"type": "string", "description": "Meeting status (e.g. ready, processing, error)"},
    "speakerId": {"type": "string", "description": "Only meetings linked to this speaker"},
    "hasSummary": {"type": "boolean", "description": "Only meetings with/without a summary"},
    "hasTranscript": {"type": "boolean", "description": "Only meetings with/without a transcript"},
    "startTime": {"type": "number"},
    "endTime": {"type": "number"},
    "limit": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "description": "Page size (default 50, max 100). Results are newest-first.",
    },
    "offset": {
        "type": "integer",
        "minimum": 0,
        "description": "Items to skip. Use pagination.nextOffset while pagination.hasMore is true to page through all results.",
    },
}


# `search_transcripts` already has `query` and `get_action_items` scopes by
# meeting selection: giving them a second free-text `search` field would leave
# agents guessing which one scopes what, so both tools get every meeting
# filter EXCEPT `search`. `list_meetings` keeps the full set.
_MEETING_FILTER_PROPERTIES_NO_SEARCH: dict[str, Any] = {
    k: v for k, v in _MEETING_FILTER_PROPERTIES.items() if k != "search"
}


def _pagination_hints(limit: int, offset: int, total: int) -> dict[str, Any]:
    """Build a pagination envelope with follow-up hints.

    ``hasMore``/``nextOffset`` tell REST and MCP clients exactly how to
    fetch the next page (``offset=nextOffset`` while ``hasMore`` is true).
    """
    end = offset + min(max(limit, 0), max(total - offset, 0))
    has_more = end < total
    return {
        "limit": limit,
        "offset": offset,
        "total": total,
        "hasMore": has_more,
        "nextOffset": end if has_more else None,
    }


def _meeting_filter_args(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "search": arguments.get("search"),
        "tags": arguments.get("tags"),
        "source": arguments.get("source"),
        "status": arguments.get("status"),
        "speakerId": arguments.get("speakerId"),
        "hasSummary": arguments.get("hasSummary"),
        "hasTranscript": arguments.get("hasTranscript"),
        "startTime": arguments.get("startTime"),
        "endTime": arguments.get("endTime"),
        "limit": arguments.get("limit"),
        "offset": arguments.get("offset"),
    }


MCP_TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "list_meetings",
        "description": (
            "List Olive meetings (newest-first) with optional text, tag, speaker, status, and date filters. "
            "Paginated: limit defaults to 50 (max 100); response includes "
            "pagination { limit, offset, total, hasMore, nextOffset }. "
            "When hasMore is true, call again with offset=nextOffset until all results are retrieved."
        ),
        "inputSchema": {
            "type": "object",
            "properties": _MEETING_FILTER_PROPERTIES,
        },
    },
    {
        "name": "list_tags",
        "description": "List distinct meeting tags with meeting counts.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_meeting",
        "description": "Get a complete Olive meeting record, including transcript and summaries.",
        "inputSchema": {
            "type": "object",
            "required": ["meetingId"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
            },
        },
    },
    {
        "name": "search_transcripts",
        "description": (
            "Search transcript text across Olive meetings and return matching snippets. "
            "Paginated: limit defaults to 50; response includes "
            "pagination { limit, offset, total, hasMore, nextOffset }. "
            "When hasMore is true, call again with offset=nextOffset."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Text to search for"},
                **_MEETING_FILTER_PROPERTIES_NO_SEARCH,
            },
        },
    },
    {
        "name": "get_action_items",
        "description": (
            "Extract checklist-style action items from Olive meeting summaries. "
            "Paginated: limit defaults to 50; response includes "
            "pagination { limit, offset, total, hasMore, nextOffset }. "
            "When hasMore is true, call again with offset=nextOffset."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "meetingId": {"type": "string"},
                "includeCompleted": {"type": "boolean", "default": True},
                **_MEETING_FILTER_PROPERTIES_NO_SEARCH,
            },
        },
    },
    {
        "name": "list_speakers",
        "description": "List Olive speaker profiles and meeting counts.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_speaker_profile",
        "description": "Get a speaker profile and associated meetings.",
        "inputSchema": {
            "type": "object",
            "required": ["speakerId"],
            "properties": {
                "speakerId": {"type": "string", "description": "Speaker ID"},
            },
        },
    },
    {
        "name": "export_speaker_library",
        "description": (
            "Export the enrolled speaker library as a portable JSON object "
            "(voiceprint vectors + metadata, no audio). Copy the returned "
            "`library` value to another Olive instance via import_speaker_library "
            "or POST /api/speakers/import to share voiceprints without re-enrolling."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "speakerIds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional subset of speaker IDs (default: all enrolled speakers)",
                },
            },
        },
    },
    {
        "name": "list_templates",
        "description": "List summary prompt templates (built-in and custom).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_template",
        "description": "Get a summary prompt template by ID.",
        "inputSchema": {
            "type": "object",
            "required": ["templateId"],
            "properties": {
                "templateId": {"type": "string", "description": "Template ID"},
            },
        },
    },
    {
        "name": "get_transcription_status",
        "description": (
            "Get transcription progress for a meeting (stage, percent, message). "
            "Poll this instead of the SSE stream; mirrors GET /api/meetings/{id}/transcribe/progress."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
            },
        },
    },
    {
        "name": "list_active_transcriptions",
        "description": (
            "List currently active transcription jobs across all meetings "
            "(mirrors GET /api/transcriptions/active)."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_summary_status",
        "description": (
            "Get summary generation status for a meeting (mirrors GET /api/meetings/{id}/summarize/status)."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
            },
        },
    },
    {
        "name": "plaud_status",
        "description": "Get Plaud Cloud sync status (connected, lastPollAt, pcsPending).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

MCP_WRITE_TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "update_speaker",
        "description": (
            "Rename an Olive speaker. Requires OLIVE_MCP_WRITE=1. "
            "Pass dryRun=true to preview without mutating; pass expectedName to abort "
            "unless the live name still matches."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["speakerId", "newName"],
            "properties": {
                "speakerId": {"type": "string", "description": "Speaker ID"},
                "newName": {"type": "string", "description": "New full name"},
                "expectedName": {
                    "type": "string",
                    "description": "Optional current name that must match live data or the operation aborts",
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, return the plan without mutating",
                },
            },
        },
    },
    {
        "name": "merge_speakers",
        "description": (
            "Merge the source speaker into the target speaker (moves meeting links, "
            "voiceprints, and clips, then deletes the source). Requires OLIVE_MCP_WRITE=1. "
            "Pass dryRun=true to preview; pass expectedSourceName/expectedTargetName to abort "
            "on mismatch."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["sourceSpeakerId", "targetSpeakerId"],
            "properties": {
                "sourceSpeakerId": {"type": "string", "description": "Speaker ID to merge away"},
                "targetSpeakerId": {"type": "string", "description": "Surviving speaker ID"},
                "expectedSourceName": {
                    "type": "string",
                    "description": "Optional live source name that must match or the merge aborts",
                },
                "expectedTargetName": {
                    "type": "string",
                    "description": "Optional live target name that must match or the merge aborts",
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, return the plan without mutating",
                },
            },
        },
    },
    {
        "name": "delete_speaker",
        "description": (
            "Delete an Olive speaker and its meeting links. Requires OLIVE_MCP_WRITE=1. "
            "Pass dryRun=true to preview; pass expectedName to abort unless the live name matches."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["speakerId"],
            "properties": {
                "speakerId": {"type": "string", "description": "Speaker ID"},
                "expectedName": {
                    "type": "string",
                    "description": "Optional current name that must match live data or the delete aborts",
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, return the plan without mutating",
                },
            },
        },
    },
    {
        "name": "set_meeting_tags",
        "description": (
            "Replace a meeting's tags. Requires OLIVE_MCP_WRITE=1. "
            "Pass dryRun=true to preview; pass expectedTags to abort unless the live tags "
            "still match (order-insensitive, single-meeting only). Pass meetingIds for bulk "
            "tagging (up to 500 meetings per call); bulk does not accept expectedTags."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["tags"],
            "properties": {
                "meetingId": {"type": "string", "description": "Single meeting ID"},
                "meetingIds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Bulk meeting IDs (alternative to meetingId, up to 500)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Replacement tag list (may be empty to clear)",
                },
                "expectedTags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional live tags that must match or the update aborts (single-meeting only)",
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, return the plan without mutating",
                },
            },
        },
    },
    {
        "name": "update_meeting_title",
        "description": (
            "Rename a meeting title (also renames its on-disk folder). Requires OLIVE_MCP_WRITE=1. "
            "Pass dryRun=true to preview; pass expectedTitle to abort unless the live title matches."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId", "newTitle"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
                "newTitle": {"type": "string", "description": "New meeting title"},
                "expectedTitle": {
                    "type": "string",
                    "description": "Optional live title that must match or the rename aborts",
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, return the plan without mutating",
                },
            },
        },
    },
    {
        "name": "transcribe_meeting",
        "description": (
            "Queue (re)transcription for one meeting on its provider queue (async, returns jobId). "
            "Requires OLIVE_MCP_WRITE=1. Pass dryRun=true to validate the meeting/recording without queueing."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
                "provider": {
                    "type": "string",
                    "enum": ["local", "speechmatics", "assemblyai", "muse"],
                    "description": "Transcription engine (defaults to server default)",
                },
                "language": {"type": "string", "description": "BCP-47 language hint (e.g. en)"},
                "force": {
                    "type": "boolean",
                    "default": True,
                    "description": "Re-transcribe even when a transcript already exists",
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, validate without queueing",
                },
            },
        },
    },
    {
        "name": "bulk_transcribe",
        "description": (
            "Queue re-transcription for many meetings with one engine (async, up to 2000). "
            "Requires OLIVE_MCP_WRITE=1. Provide meetingIds or meeting filters; "
            "dryRun=true resolves the target list without queueing."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["provider"],
            "properties": {
                "provider": {
                    "type": "string",
                    "enum": ["local", "speechmatics", "assemblyai", "muse"],
                    "description": "Transcription engine for all targets",
                },
                "meetingIds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit meeting IDs (alternative to filters)",
                },
                "search": {"type": "string", "description": "Filter: text search"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter: exact tags (AND-combined)",
                },
                "status": {"type": "string", "description": "Filter: meeting status"},
                "speakerId": {"type": "string", "description": "Filter: speaker ID"},
                "hasSummary": {"type": "boolean", "description": "Filter: with/without summary"},
                "hasTranscript": {
                    "type": "boolean",
                    "description": "Filter: with/without transcript",
                },
                "startTime": {"type": "number", "description": "Filter: start_time >= (epoch ms)"},
                "endTime": {"type": "number", "description": "Filter: start_time <= (epoch ms)"},
                "force": {
                    "type": "boolean",
                    "default": True,
                    "description": "Re-transcribe even when transcripts exist",
                },
                "language": {"type": "string", "description": "BCP-47 language hint"},
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, resolve targets without queueing",
                },
            },
        },
    },
    {
        "name": "cancel_transcription",
        "description": (
            "Cancel queued/running transcription for a meeting. Requires OLIVE_MCP_WRITE=1. "
            "Pass dryRun=true to preview without cancelling."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, return the plan without mutating",
                },
            },
        },
    },
    {
        "name": "retry_meeting_audio",
        "description": (
            "Re-queue Plaud Cloud audio download for a meeting (async, returns 202-style status). "
            "Requires OLIVE_MCP_WRITE=1. Pass dryRun=true to validate without queueing."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, validate without queueing",
                },
            },
        },
    },
    {
        "name": "reidentify_speakers",
        "description": (
            "Re-match a meeting's transcript speaker labels against enrolled voiceprints "
            "(async queued job). Requires OLIVE_MCP_WRITE=1. Pass dryRun=true to validate without queueing."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, validate without queueing",
                },
            },
        },
    },
    {
        "name": "delete_meeting",
        "description": (
            "Delete a meeting, its recordings, artifacts, and on-disk folder. Requires OLIVE_MCP_WRITE=1. "
            "Pass dryRun=true to preview; pass expectedTitle to abort unless the live title matches."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
                "expectedTitle": {
                    "type": "string",
                    "description": "Optional live title that must match or the delete aborts",
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, return the plan without mutating",
                },
            },
        },
    },
    {
        "name": "update_meeting_start_time",
        "description": (
            "Shift a meeting's start time (duration preserved, on-disk folder moved when the date "
            "changes). Requires OLIVE_MCP_WRITE=1. Pass dryRun=true to preview; pass "
            "expectedStartTime to abort unless the live start time matches."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId", "newStartTime"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
                "newStartTime": {
                    "type": "number",
                    "description": "New start time (epoch milliseconds)",
                },
                "expectedStartTime": {
                    "type": "number",
                    "description": "Optional live start time that must match or the update aborts",
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, return the plan without mutating",
                },
            },
        },
    },
    {
        "name": "auto_title_meeting",
        "description": (
            "Generate a concise title from the transcript with the default LLM and persist it "
            "(renames the on-disk folder). Requires OLIVE_MCP_WRITE=1. Pass dryRun=true to "
            "validate without calling the LLM."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
                "provider": {"type": "string", "description": "LLM provider override"},
                "model": {"type": "string", "description": "LLM model override"},
                "expectedTitle": {
                    "type": "string",
                    "description": "Optional live title that must match or the generation aborts",
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, validate without calling the LLM",
                },
            },
        },
    },
    {
        "name": "summarize_meeting",
        "description": (
            "Queue LLM summary generation for a meeting (async, returns generation status). "
            "Requires OLIVE_MCP_WRITE=1. Pass dryRun=true to validate without queueing."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
                "templateId": {"type": "string", "description": "Summary template ID"},
                "provider": {"type": "string", "description": "LLM provider override"},
                "model": {"type": "string", "description": "LLM model override"},
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, validate without queueing",
                },
            },
        },
    },
    {
        "name": "delete_summary",
        "description": (
            "Delete an individual meeting summary artifact (file + DB row; when the primary "
            "summary is deleted the newest remaining summary becomes primary). "
            "Requires OLIVE_MCP_WRITE=1. Pass dryRun=true to preview; pass expectedIsPrimary "
            "to abort unless the target's primary flag still matches."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId", "summaryId"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
                "summaryId": {
                    "type": "string",
                    "description": "Summary artifact ID (summaries[].id from get_meeting; artifactId accepted as an alias)",
                },
                "expectedIsPrimary": {
                    "type": "boolean",
                    "description": "Optional live primary flag that must match or the delete aborts",
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, return the plan without mutating",
                },
            },
        },
    },
    {
        "name": "set_primary_summary",
        "description": (
            "Change which summary is the meeting's primary (shown as summaryContent). "
            "Requires OLIVE_MCP_WRITE=1. Pass dryRun=true to preview; pass "
            "expectedPrimarySummaryId to abort unless the live primary still matches."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId", "summaryId"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
                "summaryId": {
                    "type": "string",
                    "description": "Summary artifact ID to promote (summaries[].id from get_meeting; artifactId accepted as an alias)",
                },
                "expectedPrimarySummaryId": {
                    "type": "string",
                    "description": "Optional live primary summary ID that must match or the update aborts",
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, return the plan without mutating",
                },
            },
        },
    },
    {
        "name": "import_speaker_library",
        "description": (
            "Import a portable speaker library produced by export_speaker_library "
            "(voiceprint vectors, no audio). Requires OLIVE_MCP_WRITE=1. "
            "mode='merge' unions vectors into same-name speakers and creates the rest; "
            "mode='replace' wipes the local library first (meeting attributions carry "
            "over by speaker name). Pass dryRun=true to preview without mutating."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["library"],
            "properties": {
                "library": {
                    "type": "object",
                    "description": "Speaker library object from export_speaker_library",
                },
                "mode": {
                    "type": "string",
                    "enum": ["merge", "replace"],
                    "default": "merge",
                    "description": "merge unions into the local library; replace wipes it first",
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, return the plan without mutating",
                },
            },
        },
    },
    {
        "name": "reassign_speaker",
        "description": (
            "Move all transcript segments labelled `fromLabel` in one meeting to a target "
            "speaker (by name or id). Optionally adopt a fresh voiceprint from those segments. "
            "Requires OLIVE_MCP_WRITE=1. Pass dryRun=true to preview; "
            "pass expectedFromLabel to abort unless the source label still exists."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["meetingId", "fromLabel"],
            "properties": {
                "meetingId": {"type": "string", "description": "Meeting ID"},
                "fromLabel": {
                    "type": "string",
                    "description": "Source speaker label in the transcript (e.g. 'Speaker 1')",
                },
                "toSpeakerName": {
                    "type": "string",
                    "description": "Target speaker display name (creates if missing; exclusive with toSpeakerId)",
                },
                "toSpeakerId": {
                    "type": "string",
                    "description": "Existing target speaker ID (exclusive with toSpeakerName)",
                },
                "adoptVoiceprint": {
                    "type": "boolean",
                    "description": "Adopt a voiceprint sample from the matched segments (default true)",
                    "default": True,
                },
                "localOnly": {
                    "type": "boolean",
                    "description": "Only rename in this meeting's transcript, don't grow the speaker library",
                    "default": False,
                },
                "segmentIndex": {
                    "type": "integer",
                    "description": "When set with scope='single', reassign only this segment index",
                },
                "scope": {
                    "type": "string",
                    "enum": ["all", "single"],
                    "description": "all (default) reassigns every segment with the label; single reassigns one",
                },
                "expectedFromLabel": {
                    "type": "string",
                    "description": "Optional live label that must still match or the update aborts",
                },
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, return the plan without mutating",
                },
            },
        },
    },
    {
        "name": "plaud_sync_trigger",
        "description": (
            "Trigger a Plaud Cloud poll cycle (background job, returns 202-style status). "
            "Requires OLIVE_MCP_WRITE=1. Pass dryRun=true to validate without triggering."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dryRun": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, validate without triggering",
                }
            },
        },
    },
]

MCP_WRITE_TOOL_NAMES = frozenset(t["name"] for t in MCP_WRITE_TOOLS_SPEC)


def get_mcp_tools_spec(*, write_enabled: bool = False) -> list[dict[str, Any]]:
    """Full tool listing: read tools always, write tools only when explicitly enabled."""
    if write_enabled:
        return [*MCP_TOOLS_SPEC, *MCP_WRITE_TOOLS_SPEC]
    return list(MCP_TOOLS_SPEC)


async def _handle_transcription_mcp_tools(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    db: Engine,
    meetings_dir: str,
    transcription_service: Any | None,
    plaud_poller: Any | None,
) -> dict[str, Any] | None:
    """Queue/cancel transcription work (async). Returns None when tool not handled."""
    if tool_name == "transcribe_meeting":
        tr_mid = str(arguments.get("meetingId") or "").strip()
        tr_provider = arguments.get("provider")
        tr_language = arguments.get("language")
        tr_force = arguments.get("force", True)
        tr_force = True if tr_force is None else bool(tr_force)
        tr_dry = bool(arguments.get("dryRun"))
        if not tr_mid:
            return _error_result("meetingId is required")
        if tr_provider is not None and tr_provider not in ("local", "speechmatics", "assemblyai", "muse"):
            return _error_result("provider must be one of 'local', 'speechmatics', 'assemblyai', 'muse'")
        tr_detail = get_meeting(db, tr_mid, meetings_dir)
        if not tr_detail:
            return _error_result("Meeting not found")
        tr_has_audio = bool(tr_detail.recordings)
        if tr_dry:
            return _text_result(
                {
                    "dryRun": True,
                    "operation": "transcribe_meeting",
                    "meetingId": tr_mid,
                    "meetingTitle": str(tr_detail.meeting.title or ""),
                    "provider": tr_provider,
                    "force": tr_force,
                    "hasRecording": tr_has_audio,
                    "wouldQueue": tr_has_audio,
                }
            )
        if transcription_service is None:
            return _error_result("Transcription service unavailable")
        tr_fn = getattr(transcription_service, "start_transcription", None) or getattr(
            transcription_service, "startTranscription", None
        )
        if not callable(tr_fn):
            return _error_result("Transcription service unavailable")
        tr_opts: dict[str, Any] = {"force": tr_force}
        if tr_provider is not None:
            tr_opts["provider"] = tr_provider
        if isinstance(tr_language, str) and tr_language.strip():
            tr_opts["language"] = tr_language.strip()
        tr_raw: Any = tr_fn(tr_mid, tr_opts)
        tr_started = await tr_raw if hasattr(tr_raw, "__await__") else tr_raw
        if not isinstance(tr_started, dict):
            return _error_result("Transcription service returned an unexpected result")
        return _text_result({"operation": "transcribe_meeting", "meetingId": tr_mid, **tr_started})

    if tool_name == "bulk_transcribe":
        bt_provider = arguments.get("provider")
        bt_force = arguments.get("force", True)
        bt_force = True if bt_force is None else bool(bt_force)
        bt_language = arguments.get("language")
        bt_dry = bool(arguments.get("dryRun"))
        if bt_provider not in ("local", "speechmatics", "assemblyai", "muse"):
            return _error_result("provider must be one of 'local', 'speechmatics', 'assemblyai', 'muse'")
        raw_bt_ids = arguments.get("meetingIds")
        bt_ids: list[str] = []
        if raw_bt_ids is not None:
            if not isinstance(raw_bt_ids, list) or not raw_bt_ids:
                return _error_result("meetingIds must be a non-empty array of meeting id strings")
            for entry in raw_bt_ids:
                if not isinstance(entry, str) or not entry.strip():
                    return _error_result("meetingIds must be a non-empty array of meeting id strings")
                cleaned = entry.strip()
                if cleaned not in bt_ids:
                    bt_ids.append(cleaned)
            if len(bt_ids) > 2000:
                return _error_result("Bulk transcribe supports at most 2000 meetings per request")
        bt_filters: dict[str, Any] = {}
        for key in (
            "search",
            "tags",
            "status",
            "speakerId",
            "hasSummary",
            "hasTranscript",
            "startTime",
            "endTime",
        ):
            if arguments.get(key) is not None:
                bt_filters[key] = arguments.get(key)
        if not bt_ids and not bt_filters:
            return _error_result("Provide meetingIds or at least one meeting filter")
        if bt_dry:
            if bt_ids:
                return _text_result(
                    {
                        "dryRun": True,
                        "operation": "bulk_transcribe",
                        "provider": bt_provider,
                        "mode": "ids",
                        "meetingIds": bt_ids,
                        "wouldQueue": len(bt_ids),
                    }
                )
            bt_candidates = _list_all_meetings(
                db, meetings_dir, {**bt_filters, "limit": None, "offset": None}
            )
            bt_resolved = [m["id"] for m in bt_candidates if m.get("id")]
            return _text_result(
                {
                    "dryRun": True,
                    "operation": "bulk_transcribe",
                    "provider": bt_provider,
                    "mode": "filters",
                    "filters": bt_filters,
                    "meetingIds": bt_resolved,
                    "wouldQueue": len(bt_resolved),
                }
            )
        if transcription_service is None:
            return _error_result("Transcription service unavailable")
        bt_fn = getattr(transcription_service, "start_bulk_transcription", None) or getattr(
            transcription_service, "startBulkTranscription", None
        )
        if not callable(bt_fn):
            return _error_result("Transcription service unavailable")
        bt_opts: dict[str, Any] = {"provider": bt_provider, "force": bt_force}
        if isinstance(bt_language, str) and bt_language.strip():
            bt_opts["language"] = bt_language.strip()
        if bt_ids:
            bt_opts["meetingIds"] = bt_ids
        else:
            bt_opts["filters"] = bt_filters
        bt_raw: Any = bt_fn(bt_opts)
        bt_result = await bt_raw if hasattr(bt_raw, "__await__") else bt_raw
        if not isinstance(bt_result, dict):
            return _error_result("Transcription service returned an unexpected result")
        return _text_result({"operation": "bulk_transcribe", **bt_result})

    if tool_name == "cancel_transcription":
        cx_mid = str(arguments.get("meetingId") or "").strip()
        cx_dry = bool(arguments.get("dryRun"))
        if not cx_mid:
            return _error_result("meetingId is required")
        cx_detail = get_meeting(db, cx_mid, meetings_dir)
        if not cx_detail:
            return _error_result("Meeting not found")
        if cx_dry:
            return _text_result(
                {
                    "dryRun": True,
                    "operation": "cancel_transcription",
                    "meetingId": cx_mid,
                    "wouldCancel": True,
                }
            )
        if transcription_service is None:
            return _error_result("Transcription service unavailable")
        cx_fn = getattr(transcription_service, "cancel_transcription", None) or getattr(
            transcription_service, "cancelTranscription", None
        )
        if not callable(cx_fn):
            return _error_result("Transcription service unavailable")
        cx_raw: Any = cx_fn(cx_mid)
        cx_res = await cx_raw if hasattr(cx_raw, "__await__") else cx_raw
        if not isinstance(cx_res, dict):
            cx_res = {"result": cx_res}
        return _text_result({"operation": "cancel_transcription", "meetingId": cx_mid, **cx_res})

    if tool_name == "retry_meeting_audio":
        ra_mid = str(arguments.get("meetingId") or "").strip()
        ra_dry = bool(arguments.get("dryRun"))
        if not ra_mid:
            return _error_result("meetingId is required")
        ra_detail = get_meeting(db, ra_mid, meetings_dir)
        if not ra_detail:
            return _error_result("Meeting not found")
        if ra_dry:
            return _text_result(
                {
                    "dryRun": True,
                    "operation": "retry_meeting_audio",
                    "meetingId": ra_mid,
                    "wouldRetry": True,
                }
            )
        ra_fn = getattr(plaud_poller, "retry_meeting_audio", None) or getattr(
            plaud_poller, "retryMeetingAudio", None
        )
        if plaud_poller is None or not callable(ra_fn):
            return _error_result("Plaud sync is not configured; cannot retry audio")
        ra_raw: Any = ra_fn(ra_mid)
        ra_res = await ra_raw if hasattr(ra_raw, "__await__") else ra_raw
        if not isinstance(ra_res, dict):
            ra_res = {"result": ra_res}
        return _text_result({"operation": "retry_meeting_audio", "meetingId": ra_mid, **ra_res})

    if tool_name == "reidentify_speakers":
        ri_mid = str(arguments.get("meetingId") or "").strip()
        ri_dry = bool(arguments.get("dryRun"))
        if not ri_mid:
            return _error_result("meetingId is required")
        ri_detail = get_meeting(db, ri_mid, meetings_dir)
        if not ri_detail:
            return _error_result("Meeting not found")
        if ri_dry:
            return _text_result(
                {
                    "dryRun": True,
                    "operation": "reidentify_speakers",
                    "meetingId": ri_mid,
                    "wouldQueue": True,
                }
            )
        if transcription_service is None:
            return _error_result("Transcription service unavailable")
        ri_fn = getattr(transcription_service, "request_speaker_reidentification", None) or getattr(
            transcription_service, "requestSpeakerReidentification", None
        )
        if not callable(ri_fn):
            return _error_result("Transcription service unavailable")
        ri_raw: Any = ri_fn(ri_mid)
        ri_res = await ri_raw if hasattr(ri_raw, "__await__") else ri_raw
        if not isinstance(ri_res, dict):
            return _error_result("Transcription service returned an unexpected result")
        return _text_result({"operation": "reidentify_speakers", "meetingId": ri_mid, **ri_res})

    return None


def _resolve_summary_id(arguments: dict[str, Any]) -> str:
    """Summary artifact id from MCP args (accepts summaryId, artifactId alias)."""
    for key in ("summaryId", "artifactId", "summaryArtifactId"):
        raw = arguments.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


def _fetch_summary_state(db: Engine, meeting_id: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Authoritative summary list + live primary id from the DB."""
    with db.connect() as conn:
        meeting_row = (
            conn.execute(
                text("SELECT primary_summary_artifact_id FROM meetings WHERE id = :id"),
                {"id": meeting_id},
            )
            .mappings()
            .first()
        )
        rows = (
            conn.execute(
                text(
                    """
                SELECT id, provider, created_at FROM artifacts
                WHERE meeting_id = :mid AND kind = 'summary'
                ORDER BY created_at DESC
                """
                ),
                {"mid": meeting_id},
            )
            .mappings()
            .all()
        )
    primary_id = (
        str(meeting_row["primary_summary_artifact_id"])
        if meeting_row and meeting_row["primary_summary_artifact_id"]
        else None
    )
    summaries = [
        {"id": str(r["id"]), "provider": str(r["provider"]), "createdAt": int(r["created_at"])} for r in rows
    ]
    return primary_id, summaries


async def _handle_lifecycle_mcp_tools(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    db: Engine,
    meetings_dir: str,
    title_service: Any | None,
    summary_service: Any | None,
) -> dict[str, Any] | None:
    """Meeting lifecycle edits (delete, start-time, auto-title, summarize, summary artifacts)."""
    if tool_name == "delete_meeting":
        dm_mid = str(arguments.get("meetingId") or "").strip()
        dm_expected = arguments.get("expectedTitle")
        dm_dry = bool(arguments.get("dryRun"))
        if not dm_mid:
            return _error_result("meetingId is required")
        dm_detail = get_meeting(db, dm_mid, meetings_dir)
        if not dm_detail:
            return _error_result("Meeting not found")
        dm_title = str(dm_detail.meeting.title or "")
        dm_tags = list(dm_detail.meeting.tags or [])
        if dm_expected is not None and str(dm_expected) != dm_title:
            return _error_result(
                f"Precondition failed: expectedTitle {str(dm_expected)!r} does not match "
                f"live title {dm_title!r}; aborting without changes"
            )
        if dm_dry:
            return _text_result(
                {
                    "dryRun": True,
                    "operation": "delete_meeting",
                    "meetingId": dm_mid,
                    "title": dm_title,
                    "tags": dm_tags,
                    "recordingCount": len(dm_detail.recordings or []),
                    "wouldDelete": True,
                }
            )
        dm_deleted = delete_meeting(db, dm_mid, meetings_dir)
        if not dm_deleted:
            return _error_result("Meeting not found")
        return _text_result(
            {
                "operation": "delete_meeting",
                "meetingId": dm_mid,
                "title": dm_title,
                "deleted": True,
            }
        )

    if tool_name == "update_meeting_start_time":
        st_mid = str(arguments.get("meetingId") or "").strip()
        st_raw_new = arguments.get("newStartTime")
        st_raw_exp = arguments.get("expectedStartTime")
        st_dry = bool(arguments.get("dryRun"))
        if not st_mid:
            return _error_result("meetingId is required")
        if not isinstance(st_raw_new, (int, float)) or isinstance(st_raw_new, bool):
            return _error_result("newStartTime must be epoch milliseconds")
        st_new = int(st_raw_new)
        if st_new < 0:
            return _error_result("newStartTime must be epoch milliseconds")
        st_detail = get_meeting(db, st_mid, meetings_dir)
        if not st_detail:
            return _error_result("Meeting not found")
        st_current = int(st_detail.meeting.startTime)
        if st_raw_exp is not None:
            if not isinstance(st_raw_exp, (int, float)) or isinstance(st_raw_exp, bool):
                return _error_result("expectedStartTime must be epoch milliseconds")
            if int(st_raw_exp) != st_current:
                return _error_result(
                    f"Precondition failed: expectedStartTime {int(st_raw_exp)!r} does not match "
                    f"live start time {st_current!r}; aborting without changes"
                )
        if st_dry:
            return _text_result(
                {
                    "dryRun": True,
                    "operation": "update_meeting_start_time",
                    "meetingId": st_mid,
                    "currentStartTime": st_current,
                    "newStartTime": st_new,
                    "wouldUpdate": st_current != st_new,
                }
            )
        st_updated = update_meeting_start_time(db, st_mid, st_new, meetings_dir)
        if not st_updated:
            return _error_result("Meeting not found")
        return _text_result(
            {
                "operation": "update_meeting_start_time",
                "meetingId": st_mid,
                "previousStartTime": st_current,
                "meeting": st_updated.model_dump(),
            }
        )

    if tool_name == "auto_title_meeting":
        at_mid = str(arguments.get("meetingId") or "").strip()
        at_provider = arguments.get("provider")
        at_model = arguments.get("model")
        at_expected = arguments.get("expectedTitle")
        at_dry = bool(arguments.get("dryRun"))
        if not at_mid:
            return _error_result("meetingId is required")
        at_detail = get_meeting(db, at_mid, meetings_dir)
        if not at_detail:
            return _error_result("Meeting not found")
        at_current = str(at_detail.meeting.title or "")
        if at_expected is not None and str(at_expected) != at_current:
            return _error_result(
                f"Precondition failed: expectedTitle {str(at_expected)!r} does not match "
                f"live title {at_current!r}; aborting without changes"
            )
        if at_dry:
            return _text_result(
                {
                    "dryRun": True,
                    "operation": "auto_title_meeting",
                    "meetingId": at_mid,
                    "currentTitle": at_current,
                    "wouldGenerate": True,
                }
            )
        if title_service is None:
            return _error_result("Title service unavailable")
        at_fn = getattr(title_service, "generate_and_apply", None) or getattr(
            title_service, "generateAndApply", None
        )
        if not callable(at_fn):
            return _error_result("Title service unavailable")
        at_raw: Any = at_fn(
            at_mid,
            provider=at_provider if isinstance(at_provider, str) else None,
            model=at_model if isinstance(at_model, str) else None,
        )
        at_res = await at_raw if hasattr(at_raw, "__await__") else at_raw
        if not isinstance(at_res, dict):
            return _error_result("Title service returned an unexpected result")
        return _text_result(
            {
                "operation": "auto_title_meeting",
                "meetingId": at_mid,
                "previousTitle": at_current,
                **at_res,
            }
        )

    if tool_name == "summarize_meeting":
        sm_mid = str(arguments.get("meetingId") or "").strip()
        sm_template = arguments.get("templateId")
        sm_provider = arguments.get("provider")
        sm_model = arguments.get("model")
        sm_dry = bool(arguments.get("dryRun"))
        if not sm_mid:
            return _error_result("meetingId is required")
        sm_detail = get_meeting(db, sm_mid, meetings_dir)
        if not sm_detail:
            return _error_result("Meeting not found")
        sm_has_transcript = sm_detail.meeting.primaryTranscriptArtifactId is not None
        if sm_dry:
            return _text_result(
                {
                    "dryRun": True,
                    "operation": "summarize_meeting",
                    "meetingId": sm_mid,
                    "hasTranscript": sm_has_transcript,
                    "wouldQueue": sm_has_transcript,
                }
            )
        if summary_service is None:
            return _error_result("Summary service unavailable")
        sm_fn = getattr(summary_service, "start_summary_generation", None) or getattr(
            summary_service, "startSummaryGeneration", None
        )
        if not callable(sm_fn):
            return _error_result("Summary service unavailable")
        sm_opts: dict[str, Any] = {"meetingId": sm_mid}
        if isinstance(sm_template, str) and sm_template.strip():
            sm_opts["templateId"] = sm_template.strip()
        if isinstance(sm_provider, str) and sm_provider.strip():
            sm_opts["provider"] = sm_provider.strip()
        if isinstance(sm_model, str) and sm_model.strip():
            sm_opts["model"] = sm_model.strip()
        sm_raw: Any = sm_fn(sm_opts)
        sm_status = await sm_raw if hasattr(sm_raw, "__await__") else sm_raw
        if hasattr(sm_status, "model_dump"):
            try:
                sm_status = sm_status.model_dump(exclude_none=True)
            except Exception:
                pass
        return _text_result(
            {
                "operation": "summarize_meeting",
                "meetingId": sm_mid,
                "started": True,
                "status": sm_status,
            }
        )

    if tool_name == "delete_summary":
        ds_mid = str(arguments.get("meetingId") or "").strip()
        ds_sid = _resolve_summary_id(arguments)
        ds_dry = bool(arguments.get("dryRun"))
        ds_expected = arguments.get("expectedIsPrimary")
        if not ds_mid:
            return _error_result("meetingId is required")
        if not ds_sid:
            return _error_result(
                "summaryId is required (summary artifact ID from get_meeting summaries[].id)"
            )
        ds_detail = get_meeting(db, ds_mid, meetings_dir)
        if not ds_detail:
            return _error_result("Meeting not found")
        ds_primary, ds_summaries = _fetch_summary_state(db, ds_mid)
        ds_ids = [s["id"] for s in ds_summaries]
        if ds_sid not in ds_ids:
            return _error_result(f"Summary artifact not found: {ds_sid}")
        ds_is_primary = ds_primary == ds_sid
        if ds_expected is not None and bool(ds_expected) != ds_is_primary:
            return _error_result(
                f"Precondition failed: expectedIsPrimary {bool(ds_expected)!r} does not match "
                f"live value {ds_is_primary!r}; aborting without changes"
            )
        ds_remaining = [sid for sid in ds_ids if sid != ds_sid]
        ds_next_primary = (
            ds_remaining[0] if ds_remaining and ds_is_primary else ds_primary if not ds_is_primary else None
        )
        if ds_dry:
            return _text_result(
                {
                    "dryRun": True,
                    "operation": "delete_summary",
                    "meetingId": ds_mid,
                    "summaryId": ds_sid,
                    "isPrimary": ds_is_primary,
                    "primarySummaryId": ds_primary,
                    "summaryIds": ds_ids,
                    "remainingSummaryIds": ds_remaining,
                    "nextPrimarySummaryId": ds_next_primary,
                    "wouldDelete": True,
                }
            )
        if summary_service is None:
            return _error_result("Summary service unavailable")
        ds_fn = getattr(summary_service, "delete_summary", None) or getattr(
            summary_service, "deleteSummary", None
        )
        if not callable(ds_fn):
            return _error_result("Summary service unavailable")
        try:
            ds_raw: Any = ds_fn(ds_mid, ds_sid)
            if hasattr(ds_raw, "__await__"):
                await ds_raw
        except ValueError as exc:
            return _error_result(str(exc))
        ds_new_primary, ds_new_list = _fetch_summary_state(db, ds_mid)
        return _text_result(
            {
                "operation": "delete_summary",
                "meetingId": ds_mid,
                "summaryId": ds_sid,
                "deleted": True,
                "wasPrimary": ds_is_primary,
                "primarySummaryId": ds_new_primary,
                "remainingSummaryIds": [s["id"] for s in ds_new_list],
            }
        )

    if tool_name == "set_primary_summary":
        sp_mid = str(arguments.get("meetingId") or "").strip()
        sp_sid = _resolve_summary_id(arguments)
        sp_dry = bool(arguments.get("dryRun"))
        sp_expected = arguments.get("expectedPrimarySummaryId")
        if sp_expected is None:
            sp_expected = arguments.get("expectedPrimaryArtifactId")
        if not sp_mid:
            return _error_result("meetingId is required")
        if not sp_sid:
            return _error_result(
                "summaryId is required (summary artifact ID from get_meeting summaries[].id)"
            )
        sp_detail = get_meeting(db, sp_mid, meetings_dir)
        if not sp_detail:
            return _error_result("Meeting not found")
        sp_primary, sp_summaries = _fetch_summary_state(db, sp_mid)
        sp_ids = [s["id"] for s in sp_summaries]
        if sp_sid not in sp_ids:
            return _error_result(f"Summary artifact not found: {sp_sid}")
        if sp_expected is not None and str(sp_expected) != str(sp_primary or ""):
            return _error_result(
                f"Precondition failed: expectedPrimarySummaryId {str(sp_expected)!r} does not match "
                f"live primary {str(sp_primary or '')!r}; aborting without changes"
            )
        if sp_dry:
            return _text_result(
                {
                    "dryRun": True,
                    "operation": "set_primary_summary",
                    "meetingId": sp_mid,
                    "summaryId": sp_sid,
                    "currentPrimarySummaryId": sp_primary,
                    "isAlreadyPrimary": sp_primary == sp_sid,
                    "wouldUpdate": sp_primary != sp_sid,
                }
            )
        if sp_primary == sp_sid:
            return _text_result(
                {
                    "operation": "set_primary_summary",
                    "meetingId": sp_mid,
                    "summaryId": sp_sid,
                    "previousPrimarySummaryId": sp_primary,
                    "primarySummaryId": sp_primary,
                    "updated": False,
                }
            )
        if summary_service is None:
            return _error_result("Summary service unavailable")
        sp_fn = getattr(summary_service, "set_primary_summary", None) or getattr(
            summary_service, "setPrimarySummary", None
        )
        if not callable(sp_fn):
            return _error_result("Summary service unavailable")
        try:
            sp_raw: Any = sp_fn(sp_mid, sp_sid)
            if hasattr(sp_raw, "__await__"):
                await sp_raw
        except ValueError as exc:
            return _error_result(str(exc))
        sp_final, _ = _fetch_summary_state(db, sp_mid)
        return _text_result(
            {
                "operation": "set_primary_summary",
                "meetingId": sp_mid,
                "summaryId": sp_sid,
                "previousPrimarySummaryId": sp_primary,
                "primarySummaryId": sp_final,
                "updated": True,
            }
        )

    return None


async def execute_mcp_tool(  # pyright: ignore[reportGeneralTypeIssues]
    tool_name: str,
    arguments: dict[str, Any],
    *,
    db: Engine,
    meetings_dir: str,
    speaker_service: Any,
    write_enabled: bool | None = None,
    transcription_service: Any | None = None,
    summary_service: Any | None = None,
    title_service: Any | None = None,
    plaud_poller: Any | None = None,
    template_service: Any | None = None,
) -> dict[str, Any]:
    try:
        if tool_name in MCP_WRITE_TOOL_NAMES and not is_mcp_write_enabled(write_enabled):
            return _write_disabled_error()

        if tool_name == "list_meetings":
            resp = list_meetings(db, _meeting_filter_args(arguments), meetings_dir)
            return _text_result({"meetings": resp["meetings"], "pagination": resp["pagination"]})

        if tool_name == "list_tags":
            return _text_result({"tags": list_tags(db)})

        if tool_name == "get_meeting":
            meeting_id = str(arguments.get("meetingId") or "")
            detail = get_meeting(db, meeting_id, meetings_dir)
            if not detail:
                return _error_result("Meeting not found")
            return _text_result(detail.model_dump())

        if tool_name == "search_transcripts":
            query = str(arguments.get("query") or "").strip()
            if not query:
                return _error_result("query is required")
            limit = int(arguments.get("limit") or 50)
            offset = int(arguments.get("offset") or 0)
            candidates = _list_all_meetings(
                db,
                meetings_dir,
                {**_meeting_filter_args(arguments), "limit": None, "offset": None},
            )
            matches: list[dict[str, Any]] = []
            for m in candidates:
                txt = _load_transcript_text(db, meetings_dir, m)
                snippet = _transcript_snippet(txt, query)
                if snippet:
                    matches.append({"meeting": m, "snippet": snippet})
            return _text_result(
                {
                    "matches": matches[offset : offset + limit],
                    "pagination": _pagination_hints(limit, offset, len(matches)),
                }
            )

        if tool_name == "get_action_items":
            meeting_id = arguments.get("meetingId")
            include_completed = arguments.get("includeCompleted", True)
            limit = int(arguments.get("limit") or 50)
            offset = int(arguments.get("offset") or 0)
            if meeting_id:
                d = get_meeting(db, str(meeting_id), meetings_dir)
                if not d:
                    return _error_result("Meeting not found")
                meetings_list = [d.meeting.model_dump()]
            else:
                meetings_list = _list_all_meetings(
                    db,
                    meetings_dir,
                    {**_meeting_filter_args(arguments), "limit": None, "offset": None},
                )
            action_items: list[dict[str, Any]] = []
            for m in meetings_list:
                d = get_meeting(db, m["id"], meetings_dir)
                for item in _parse_summary_action_items(d.summaryContent if d else None):
                    if include_completed or not item["completed"]:
                        action_items.append(
                            {
                                "meetingId": m["id"],
                                "meetingTitle": m["title"],
                                **item,
                            }
                        )
            return _text_result(
                {
                    "actionItems": action_items[offset : offset + limit],
                    "pagination": _pagination_hints(limit, offset, len(action_items)),
                }
            )

        if tool_name == "list_speakers":
            list_fn = getattr(speaker_service, "list_speakers", None) or getattr(
                speaker_service, "listSpeakers", None
            )
            if not callable(list_fn):
                return _error_result("Speaker service unavailable")
            s_res: Any = list_fn()
            speakers = await s_res if hasattr(s_res, "__await__") else s_res
            return _text_result({"speakers": speakers})

        if tool_name == "get_speaker_profile":
            speaker_id = str(arguments.get("speakerId") or "")
            get_fn = getattr(speaker_service, "get_speaker", None) or getattr(
                speaker_service, "getSpeaker", None
            )
            if not callable(get_fn):
                return _error_result("Speaker service unavailable")
            g_res: Any = get_fn(speaker_id)
            spk = await g_res if hasattr(g_res, "__await__") else g_res
            if not spk:
                return _error_result("Speaker not found")
            return _text_result(spk)

        if tool_name == "export_speaker_library":
            raw_ids = arguments.get("speakerIds")
            exp_fn = getattr(speaker_service, "export_speaker_library", None) or getattr(
                speaker_service, "exportSpeakerLibrary", None
            )
            if not callable(exp_fn):
                return _error_result("Speaker service unavailable")
            exp_opts: dict[str, Any] = {}
            if isinstance(raw_ids, list):
                exp_opts["speakerIds"] = [str(v) for v in raw_ids if isinstance(v, str) and v.strip()]
            exp_res: Any = exp_fn(exp_opts)
            library = await exp_res if hasattr(exp_res, "__await__") else exp_res
            return _text_result({"library": library})

        if tool_name == "list_templates":
            if template_service is None:
                return _error_result("Template service unavailable")
            t_fn = getattr(template_service, "list_templates", None) or getattr(
                template_service, "listTemplates", None
            )
            if not callable(t_fn):
                return _error_result("Template service unavailable")
            t_res: Any = t_fn()
            if hasattr(t_res, "__await__"):
                t_res = await t_res
            templates: list[Any] = []
            for t in t_res or []:
                templates.append(t.model_dump() if hasattr(t, "model_dump") else t)
            return _text_result({"templates": templates})

        if tool_name == "get_template":
            tid = str(arguments.get("templateId") or arguments.get("template_id") or "").strip()
            if not tid:
                return _error_result("templateId is required")
            if template_service is None:
                return _error_result("Template service unavailable")
            tpl_fn = getattr(template_service, "get_template", None) or getattr(
                template_service, "getTemplate", None
            )
            if not callable(tpl_fn):
                return _error_result("Template service unavailable")
            tpl_res: Any = tpl_fn(tid)
            if hasattr(tpl_res, "__await__"):
                tpl_res = await tpl_res
            if not tpl_res:
                return _error_result("Template not found")
            tpl = tpl_res.model_dump() if hasattr(tpl_res, "model_dump") else tpl_res
            return _text_result({"template": tpl})

        if tool_name == "get_transcription_status":
            trs_mid = str(arguments.get("meetingId") or "").strip()
            if not trs_mid:
                return _error_result("meetingId is required")
            trs_detail = get_meeting(db, trs_mid, meetings_dir)
            if not trs_detail:
                return _error_result("Meeting not found")
            if transcription_service is None:
                return _error_result("Transcription service unavailable")
            trs_fn = getattr(transcription_service, "get_transcription_progress", None) or getattr(
                transcription_service, "getTranscriptionProgress", None
            )
            if not callable(trs_fn):
                return _error_result("Transcription service unavailable")
            trs_res: Any = trs_fn(trs_mid)
            if hasattr(trs_res, "__await__"):
                trs_res = await trs_res
            trs_dict = trs_res.model_dump(exclude_none=True) if hasattr(trs_res, "model_dump") else trs_res
            return _text_result({"meetingId": trs_mid, "progress": trs_dict})

        if tool_name == "list_active_transcriptions":
            if transcription_service is None:
                return _error_result("Transcription service unavailable")
            la_fn = getattr(transcription_service, "get_all_active_transcription_progress", None) or getattr(
                transcription_service, "getAllActiveTranscriptionProgress", None
            )
            if not callable(la_fn):
                return _error_result("Transcription service unavailable")
            la_res: Any = la_fn()
            if hasattr(la_res, "__await__"):
                la_res = await la_res
            active: dict[str, Any] = {}
            if isinstance(la_res, dict):
                for kv_k, kv_v in la_res.items():
                    active[kv_k] = kv_v.model_dump(exclude_none=True) if hasattr(kv_v, "model_dump") else kv_v
            return _text_result({"active": active})

        if tool_name == "get_summary_status":
            ss_mid = str(arguments.get("meetingId") or "").strip()
            if not ss_mid:
                return _error_result("meetingId is required")
            ss_detail = get_meeting(db, ss_mid, meetings_dir)
            if not ss_detail:
                return _error_result("Meeting not found")
            if summary_service is None:
                return _error_result("Summary service unavailable")
            ss_fn = getattr(summary_service, "get_summary_generation_status", None) or getattr(
                summary_service, "getSummaryGenerationStatus", None
            )
            if not callable(ss_fn):
                return _error_result("Summary service unavailable")
            ss_res: Any = ss_fn(ss_mid)
            if hasattr(ss_res, "__await__"):
                ss_res = await ss_res
            ss_dict = ss_res.model_dump(exclude_none=True) if hasattr(ss_res, "model_dump") else ss_res
            ss_generating = isinstance(ss_dict, dict) and ss_dict.get("stage") == "running"
            return _text_result({"meetingId": ss_mid, "generating": ss_generating, "status": ss_dict})

        if tool_name == "plaud_status":
            if plaud_poller is None:
                return _error_result("Plaud sync is not configured")
            ps_fn = getattr(plaud_poller, "get_status", None) or getattr(plaud_poller, "getStatus", None)
            if not callable(ps_fn):
                return _error_result("Plaud sync is not configured")
            ps_res: Any = ps_fn()
            if hasattr(ps_res, "__await__"):
                ps_res = await ps_res
            return _text_result(ps_res if isinstance(ps_res, dict) else {"status": ps_res})

        if tool_name == "update_speaker":
            sid = str(arguments.get("speakerId") or "").strip()
            new_name = str(arguments.get("newName") or "").strip()
            expected = arguments.get("expectedName")
            dry_run = bool(arguments.get("dryRun"))
            if not sid:
                return _error_result("speakerId is required")
            if not new_name:
                return _error_result("newName must be a non-empty string")
            get_fn = getattr(speaker_service, "get_speaker", None) or getattr(
                speaker_service, "getSpeaker", None
            )
            if not callable(get_fn):
                return _error_result("Speaker service unavailable")
            cur_res: Any = get_fn(sid)
            cur = await cur_res if hasattr(cur_res, "__await__") else cur_res
            if not cur:
                return _error_result("Speaker not found")
            current_name = str((cur.get("speaker") or {}).get("name") or "")
            if expected is not None and str(expected) != current_name:
                return _error_result(
                    f"Precondition failed: expectedName {str(expected)!r} does not match "
                    f"live name {current_name!r}; aborting without changes"
                )
            if dry_run:
                return _text_result(
                    {
                        "dryRun": True,
                        "operation": "update_speaker",
                        "speakerId": sid,
                        "currentName": current_name,
                        "newName": new_name,
                        "wouldUpdate": current_name != new_name,
                    }
                )
            if current_name == new_name:
                return _text_result(
                    {
                        "operation": "update_speaker",
                        "speakerId": sid,
                        "previousName": current_name,
                        "speaker": cur.get("speaker"),
                        "updated": False,
                    }
                )
            upd_fn = getattr(speaker_service, "update_speaker", None) or getattr(
                speaker_service, "updateSpeaker", None
            )
            if not callable(upd_fn):
                return _error_result("Speaker service unavailable")
            u_res: Any = upd_fn(sid, {"name": new_name})
            updated = await u_res if hasattr(u_res, "__await__") else u_res
            return _text_result(
                {
                    "operation": "update_speaker",
                    "speakerId": sid,
                    "previousName": current_name,
                    "speaker": updated,
                    "updated": True,
                }
            )

        if tool_name == "merge_speakers":
            src_id = str(arguments.get("sourceSpeakerId") or "").strip()
            tgt_id = str(arguments.get("targetSpeakerId") or "").strip()
            exp_src = arguments.get("expectedSourceName")
            exp_tgt = arguments.get("expectedTargetName")
            dry_run = bool(arguments.get("dryRun"))
            if not src_id or not tgt_id:
                return _error_result("sourceSpeakerId and targetSpeakerId are required")
            if src_id == tgt_id:
                return _error_result("Cannot merge speaker with themselves")
            get_fn = getattr(speaker_service, "get_speaker", None) or getattr(
                speaker_service, "getSpeaker", None
            )
            if not callable(get_fn):
                return _error_result("Speaker service unavailable")
            src_res: Any = get_fn(src_id)
            s_cur = await src_res if hasattr(src_res, "__await__") else src_res
            if not s_cur:
                return _error_result("Source speaker not found")
            tgt_res: Any = get_fn(tgt_id)
            t_cur = await tgt_res if hasattr(tgt_res, "__await__") else tgt_res
            if not t_cur:
                return _error_result("Target speaker not found")
            src_name = str((s_cur.get("speaker") or {}).get("name") or "")
            tgt_name = str((t_cur.get("speaker") or {}).get("name") or "")
            if exp_src is not None and str(exp_src) != src_name:
                return _error_result(
                    f"Precondition failed: expectedSourceName {str(exp_src)!r} does not match "
                    f"live name {src_name!r}; aborting without changes"
                )
            if exp_tgt is not None and str(exp_tgt) != tgt_name:
                return _error_result(
                    f"Precondition failed: expectedTargetName {str(exp_tgt)!r} does not match "
                    f"live name {tgt_name!r}; aborting without changes"
                )
            src_meetings = s_cur.get("meetings") or []
            tgt_meetings = t_cur.get("meetings") or []
            src_ids: list[str] = [
                str(m.get("id")) for m in src_meetings if isinstance(m, dict) and m.get("id")
            ]
            tgt_ids: set[str] = {
                str(m.get("id")) for m in tgt_meetings if isinstance(m, dict) and m.get("id")
            }
            to_move = [mid for mid in src_ids if mid not in tgt_ids]
            if dry_run:
                return _text_result(
                    {
                        "dryRun": True,
                        "operation": "merge_speakers",
                        "source": {
                            "id": src_id,
                            "name": src_name,
                            "meetingCount": len(src_ids),
                            "meetingIds": src_ids,
                        },
                        "target": {
                            "id": tgt_id,
                            "name": tgt_name,
                            "meetingCount": len(tgt_ids),
                            "meetingIds": sorted(tgt_ids),
                        },
                        "wouldMoveMeetingIds": to_move,
                        "wouldDeleteSourceId": src_id,
                    }
                )
            merge_fn = getattr(speaker_service, "merge_speakers", None) or getattr(
                speaker_service, "mergeSpeakers", None
            )
            if not callable(merge_fn):
                return _error_result("Speaker service unavailable")
            m_res: Any = merge_fn(src_id, tgt_id)
            merged = await m_res if hasattr(m_res, "__await__") else m_res
            return _text_result(
                {
                    "operation": "merge_speakers",
                    "sourceSpeakerId": src_id,
                    "sourceName": src_name,
                    "mergedMeetingIds": to_move,
                    "speaker": merged,
                }
            )

        if tool_name == "delete_speaker":
            sid = str(arguments.get("speakerId") or "").strip()
            expected = arguments.get("expectedName")
            dry_run = bool(arguments.get("dryRun"))
            if not sid:
                return _error_result("speakerId is required")
            get_fn = getattr(speaker_service, "get_speaker", None) or getattr(
                speaker_service, "getSpeaker", None
            )
            if not callable(get_fn):
                return _error_result("Speaker service unavailable")
            del_cur_res: Any = get_fn(sid)
            del_cur = await del_cur_res if hasattr(del_cur_res, "__await__") else del_cur_res
            if not del_cur:
                return _error_result("Speaker not found")
            current_name = str((del_cur.get("speaker") or {}).get("name") or "")
            if expected is not None and str(expected) != current_name:
                return _error_result(
                    f"Precondition failed: expectedName {str(expected)!r} does not match "
                    f"live name {current_name!r}; aborting without changes"
                )
            meetings = del_cur.get("meetings") or []
            mids: list[str] = [str(m.get("id")) for m in meetings if isinstance(m, dict) and m.get("id")]
            if dry_run:
                return _text_result(
                    {
                        "dryRun": True,
                        "operation": "delete_speaker",
                        "speakerId": sid,
                        "name": current_name,
                        "meetingCount": len(mids),
                        "meetingIds": mids,
                        "wouldDelete": True,
                    }
                )
            del_fn = getattr(speaker_service, "delete_speaker", None) or getattr(
                speaker_service, "deleteSpeaker", None
            )
            if not callable(del_fn):
                return _error_result("Speaker service unavailable")
            d_res: Any = del_fn(sid)
            if hasattr(d_res, "__await__"):
                await d_res
            return _text_result(
                {
                    "operation": "delete_speaker",
                    "speakerId": sid,
                    "name": current_name,
                    "deleted": True,
                    "releasedMeetingIds": mids,
                }
            )

        if tool_name == "import_speaker_library":
            imp_library = arguments.get("library")
            imp_mode = str(arguments.get("mode") or "merge")
            imp_dry = bool(arguments.get("dryRun"))
            if not isinstance(imp_library, dict):
                return _error_result("library must be the speaker library object from export_speaker_library")
            if imp_mode not in ("merge", "replace"):
                return _error_result('mode must be "merge" or "replace"')
            imp_fn = getattr(speaker_service, "import_speaker_library", None) or getattr(
                speaker_service, "importSpeakerLibrary", None
            )
            if not callable(imp_fn):
                return _error_result("Speaker service unavailable")
            try:
                imp_res: Any = imp_fn({"library": imp_library, "mode": imp_mode, "dryRun": imp_dry})
                imported = await imp_res if hasattr(imp_res, "__await__") else imp_res
            except ValueError as exc:
                return _error_result(str(exc))
            return _text_result({"operation": "import_speaker_library", **imported})

        if tool_name == "reassign_speaker":
            rs_mid = str(arguments.get("meetingId") or "").strip()
            rs_from = str(arguments.get("fromLabel") or "").strip()
            rs_to_name = arguments.get("toSpeakerName")
            rs_to_id = arguments.get("toSpeakerId")
            rs_adopt = arguments.get("adoptVoiceprint")
            rs_local = bool(arguments.get("localOnly"))
            rs_seg = arguments.get("segmentIndex")
            rs_scope = arguments.get("scope")
            rs_expected = arguments.get("expectedFromLabel")
            rs_dry = bool(arguments.get("dryRun"))
            if not rs_mid:
                return _error_result("meetingId is required")
            if not rs_from:
                return _error_result("fromLabel is required")
            if rs_expected is not None and str(rs_expected).strip() != rs_from:
                return _error_result(
                    f"Precondition failed: expectedFromLabel {str(rs_expected)!r} does not match "
                    f"live fromLabel {rs_from!r}; aborting without changes"
                )
            if rs_to_name is not None and not isinstance(rs_to_name, str):
                return _error_result("toSpeakerName must be a string")
            if rs_to_id is not None and not isinstance(rs_to_id, str):
                return _error_result("toSpeakerId must be a string")
            rs_has_name = isinstance(rs_to_name, str) and rs_to_name.strip() != ""
            rs_has_id = isinstance(rs_to_id, str) and rs_to_id.strip() != ""
            if not rs_has_name and not rs_has_id:
                return _error_result("Provide toSpeakerName or toSpeakerId")
            if rs_has_name and rs_has_id:
                return _error_result("Provide either toSpeakerName or toSpeakerId, not both")
            rs_detail = get_meeting(db, rs_mid, meetings_dir)
            if not rs_detail:
                return _error_result("Meeting not found")
            rs_adopt_bool = True if rs_adopt is None else bool(rs_adopt)
            if rs_dry:
                return _text_result(
                    {
                        "dryRun": True,
                        "operation": "reassign_speaker",
                        "meetingId": rs_mid,
                        "fromLabel": rs_from,
                        "toSpeakerName": rs_to_name.strip() if rs_has_name else None,
                        "toSpeakerId": rs_to_id.strip() if rs_has_id else None,
                        "adoptVoiceprint": rs_adopt_bool,
                        "localOnly": rs_local,
                        "segmentIndex": rs_seg,
                        "scope": rs_scope,
                        "wouldReassign": True,
                    }
                )
            if speaker_service is None:
                return _error_result("Speaker service unavailable")
            rs_fn = getattr(speaker_service, "reassign_meeting_speaker", None) or getattr(
                speaker_service, "reassignMeetingSpeaker", None
            )
            if not callable(rs_fn):
                return _error_result("Speaker service unavailable")
            rs_opts: dict[str, Any] = {
                "meetingId": rs_mid,
                "fromLabel": rs_from,
                "adoptVoiceprint": rs_adopt_bool,
                "localOnly": rs_local,
            }
            if rs_has_name:
                rs_opts["toSpeakerName"] = rs_to_name.strip()
            if rs_has_id:
                rs_opts["toSpeakerId"] = rs_to_id.strip()
            if rs_seg is not None:
                rs_opts["segmentIndex"] = rs_seg
            if isinstance(rs_scope, str) and rs_scope.strip():
                rs_opts["scope"] = rs_scope.strip()
            try:
                rs_raw: Any = rs_fn(rs_opts)
                rs_res = await rs_raw if hasattr(rs_raw, "__await__") else rs_raw
            except Exception as exc:
                return _error_result(str(exc))
            if isinstance(rs_res, dict):
                return _text_result({"operation": "reassign_speaker", "meetingId": rs_mid, **rs_res})
            return _text_result({"operation": "reassign_speaker", "meetingId": rs_mid, "result": rs_res})

        if tool_name == "plaud_sync_trigger":
            ps_dry = bool(arguments.get("dryRun"))
            if plaud_poller is None:
                return _error_result("Plaud sync is not configured")
            if ps_dry:
                return _text_result({"dryRun": True, "operation": "plaud_sync_trigger", "wouldTrigger": True})
            ps_bg_fn = getattr(plaud_poller, "trigger_background", None) or getattr(
                plaud_poller, "triggerBackground", None
            )
            if callable(ps_bg_fn):
                try:
                    ps_started = bool(ps_bg_fn())
                    return _text_result(
                        {
                            "operation": "plaud_sync_trigger",
                            "status": "started",
                            "alreadyRunning": not ps_started,
                        }
                    )
                except Exception as exc:
                    return _error_result(str(exc))
            ps_fn2 = getattr(plaud_poller, "trigger", None)
            if not callable(ps_fn2):
                return _error_result("Plaud sync is not configured")
            try:
                ps_raw2: Any = ps_fn2()
                ps_res2 = await ps_raw2 if hasattr(ps_raw2, "__await__") else ps_raw2
                if isinstance(ps_res2, dict):
                    return _text_result({"operation": "plaud_sync_trigger", **ps_res2})
                return _text_result({"operation": "plaud_sync_trigger", "result": ps_res2})
            except Exception as exc:
                return _error_result(str(exc))

        if tool_name == "set_meeting_tags":
            raw_single_id = arguments.get("meetingId")
            raw_bulk_ids = arguments.get("meetingIds")
            raw_tags = arguments.get("tags")
            raw_expected = arguments.get("expectedTags")
            tags_dry_run = bool(arguments.get("dryRun"))
            has_single = bool(str(raw_single_id or "").strip())
            has_bulk = raw_bulk_ids is not None
            if has_single and has_bulk:
                return _error_result("Provide either meetingId or meetingIds, not both")
            new_tags = _normalize_tag_list_for_mcp(raw_tags)
            if new_tags is None:
                return _error_result("tags must be an array of strings")
            if has_bulk:
                if not isinstance(raw_bulk_ids, list) or not raw_bulk_ids:
                    return _error_result("meetingIds must be a non-empty array of meeting id strings")
                bulk_ids: list[str] = []
                for entry in raw_bulk_ids:
                    if not isinstance(entry, str) or not entry.strip():
                        return _error_result("meetingIds must be a non-empty array of meeting id strings")
                    cleaned = entry.strip()
                    if cleaned not in bulk_ids:
                        bulk_ids.append(cleaned)
                if len(bulk_ids) > 500:
                    return _error_result("Bulk tagging supports at most 500 meetings per call")
                if raw_expected is not None:
                    return _error_result("expectedTags is only supported for single-meeting updates")
                if tags_dry_run:
                    bulk_plan: list[dict[str, Any]] = []
                    for bulk_mid in bulk_ids:
                        bulk_detail = get_meeting(db, bulk_mid, meetings_dir)
                        if not bulk_detail:
                            bulk_plan.append({"meetingId": bulk_mid, "found": False})
                        else:
                            bulk_current = list(bulk_detail.meeting.tags or [])
                            bulk_plan.append(
                                {
                                    "meetingId": bulk_mid,
                                    "found": True,
                                    "currentTags": bulk_current,
                                    "newTags": new_tags,
                                    "wouldUpdate": set(new_tags) != set(bulk_current),
                                }
                            )
                    return _text_result(
                        {
                            "dryRun": True,
                            "operation": "set_meeting_tags",
                            "mode": "bulk",
                            "results": bulk_plan,
                            "summary": {
                                "total": len(bulk_ids),
                                "wouldUpdate": sum(1 for r in bulk_plan if r.get("wouldUpdate")),
                                "notFound": sum(1 for r in bulk_plan if not r.get("found")),
                            },
                        }
                    )
                bulk_results: list[dict[str, Any]] = []
                for bulk_mid in bulk_ids:
                    bulk_detail = get_meeting(db, bulk_mid, meetings_dir)
                    if not bulk_detail:
                        bulk_results.append({"meetingId": bulk_mid, "error": "Meeting not found"})
                        continue
                    bulk_prev = list(bulk_detail.meeting.tags or [])
                    bulk_updated = update_meeting_tags(db, bulk_mid, new_tags)
                    if not bulk_updated:
                        bulk_results.append({"meetingId": bulk_mid, "error": "Meeting not found"})
                    else:
                        bulk_results.append(
                            {
                                "meetingId": bulk_mid,
                                "previousTags": bulk_prev,
                                "meeting": bulk_updated.model_dump(),
                            }
                        )
                return _text_result(
                    {
                        "operation": "set_meeting_tags",
                        "mode": "bulk",
                        "results": bulk_results,
                        "summary": {
                            "total": len(bulk_ids),
                            "updated": sum(1 for r in bulk_results if "meeting" in r),
                            "notFound": sum(1 for r in bulk_results if r.get("error")),
                        },
                    }
                )
            single_mid = str(raw_single_id or "").strip()
            if not single_mid:
                return _error_result("meetingId is required (or provide meetingIds for bulk)")
            single_detail = get_meeting(db, single_mid, meetings_dir)
            if not single_detail:
                return _error_result("Meeting not found")
            single_current = list(single_detail.meeting.tags or [])
            if raw_expected is not None:
                exp_tags = _normalize_tag_list_for_mcp(raw_expected)
                if exp_tags is None:
                    return _error_result("expectedTags must be an array of strings")
                if set(exp_tags) != set(single_current):
                    return _error_result(
                        f"Precondition failed: expectedTags {exp_tags!r} does not match "
                        f"live tags {single_current!r}; aborting without changes"
                    )
            if tags_dry_run:
                return _text_result(
                    {
                        "dryRun": True,
                        "operation": "set_meeting_tags",
                        "meetingId": single_mid,
                        "currentTags": single_current,
                        "newTags": new_tags,
                        "wouldUpdate": set(new_tags) != set(single_current),
                    }
                )
            single_updated = update_meeting_tags(db, single_mid, new_tags)
            if not single_updated:
                return _error_result("Meeting not found")
            return _text_result(
                {
                    "operation": "set_meeting_tags",
                    "meetingId": single_mid,
                    "previousTags": single_current,
                    "meeting": single_updated.model_dump(),
                }
            )

        if tool_name == "update_meeting_title":
            mid = str(arguments.get("meetingId") or "").strip()
            new_title = str(arguments.get("newTitle") or "").strip()
            expected = arguments.get("expectedTitle")
            dry_run = bool(arguments.get("dryRun"))
            if not mid:
                return _error_result("meetingId is required")
            if not new_title:
                return _error_result("newTitle must be a non-empty string")
            detail = get_meeting(db, mid, meetings_dir)
            if not detail:
                return _error_result("Meeting not found")
            current_title = str(detail.meeting.title or "")
            if expected is not None and str(expected) != current_title:
                return _error_result(
                    f"Precondition failed: expectedTitle {str(expected)!r} does not match "
                    f"live title {current_title!r}; aborting without changes"
                )
            if dry_run:
                return _text_result(
                    {
                        "dryRun": True,
                        "operation": "update_meeting_title",
                        "meetingId": mid,
                        "currentTitle": current_title,
                        "newTitle": new_title,
                        "wouldUpdate": current_title != new_title,
                    }
                )
            updated = update_meeting_title(db, mid, new_title, meetings_dir)
            if not updated:
                return _error_result("Meeting not found")
            return _text_result(
                {
                    "operation": "update_meeting_title",
                    "meetingId": mid,
                    "previousTitle": current_title,
                    "meeting": updated.model_dump(),
                }
            )

        tr_handled = await _handle_transcription_mcp_tools(
            tool_name,
            arguments,
            db=db,
            meetings_dir=meetings_dir,
            transcription_service=transcription_service,
            plaud_poller=plaud_poller,
        )
        if tr_handled is not None:
            return tr_handled

        lc_handled = await _handle_lifecycle_mcp_tools(
            tool_name,
            arguments,
            db=db,
            meetings_dir=meetings_dir,
            title_service=title_service,
            summary_service=summary_service,
        )
        if lc_handled is not None:
            return lc_handled

        return _error_result(f"Unknown tool: {tool_name}")
    except Exception as exc:
        return _error_result(exc)


def create_mcp_server(options: dict[str, Any]) -> MCPServer:
    server = MCPServer("olive")
    return server


createMcpServer = create_mcp_server


async def handle_mcp_request(
    *,
    request: Request,
    db: Engine,
    meetings_dir: str,
    speaker_service: Any,
    mcp_token: str | None,
    bind_host: str,
    write_enabled: bool | None = None,
    transcription_service: Any | None = None,
    summary_service: Any | None = None,
    title_service: Any | None = None,
    plaud_poller: Any | None = None,
    template_service: Any | None = None,
) -> Response:
    resolved_write_enabled = is_mcp_write_enabled(write_enabled)
    # 1. Host / Origin validation when bound to loopback
    if bind_host in LOOPBACK_HOSTS:
        host_hdr = (request.headers.get("host") or "127.0.0.1").split(":", 1)[0].strip().lower()
        if host_hdr and host_hdr not in LOOPBACK_HOSTS:
            return JSONResponse({"error": "Forbidden host header"}, status_code=403)
        origin_hdr = request.headers.get("origin")
        if origin_hdr:
            parsed_orig = urllib.parse.urlparse(origin_hdr)
            orig_host = (parsed_orig.hostname or "").lower()
            if orig_host not in LOOPBACK_HOSTS:
                return JSONResponse({"error": "Forbidden origin header"}, status_code=403)

    # 2. Bearer token authentication
    if mcp_token:
        auth_header = request.headers.get("authorization") or ""
        if auth_header != f"Bearer {mcp_token}":
            return JSONResponse(
                {"error": "Unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

    if request.method == "OPTIONS":
        return Response(status_code=204)

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
            status_code=400,
        )

    rpc_id = payload.get("id") if isinstance(payload, dict) else None
    method = payload.get("method") if isinstance(payload, dict) else None
    params = (payload.get("params") or {}) if isinstance(payload, dict) else {}

    if method == "initialize":
        result = {
            "protocolVersion": params.get("protocolVersion") or "2025-11-25",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "olive", "version": "0.2.0"},
        }
        return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": result})

    if method == "notifications/initialized":
        return Response(status_code=202)

    if method == "tools/list":
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {"tools": get_mcp_tools_spec(write_enabled=resolved_write_enabled)},
            }
        )

    if method == "tools/call":
        tool_name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        tool_res = await execute_mcp_tool(
            tool_name,
            arguments if isinstance(arguments, dict) else {},
            db=db,
            meetings_dir=meetings_dir,
            speaker_service=speaker_service,
            write_enabled=resolved_write_enabled,
            transcription_service=transcription_service,
            summary_service=summary_service,
            title_service=title_service,
            plaud_poller=plaud_poller,
            template_service=template_service,
        )
        return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": tool_res})

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        },
        status_code=400,
    )
