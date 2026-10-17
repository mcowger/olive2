# Olive API

## `POST /api/ingest`

Ingests arbitrary audio recordings into Olive from web uploads or iOS Shortcuts.

* **Authentication:** `Authorization: Bearer <OLIVE_INGEST_TOKEN>` (required if `OLIVE_INGEST_TOKEN` is set in environment).
* **Content-Type:** `multipart/form-data`
* **Form Fields:**
  * `file`: Audio file blob (required). Supported formats: `.m4a`, `.mp3`, `.wav`, `.aac`, `.ogg`, `.flac`, `.webm`.
  * `title`: Optional custom meeting title string.
  * `source`: Optional source (`'upload'` | `'ios-shortcut'`, default inferred from User-Agent / parameter).
  * `autoTranscribe`: Optional boolean (`true` / `false`).
  * `provider`: Optional transcription provider (`'speechmatics'` | `'local'`).

Response:
```json
{
  "meetingId": "uuid",
  "recordingId": "uuid",
  "deduped": false,
  "meeting": { ... },
  "audioPath": "audio/uuid.m4a"
}
```

*Note on Deduplication:* Computes SHA-256 hash of the uploaded audio payload. If the exact same recording was previously uploaded, responds with HTTP 200 `{ "deduped": true }` and returns the existing meeting without re-allocating disk storage.

## `GET /api/meetings/:id/audio`

Streams the audio recording for in-browser playback. Supports HTTP `Range` requests for seekable playback.

## `POST /api/meetings/:id/audio/retry`

Plaud meetings only. Resets the Plaud ingest state so the next poll refetches the audio from Plaud Cloud, and returns HTTP 202. Non-Plaud meetings get HTTP 400; unknown meetings (or Plaud meetings with no ingest state) get HTTP 404. Meeting payloads carry `audioStatus` (`ready` | `retrieving` | `missing`) and `canRetryAudio` so the UI can grey out audio-pending meetings.

---

## iOS Shortcut Setup (Voice Memos to Olive)

You can easily send Apple Voice Memos directly into Olive via an iOS Share-Sheet Shortcut:

1. Open the **Shortcuts** app on iOS.
2. Create a new Shortcut with:
   * **Receive:** Audio / Files from Share Sheet.
   * **Action 1:** `Get Contents of URL`
     * **URL:** `https://<your-olive-host>/api/ingest`
     * **Method:** `POST`
     * **Headers:** `Authorization: Bearer <your-OLIVE_INGEST_TOKEN>`
     * **Request Body:** `Form`
       * `file`: `Shortcut Input` (Type: File)
       * `source`: `ios-shortcut`
       * `autoTranscribe`: `true`
   * **Action 2:** `Show Notification` with upload status.
3. In Apple Voice Memos, tap **Share** $\rightarrow$ select your Olive shortcut.

---

## `GET /api/health`

Returns HTTP 200 with:

```json
{"status":"ok"}
```

## `GET /api/meetings?limit&offset&search&tags&status&speakerId&hasSummary&hasTranscript&includeDetails`

Returns HTTP 200 with a stable envelope. `meetings` is always an array, including when the
database is empty. `limit` defaults to `50` and is capped at `100`; `offset` defaults to `0`.
Results are newest-first. Every response includes
`pagination: { limit, offset, total, hasMore, nextOffset }` — while `hasMore`
is true, fetch the next page with `offset=nextOffset` until all results are loaded.

By default each meeting is slim `{ id, title, startTime, endTime, source, tags }`
(~65% smaller — enough for listing, matching, and counting). Pass
`includeDetails=true` only when you need `status`, artifact ids, timestamps,
`lastError`, `audioStatus`, or `canRetryAudio` (the web UI always does).

* `tags`: exact tag match (case-insensitive), repeatable (`?tags=acme&tags=q3`, AND semantics)
  or comma-separated (`?tags=acme,q3`). Unlike `search`, never matches transcript text —
  use it to scope to a tagged subset (e.g. all calls for one client).
* `status`: exact meeting status (`ready`, `processing`, `error`, …).
* `speakerId`: only meetings linked to this speaker.
* `hasSummary` / `hasTranscript`: `true` / `false`.
* All filters compose as AND with `search`, `source`, `startTime`, `endTime`.

## `GET /api/tags`

Returns distinct meeting tags with meeting counts, sorted alphabetically:

```json
{
  "tags": [
    { "tag": "acme", "meetingCount": 4 }
  ]
}
```

```json
{
  "meetings": [
    {
      "id": "uuid",
      "title": "Design review",
      "startTime": 1720000000000,
      "endTime": 1720003600000,
      "source": "upload",
      "status": "ready",
      "tags": ["project"],
      "primaryTranscriptArtifactId": null,
      "primarySummaryArtifactId": null,
      "lastError": null,
      "createdAt": 1720000000000,
      "updatedAt": 1720000000000
    }
  ],
  "pagination": {
    "limit": 50,
    "offset": 0,
    "total": 1,
    "hasMore": false,
    "nextOffset": null
  }
}
```

## `/mcp`

Provides Olive's Model Context Protocol endpoint on the same port as the HTTP API. Read tools (`list_meetings`, `list_tags`, `get_meeting`, `search_transcripts`, `get_action_items`, `list_speakers`, `get_speaker_profile`, `export_speaker_library`, `list_templates`, `get_template`, `get_transcription_status`, `list_active_transcriptions`, `get_summary_status`, `plaud_status`) are always available. Write tools are off by default and are listed and callable only when the server runs with `OLIVE_MCP_WRITE=1`: speaker edits (`update_speaker`, `merge_speakers`, `delete_speaker`, `reassign_speaker`), tags (`set_meeting_tags`, single or bulk via `meetingIds` up to 500), titles and lifecycle (`update_meeting_title`, `update_meeting_start_time`, `auto_title_meeting`, `summarize_meeting`, `delete_summary`, `set_primary_summary`, `delete_meeting`), and transcription + Plaud (`transcribe_meeting`, `bulk_transcribe` up to 2000, `cancel_transcription`, `retry_meeting_audio`, `reidentify_speakers`, `plaud_sync_trigger`).

`list_meetings` accepts `search` plus the meeting filters: `tags` (exact,
AND-combined), `source`, `status`, `speakerId`, `hasSummary`,
`hasTranscript`, `startTime`, `endTime`, plus `limit` / `offset`.
Each meeting is slim `{ id, title, startTime, endTime, source, tags }` by default
(~65% smaller — enough for listing, matching, and counting). Pass
`includeDetails: true` only when you need `status`, artifact ids, timestamps,
`lastError`, `audioStatus`, or `canRetryAudio`; prefer `get_meeting` for full
detail on one meeting instead of `includeDetails` on a whole list.
All three paginated tools (`list_meetings`, `search_transcripts`,
`get_action_items`) return `pagination: { limit, offset, total, hasMore,
nextOffset }` — while `hasMore` is true, call again with
`offset=nextOffset` until everything is retrieved.
`search_transcripts` takes `query` (the transcript text to find) plus the same
filters *except* `search` (nested meetings are slim unless `includeDetails`),
and `get_action_items` takes the same filters
*except* `search` — one free-text field per tool, so agents never have to
guess which one scopes what. Point an agent at a tagged subset (e.g.
`{"tags": ["acme"]}`) to review one client or project without keyword
collisions from transcript text.

### MCP search: future considerations (deferred)

Deliberately out of scope for the tag-filtering work above; noted here so the
trade-offs are recorded:

* **Richer structured facets (cheap, SQL-only):** `provider`/engine (from
  `recordings` / `artifacts` / `stage_runs`), duration range (from
  `recordings.duration_ms`), stage state (from `stage_runs`), and chat
  presence (from `chat_messages`) are all queryable today and can be added
  as filters if agent workflows ask for them.
* **Segment-level transcript search (expensive):** `search_transcripts`
  currently scans transcript files per call and returns a plain-text window
  with no timestamps or speaker attribution. Returning per-segment hits
  (`startMs` / `endMs`, speaker, snippet, hit counts) requires parsing
  segments on scan; a persistent FTS5 segment index would be the step after
  that if libraries grow large enough for scan cost to matter.
* **Beyond FTS:** semantic/embedding search. Not recommended until filtered
exact + full-text search proves insufficient.

Authentication uses `Authorization: Bearer <OLIVE_MCP_TOKEN>`. If `OLIVE_MCP_TOKEN` is unset, `OLIVE_INGEST_TOKEN` is used as a fallback. When Olive binds to loopback, the endpoint may be used without a token; non-loopback binds require one.

### MCP writes: plan then apply

Every write tool accepts `dryRun: true` to preview without mutating, and an `expected*` guard validated against live data before any mutation:

- `update_speaker { speakerId, newName, expectedName?, dryRun? }` — rename; aborts unless the live name matches `expectedName`.
- `merge_speakers { sourceSpeakerId, targetSpeakerId, expectedSourceName?, expectedTargetName?, dryRun? }` — moves meeting links, voiceprints, and clips into the target, then deletes the source; `dryRun` lists the meetings that would move.
- `delete_speaker { speakerId, expectedName?, dryRun? }` — deletes the speaker and its meeting links; `dryRun` lists affected meetings.
- `import_speaker_library { library, mode?, dryRun? }` — imports a library from `export_speaker_library`; `mode` is `merge` (default) or `replace` (wipes local first, meeting attributions carry over by name); `dryRun` returns the per-speaker plan without mutating. Like other bulk tools it has no `expected*` guard.
- `set_meeting_tags { meetingId?, meetingIds?, tags, expectedTags?, dryRun? }` — replaces tags (single, or bulk up to 500 via `meetingIds`); single aborts unless live tags match `expectedTags` (order-insensitive). For bulk tagging, scope with `list_meetings` or pass `meetingIds` directly.
- `update_meeting_title { meetingId, newTitle, expectedTitle?, dryRun? }` — renames the title and its on-disk folder; aborts unless the live title matches `expectedTitle`.
- `transcribe_meeting { meetingId, provider?, language?, force?, dryRun? }` — queues (re)transcription async, returns `jobId`; `dryRun` validates the meeting/recording without queueing.
- `bulk_transcribe { provider, meetingIds?, search?, tags?, status?, speakerId?, hasSummary?, hasTranscript?, startTime?, endTime?, force?, language?, dryRun? }` — queues one engine for up to 2000 meetings; `dryRun` resolves the target list without queueing.
- `cancel_transcription { meetingId, dryRun? }`, `retry_meeting_audio { meetingId, dryRun? }` (Plaud re-download), `reidentify_speakers { meetingId, dryRun? }` (re-match labels to voiceprints) — all async with `dryRun` validation.
- `delete_meeting { meetingId, expectedTitle?, dryRun? }` — deletes recordings, artifacts, and folder; aborts unless the live title matches.
- `update_meeting_start_time { meetingId, newStartTime, expectedStartTime?, dryRun? }` — shifts start (duration preserved, folder moved on date change).
- `auto_title_meeting { meetingId, provider?, model?, expectedTitle?, dryRun? }` — LLM title + persist; `dryRun` validates without calling the LLM.
- `summarize_meeting { meetingId, templateId?, provider?, model?, dryRun? }` — queues summary generation, returns generation status.
- `delete_summary { meetingId, summaryId, expectedIsPrimary?, dryRun? }` — deletes one summary artifact (file + DB row; deleting the primary promotes the newest remaining summary); `summaryId` is `summaries[].id` from `get_meeting` (`artifactId` accepted as an alias); aborts unless the live primary flag matches `expectedIsPrimary`.
- `set_primary_summary { meetingId, summaryId, expectedPrimarySummaryId?, dryRun? }` — promotes a summary to primary; aborts unless the live primary matches `expectedPrimarySummaryId`.
- `reassign_speaker { meetingId, fromLabel, toSpeakerName?/toSpeakerId?, adoptVoiceprint?, localOnly?, segmentIndex?, scope?, expectedFromLabel?, dryRun? }` — moves transcript segments labelled `fromLabel` to a target speaker (creates by name or resolves by id); `adoptVoiceprint` (default true) grows the library from the matched audio.
- `plaud_sync_trigger { dryRun? }` — triggers a Plaud Cloud poll cycle (202-style, background job).

Read helpers that mirror the REST polling/debug surface (always available, no `OLIVE_MCP_WRITE=1` needed):

- `list_templates {}` / `get_template { templateId }` — discover summary prompt templates.
- `get_transcription_status { meetingId }` — per-meeting transcription progress (`GET /api/meetings/{id}/transcribe/progress` without SSE).
- `list_active_transcriptions {}` — all active transcriptions (`GET /api/transcriptions/active`).
- `get_summary_status { meetingId }` — summary generation status (`GET /api/meetings/{id}/summarize/status`).
- `plaud_status {}` — Plaud sync state (`GET /api/plaud/status`).
- `list_speakers { includeVoiceprints? }` / `get_speaker_profile { speakerId, includeVoiceprints? }` — speakers are slim `{ id, name, enrolled, voiceprintCounts, enrolledAt, createdAt, meetingCount }` by default (no voiceprint vectors); pass `includeVoiceprints: true` only to inspect raw vectors. Bulk vectors live in `export_speaker_library`.
- `get_meeting { meetingId, includeTranscript?, includeSummaries?, includeVoiceprints? }` — transcript/summaries default to included; pass `includeTranscript: false` / `includeSummaries: false` for metadata-only fetches. Meeting speakers are vector-free by default.

Recommended agent flow: `dryRun` first (or read via `get_speaker_profile` / `get_meeting`), show the plan to the user, then apply with the `expected*` values from the plan so concurrent edits abort instead of clobbering.

## `GET /api/meetings/:id`

Returns HTTP 200 with the full meeting aggregate (all sections included by default):

```json
{
  "meeting": { "id": "uuid", "title": "Design review", ... },
  "recordings": [ ... ],
  "artifacts": [ ... ],
  "speakers": [ ... ],
  "stageRuns": [ ... ],
  "transcriptContent": "...",
  "summaryContent": "..."
}
```

Large sections are opt-out via query flags: `?includeTranscript=false` skips
`transcriptContent`, `?includeSummaries=false` skips `summaryContent` and
`summaries[]` bodies, `?includeVoiceprints=false` strips raw voiceprint vectors
from `speakers` (identity + `enrolledAt` kept).

## `POST /api/meetings/:id/transcribe`

Triggers Speechmatics transcription for a meeting's recording.

Optional JSON body:
```json
{
  "language": "en",
  "poll": false,
  "force": false
}
```

Returns:
```json
{
  "stageRunId": "uuid",
  "jobId": "speechmatics-job-id",
  "status": "running"
}
```

## `POST /api/webhooks/speechmatics`

Webhook destination for Speechmatics job completion notifications. Receives transcript payload, normalizes to canonical format, stores `speechmatics.json` / `speechmatics.txt` artifacts, links speakers, and sets primary transcript.

## `GET /api/speakers`

Returns a list of all enrolled/discovered speakers with meeting participation counts.

```json
{
  "speakers": [
    {
      "id": "uuid",
      "name": "Matt",
      "providerIds": { "speechmatics": ["id1", "id2"] },
      "enrolledAt": 1720000000000,
      "enrollmentClipPaths": ["speakers/uuid/clip_1720000000000.wav"],
      "meetingCount": 4,
      "createdAt": 1720000000000
    }
  ]
}
```

## `POST /api/speakers/enroll`

Accepts `multipart/form-data` with:
* `name`: Speaker name (required)
* `file`: 5–30s solo audio clip (required)
* `speakerId`: Optional existing speaker ID to add additional audio clip

Submits audio to Speechmatics with `get_speakers: true`, extracts voiceprint identifiers, saves the clip to disk, and persists the speaker with their Speechmatics identifiers.

## `GET /api/speakers/:id`

Returns speaker profile, enrolled audio clips, and list of associated meetings.

## `GET /api/speakers/export?speakerIds&download`

Exports the enrolled speaker library as a portable JSON object (voiceprint vectors + metadata, no audio). Copy the returned `library` to another Olive instance to share voiceprints without re-enrolling. Optional `speakerIds` (comma-separated) exports a subset; `download=1` adds a `Content-Disposition: attachment` header.

```json
{
  "library": {
    "version": "1.0",
    "createdAt": 1720000000000,
    "embeddingModel": "Wespeaker/…",
    "speakers": [
      { "id": "uuid", "name": "Matt", "enrolledAt": 1720000000000, "createdAt": 1720000000000, "voiceprints": [[0.01, …]] }
    ]
  }
}
```

## `POST /api/speakers/import`

Imports a portable speaker library from `GET /api/speakers/export`. Accepts either `{ "library", "mode", "dryRun" }` or the library object itself as the body. `mode` is `merge` (default: union vectors into same-name speakers, create the rest) or `replace` (wipe the local library first; meeting attributions carry over by speaker name). `dryRun: true` previews via `plan` without writing. Meeting links are never imported — meeting IDs differ per instance.

```json
{
  "mode": "merge",
  "dryRun": false,
  "stats": { "speakersImported": 2, "created": 1, "merged": 1, "skipped": 0, "vectorsImported": 3, "vectorsSkippedRedundant": 1 },
  "warnings": ["…"],
  "speakers": [ … ]
}
```

## `POST /api/meetings/:id/speakers/link`

Links a speaker to a meeting. Optionally transfers voiceprint identifiers from a generic label (e.g. `S1`) in the meeting's transcript to the speaker's profile.

JSON body:
```json
{
  "speakerId": "uuid",
  "speechmaticsLabel": "S1"
}
```

---

## `GET /api/templates`

Returns a list of all summary prompt templates (built-in and custom), sorted with default first.

```json
{
  "templates": [
    {
      "id": "00000000-0000-0000-0000-000000000001",
      "name": "Executive Summary",
      "description": "Structured meeting overview...",
      "systemPrompt": "You are an expert executive assistant...",
      "userPrompt": "# Meeting Summary: {{title}}\n...",
      "isDefault": true,
      "isBuiltin": true,
      "createdAt": 1720000000000,
      "updatedAt": 1720000000000
    }
  ]
}
```

## `GET /api/templates/:id`

Returns a single summary prompt template by its UUID.

## `POST /api/templates`

Creates a custom summary prompt template (`isBuiltin: false`).

JSON body:
```json
{
  "name": "Sprint Retrospective",
  "description": "Highlights wins, blockers, and next sprint commitments",
  "systemPrompt": "You are an agile coach summarizing team retrospective.",
  "userPrompt": "# Retro: {{title}}\n**Date:** {{date}}\n\n## Summary\n{{transcript}}",
  "isDefault": false
}
```

## `PATCH /api/templates/:id` or `PUT /api/templates/:id`

Updates fields on a template. Built-in status cannot be altered.

## `POST /api/templates/:id/default`

Designates the specified template as the default summary template and removes default status from all others.

## `DELETE /api/templates/:id`

Deletes a custom template. Built-in templates cannot be deleted (returns HTTP 400).

---

## `GET /api/llm/providers`

Returns the list of supported LLM providers, whether each is configured with an API key, custom base URL status, and available model counts.

## `GET /api/llm/models?provider=&search=&limit=`

Lists models from the unified `pi-ai` model catalog. Supports filtering by provider, fuzzy search by model name/ID, and result limits.

## `POST /api/llm/refresh-catalog`

Fetches and refreshes the web model catalog from online registries (`models.dev`) and dynamic providers.

## `GET /api/llm/config`

Returns current LLM settings, including the active default provider, default model, configured provider endpoints, and custom model definitions (with API keys securely masked).

## `POST /api/llm/config`

Updates LLM settings:
```json
{
  "defaultProvider": "anthropic",
  "defaultModel": "claude-3-7-sonnet-20250219",
  "providers": {
    "openai": {
      "baseUrl": "http://localhost:11434/v1"
    }
  }
}
```

## `POST /api/llm/generate`

Executes a prompt generation using `pi-ai`:
```json
{
  "provider": "google",
  "model": "gemini-2.5-flash",
  "systemPrompt": "You are a concise executive assistant.",
  "prompt": "Summarize the key points: ..."
}
```

## `POST /api/llm/test`

Tests connectivity to an LLM provider and model by executing a quick ping prompt.
```json
{
  "provider": "google",
  "model": "gemini-2.5-flash"
}
```

---

## Backup & Restore API

### `POST /api/backup`

Creates a complete, self-contained backup archive (`.tar.gz`) containing:
- Consistent SQLite database snapshot (`VACUUM INTO` + WAL checkpoint)
- User settings (`settings.json`) and model cache (`models.json`)
- All meeting folders with **all audio recordings**, transcripts, and summaries
- Metadata manifest (`manifest.json`)

Optional JSON payload:
```json
{
  "filename": "my-custom-backup-name.tar.gz"
}
```

Response (HTTP 201):
```json
{
  "ok": true,
  "backup": {
    "filename": "olive-backup-20260817-003015.tar.gz",
    "path": "/app/data/config/backups/olive-backup-20260817-003015.tar.gz",
    "sizeBytes": 5839201,
    "createdAt": "2026-08-17T00:30:15.000Z",
    "manifest": {
      "version": "1.0.0",
      "createdAt": "2026-08-17T00:30:15.000Z",
      "oliveVersion": "0.1.0",
      "app": "olive",
      "stats": {
        "meetingCount": 14,
        "recordingCount": 14,
        "audioFilesCount": 14,
        "totalAudioSizeBytes": 154829102,
        "summaryCount": 28,
        "speakerCount": 6,
        "templateCount": 4
      }
    }
  }
}
```

### `GET /api/backup/list`

Lists all stored backup archives on the server with metadata and manifest statistics.

### `GET /api/backup/download/:filename`

Streams the specified backup `.tar.gz` archive as an attachment download.

### `GET /api/backup/export`

Creates a fresh backup archive and immediately streams it as a browser download attachment (`Content-Disposition: attachment; filename="olive-backup-..."`).

### `DELETE /api/backup/:filename`

Deletes a stored backup archive from the server.

### `POST /api/backup/restore`

Restores the database, settings, and all audio files from a backup archive. Supports:
1. **Multipart File Upload:** `multipart/form-data` with `file: <backup.tar.gz>`
2. **Server-Stored File:** `application/json` with `{ "filename": "olive-backup-..." }`

Response (HTTP 200):
```json
{
  "ok": true,
  "result": {
    "success": true,
    "restoredAt": "2026-08-17T00:35:00.000Z",
    "manifest": { ... },
    "stats": {
      "meetings": 14,
      "recordings": 14,
      "audioFiles": 14,
      "summaries": 28,
      "speakers": 6,
      "templates": 4
    }
  }
}
```

---

## Logs API

### `GET /api/logs?level=&category=&meetingId=&search=&limit=&offset=`

Queries structured system and job logs with severity level filtering, category filtering, meeting scoping, text search, and pagination.

* **Parameters:**
  * `level`: Minimum severity level (`"debug"` | `"info"` | `"warn"` | `"error"`, defaults to `"debug"`). Filters for all logs at or above this severity.
  * `category`: Optional category filter (e.g. `"ingest"`, `"transcription"`, `"diarization"`, `"summary"`, `"speakers"`, `"plaud"`, `"backup"`, `"llm"`).
  * `meetingId`: Optional meeting UUID to scope logs to a specific meeting.
  * `search`: Fuzzy search substring against log message and JSON details.
  * `limit`: Page limit (default `100`, max `500`).
  * `offset`: Page offset (default `0`).

Response (HTTP 200):
```json
{
  "ok": true,
  "logs": [
    {
      "id": "uuid",
      "level": "info",
      "category": "transcription",
      "message": "Transcription completed successfully",
      "meetingId": "uuid",
      "meetingTitle": "Architecture Sync",
      "details": {
        "segmentCount": 24,
        "durationMs": 312000
      },
      "createdAt": 1786933000000
    }
  ],
  "pagination": {
    "total": 150,
    "limit": 100,
    "offset": 0
  },
  "categories": ["backup", "diarization", "ingest", "llm", "plaud", "speakers", "summary", "system", "transcription"]
}
```

### `DELETE /api/logs?olderThanMs=`

Clears stored logs from SQLite database, with optional `olderThanMs` timestamp cutoff.

---

## Command Line Utilities

### Create Backup
```bash
cd backend && uv run olive backup [optional-output-path.tar.gz]
```

### Restore Backup
```bash
cd backend && uv run olive restore <path-to-backup.tar.gz>
```
