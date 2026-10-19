# Changelog

All notable changes to Olive are documented in this file, newest first.

Olive releases are CalVer-tagged Docker images (`YYYY.MM.DD`, with an
optional `.N` same-day counter). Each section below is dated with the UTC
day its changes merged (`## [YYYY-MM-DD]`), with entries grouped under
`Added` / `Changed` / `Fixed` — pushing a tag publishes the image and turns
every section newer than the previous tag into the GitHub release notes.

## [2026-10-19]

### Fixed

- **Large local-transcription backlogs no longer deadlock or exhaust threads.** Two bugs made a big local re-transcription backlog hang for hours with a runaway thread count:
  1. **Queue deadlock** — the local transcription driver shared the 2-slot `pipeline` queue with the child workflow it waits on, so two or more concurrent transcriptions held both slots while their children waited for a slot that never freed. The driver now runs on its own queue; `pipeline` bounds only the actual concurrent-meeting work.
  2. **Unbounded DBOS executor** — DBOS's workflow thread pool defaults to unbounded, and because the driver blocks awaiting its child (which blocks awaiting each stage) with frequent polling, a backlog spawned thousands of executor threads until the process ran out of memory. The executor is now capped (sized from the stage-queue concurrencies).
  The ML engines are also loaded once and reused across meetings instead of rebuilt per meeting, and a stalled stage is cancelled after `OLIVE_STAGE_STALL_TIMEOUT_SEC` (default 900s) of no progress (jobs merely waiting their turn are never cancelled) so one wedged stage can't block the queue.
- **Queued background work now resumes automatically on restart.** The DBOS durable-execution runtime was launched lazily, only when a transcription request first arrived, so after a restart (redeploy, reboot, or an OOM-kill) the queue-runner thread never started on its own: an enqueued backlog sat idle and in-flight jobs were never recovered until some API call happened to wake it. DBOS is now launched at server startup, so the queue runner and crash recovery come up immediately and a pending backlog drains with no manual intervention.
- **Recovered transcriptions no longer fail to decode after a restart.** A transcription whose audio was held only in memory (not yet written with an on-disk path in its job record) would die on recovery with "Either audio_path or audio_bytes must be provided", because the in-memory audio is gone once the process restarts. The audio-decode step now falls back to the recording's file on disk, looked up from the meeting, so a transcription interrupted by a restart or OOM-kill resumes instead of erroring out.

## [2026-10-16]

### Changed

- **Slim meeting lists by default** — `GET /api/meetings` and the MCP `list_meetings` / `search_transcripts` / `get_action_items` tools now return slim meetings `{ id, title, startTime, endTime, source, tags }` (~65% smaller — a 100-row MCP list drops from ~60KB to ~21KB). Pass `includeDetails=true` (REST) or `includeDetails: true` (MCP) only when you need `status`, artifact ids, timestamps, `lastError`, `audioStatus`, or `canRetryAudio`; prefer `get_meeting` for full detail on one meeting. The web UI requests full detail so the list still shows processing state and audio badges.
- **Slim speakers + meeting sections** — MCP `list_speakers` / `get_speaker_profile` no longer embed raw voiceprint vectors (each 256-float embedding is ~2-3KB of JSON): speakers are `{ id, name, enrolled, voiceprintCounts, enrolledAt, createdAt, meetingCount }` by default, with `includeVoiceprints: true` to opt into raw `providerIds`; bulk vectors remain available via `export_speaker_library`. MCP `get_meeting` defaults to full text but accepts `includeTranscript: false` / `includeSummaries: false` for metadata-only fetches, and its `speakers[]` are vector-free unless `includeVoiceprints: true`. Same flags exist on the REST routes (`GET /api/meetings/:id?includeTranscript=false&includeSummaries=false&includeVoiceprints=false`); the web UI keeps requesting everything, so nothing visibly changes.

- **Interface moved to daisyUI components with light + dark themes** — buttons, inputs, badges, cards, tables, tabs, modals, alerts, progress, skeletons, and the mobile navigation now use shared daisyUI components instead of one-off styling, so theming and future changes land in one place. The header has a sun/moon toggle switching between the default dark `olive-dark` theme and a new `olive-light` theme (remembered per browser). Native `alert()`/`confirm()` popups are replaced with in-app toasts and confirmation dialogs, the meeting chat is now a side drawer that closes on navigation, and the mobile tab bar is a 5-item dock with Logs/Backup under More.

### Added

- **Split meeting in two** — over-long recordings (meeting + forgotten commute) can be cut at any point via `POST /api/meetings/{id}/split { splitMs }`. The audio is cut with ffmpeg, every JSON transcript is divided on the same boundary and re-anchored (no re-transcription), summaries are discarded on both halves as stale, and the second half becomes a new "(Part 2)" meeting. In the app, drag the split slider under the audio player (or "Use current position"), or hover between transcript turns and hit "✂ Split meeting here", then confirm — delete the half you don't want afterwards. Also available to agents as the opt-in MCP write tool `split_meeting { meetingId, splitMs, expectedTitle?, dryRun? }` (`OLIVE_MCP_WRITE=1`).

### Fixed

- **Checkbox rendered as an empty text box** — the "Set as primary summary for this meeting" option in the Generate AI Summary modal (and the "Set as default summary template" toggle in the summary-template editor) used the wrong style class, so the checkbox appeared as a stretched, empty input instead of a checked box.
- **Split meeting controls wrapped oddly** — the scissors icon in the audio-player "Split Meeting" button, the split-panel header, the transcript hover "Split meeting here" chip, and the split-turn modal title now sit inline with their labels instead of stacking above them.
- **Auto-title failing on reasoning models** — generating a meeting title no longer sends a 32-token output cap, so reasoning models that spend their budget on internal reasoning before emitting the title no longer fail with "LLM returned an unusable title". The LLM connection test (`POST /api/llm/test`) no longer caps output either.
- **Split audio preview still played the full recording** — splitting a meeting rewrote the original audio but left the cached voice-enhanced file (`audio/{recordingId}.enhanced.mp3`) untouched, and the player defaults to the enhanced stream — so it kept showing the pre-split duration (e.g. 2:27:18 on a 24-minute half). Splitting now deletes the stale enhanced cache so it rebuilds from the cut audio, the audio endpoint rebuilds the cache whenever the source is newer than it, and the UI reloads the player and drops its cached duration after a split.
- **Upload durations (`0m 0s`)** — `POST /api/ingest` now reads `durationMs` (or `duration`) off the multipart form and the server probes the audio with `ffprobe` when the client doesn't send one, so uploaded meetings get a real `end_time` and recording `duration_ms` instead of showing 0m 0s.
- **Meetings list pagination** — the meetings page no longer dead-ends at the newest 50: it loads 100 at a time with infinite scroll (plus a Load more button and a "Showing X of Y" count), so large archives are fully browsable and searchable. `GET /api/meetings` and the MCP `list_meetings` / `search_transcripts` / `get_action_items` tools now return `pagination: { limit, offset, total, hasMore, nextOffset }` — follow `offset=nextOffset` while `hasMore` is true to page through everything.

## [2026-10-15]

<!-- Add new entries under today's date heading (newest date first) under Added / Changed / Fixed. See AGENTS.md "Changelog policy". -->

### Added

- **MCP read + control surface** — the MCP endpoint at `/mcp` now exposes `list_templates` / `get_template`, `get_transcription_status` / `list_active_transcriptions` / `get_summary_status`, and `plaud_status` as always-on read tools (no opt-in needed) so agents can discover prompt templates and poll job progress without subscribing to SSE. Two new opt-in write tools (`OLIVE_MCP_WRITE=1`): `reassign_speaker { meetingId, fromLabel, toSpeakerName?/toSpeakerId?, adoptVoiceprint?, localOnly?, segmentIndex?, scope?, expectedFromLabel?, dryRun? }` (moves transcript segments to a target speaker, optionally growing the voiceprint library) and `plaud_sync_trigger` (kicks a Plaud Cloud poll in the background).
- **MCP summary management** — the opt-in MCP write toolset (`OLIVE_MCP_WRITE=1`) now exposes `delete_summary { meetingId, summaryId, expectedIsPrimary?, dryRun? }` and `set_primary_summary { meetingId, summaryId, expectedPrimarySummaryId?, dryRun? }`, matching the existing `DELETE /api/meetings/{id}/summaries/{artifactId}` and `POST .../primary` routes. Use `get_meeting` (`summaries[].id`, `isPrimary`) to pick a target, preview with `dryRun`, then apply with the `expected*` guard from the plan so concurrent edits abort instead of clobbering.
- **Portable speaker library export/import** — share enrolled voiceprints between Olive instances without re-enrolling: `GET /api/speakers/export` returns the library as JSON (vectors + metadata, no audio) and `POST /api/speakers/import` applies it in `merge` mode (union into same-name speakers, create the rest) or `replace` mode (wipe local first, meeting attributions carry over by name), both with `dryRun` previews. Available in MCP too: `export_speaker_library` is a read tool, `import_speaker_library` is an opt-in write tool (`OLIVE_MCP_WRITE=1`). Meeting links are never transferred — meeting IDs differ per instance.
- **MCP write tools (opt-in)** — `/mcp` now offers `update_speaker`, `merge_speakers`, `delete_speaker`, `set_meeting_tags` (single or bulk via `meetingIds` up to 500), `update_meeting_title`, plus meeting lifecycle (`delete_meeting`, `update_meeting_start_time`, `auto_title_meeting`, `summarize_meeting`) and transcription (`transcribe_meeting`, `bulk_transcribe`, `cancel_transcription`, `retry_meeting_audio`, `reidentify_speakers`) alongside the read tools. They stay hidden and rejected unless the server runs with `OLIVE_MCP_WRITE=1`, and every tool accepts `dryRun` to preview plus `expected*` guards that abort on mismatch (long-running ops queue async and return job status), so agents can plan a speaker/tag/title cleanup against live data and apply on confirm.
- **Audio-pending meetings + Plaud audio refetch** — meetings whose audio hasn't landed yet now show a dimmed card with a "Retrieving audio…" (or "Audio missing") badge in the list instead of looking ready, and the detail page shows a banner explaining the state with Transcribe and Generate Summary disabled until audio attaches. Plaud meetings can re-queue the cloud download via a Retry download button (new `POST /api/meetings/{id}/audio/retry`), backed by `audioStatus`/`canRetryAudio` on the meeting payload.
- **Transcription API keys in Settings UI** — Speechmatics, AssemblyAI, and Muse Voice keys can now be entered in Settings alongside the transcription engine picker instead of only via environment variables. Keys are stored in `settings.json`, masked in API responses (re-saving a masked value keeps the stored key), take effect immediately without a restart, and environment variables still override them when set.
- **Muse Voice transcription** — transcribe meetings with Meta's
  Muse Voice Transcribe (`muse-voice-transcribe-1.0`, $0.18/hr) via
  `MUSE_API_KEY`. Uses speaker diarization by default, auto-chunks
  recordings over the 10-minute API limit, and supports keyword and
  language biasing from the existing transcription options. Speaker
  labels are kept distinct per chunk (labels are request-scoped) and
  reconciled to enrolled names by voiceprint matching. Selectable
  anywhere AssemblyAI/Speechmatics are (upload, meeting detail, bulk
  re-transcribe, default engine setting).
- **Auto-title meetings** — generate a concise meeting title from the
  transcript with the configured default LLM via the Auto-title button
  next to the meeting title.
- **Always re-transcribe Plaud recordings with the configured engine** — Settings now offers a three-way Plaud recordings policy (use Plaud transcript, re-transcribe when speakers are unnamed, or always re-transcribe). The always mode keeps Plaud's transcript as a fallback and makes the configured engine authoritative, takes effect without a restart, and skips the redundant Plaud speaker-identification pass. Meetings that arrive with a summary but no transcript segments now always hand off to the configured engine, which also covers running Plaud hardware without a transcription subscription (turn AutoFlow off in Plaud and Olive transcribes from audio).
- **Tag + facet filtering (API + MCP)** — `GET /api/meetings` accepts exact `tags` (repeatable or comma-separated, AND-combined, case-insensitive), `status`, `speakerId`, `hasSummary`, and `hasTranscript` filters that compose with the existing search/source/date filters; new `GET /api/tags` lists distinct tags with meeting counts. The MCP `list_meetings` tool accepts `search` plus the same filters, `search_transcripts` takes `query` plus the same filters except `search`, and `get_action_items` takes the filters except `search`, plus a new `list_tags` tool — so agents can scope "all calls for this client" to a tagged subset instead of keyword-matching transcripts. Bulk re-transcribe filters accept the same set.
- **Transcript-only speaker names** — the Assign Speaker dialog now offers
  "This transcript only" alongside "Save to People", so one-off voices
  (e.g. an interview guest) can be relabeled in that meeting without
  creating a People entry, linking the meeting, or learning a voiceprint.
- **Editable meeting time** — the meeting date on the detail page can now
  be changed via an edit button next to it. The duration is preserved
  (end time shifts with the start) and the on-disk folder is moved when
  the UTC date stamp changes, so shifted imports can be corrected in place.
- **CalVer releases** — pushing a `YYYY.MM.DD` tag (`.N` suffix for same-day
  follow-ups, no `v` prefix) runs CI, publishes the multi-arch image to GHCR
  pinned to that tag (plus `:latest`), and creates a GitHub release whose
  notes are the changelog sections since the previous tag. Release builds
  also stamp the tag into the image, so What's New shows e.g.
  `2026.10.15 · 4d5ea70` instead of just the build date and commit.

### Changed

- **Build identity instead of version number** — the What's New header now shows the build's UTC date/time and commit hash (e.g. `2026-10-15 14:32 UTC · 4d5ea70`) instead of a static version, matching continuous delivery with no releases.
- **Faster container startup** — model preflight verifies the four local
  models from the on-disk cache with no Hugging Face Hub round-trips and
  without importing the heavy ML stack (`torch`, `transformers`) on a warm
  boot, checks them concurrently with per-model timing, and the image now
  starts via the venv binary instead of `uv run`.

### Fixed

- **Deleted Plaud recordings stay deleted** — deleting a Plaud-synced
  meeting now tombstones its cloud recording ID, so the next poll or
  restart no longer re-imports the same recording as a new meeting.
- **Human-verified voiceprints always enroll** — confirming a speaker turn,
  reassigning a voice, or rapid-enrolling checked samples now trains the
  voiceprint even when the clip scores below the outlier threshold; a low
  score expands the voiceprint instead of rejecting your confirmation.
- **Speaker re-identification works on Plaud transcripts** — re-running
  speaker matching no longer fails with "could not be parsed" on meetings
  whose transcript was already rewritten in Plaud's list format.
- **Plaud timestamps parsed as UTC** — timezone-less `start_at` values from
  Plaud are now interpreted as UTC instead of the server's local timezone,
  so meetings no longer shift by the host UTC offset on ingest.

## [2026-08-11 — 2026-10-14]

Covers the full history from project start (2026-08-11) through 2026-10-14,
including the original TypeScript/Bun prototype era and the Python
FastAPI + Vite rewrite (2026-10-13).

### Added

- **Meetings management** — meeting list/detail views; rename with atomic
  on-disk folder move (`YYYY-MM-DD_<title>__<id8>`) and no re-import; tags
  (add/edit, trimmed and de-duplicated); delete from UI removing DB rows,
  artifacts, recordings, chat, Plaud state, logs, and the meeting folder.
- **Ingest** — audio upload API (`POST /api/ingest`, iOS Shortcut compatible);
  Plaud Cloud sync with OAuth 2.0 PKCE modal, 15-minute poller, auto-transcription,
  duplicate-audio detection (`duplicate` status, no re-download), migrated `of_`
  file-ID handling, HTTP 429 backoff with `Retry-After`, and background
  "Sync Now" (202 immediately).
- **Transcription providers** — Speechmatics batch STT with diarization and
  webhook; local on-device pipeline (Cohere Transcribe INT8 ONNX, IBM Granite
  Speech, Qwen3-ASR default, CTC forced alignment); AssemblyAI Universal-3.5
  provider with live upload/polling progress; provider selection honored with a
  UI dropdown and per-request override.
- **Transcription UX** — real-time SSE progress streaming that survives page
  navigation and covers runs started elsewhere; background execution (worker
  thread, then durable DBOS stage queues); cancellation that also cancels the
  underlying DBOS workflow and stays cancelled; bulk re-transcribe across
  meetings (`POST /api/meetings/bulk-transcribe`, selection checkboxes, queue
  backed); whole-number progress with realistic speaker-ID weighting.
- **Speakers & diarization** — speaker registry and People dashboard; in-meeting
  enrollment, reassignment, merging, and renaming; neural diarization
  (pyannote community-1) with WeSpeaker ResNet293-LM voiceprints as the single
  speaker-identity store; cross-recording voiceprint matching and ASR-first
  cloud speaker identification; Plaud transcript speakers matched to enrolled
  voiceprints; post-facto "Re-identify" for any transcript
  (`POST /api/meetings/{id}/speakers/reidentify`); rapid enrollment flow for
  unmatched groups; 12-second trim for over-long samples; voiceprint cap with
  outlier/near-duplicate filtering and UI notices; generic "Speaker N" labels
  treated as transcript-only (never people); segment coalescing, pruning, and
  inline segment add/edit/delete; tunable diarization/voiceprint thresholds.
- **Summaries, chat & search** — prompt templates and multi-provider LLM calling
  with per-model thinking levels; asynchronous summary generation with status
  polling; multi-summary management with markdown rendering and raw/rendered
  toggle; per-meeting chat grounded in summary, transcript, and speakers;
  full-text content search with auto-summarization on transcription completion.
- **Usage tracking** — cloud transcription (per audio hour) and LLM (per token)
  spend recorded with estimated USD, history backfilled, Usage tab plus
  per-meeting cost line.
- **System & platform** — backup/restore via API, UI, and CLI (root and legacy
  archive layouts); structured system logging with Logs API and UI view;
  read-only MCP server at `/mcp` (search meetings, read transcripts/summaries,
  action items); `/debug` DBOS stage-queues page and API with live per-stage
  display, bounded pipeline concurrency, and inter-stage queue status; favicons
  and PWA manifest; mobile experience (bottom tab bar, modal shell, no iOS zoom
  on inputs, select mode, responsive cards); FFmpeg speech-enhancement filter
  with player toggle.
- **Distribution & runtime** — Docker as primary distribution: first start
  downloads speech models (~4.5 GB) into a persistent `/app/models` volume and
  refuses to run without it; multi-arch (amd64/arm64) images built on native
  runners; Python 3.12 FastAPI/SQLAlchemy backend with Vite 6 + pnpm frontend
  and OpenAPI-generated types; durable DBOS SQLite stage queues
  (`pipeline`, `plaud_sync`, `ffmpeg`, `diarize`, `speaker_embed`, `asr`,
  `align`, `speechmatics`, `assemblyai`, `summarize`); ONNX inference offloaded
  to threads; clean shutdown of workflows, poller, SSE, and DBOS on SIGINT.

### Fixed

- Plaud duplicate-audio re-download loop (hundreds of GB re-fetched every poll
  cycle) resolved with permanent duplicate marking.
- Plaud auto-poll never firing (scheduler bound to a dead loop); now starts in
  the FastAPI lifespan. Manual sync, legacy ID migration, and 404 healing fixed.
- Headerless Plaud Opus recordings rejected by ffmpeg/cloud ASR; raw packets are
  now canonicalized to valid Ogg on disk (with startup migration of legacy files).
- Transcription provider config ignored (hard-defaulted to Speechmatics) and
  upload failures silent; now controlled and surfaced as explicit error events.
- Cancelled transcriptions flipping back to in-progress on late updates;
  recovered transcripts after restart now persist and mark ready.
- LLM settings wiping stored API keys on blank save; custom provider base URLs
  breaking across SDKs (`/v1` double-prefix).
- Speaker rename reverting until reload; deleting a meeting resetting page
  scroll/loading state; long transcription progress decimals.
- Plaud import showing redundant `Auto_sum_note` heading and leading separators;
  transcript parser rejecting `item.text` / `startMs` shapes and top-level array
  transcripts.
- Container restarts losing Plaud OAuth tokens (now persisted in `configDir` and
  included in backups); custom Speechmatics client requiring an API key.

### Changed

- Backend migrated from Bun/Hono TypeScript monorepo to Python FastAPI; frontend
  build migrated to Vite + pnpm (SQLite schema, config paths, and meeting folder
  layout preserved).
- Background work moved from subprocess/worker-thread to the DBOS durable
  job-queue system (the sole enqueue path; see AGENTS.md rule 5).
- Default local ASR model switched to Qwen3-ASR; diarizer/embedding throughput
  scales with host CPU cores.
- Test suite runs in ~17s via fake neural models and live-test gating.

### Infrastructure / chores

- Dockerfile with model pre-caching and GHCR publish; isolated multi-stage model
  layer with registry caching; staging deploy script and image tags.
- GitHub Actions bumped to latest majors (Node 24); Bun runtime 1.4 (TS era);
  pinned llama.cpp builds per platform (TS era).
- Docs: planning docs (HIGHLEVEL, MILESTONES, DESIGN), diarization-fix notes, ML
  acceleration plan, README features/architecture/screenshots, API docs,
  verification policy for coding agents.
