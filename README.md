<div align="center">

# Olive

**Self-hosted meeting transcription, speaker diarization, and audio archive**

Ingests audio from Plaud hardware recorders, iOS Voice Memos, and browser uploads. Runs local or cloud transcription, matches speakers across meetings with neural voiceprints, and generates Markdown summaries from prompt templates.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-blue?style=flat-square&logo=typescript)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19-61dafb?style=flat-square&logo=react)](https://react.dev/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-v3-38bdf8?style=flat-square&logo=tailwindcss)](https://tailwindcss.com/)
[![SQLite](https://img.shields.io/badge/SQLite-SQLAlchemy%202-003B57?style=flat-square&logo=sqlite)](https://sqlite.org/)
[![pyannote.audio](https://img.shields.io/badge/Diarization-pyannote%20community--1-green?style=flat-square)](https://github.com/pyannote/pyannote-audio)
[![WeSpeaker](https://img.shields.io/badge/Voiceprints-WeSpeaker%20ResNet293-orange?style=flat-square)](https://github.com/wenet-e2e/wespeaker)
[![Cohere Transcribe](https://img.shields.io/badge/Local%20ASR-Cohere%20Transcribe%20INT8-purple?style=flat-square)](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)](https://docker.com)

<br/>

<img src="docs/screenshots/meeting-detail.png" alt="Olive Meeting Detail View" width="100%" />

</div>

---

## Features

| Feature | Description |
| :--- | :--- |
| **Multi-source audio ingestion** | Upload `.m4a`, `.mp3`, `.wav`, `.aac`, `.ogg`, `.flac`, `.webm` with SHA-256 deduplication and optional auto-transcription. |
| **Plaud Note integration** | OAuth 2.0 PKCE sync with Plaud Cloud. Polls in the background for recordings, transcripts, and AI notes. |
| **Apple Voice Memos shortcut** | One-tap iOS Share-Sheet shortcut to upload recordings directly from iPhone or iPad. |
| **Dual transcription pipeline** | Local neural STT (Cohere Transcribe 03-2026 INT8 ONNX + CTC forced alignment) or cloud Speechmatics. |
| **Speaker diarization & voiceprints** | Local `pyannote.audio` (`speaker-diarization-community-1`) + WeSpeaker ResNet293-LM 256-dim embeddings for cross-meeting speaker recognition. |
| **Multi-provider LLM summaries** | Powered by `any-llm` and the `models.dev` catalog. Works with Gemini, Claude, OpenAI, Groq, OpenRouter, Ollama, or custom OpenAI-compatible proxies. |
| **Summary templates** | Prompt templates (`Executive Summary`, `1-on-1 Catchup`, `Technical Architecture & Design Review`, `Action Items & Decisions`) with `{{title}}`, `{{date}}`, `{{speakers}}`, and `{{transcript}}` variables. |
| **System & job logs** | Streaming log viewer with category filters, severity levels (`DEBUG` to `ERROR`), and JSON detail inspection. |
| **Backup & restore** | Single-click `.tar.gz` export and restore containing the SQLite snapshot, settings, speaker clips, transcripts, summaries, and raw audio files. |

---

## User guides

- [Voiceprint matching: what it is and how to tune it](docs/VOICEPRINT_MATCHING.md) — how Olive learns voices, what the clustering and voiceprint-matching sliders do, and how to fix mislabeled speakers (including the "favorite speaker" effect).

---

## Screenshots

### Meeting archive & detail view

<div align="center">
  <img src="docs/screenshots/meetings-dashboard.png" alt="Olive Meetings Dashboard" width="100%" />
  <p><em>Meetings dashboard with full-text search across titles, tags, transcripts, and summaries.</em></p>
</div>

<br/>

<div align="center">
  <img src="docs/screenshots/meeting-detail.png" alt="Meeting Detail with Audio and Summary" width="100%" />
  <p><em>Meeting detail view with seekable audio player, speaker assignment pills, and Markdown summaries.</em></p>
</div>

---

### LLM provider & local ASR settings

<div align="center">
  <img src="docs/screenshots/llm-settings.png" alt="LLM Provider and ASR Settings" width="70%" />
  <p><em>Model catalog, reasoning/thinking level config, Hugging Face token, and diarization/voiceprint thresholds.</em></p>
</div>

---

### Ingestion & Plaud sync

<div align="center">
  <table>
    <tr>
      <td width="50%" align="center">
        <img src="docs/screenshots/upload-modal.png" alt="Upload Audio Recording" />
        <br/>
        <em>Direct audio upload with auto-transcribe toggle and engine selector.</em>
      </td>
      <td width="50%" align="center">
        <img src="docs/screenshots/plaud-integration.png" alt="Plaud Device Sync" />
        <br/>
        <em>Plaud Cloud OAuth 2.0 PKCE login and background sync.</em>
      </td>
    </tr>
  </table>
</div>

---

### Summary templates, logs & backups

<div align="center">
  <img src="docs/screenshots/summary-templates.png" alt="Summary Prompt Templates" width="100%" />
</div>

<br/>

<div align="center">
  <img src="docs/screenshots/system-logs.png" alt="System and Job Logs" width="100%" />
</div>

<br/>

<div align="center">
  <img src="docs/screenshots/backup-restore.png" alt="Backup and Restore" width="100%" />
</div>

---

## Architecture

```
SOURCES                           PIPELINE & ARTIFACT STORE                       INTELLIGENCE & EXPORT
───────                           ─────────────────────────                       ─────────────────────

 [ Plaud Note AI ] ──(OAuth/PCS)─┐
                                 │      ┌─────────────────────────────┐
 [ iOS Shortcut ]  ──(Ingest)───┼────► │   Meeting Aggregate Root    │
                                 │      │  ─────────────────────────  │
 [ Web UI Upload ] ──(Upload)───┘      │  • sha256 deduplicated      │
                                        │  • audio/<id>.<ext>         │
                                        │  • SQLite metadata + WAL    │
                                        └──────────────┬──────────────┘
                                                       │
                                 ┌─────────────────────┴─────────────────────┐
                                 ▼                                           ▼
                    ┌─────────────────────────┐                 ┌─────────────────────────┐
                    │     Transcription &     │                 │   LLM Summarization &   │
                    │   Speaker Diarization   │                 │      Meeting Chat       │
                    │ ─────────────────────── │                 │ ─────────────────────── │
                    │ • Cohere Transcribe INT8│                 │ • any-llm + models.dev  │
                    │ • MMS CTC Forced Aligner│                 │ • Gemini / Claude / GPT │
                    │ • pyannote community-1  │                 │ • Local Ollama / vLLM   │
                    │ • WeSpeaker ResNet293   │                 │ • Structured Templates  │
                    │ • Cloud: Speechmatics   │                 │ • /mcp Server (opt-in writes) │
                    └─────────────────────────┘                 └─────────────────────────┘
                                 │                                           │
                                 └─────────────────────┬─────────────────────┘
                                                       │
                                                       ▼
                                        ┌─────────────────────────────┐
                                        │     React 19 + Vite SPA     │
                                        │  Markdown • Audio • Backup  │
                                        └─────────────────────────────┘
```

---

## Quickstart (Docker — recommended)

The multi-arch image (`linux/amd64`, `linux/arm64`) is the primary way to run Olive. On first start it downloads ~4.5 GB of local speech models (WeSpeaker, MMS aligner, pyannote diarization, Cohere Transcribe INT8) into a persistent models volume, then starts the server. Model downloads survive upgrades because they live on the volume, not in the image.

```bash
# Pull and start (mounts required: olive2_data for meetings/DB, olive2_models for ML models)
HF_TOKEN=hf_... OLIVE_MCP_TOKEN=... docker compose up -d

# Or with plain docker run:
docker run -d \
  --name olive2 \
  -p 4471:4470 \
  -e HF_TOKEN=hf_... \
  -e OLIVE_MCP_TOKEN=... \
  -v olive2_data:/app/data \
  -v olive2_models:/app/models \
  --restart unless-stopped \
  ghcr.io/mcowger/olive2:latest
```

Open **`http://localhost:4471`**. Follow the logs on first start (`docker logs -f olive2`) — model downloads take several minutes.

The server **refuses to start** if `/app/models` is not a mounted, writable volume, or if a model cannot be downloaded — the error message says why. Do not skip the `olive2_models` mount: without it, every upgrade would re-download gigabytes of models.

### Pinned releases

Every release is also published under a CalVer tag (`YYYY.MM.DD`, e.g. `ghcr.io/mcowger/olive2:2026.10.15`, with a `.N` suffix for same-day follow-ups) alongside a GitHub release carrying the changelog notes — no artifacts, the release points at the GHCR image. Pin the image tag in `compose.yml` (or your `docker run` command) to hold an upgrade until you're ready; `:latest` tracks the newest release.

### Local installation (advanced)

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12+), [pnpm](https://pnpm.io/) (Node.js 22+), `ffmpeg`, and `libsndfile1`.

```bash
# 1. Clone the repository
git clone https://github.com/mcowger/olive2.git
cd olive2

# 2. Install Python backend dependencies
cd backend && uv sync && cd ..

# 3. Pre-download local speech, diarization, alignment & voiceprint models
cd backend && uv run olive models download && cd ..

# 4. Install and build the React frontend
cd frontend && pnpm install && pnpm run build && cd ..

# 5. Start the server
cd backend && uv run olive serve
```

Open **`http://localhost:4471`**.

---

## Configuration & environment variables

Olive stores runtime settings in `<configDir>/settings.json` and accepts environment variables for bootstrap and secrets:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` / `OLIVE_BIND_PORT` | `4471` | HTTP server port |
| `OLIVE_BIND_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` in Docker) |
| `OLIVE_CONFIG_DIR` | `~/.config/olive` | Directory for `olive.sqlite`, `settings.json`, and speaker clips (`~/Library/Application Support/olive` on macOS) |
| `OLIVE_MEETINGS_DIR` | `<configDir>/meetings` | Storage directory for meeting folders (`audio/`, `transcripts/`, `summaries/`) |
| `OLIVE_MODELS_DIR` | `<configDir>/models` (`/app/models` in Docker) | Cache directory for local Hugging Face and ONNX models. In Docker this **must** be a mounted persistent volume — the server exits if it isn't mounted or writable |
| `HF_TOKEN` | *(optional)* | Hugging Face token for gated `pyannote` models (overrides `hfToken` in Settings UI) |
| `OLIVE_INGEST_TOKEN` | *(optional)* | Bearer token required for `POST /api/ingest` |
| `OLIVE_MCP_TOKEN` | *(optional)* | Bearer token for `/mcp`; falls back to `OLIVE_INGEST_TOKEN` and is required for non-loopback binds |
| `OLIVE_MCP_WRITE` | `0` | Set to `1` to enable MCP write tools (speakers, tags incl. bulk, titles, meeting lifecycle, transcription, summaries); off by default |
| `PLAUD_TOKEN_PATH` | `<configDir>/plaud-tokens.json` | OAuth token file for Plaud Cloud (migrates `~/.plaud/tokens.json` automatically if present) |
| `SPEECHMATICS_API_KEY` | *(optional)* | API key for Speechmatics cloud batch transcription (overrides the key in Settings UI) |
| `ASSEMBLY_API_KEY` | *(optional)* | API key for AssemblyAI cloud transcription (Universal-3.5 Pro; `ASSEMBLYAI_API_KEY` also accepted; overrides the key in Settings UI) |
| `MUSE_API_KEY` | *(optional)* | API key for Muse Voice transcription (`MODEL_API_KEY` also accepted; overrides the key in Settings UI) |
| `SPEECHMATICS_WEBHOOK_SECRET` | *(optional)* | Shared secret for `POST /api/webhooks/speechmatics` |

### MCP endpoint

Olive exposes a Model Context Protocol server at `/mcp` on the main HTTP port. Read tools are always available (`list_meetings`, `get_meeting`, `search_transcripts`, `get_action_items`, `list_speakers`, `get_speaker_profile`, `export_speaker_library`, `list_templates`, `get_template`, `get_transcription_status`, `list_active_transcriptions`, `get_summary_status`, `plaud_status`). Write tools are off by default and appear only when the server runs with `OLIVE_MCP_WRITE=1`: speaker edits (`update_speaker`, `merge_speakers`, `delete_speaker`, `reassign_speaker`), tags incl. bulk (`set_meeting_tags` with `meetingId` or `meetingIds`), titles and lifecycle (`update_meeting_title`, `update_meeting_start_time`, `auto_title_meeting`, `summarize_meeting`, `delete_summary`, `set_primary_summary`, `delete_meeting`), and transcription + Plaud (`transcribe_meeting`, `bulk_transcribe`, `cancel_transcription`, `retry_meeting_audio`, `reidentify_speakers`, `plaud_sync_trigger`). Every write tool accepts `dryRun` to preview without mutating plus `expected*` guards that abort when live data no longer matches (long-running ops queue async and return job status), so agents can plan a cleanup and apply on confirm.

Loopback clients can connect without a token. Set `OLIVE_MCP_TOKEN` (or `OLIVE_INGEST_TOKEN`) when binding to a non-loopback interface.

```json
{
  "mcpServers": {
    "olive": {
      "url": "http://127.0.0.1:4471/mcp",
      "headers": {
        "Authorization": "Bearer ${OLIVE_MCP_TOKEN}"
      }
    }
  }
}
```

---

## Integrations

### 1. Apple Voice Memos (iOS Shortcut)
1. Open **Shortcuts** on iOS and create a new shortcut.
2. Set **Receive:** `Audio` and `Files` from the Share Sheet.
3. Add **Get Contents of URL**:
   - **URL:** `https://<your-olive-host>/api/ingest`
   - **Method:** `POST`
   - **Headers:** `Authorization: Bearer <YOUR_OLIVE_INGEST_TOKEN>`
   - **Request Body:** `Form` (`file` = `Shortcut Input`, `autoTranscribe` = `true`, `source` = `ios-shortcut`)

### 2. Plaud Note & Plaud Cloud
1. Click **Plaud** in the top navigation bar.
2. Click **Start Login**, authorize in your browser, and paste the callback URL.
3. Olive polls Plaud Cloud in the background and imports recordings, transcripts, and AI notes.

---

## Backup & restore

- **Web UI:** Open the **Backup** tab to download a `.tar.gz` archive, save a snapshot on the server, or upload an archive to restore.
- **CLI:**
  ```bash
  cd backend && uv run olive backup ./my-backup.tar.gz
  cd backend && uv run olive restore ./my-backup.tar.gz
  ```

---

## Development

```bash
# Install tooling and git hooks (lefthook, managed via mise)
mise install && lefthook install

# Start backend with live reload
cd backend && uv run uvicorn olive.main:app --reload --port 4471

# Start Vite dev server (proxies /api and /mcp to http://127.0.0.1:4471)
cd frontend && pnpm run dev

# Lint, typecheck, and run backend tests
cd backend && uv run ruff check src tests eval && uv run basedpyright && uv run pytest

# Export OpenAPI schema, regenerate frontend TS types, typecheck, and build
cd backend && uv run python -c "import json; from olive.app import export_openapi_schema; open('openapi.json', 'w').write(json.dumps(export_openapi_schema(), indent=2) + '\n')"
cd ../frontend && pnpm run generate:api && pnpm run typecheck && pnpm run build
```

---

## License

MIT © [Matt Cowger](https://github.com/mcowger)
