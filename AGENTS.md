# AGENTS.md

Repository guide for coding agents working in Olive.

## Stack & repository layout

Olive is a two-part repository:
- `backend/`: Python 3.12 backend managed by `uv` (`backend/pyproject.toml`, `backend/uv.lock`).
  - `src/olive/main.py` & `app.py`: FastAPI application factory, HTTP routes, SSE streams, and SPA static serving.
  - `src/olive/cli.py`: `typer` CLI (`olive serve`, `olive backup`, `olive restore`, `olive models download`).
  - `src/olive/settings.py` & `layout.py`: OS path resolution (`OLIVE_CONFIG_DIR`, `OLIVE_MEETINGS_DIR`, `OLIVE_MODELS_DIR`), `settings.json` persistence, and meeting folder naming (`<date>_<sanitized_title>__<id8>`).
  - `src/olive/schemas.py`: Pydantic v2 domain and API models (camelCase JSON serialization).
  - `src/olive/db/`: SQLAlchemy 2.0 models (`models.py`), engine/session management (`session.py`), and migrations/seeding (`migrations.py`). Alembic lives in `backend/alembic.ini` and `backend/migrations/`.
  - `src/olive/services/`: Domain services (`meetings.py`, `ingest.py`, `transcription.py`, `speakers.py`, `summaries.py`, `chat.py`, `templates.py`, `llm.py`, `backup.py`).
  - `src/olive/ml/`: Local ML speech pipeline:
    - `protocols.py`: `AsrBackend`, `Aligner`, `Diarizer`, `SpeakerEmbedder` protocols.
    - `audio.py`: FFmpeg audio decoding, speech enhancement filter, and `soundfile` WAV codec.
    - `diarize.py`: `PyannoteSpeakerDiarizer` (`pyannote/speaker-diarization-community-1`).
    - `embed.py` & `scoring.py`: `WeSpeakerEmbeddingExtractor` (`Wespeaker/wespeaker-voxceleb-resnet293-LM` ONNX, 256-dim) and `numpy` cosine/centroid scoring.
    - `asr_cohere.py`: `CohereAsrEngine` (`vigneshlabs/cohere-transcribe-03-2026-int8-onnx` via `onnxruntime` + `transformers` `CohereAsrProcessor`).
    - `align.py`: `CtcForcedAligner` (`ctc-forced-aligner` MMS-300m ONNX).
    - `pipeline.py`: `LocalTranscriptionPipeline` (decode/diarize/embed/ASR/align engine composition). `worker.py` only holds the `AbortController`/`AbortSignal` cancellation primitives — orchestration lives in `workflows/`.
  - `src/olive/providers/speechmatics/`: Speechmatics batch client and json-v2 normalizer.
  - `src/olive/plaud/`: Vendored Plaud Cloud OAuth 2.0 PKCE client (`sdk.py`, `client.py`, `auth.py`), 429 retry backoff (`retry.py`), and background poller (`poller.py`).
  - `src/olive/mcp_server.py`: Model Context Protocol server mounted at `/mcp` (read tools always on; write tools gated by `OLIVE_MCP_WRITE=1`).
  - `src/olive/workflows/`: Durable DBOS job-queue system — **the SOLE acceptable path for enqueuing background work** (see rule 5).
    - `queues.py`: Named queues and worker concurrency (`STAGE_QUEUE_SPECS`): `plaud_sync` (1), `ffmpeg` (2), `diarize` (1), `speaker_embed` (1), `asr` (1), `align` (1), `speechmatics` (2), `assemblyai` (2), `muse` (2), `summarize` (2).
    - `transcription_workflow.py`: local transcription is a **flat stage hand-off** — `ffmpeg_decode → diarize → {speaker_embed ‖ asr} → align`, one fire-and-forget DBOS workflow per stage on its own queue, ids `stage:{meeting_id}:{run8}:{stage}`. No workflow ever awaits another workflow: a stage runs its step, records `stage_runs`, claims/enqueues its successors (the `align` fan-in is claimed atomically from `stage_runs`) and returns. Each stage rebuilds its inputs from run-scoped JSON documents in the application database's `stage_checkpoints` table (cleared on completion, failure, or cancellation) and decodes audio from the recording itself (no shared WAV on disk). Stuck stages are bounded by per-stage DBOS step timeouts (`OLIVE_STAGE_TIMEOUT_SEC` overrides), not by a watchdog; a runtime teardown cancels the task and leaves the workflow PENDING so the next launch resumes it, and the final `align` stage owns persistence. `enqueue_stage_job(queue, fn, deduplication_id=..., workflow_id=..., label=..., meeting_id=...)` wraps a callable as a durable queued step; `cancel_dbos_workflows_for_meeting()` cancels queued/running workflows per meeting; `runtime.py` launches the SQLite-backed DBOS runtime (pytest-aware: temp dirs under test, never `~/.config/olive`) with a bounded executor pool (`OLIVE_DBOS_MAX_EXECUTOR_THREADS`).
  - `tests/`: `pytest` suite (`test_*.py`).
  - `eval/`: AMI corpus evaluation harness (`run_ami_eval.py`, `RESULTS.md`).
- `frontend/`: React 19 + TypeScript + Tailwind v3 SPA built with Vite and `pnpm`.
  - `src/main.tsx`: Application UI.
  - `src/api/schema.d.ts`: Generated from `../backend/openapi.json` via `openapi-typescript`. Do not hand-edit.
  - `src/api/types.ts`: Type aliases re-exported from `schema.d.ts`.

## Common commands

### Backend (`cd backend`)
- Install/sync deps: `uv sync`
- Lint: `uv run ruff check src tests eval`
- Typecheck: `uv run basedpyright`
- Run tests: `uv run pytest`
- Start dev server: `uv run uvicorn olive.main:app --reload --port 4471`

### Frontend (`cd frontend`)
- Install deps: `pnpm install`
- Regenerate OpenAPI types (after changing Pydantic schemas in `backend/src/olive/schemas.py`):
  ```bash
  cd backend && uv run python -c "import json; from olive.app import export_openapi_schema; open('openapi.json', 'w').write(json.dumps(export_openapi_schema(), indent=2) + '\n')"
  cd ../frontend && pnpm run generate:api
  ```
- Typecheck: `pnpm run typecheck`
- Build production bundle: `pnpm run build`

### Full Dev Stack & FRP Tunnel (repo root)
- Start backend + Vite HMR + optional `frpc` HTTP tunnel: `python3 scripts/dev.py`
- Get deterministic worktree FRP tunnel URL (`FRPC_SERVER_ADDR`, `FRPC_AUTH_TOKEN`, `FRPC_SUBDOMAIN_HOST`):
  - `python3 scripts/dev_config.py frp_url`
  - `python3 scripts/dev_config.py frp_url --json`
  - `python3 scripts/dev_config.py frp_url --subdomain`
  - `python3 scripts/dev_config.py frp_url --hostname`
  - `python3 scripts/dev_config.py frp_url --file`

## Paseo worktrees (`paseo.json`)

Paseo runs `worktree.setup` entries and `scripts` commands in non-interactive `bash` (one shell per entry), inheriting the **daemon's** `PATH` — a frozen snapshot of an activated mise env from daemon start. It contains absolute install bins for globally-declared tools (`uv`, `node`, …), so those resolve fine; tools declared **only** in this repo's `mise.toml` (`pnpm`, `lefthook`) land in `~/.local/share/mise/installs` but never reach `PATH`. `mise install` installs them without touching `PATH`, and mise shims are not on the daemon `PATH` either — so bare `pnpm` fails with `pnpm: command not found`.

Wrap every repo-only tool invocation in `mise exec --`, which resolves versions from `mise.toml` at exec time (needs the config trusted — setup line 1 runs `mise trust`):
- setup entries: `cd frontend && mise exec -- pnpm install --frozen-lockfile`
- scripts: `cd frontend && mise exec -- pnpm run build` / `... pnpm run typecheck` / `... pnpm run generate:api`
- services: `PORT=$PASEO_PORT mise exec -- python3 scripts/dev.py` — the wrapper is load-bearing, because `scripts/dev.py` spawns a bare `pnpm exec vite` child that inherits the augmented `PATH`.

This applies to any tool added to `mise.toml` later: the daemon `PATH` stays frozen until the daemon restarts, so either wrap it in `mise exec --` or restart the daemon after `mise install`. To confirm what a Paseo shell actually sees: `tr '\0' '\n' < /proc/<daemon-pid>/environ | grep '^PATH=' | tr ':' '\n'` (pid is in `~/.paseo/paseo.pid` as JSON).

## Git hooks

Git hooks are managed with [lefthook](https://lefthook.dev) (`lefthook.yml` at repo root, binary via `mise.toml`). Contributors enable them with `mise install && lefthook install`. Active pre-commit jobs: backend `ruff` + `basedpyright`, frontend `pnpm run typecheck`, OpenAPI schema/TS-type regen (drift is auto-staged via `git add` — `stage_fixed` alone doesn't stage previously-unstaged files) — all gated to staged files via `glob`. Pre-push runs backend `pytest` when backend files changed.

## Verification policy

Run the cheapest check that covers the change — never all checks reflexively.
- Test-only or single-file change → `uv run pytest tests/test_<area>.py` plus `uv run ruff check` on touched paths.
- `src/` production change → affected test file(s) + ruff on touched paths.
- Full `uv run pytest` (~17s) and `uv run basedpyright` (slow): pre-PR / cross-cutting changes only, not after every edit.
- Never run `eval/run_ami_eval.py` unless explicitly asked (real corpus, real models, minutes–hours).
- Never write ad-hoc repro scripts that import the ML stack (`torch`, `pyannote`, `onnxruntime`, DBOS runners); validate via the pytest suite instead.

## Changelog policy

`CHANGELOG.md` (repo root) is user-facing and served in-app via `GET /api/changelog` + the header "What's New" button. It is also the source of GitHub release notes. Keep it up to date:
- Every user-visible change (feature, fix, behavior change) gets an entry in the same commit/PR, under a `## [YYYY-MM-DD]` heading for the UTC merge date (newest date first — create today's heading if missing), grouped under `Added`, `Changed`, or `Fixed`. One bullet per change, user-facing language, no commit SHAs.
- Infra-only, chore, and docs changes: add an entry only if they affect users/operators (Docker, deploy, backup, models); otherwise skip.
- Never create `Unreleased` or versioned sections, and never bump versions as part of a changelog update. The Dockerfile already bundles `CHANGELOG.md` into the image — no packaging change needed.
- Releases are CalVer tags (`YYYY.MM.DD`, optional `.N` same-day counter, no `v` prefix). Pushing one runs CI, publishes the GHCR image (`:<tag>` + `:latest`), and creates a GitHub release whose notes are every changelog section newer than the previous tag (`scripts/release_notes.py`, tested in `backend/tests/test_release_notes.py`). Never hand-edit release notes — fix the changelog and re-cut the tag. Runbook (preflight/cut/preview scripts + fix-forward policy): `.agents/skills/release/SKILL.md`.
- Build identity is the release tag plus commit hash when built from a tag (e.g. `2026.10.15 · 4d5ea70`), else UTC date/time + commit hash, computed in `backend/src/olive/build_info.py` from `OLIVE_BUILD_TAG` / `OLIVE_BUILD_SHA` / `OLIVE_BUILD_DATE` env (baked via Docker build args) with a git `HEAD` fallback in dev. Docker builds (GH publish workflow, staging deploy) must keep passing those build args.

## Invariants & rules

1. **Never touch `~/.config/olive` in tests.** All tests must use `:memory:` SQLite databases and `tmp_path` directories for `configDir`, `meetingsDir`, and `backupsDir`.
2. **Preserve on-disk & DB compatibility.**
   - SQLite timestamps are epoch milliseconds (`INTEGER`), IDs are UUID strings (`TEXT`), and JSON fields (`tags`, `provider_ids`, `enrollment_clip_paths`, `details`, `usage`) are stored as `TEXT`.
   - Meeting folder names (`layout.py`) must remain byte-identical to `YYYY-MM-DD_<sanitized_title>__<id[:8]>`.
   - Backup archives (`.tar.gz`) must remain compatible with both root `olive.sqlite`/`manifest.json` and legacy `database/olive.sqlite`/`backup_manifest.json` layouts.
3. **External API contracts.** Keep `POST /api/ingest` (used by the iOS Shortcut), `/mcp` (used by MCP clients), and `POST /api/webhooks/speechmatics` backward-compatible.
4. **OpenAPI drift check.** CI runs `pnpm run generate:api` and `git diff --exit-code src/api/schema.d.ts`. Whenever you change `backend/src/olive/schemas.py` or `export_openapi_schema()`, regenerate `backend/openapi.json` and `frontend/src/api/schema.d.ts`.
5. **The DBOS job queue is the SOLE acceptable path for enqueuing background work.** Durable work (transcription, summarization, syncs) MUST be enqueued via `enqueue_stage_job()` / `enqueue_driver_job()` — or `DBOS.enqueue_workflow_async` for staged pipelines — on a named queue in `STAGE_QUEUE_SPECS` (`backend/src/olive/workflows/queues.py`), always with a `deduplication_id` (and `meeting_id` + `label` for per-meeting jobs). `asyncio.create_task` is NEVER an enqueue mechanism: it is only acceptable for request-scoped pumps (SSE streaming, cancellable-upload heartbeats), scheduler internals, and tiny scheduling glue that awaits an enqueue call. Rationale: bare tasks are unbounded, invisible in the queue snapshot, un-cancellable as a unit, and lost on restart; queue jobs get bounded worker concurrency, dedup, restart recovery, and `cancel_dbos_workflows_for_meeting` coverage.
   - Leaf work (no nested enqueues, e.g. auto-summarize) → `enqueue_stage_job()` (step wrapper). Work whose body itself enqueues child workflows (cloud transcription drivers) → `enqueue_driver_job()` (workflow wrapper). DBOS forbids child-workflow enqueue from inside a step (`assert cur_ctx.is_workflow()`), so wrapping such a body in a step fails.
   - Local transcription stages are enqueued with `DBOS.enqueue_workflow_async(queue, run_local_stage, stage, payload)` via `enqueue_local_transcription()` / `_enqueue_local_stage()`, with the deterministic `stage:{meeting_id}:{run8}:{stage}` id. **Never make a stage await another stage's result** (`get_result()` inside a workflow): that re-creates the slot-holding deadlock this design replaced. A test (`tests/test_stage_handoff_pipeline.py`) enforces it.
   - Transcription routing (`services/transcription.py`, `TRANSCRIBE_QUEUE_BY_PROVIDER`): cloud providers keep a single job — `speechmatics` → `speechmatics`, `assemblyai` → `assemblyai`, `muse` → `muse`, ids `transcribe:{meeting_id}:{hex8}` with dedup id `transcribe:{meeting_id}`. `local` is no longer a single queue route: it goes through `enqueue_local_transcription()` and the flat stage chain (`stage:{meeting_id}:{run8}:{stage}`). When adding a new job-id format, keep the prefix matchers in `cancel_dbos_workflows_for_meeting` in sync (covered by a table-driven test) or queued jobs become un-cancellable.
