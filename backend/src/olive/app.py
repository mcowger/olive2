"""FastAPI application factory and HTTP routes (port of packages/server/src/app.ts)."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sse_starlette.sse import EventSourceResponse

from . import __version__
from .build_info import get_build_display
from .db import DbHandle, get_db
from .layout import meeting_paths
from .ml.audio import enhance_audio_file_async
from .schemas import LogItem
from .services.ingest import IngestService
from .services.insights.llm_query import MAX_FIND_EVENTS_LIMIT
from .services.meetings import (
    delete_meeting,
    get_meeting,
    list_meetings,
    list_tags,
    parse_json_field,
    update_meeting_start_time,
    update_meeting_tags,
    update_meeting_title,
)
from .services.templates import TemplateService
from .settings import (
    OlivePaths,
    get_effective_assemblyai_key,
    get_effective_muse_key,
    get_effective_speechmatics_key,
    is_masked_secret,
    load_app_config,
    mask_app_config,
    resolve_paths,
    save_app_config,
)


def _numeric_query_param(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _bool_query_param(value: str | None, default: bool) -> bool:
    """Parse an opt-out section flag (absent/blank → default)."""
    if value is None or value == "":
        return default
    return value.strip().lower() in ("true", "1", "yes")


DEFAULT_MAX_TRANSCRIPT_IMPORT_BYTES = 10 * 1024 * 1024
# Allowance for multipart framing / JSON escaping on top of the transcript's own byte cap.
_IMPORT_BODY_OVERHEAD_BYTES = 64 * 1024


async def _read_capped_body(request: Request, limit: int) -> bytes | None:
    """Read the request body, returning ``None`` as soon as more than ``limit`` bytes arrive."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _replayed_request(request: Request, body: bytes) -> Request:
    """A request over the same scope whose body is the already-read ``body``."""
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(request.scope, receive)


def _time_bound_param(value: Any) -> int | None:
    """Parse a from/to bound (epoch ms or ISO-8601 date/datetime, UTC when naive); blank is None."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"Invalid time bound: {value!r}")
    if isinstance(value, int | float):
        return int(value)
    raw = str(value).strip()
    if raw.lstrip("-").isdigit():
        return int(raw)
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def _alias_list(values: Any) -> list[str]:
    """Normalise ``meAliases`` (list, or comma/newline separated string(s)) into a clean list."""
    items = values if isinstance(values, list | tuple) else [values]
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        out.extend(a.strip() for a in item.replace("\n", ",").split(",") if a.strip())
    return out


def _is_loopback_host(hostname: str) -> bool:
    return hostname in ("127.0.0.1", "localhost", "::1")


def _resolve_changelog_path() -> Path | None:
    """Locate the repo CHANGELOG.md in dev, Docker, or via env override."""
    override = os.environ.get("OLIVE_CHANGELOG_PATH")
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
    docker_path = Path("/app/CHANGELOG.md")
    if docker_path.is_file():
        return docker_path
    repo_candidate = Path(__file__).resolve().parents[3] / "CHANGELOG.md"
    if repo_candidate.is_file():
        return repo_candidate
    return None


def _safe_web_path(web_root: str, request_path: str) -> Path | None:
    root = Path(web_root).resolve()
    rel = request_path.lstrip("/\\") or "index.html"
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def refresh_transcription_client_keys(
    clients: list[tuple[str, Any]],
    config: Any = None,
    env: dict[str, str] | None = None,
) -> None:
    """Pushes effective (env-wins) transcription keys onto provider clients.

    Provider clients are constructed once and reused; without this a key
    entered in the Settings UI would only take effect after a restart.
    Duplicate client objects are only updated once. Failures are swallowed
    so a config save never fails because of a key refresh.
    """
    getters = {
        "speechmatics": get_effective_speechmatics_key,
        "assemblyai": get_effective_assemblyai_key,
        "muse": get_effective_muse_key,
    }
    seen: set[int] = set()
    for kind, client in clients:
        if client is None or id(client) in seen:
            continue
        seen.add(id(client))
        getter = getters.get(kind)
        if getter is None:
            continue
        try:
            key = getter(config, env) if env is not None else getter(config) or ""
        except Exception:
            continue
        setter = getattr(client, "set_api_key", None) or getattr(client, "setApiKey", None)
        if callable(setter):
            try:
                setter(key)
            except Exception:
                pass
        elif hasattr(client, "api_key"):
            try:
                client.api_key = key
            except Exception:
                pass


def create_app(options: dict[str, Any] | None = None, **kwargs: Any) -> FastAPI:
    opts = {**(options or {}), **kwargs}
    raw_db = opts.get("db")
    if isinstance(raw_db, DbHandle):
        db: Engine = raw_db.engine
    elif isinstance(raw_db, Engine):
        db = raw_db
    else:
        db = get_db()

    base_paths = resolve_paths()
    config_dir = str(opts.get("configDir") or opts.get("config_dir") or base_paths.config_dir)
    meetings_dir = str(opts.get("meetingsDir") or opts.get("meetings_dir") or base_paths.meetings_dir)
    backups_dir = str(opts.get("backupsDir") or opts.get("backups_dir") or base_paths.backups_dir)
    models_dir = str(opts.get("modelsDir") or opts.get("models_dir") or base_paths.models_dir)
    plaud_tokens_path = str(
        opts.get("plaudTokensPath") or opts.get("plaud_tokens_path") or base_paths.plaud_tokens_path
    )

    app_config_paths = OlivePaths(
        config_dir=config_dir,
        meetings_dir=meetings_dir,
        backups_dir=backups_dir,
        models_dir=models_dir,
        database_path=str(opts.get("databasePath") or Path(config_dir) / "olive.sqlite"),
        settings_path=str(Path(config_dir) / "settings.json"),
        plaud_tokens_path=plaud_tokens_path,
    )
    # Canonicalize any legacy headerless raw Plaud Opus recordings on disk
    # (idempotent; wrapped files are valid Ogg Opus for every consumer).
    try:
        from .ml.opus import migrate_raw_opus_recordings

        migrate_raw_opus_recordings(db, meetings_dir)
    except Exception:
        pass

    configured_provider = load_app_config(app_config_paths).transcriptionProvider
    default_provider = (
        opts.get("defaultTranscriptionProvider")
        or opts.get("default_transcription_provider")
        or configured_provider
    )

    speechmatics_webhook_secret = (
        opts.get("speechmaticsWebhookSecret")
        or opts.get("speechmatics_webhook_secret")
        or os.environ.get("SPEECHMATICS_WEBHOOK_SECRET")
    )
    ingest_token = opts.get("ingestToken") or opts.get("ingest_token") or os.environ.get("OLIVE_INGEST_TOKEN")
    mcp_token = (
        opts.get("mcpToken") or opts.get("mcp_token") or os.environ.get("OLIVE_MCP_TOKEN") or ingest_token
    )
    _mcp_write_opt = opts.get("mcpWriteEnabled", opts.get("mcp_write_enabled", None))
    if isinstance(_mcp_write_opt, bool):
        mcp_write_enabled = _mcp_write_opt
    elif isinstance(_mcp_write_opt, str):
        mcp_write_enabled = _mcp_write_opt.strip().lower() in ("1", "true", "yes", "on", "y")
    elif _mcp_write_opt is not None:
        mcp_write_enabled = bool(_mcp_write_opt)
    else:
        mcp_write_enabled = os.environ.get("OLIVE_MCP_WRITE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
            "y",
        )
    bind_host = str(
        opts.get("bindHost") or opts.get("bind_host") or os.environ.get("OLIVE_BIND_HOST") or "127.0.0.1"
    )
    if not mcp_token and not _is_loopback_host(bind_host):
        raise ValueError(
            "OLIVE_MCP_TOKEN or OLIVE_INGEST_TOKEN is required when OLIVE_BIND_HOST is not loopback"
        )

    speechmatics_client = opts.get("speechmaticsClient") or opts.get("speechmatics_client")
    assemblyai_client = opts.get("assemblyaiClient") or opts.get("assemblyai_client")
    muse_client = opts.get("museClient") or opts.get("muse_client")
    transcription_service = opts.get("transcriptionService") or opts.get("transcription_service")
    speaker_service = opts.get("speakerService") or opts.get("speaker_service")
    llm_service = opts.get("llmService") or opts.get("llm_service")
    summary_service = opts.get("summaryService") or opts.get("summary_service")
    title_service = opts.get("titleService") or opts.get("title_service")
    insights_service = opts.get("insightsService") or opts.get("insights_service")
    insights_query_service = opts.get("insightsQueryService") or opts.get("insights_query_service")
    insights_llm_service = opts.get("insightsLlmService") or opts.get("insights_llm_service")
    transcript_import_service = opts.get("transcriptImportService") or opts.get(
        "transcript_import_service"
    )
    max_transcript_import_bytes = int(
        opts.get("maxTranscriptImportBytes")
        or opts.get("max_transcript_import_bytes")
        or DEFAULT_MAX_TRANSCRIPT_IMPORT_BYTES
    )
    chat_service = opts.get("chatService") or opts.get("chat_service")
    backup_service = opts.get("backupService") or opts.get("backup_service")
    plaud_poller = opts.get("plaudPoller") or opts.get("plaud_poller")
    plaud_client = opts.get("plaudClient") or opts.get("plaud_client")
    oauth_manager = opts.get("oauthManager") or opts.get("oauth_manager")
    auth_sessions = opts.get("authSessions") or opts.get("auth_sessions")

    def _current_app_config() -> Any:
        try:
            return load_app_config(app_config_paths)
        except Exception:
            return None

    def _apply_stored_transcription_keys(config: Any = None) -> None:
        """Pushes effective (env-wins) transcription keys onto live clients."""
        cfg = config or _current_app_config()
        pairs: list[tuple[str, Any]] = [
            ("speechmatics", speechmatics_client),
            ("assemblyai", assemblyai_client),
            ("muse", muse_client),
        ]
        if transcription_service is not None:
            pairs.extend(
                [
                    ("speechmatics", getattr(transcription_service, "client", None)),
                    ("assemblyai", getattr(transcription_service, "assemblyai_client", None)),
                    ("muse", getattr(transcription_service, "muse_client", None)),
                ]
            )
        refresh_transcription_client_keys(pairs, cfg)

    def get_speechmatics_client() -> Any:
        nonlocal speechmatics_client
        if speechmatics_client is None:
            from .providers.speechmatics import SpeechmaticsClient

            cfg = _current_app_config()
            try:
                initial_key = get_effective_speechmatics_key(cfg) or ""
            except Exception:
                initial_key = ""
            speechmatics_client = SpeechmaticsClient({"apiKey": initial_key})
        return speechmatics_client

    def get_assemblyai_client() -> Any:
        nonlocal assemblyai_client
        if assemblyai_client is None:
            from .providers.assemblyai import AssemblyAIClient

            cfg = _current_app_config()
            try:
                initial_key = get_effective_assemblyai_key(cfg) or ""
            except Exception:
                initial_key = ""
            assemblyai_client = AssemblyAIClient({"apiKey": initial_key})
        return assemblyai_client

    def get_muse_client() -> Any:
        nonlocal muse_client
        if muse_client is None:
            from .providers.muse import MuseClient

            cfg = _current_app_config()
            try:
                initial_key = get_effective_muse_key(cfg) or ""
            except Exception:
                initial_key = ""
            muse_client = MuseClient({"apiKey": initial_key})
        return muse_client

    def get_llm_service() -> Any:
        nonlocal llm_service
        if llm_service is None:
            from .services.llm import LlmService

            llm_service = LlmService(db=db, config_dir=config_dir)
        return llm_service

    def get_summary_service() -> Any:
        nonlocal summary_service
        if summary_service is None:
            from .services.summaries import SummaryService

            summary_service = SummaryService(
                db=db,
                meetings_dir=meetings_dir,
                llm_service=get_llm_service(),
                template_service=template_service,
            )
        return summary_service

    def get_insights_service() -> Any:
        nonlocal insights_service
        if insights_service is None:
            from .services.insights.extract import InsightsService
            from .workflows import configure_insights_queue

            insights_service = InsightsService(
                db=db,
                meetings_dir=meetings_dir,
                config_dir=config_dir,
                llm_service=get_llm_service(),
            )
            configure_insights_queue(insights_service)
        return insights_service

    def get_insights_query_service() -> Any:
        nonlocal insights_query_service
        if insights_query_service is None:
            from .services.insights.query import InsightsQueryService

            insights_query_service = InsightsQueryService(db=db, config_dir=config_dir)
        return insights_query_service

    def get_insights_llm_service() -> Any:
        nonlocal insights_llm_service
        if insights_llm_service is None:
            from .services.insights.llm_query import InsightsLlmService

            insights_llm_service = InsightsLlmService(
                db=db,
                meetings_dir=meetings_dir,
                config_dir=config_dir,
                llm_service=get_llm_service(),
            )
        return insights_llm_service

    def get_transcript_import_service() -> Any:
        nonlocal transcript_import_service
        if transcript_import_service is None:
            from .services.transcript_import import TranscriptImportService

            transcript_import_service = TranscriptImportService(
                db=db,
                meetings_dir=meetings_dir,
                config_dir=config_dir,
                insights_service=get_insights_service(),
            )
        return transcript_import_service

    def get_title_service() -> Any:
        nonlocal title_service
        if title_service is None:
            from .services.titles import TitleService

            title_service = TitleService(
                db=db,
                meetings_dir=meetings_dir,
                llm_service=get_llm_service(),
            )
        return title_service

    def get_chat_service() -> Any:
        nonlocal chat_service
        if chat_service is None:
            from .services.chat import ChatService

            chat_service = ChatService(
                db=db,
                meetings_dir=meetings_dir,
                llm_service=get_llm_service(),
            )
        return chat_service

    def get_transcription_service() -> Any:
        nonlocal transcription_service
        if transcription_service is None:
            from .services.transcription import TranscriptionService

            transcription_service = TranscriptionService(
                db=db,
                meetings_dir=meetings_dir,
                speechmatics_client=get_speechmatics_client(),
                assemblyai_client=get_assemblyai_client(),
                muse_client=get_muse_client(),
                default_provider=default_provider,
                webhook_secret=speechmatics_webhook_secret,
                config_dir=config_dir,
            )
            transcription_service.set_summary_service(get_summary_service())
            transcription_service.set_insights_service(get_insights_service())
        return transcription_service

    def get_speaker_service() -> Any:
        nonlocal speaker_service
        if speaker_service is None:
            from .services.speakers import SpeakerService

            speaker_service = SpeakerService(
                db=db,
                config_dir=config_dir,
                meetings_dir=meetings_dir,
            )
        return speaker_service

    def get_backup_service() -> Any:
        nonlocal backup_service
        if backup_service is None:
            from .services.backup import BackupService

            backup_service = BackupService(
                db=db,
                paths=app_config_paths,
                config_dir=config_dir,
                meetings_dir=meetings_dir,
                backups_dir=backups_dir,
            )
        return backup_service

    def get_plaud_poller() -> Any:
        nonlocal plaud_poller, plaud_client
        if plaud_poller is None:
            from .plaud.client import create_plaud_client
            from .plaud.poller import PlaudPoller

            if plaud_client is None:
                plaud_client = create_plaud_client(app_config_paths)
            plaud_poller = PlaudPoller(
                db=db,
                meetings_dir=meetings_dir,
                client=plaud_client,
                transcription_service=get_transcription_service(),
                config_dir=config_dir,
            )
            tsvc = get_transcription_service()
            setter = getattr(tsvc, "set_plaud_poller", None) or getattr(tsvc, "setPlaudPoller", None)
            if callable(setter):
                setter(plaud_poller)
        return plaud_poller

    def get_oauth_manager() -> Any:
        nonlocal oauth_manager
        if oauth_manager is None:
            oauth_manager = getattr(get_plaud_poller(), "oauth", None)
        return oauth_manager

    def get_auth_sessions() -> Any:
        nonlocal auth_sessions
        if auth_sessions is None:
            from .plaud.auth import PlaudAuthSessionStore

            auth_sessions = PlaudAuthSessionStore()
        return auth_sessions

    if transcription_service is not None and summary_service is not None:
        if hasattr(transcription_service, "set_summary_service"):
            transcription_service.set_summary_service(summary_service)
        elif hasattr(transcription_service, "setSummaryService"):
            transcription_service.setSummaryService(summary_service)

    if transcription_service is not None:
        # Injected transcription service: same insights auto-enqueue wiring as the lazy path.
        _set_insights = getattr(transcription_service, "set_insights_service", None) or getattr(
            transcription_service, "setInsightsService", None
        )
        if callable(_set_insights):
            _set_insights(get_insights_service())

    ingest_service: IngestService = (
        opts.get("ingestService")
        or opts.get("ingest_service")
        or IngestService(
            db=db,
            meetings_dir=meetings_dir,
            transcription_service=transcription_service,
            default_transcription_provider=default_provider,
        )
    )

    template_service: TemplateService = (
        opts.get("templateService") or opts.get("template_service") or TemplateService(db)
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        from .workflows import reset_shutdown_flag, shutdown_dbos

        reset_shutdown_flag()
        # The Plaud poller must start here — inside the server's running
        # event loop. Starting it during app construction leaves
        # AsyncIOScheduler bound to a loop that never runs, so interval
        # polls (and the boot backfill/trigger) silently never fire and only
        # manual "Sync Now" works.
        if opts.get("startPlaudPoller") or opts.get("start_plaud_poller"):
            try:
                poll_interval = int(
                    opts.get("pollIntervalMinutes") or opts.get("poll_interval_minutes") or 15
                )
                get_plaud_poller().start(poll_interval)
            except Exception as exc:
                from .logging_db import logger as _lifespan_logger

                _lifespan_logger.warn("Failed to start Plaud poller", {"error": str(exc)})

        # Best-effort LLM web-catalog refresh so fresh installs see current
        # models without a manual "Sync Web Catalog". Skipped while the
        # on-disk cache is fresh; fire-and-forget so a slow/offline
        # models.dev never delays startup. This is lightweight scheduling
        # glue (single HTTP fetch + file write), not durable queue work.
        # Opt out with OLIVE_LLM_CATALOG_AUTO_REFRESH=0 or
        # llmCatalogAutoRefresh=False.
        async def _refresh_llm_catalog_on_startup() -> None:
            try:
                await get_llm_service().refresh_catalog_if_stale()
            except Exception as exc:
                from .logging_db import logger as _lifespan_logger

                _lifespan_logger.warn("LLM catalog auto-refresh failed", {"error": str(exc)})

        auto_refresh_opt = opts.get("llmCatalogAutoRefresh", opts.get("llm_catalog_auto_refresh", None))
        auto_refresh_env = os.environ.get("OLIVE_LLM_CATALOG_AUTO_REFRESH", "")
        auto_refresh_disabled = auto_refresh_opt is False or auto_refresh_env.strip().lower() in (
            "0",
            "false",
            "off",
            "no",
        )
        if not auto_refresh_disabled:
            try:
                if get_llm_service().is_web_catalog_stale():
                    asyncio.create_task(_refresh_llm_catalog_on_startup())
            except Exception:
                pass
        try:
            yield
        finally:
            if plaud_poller is not None and hasattr(plaud_poller, "stop"):
                try:
                    plaud_poller.stop()
                except Exception:
                    pass
            if transcription_service is not None and hasattr(transcription_service, "stop"):
                try:
                    transcription_service.stop()
                except Exception:
                    pass
            await asyncio.to_thread(shutdown_dbos)

    from .workflows.transcription_workflow import configure_recovery_runtime

    configure_recovery_runtime(
        db=db,
        progress_sink=lambda meeting_id, update: get_transcription_service().emit_progress(
            meeting_id, update
        ),
        complete_sink=lambda meeting_id, result, wf_id: (
            get_transcription_service().complete_recovered_local_transcript(meeting_id, result, wf_id)
        ),
        config_dir=config_dir,
        meetings_dir=meetings_dir,
    )

    app = FastAPI(title="Olive API", version=__version__, lifespan=lifespan)

    @app.api_route("/mcp", methods=["GET", "POST", "DELETE", "OPTIONS"], include_in_schema=False)
    async def mcp_endpoint(request: Request) -> Response:
        from .mcp_server import handle_mcp_request

        return await handle_mcp_request(
            request=request,
            db=db,
            meetings_dir=meetings_dir,
            speaker_service=get_speaker_service(),
            mcp_token=mcp_token,
            bind_host=bind_host,
            write_enabled=mcp_write_enabled,
            transcription_service=get_transcription_service(),
            summary_service=get_summary_service(),
            title_service=get_title_service(),
            plaud_poller=plaud_poller,
            template_service=template_service,
            insights_service=get_insights_service(),
            insights_query_service=get_insights_query_service(),
            insights_llm_service=get_insights_llm_service(),
        )

    if (
        opts.get("startPlaudPoller")
        or opts.get("start_plaud_poller")
        or opts.get("syncOnStartup")
        or opts.get("sync_on_startup")
    ):
        try:
            get_speaker_service().sync_meeting_speakers_and_transcripts_sync()
        except Exception:
            pass

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/meetings")
    async def get_meetings_route(request: Request) -> JSONResponse:
        q = request.query_params
        # `tags` accepts repeatable params (?tags=a&tags=b, AND semantics)
        # as well as a single comma-separated value (?tags=a,b).
        raw_tags: list[str] = []
        for entry in q.getlist("tags"):
            raw_tags.extend(str(entry).split(","))
        resp = list_meetings(
            db,
            {
                "limit": _numeric_query_param(q.get("limit")),
                "offset": _numeric_query_param(q.get("offset")),
                "search": q.get("search"),
                "tags": raw_tags,
                "source": q.get("source"),
                "status": q.get("status"),
                "speakerId": q.get("speakerId"),
                "hasSummary": q.get("hasSummary"),
                "hasTranscript": q.get("hasTranscript"),
                "startTime": _numeric_query_param(q.get("startTime")),
                "endTime": _numeric_query_param(q.get("endTime")),
                "includeDetails": q.get("includeDetails")
                if q.get("includeDetails") is not None
                else q.get("include_details"),
            },
            meetings_dir,
        )
        return JSONResponse(resp)

    @app.get("/api/tags")
    async def get_tags_route() -> JSONResponse:
        return JSONResponse({"tags": list_tags(db)})

    @app.post("/api/ingest")
    async def post_ingest_route(request: Request) -> JSONResponse:
        if ingest_token:
            auth_header = request.headers.get("authorization") or ""
            if auth_header != f"Bearer {ingest_token}":
                return JSONResponse(
                    {"error": "Unauthorized: invalid or missing ingest token"},
                    status_code=401,
                )

        content_type = request.headers.get("content-type") or ""
        if "multipart/form-data" not in content_type.lower():
            return JSONResponse({"error": "Multipart form data required"}, status_code=400)

        try:
            form = await request.form()
        except Exception:
            return JSONResponse({"error": "Multipart form data required"}, status_code=400)

        file_field = form.get("file")
        if file_field is None or not hasattr(file_field, "read"):
            return JSONResponse({"error": "file field (audio blob) is required"}, status_code=400)

        audio_bytes = await file_field.read()  # type: ignore[union-attr]
        filename = getattr(file_field, "filename", None) or "audio.m4a"
        mime = getattr(file_field, "content_type", None) or "audio/mp4"

        title = form.get("title")
        source_param = form.get("source")
        auto_transcribe_param = form.get("autoTranscribe")
        provider_param = form.get("provider")
        model_id_param = form.get("modelId")
        duration_ms_param = form.get("durationMs")
        if duration_ms_param is None:
            duration_ms_param = form.get("duration_ms")
        duration_param = form.get("duration")

        user_agent = request.headers.get("user-agent") or ""
        is_shortcut = ("Shortcuts" in user_agent) or (source_param == "ios-shortcut")
        source = "ios-shortcut" if is_shortcut else "plaud" if source_param == "plaud" else "upload"

        auto_transcribe = str(auto_transcribe_param) in ("true", "1")
        transcription_provider = load_app_config(app_config_paths).transcriptionProvider
        if provider_param in ("local", "speechmatics", "assemblyai", "muse"):
            transcription_provider = provider_param  # type: ignore[assignment]
        model_id = (
            str(model_id_param).strip()
            if isinstance(model_id_param, str) and model_id_param.strip()
            else None
        )

        if auto_transcribe and ingest_service.transcription_service is None:
            ingest_service.transcription_service = get_transcription_service()
        try:
            result = await _call_maybe_async(
                ingest_service,
                ("ingest_audio", "ingestAudio"),
                {
                    "audioBytes": audio_bytes,
                    "filename": filename,
                    "mime": mime,
                    "title": str(title) if isinstance(title, str) else None,
                    "source": source,
                    "autoTranscribe": auto_transcribe,
                    "transcriptionProvider": transcription_provider,
                    "modelId": model_id,
                    "durationMs": (
                        str(duration_ms_param)
                        if duration_ms_param is not None and not hasattr(duration_ms_param, "read")
                        else None
                    ),
                    "duration": (
                        str(duration_param)
                        if duration_param is not None and not hasattr(duration_param, "read")
                        else None
                    ),
                },
            )
            return JSONResponse(result, status_code=200 if result.get("deduped") else 201)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # -- Olive Insights -------------------------------------------------------------

    def _insights_disabled() -> JSONResponse | None:
        """404 for every /api/insights/* route while the feature is off (it ships disabled)."""
        if get_insights_service().is_enabled():
            return None
        return JSONResponse({"error": "insights disabled"}, status_code=404)

    @app.get("/api/insights/overview")
    async def get_insights_overview_route(request: Request) -> JSONResponse:
        if (off := _insights_disabled()) is not None:
            return off
        q = request.query_params
        try:
            res = await asyncio.to_thread(
                get_insights_query_service().overview,
                q.get("from") or None,
                q.get("to") or None,
                q.get("speaker") or None,
                q.get("tag") or None,
                q.get("bucket") or "month",
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(res)

    @app.get("/api/insights/person/{speaker_id}")
    async def get_insights_person_route(speaker_id: str, request: Request) -> JSONResponse:
        if (off := _insights_disabled()) is not None:
            return off
        q = request.query_params
        try:
            res = await asyncio.to_thread(
                get_insights_query_service().person,
                speaker_id,
                q.get("from") or None,
                q.get("to") or None,
                q.get("bucket") or "month",
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(res)

    @app.get("/api/insights/meeting/{meeting_id}")
    async def get_insights_meeting_route(meeting_id: str) -> JSONResponse:
        if (off := _insights_disabled()) is not None:
            return off
        res = await asyncio.to_thread(get_insights_query_service().meeting, meeting_id)
        if res is None:
            return JSONResponse({"error": "Meeting not found"}, status_code=404)
        return JSONResponse(res)

    @app.get("/api/insights/dyads")
    async def get_insights_dyads_route(request: Request) -> JSONResponse:
        if (off := _insights_disabled()) is not None:
            return off
        q = request.query_params
        try:
            res = await asyncio.to_thread(
                get_insights_query_service().dyads,
                None,
                q.get("from") or None,
                q.get("to") or None,
                q.get("bucket") or "month",
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(res)

    @app.get("/api/insights/find")
    async def get_insights_find_route(request: Request) -> JSONResponse:
        if (off := _insights_disabled()) is not None:
            return off
        q = request.query_params
        query = (q.get("q") or "").strip()
        if not query:
            return JSONResponse({"error": "q is required"}, status_code=400)
        limit = 5
        if q.get("limit"):
            try:
                limit = int(q["limit"])
            except ValueError:
                return JSONResponse({"error": "limit must be an integer"}, status_code=400)
        try:
            results = await get_insights_llm_service().find_events(
                query, min(max(limit, 1), MAX_FIND_EVENTS_LIMIT)
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        return JSONResponse({"query": query, "results": results})

    @app.get("/api/insights/playbook/{speaker_id}")
    async def get_insights_playbook_route(speaker_id: str) -> JSONResponse:
        if (off := _insights_disabled()) is not None:
            return off
        try:
            res = await get_insights_llm_service().playbook(speaker_id)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        return JSONResponse(res)

    @app.get("/api/insights/coaching")
    async def get_insights_coaching_route(request: Request) -> JSONResponse:
        if (off := _insights_disabled()) is not None:
            return off
        speaker = request.query_params.get("speaker") or None
        try:
            res = await get_insights_llm_service().coaching("speaker" if speaker else "all", speaker)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        return JSONResponse(res)

    @app.get("/api/insights/me")
    async def get_insights_me_route() -> JSONResponse:
        if (off := _insights_disabled()) is not None:
            return off
        return JSONResponse(await asyncio.to_thread(get_insights_query_service().get_me))

    @app.put("/api/insights/me")
    async def put_insights_me_route(request: Request) -> JSONResponse:
        if (off := _insights_disabled()) is not None:
            return off
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "JSON object body required"}, status_code=400)
        speaker_id = body.get("speakerId")
        aliases = body.get("aliases", [])
        if speaker_id is not None and not isinstance(speaker_id, str):
            return JSONResponse({"error": "speakerId must be a string or null"}, status_code=400)
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            return JSONResponse({"error": "aliases must be a list of strings"}, status_code=400)
        res = await asyncio.to_thread(
            get_insights_query_service().set_me, speaker_id or None, aliases
        )
        return JSONResponse(res)

    @app.post("/api/insights/backfill")
    async def post_insights_backfill_route(request: Request) -> JSONResponse:
        if (off := _insights_disabled()) is not None:
            return off
        body: Any = {}
        if (await request.body()).strip():
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "JSON object body required"}, status_code=400)
        try:
            frm = _time_bound_param(body.get("from"))
            to = _time_bound_param(body.get("to"))
            limit = int(body["limit"]) if body.get("limit") is not None else None
        except (TypeError, ValueError):
            return JSONResponse({"error": "Invalid from, to, or limit"}, status_code=400)
        from .services.insights import backfill as insights_backfill

        # Constructing the service registers the durable queue hook backfill enqueues through.
        get_insights_service()
        res = await insights_backfill.backfill(
            db,
            meetings_dir,
            config_dir,
            frm=frm,
            to=to,
            force=bool(body.get("force")),
            limit=limit,
        )
        return JSONResponse(res, status_code=202)

    @app.get("/api/insights/status")
    async def get_insights_status_route() -> JSONResponse:
        if (off := _insights_disabled()) is not None:
            return off

        def _counts() -> dict[str, int]:
            with db.connect() as conn:
                transcribed = conn.execute(
                    text("SELECT COUNT(*) FROM meetings WHERE primary_transcript_artifact_id IS NOT NULL")
                ).scalar_one()
                analyzed = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM meetings m WHERE m.primary_transcript_artifact_id IS NOT NULL "
                        "AND EXISTS (SELECT 1 FROM meeting_speaker_signals s WHERE s.meeting_id = m.id)"
                    )
                ).scalar_one()
            return {
                "transcribedMeetings": int(transcribed),
                "analyzedMeetings": int(analyzed),
                "pendingMeetings": int(transcribed) - int(analyzed),
            }

        return JSONResponse(await asyncio.to_thread(_counts))

    @app.post("/api/transcripts/import")
    async def post_transcript_import_route(request: Request) -> JSONResponse:
        if ingest_token:
            auth_header = request.headers.get("authorization") or ""
            if auth_header != f"Bearer {ingest_token}":
                return JSONResponse(
                    {"error": "Unauthorized: invalid or missing ingest token"},
                    status_code=401,
                )

        too_large = JSONResponse(
            {"error": f"Transcript exceeds the {max_transcript_import_bytes} byte limit"},
            status_code=413,
        )
        declared = _numeric_query_param(request.headers.get("content-length"))
        if declared is not None and declared > max_transcript_import_bytes:
            return too_large

        content_type = (request.headers.get("content-type") or "").lower()
        is_multipart = "multipart/form-data" in content_type
        is_json = "application/json" in content_type
        if not is_multipart and not is_json:
            return JSONResponse(
                {"error": "Multipart form data or a JSON body is required"}, status_code=400
            )

        # Content-Length can be absent (chunked) or a lie, so count bytes as they arrive and stop
        # at the cap instead of letting the framework buffer an unbounded body. The slack covers
        # multipart framing / JSON escaping around a transcript that is itself within the cap.
        raw = await _read_capped_body(request, max_transcript_import_bytes + _IMPORT_BODY_OVERHEAD_BYTES)
        if raw is None:
            return too_large

        opts_in: dict[str, Any]
        if is_multipart:
            try:
                form = await _replayed_request(request, raw).form()
            except Exception:
                return JSONResponse({"error": "Invalid multipart form data"}, status_code=400)
            file_field = form.get("file")
            if file_field is None or not hasattr(file_field, "read"):
                return JSONResponse({"error": "file field (transcript) is required"}, status_code=400)
            content = await file_field.read()  # type: ignore[union-attr]
            opts_in = {
                "content": content,
                "filename": getattr(file_field, "filename", None),
                "meAliases": _alias_list(form.getlist("meAliases")),
            }
            for key in ("title", "startTime", "subtype", "format"):
                value = form.get(key)
                if isinstance(value, str) and value.strip():
                    opts_in[key] = value.strip()
        else:
            try:
                body = json.loads(raw)
            except ValueError:
                return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
            if not isinstance(body, dict) or not isinstance(body.get("content"), str):
                return JSONResponse(
                    {"error": "JSON body with a string content field is required"}, status_code=400
                )
            content = body["content"]
            opts_in = {
                "content": content,
                "filename": body.get("filename") if isinstance(body.get("filename"), str) else None,
                "meAliases": _alias_list(body.get("meAliases") or []),
            }
            for key in ("title", "startTime", "subtype", "format"):
                value = body.get(key)
                if value is not None and value != "":
                    opts_in[key] = value

        # Measure bytes, not characters: multi-byte text is far larger on disk than len() says.
        size = len(content) if isinstance(content, bytes | bytearray) else len(content.encode("utf-8"))
        if size > max_transcript_import_bytes:
            return too_large

        from .services.transcript_import import MissingStartTimeError

        try:
            result = await get_transcript_import_service().import_transcript(opts_in)
        except MissingStartTimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            from .logging_db import logger as _import_logger

            _import_logger.error("Transcript import failed", {"category": "import", "error": str(exc)})
            return JSONResponse({"error": "Transcript import failed"}, status_code=500)
        return JSONResponse(result, status_code=200 if result.get("deduped") else 201)

    @app.get("/api/meetings/{meeting_id}")
    async def get_meeting_detail_route(meeting_id: str, request: Request) -> JSONResponse:
        q = request.query_params
        detail = get_meeting(
            db,
            meeting_id,
            meetings_dir,
            {
                # Full detail by default (the web UI renders transcript +
                # summaries + enrolled-speaker state from this route).
                "includeTranscript": _bool_query_param(q.get("includeTranscript"), True),
                "includeSummaries": _bool_query_param(q.get("includeSummaries"), True),
                "includeVoiceprints": _bool_query_param(q.get("includeVoiceprints"), True),
            },
        )
        if not detail:
            return JSONResponse({"error": "Meeting not found"}, status_code=404)

        svc = get_transcription_service()
        active_progress = (
            _call_sync(
                svc,
                ("get_transcription_progress", "getTranscriptionProgress"),
                meeting_id,
            )
            if svc is not None
            else None
        )
        summary_gen = (
            _call_sync(
                summary_service,
                ("get_summary_generation_status", "getSummaryGenerationStatus"),
                meeting_id,
            )
            if summary_service is not None
            else None
        )
        reidentify_status = (
            _call_sync(
                svc,
                ("get_speaker_reidentification_status", "getSpeakerReidentificationStatus"),
                meeting_id,
            )
            if svc is not None
            else None
        )
        payload = detail.model_dump()
        payload["transcriptionProgress"] = (
            active_progress.model_dump(exclude_none=True)
            if active_progress is not None and hasattr(active_progress, "model_dump")
            else active_progress
        )
        payload["summaryGeneration"] = (
            summary_gen.model_dump(exclude_none=True)
            if summary_gen is not None and hasattr(summary_gen, "model_dump")
            else summary_gen
        )
        payload["speakerReidentification"] = (
            reidentify_status.model_dump(exclude_none=True)
            if reidentify_status is not None and hasattr(reidentify_status, "model_dump")
            else reidentify_status
        )
        return JSONResponse(payload)

    @app.put("/api/meetings/{meeting_id}/title")
    async def put_meeting_title_route(meeting_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "JSON body required"}, status_code=400)

        if not isinstance(body, dict) or not isinstance(body.get("title"), str) or not body["title"].strip():
            return JSONResponse({"error": "title must be a non-empty string"}, status_code=400)

        try:
            updated = update_meeting_title(db, meeting_id, body["title"], meetings_dir)
            if not updated:
                return JSONResponse({"error": "Meeting not found"}, status_code=404)
            return JSONResponse({"meeting": updated.model_dump()})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.put("/api/meetings/{meeting_id}/start-time")
    async def put_meeting_start_time_route(meeting_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "JSON body required"}, status_code=400)

        if not isinstance(body, dict):
            return JSONResponse({"error": "JSON body required"}, status_code=400)
        raw = body.get("startTime", body.get("start_time"))
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return JSONResponse({"error": "startTime must be epoch milliseconds"}, status_code=400)
        start_time = int(raw)

        try:
            updated = update_meeting_start_time(db, meeting_id, start_time, meetings_dir)
            if not updated:
                return JSONResponse({"error": "Meeting not found"}, status_code=404)
            return JSONResponse({"meeting": updated.model_dump()})
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/meetings/{meeting_id}/auto-title")
    async def post_auto_title_route(meeting_id: str, request: Request) -> JSONResponse:
        body: dict[str, Any] = {}
        try:
            parsed = await request.json()
            if isinstance(parsed, dict):
                body = parsed
        except Exception:
            pass

        try:
            svc = get_title_service()
            generate_fn = getattr(svc, "generate_and_apply", None) or getattr(svc, "generateAndApply", None)
            if not callable(generate_fn):
                raise RuntimeError("Title service is not configured")
            res = generate_fn(
                meeting_id,
                provider=body.get("provider"),
                model=body.get("model"),
            )
            if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
                res = await res
            return JSONResponse(res)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.delete("/api/meetings/{meeting_id}")
    async def delete_meeting_route(meeting_id: str) -> JSONResponse:
        try:
            await _call_maybe_async(
                transcription_service,
                ("cancel_transcription", "cancelTranscription"),
                meeting_id,
            )
        except Exception:
            pass
        try:
            deleted = delete_meeting(db, meeting_id, meetings_dir)
            if not deleted:
                return JSONResponse({"error": "Meeting not found"}, status_code=404)
            return JSONResponse({"status": "deleted"})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/meetings/{meeting_id}/split")
    async def post_meeting_split_route(meeting_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "JSON body required"}, status_code=400)

        raw = body.get("splitMs", body.get("split_ms")) if isinstance(body, dict) else None
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return JSONResponse({"error": "splitMs must be milliseconds from the start"}, status_code=400)
        try:
            from .services.split import split_meeting

            result = await asyncio.to_thread(split_meeting, db, meetings_dir, meeting_id, int(raw))
            return JSONResponse(result)
        except ValueError as exc:
            msg = str(exc)
            code = 404 if "not found" in msg.lower() else 400
            return JSONResponse({"error": msg}, status_code=code)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.put("/api/meetings/{meeting_id}/tags")
    async def put_meeting_tags_route(meeting_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "JSON body required"}, status_code=400)

        tags = body.get("tags") if isinstance(body, dict) else None
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            return JSONResponse({"error": "tags must be an array of strings"}, status_code=400)

        updated = update_meeting_tags(db, meeting_id, tags)
        if not updated:
            return JSONResponse({"error": "Meeting not found"}, status_code=404)
        return JSONResponse({"meeting": updated.model_dump()})

    @app.get("/api/meetings/{meeting_id}/audio")
    async def get_meeting_audio_route(meeting_id: str, request: Request) -> Response:
        with db.connect() as conn:
            meeting = (
                conn.execute(
                    text("SELECT * FROM meetings WHERE id = :id"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not meeting:
                return JSONResponse({"error": "Meeting not found"}, status_code=404)
            recording = (
                conn.execute(
                    text("SELECT * FROM recordings WHERE meeting_id = :id ORDER BY created_at ASC LIMIT 1"),
                    {"id": meeting_id},
                )
                .mappings()
                .first()
            )
            if not recording:
                return JSONResponse({"error": "Recording not found"}, status_code=404)

        folder = Path(
            meeting_paths(meetings_dir, int(meeting["start_time"]), meeting["title"], meeting["id"]).folder
        )
        full_path = folder / recording["path"]
        if not full_path.exists():
            return JSONResponse({"error": "Audio file not found on disk"}, status_code=404)

        target_path = full_path
        target_mime = recording["mime"] or "audio/mp4"

        if request.query_params.get("enhanced") == "true":
            enhanced_full = folder / "audio" / f"{recording['id']}.enhanced.mp3"
            # Guard against a stale cache: if the source audio was rewritten
            # after the enhanced file was built (e.g. a meeting split
            # overwrote Part A in place), rebuild instead of serving it.
            if enhanced_full.exists():
                try:
                    if full_path.stat().st_mtime > enhanced_full.stat().st_mtime:
                        enhanced_full.unlink(missing_ok=True)
                except OSError:
                    pass
            if not enhanced_full.exists():
                await enhance_audio_file_async(full_path, enhanced_full)
            if enhanced_full.exists():
                target_path = enhanced_full
                target_mime = "audio/mpeg"

        size = target_path.stat().st_size
        range_header = request.headers.get("range")
        if range_header and range_header.startswith("bytes="):
            parts = range_header.replace("bytes=", "", 1).split("-")
            try:
                start = int(parts[0])
                end = int(parts[1]) if (len(parts) > 1 and parts[1]) else size - 1
            except ValueError:
                start = -1
                end = -1
            if 0 <= start <= end < size:
                chunk_size = end - start + 1
                with target_path.open("rb") as f:
                    f.seek(start)
                    data = f.read(chunk_size)
                return Response(
                    content=data,
                    status_code=206,
                    headers={
                        "Content-Range": f"bytes {start}-{end}/{size}",
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(chunk_size),
                        "Content-Type": target_mime,
                    },
                )

        return Response(
            content=target_path.read_bytes(),
            status_code=200,
            headers={
                "Content-Length": str(size),
                "Content-Type": target_mime,
                "Accept-Ranges": "bytes",
            },
        )

    @app.post("/api/meetings/{meeting_id}/audio/retry")
    async def post_meeting_audio_retry_route(meeting_id: str) -> JSONResponse:
        try:
            res = await _call_maybe_async(
                get_plaud_poller(),
                ("retry_meeting_audio", "retryMeetingAudio"),
                meeting_id,
            )
            return JSONResponse(res, status_code=202)
        except ValueError as exc:
            msg = str(exc)
            lowered = msg.lower()
            code = 404 if ("not found" in lowered or "no plaud ingest" in lowered) else 400
            return JSONResponse({"error": msg}, status_code=code)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # Chat API
    @app.get("/api/meetings/{meeting_id}/chat/messages")
    async def get_chat_messages_route(meeting_id: str) -> JSONResponse:
        try:
            msgs = await _call_maybe_async(get_chat_service(), ("get_messages", "getMessages"), meeting_id)
            return JSONResponse(
                {"messages": [m.model_dump() if hasattr(m, "model_dump") else m for m in msgs]}
            )
        except Exception as exc:
            msg = str(exc)
            return JSONResponse(
                {"error": msg},
                status_code=404 if msg.startswith("Meeting not found") else 400,
            )

    @app.delete("/api/meetings/{meeting_id}/chat/messages")
    async def delete_chat_messages_route(meeting_id: str) -> JSONResponse:
        try:
            await _call_maybe_async(get_chat_service(), ("clear_messages", "clearMessages"), meeting_id)
            return JSONResponse({"status": "cleared"})
        except Exception as exc:
            msg = str(exc)
            return JSONResponse(
                {"error": msg},
                status_code=404 if msg.startswith("Meeting not found") else 400,
            )

    @app.post("/api/meetings/{meeting_id}/chat/messages")
    async def post_chat_messages_route(meeting_id: str, request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "JSON body required"}, status_code=400)

        content = body.get("content") if isinstance(body, dict) else None
        if not isinstance(content, str) or not content.strip():
            return JSONResponse({"error": "Message content is required"}, status_code=400)

        async def event_generator():
            queue: asyncio.Queue[dict[str, str] | None] = asyncio.Queue()

            async def on_delta(delta: str) -> None:
                await queue.put({"event": "delta", "data": json.dumps({"delta": delta})})

            async def runner() -> None:
                try:
                    assistant_msg = await _call_maybe_async(
                        get_chat_service(),
                        ("stream_message", "streamMessage"),
                        {
                            "meetingId": meeting_id,
                            "content": content,
                            "onDelta": on_delta,
                        },
                    )
                    msg_dict = (
                        assistant_msg.model_dump() if hasattr(assistant_msg, "model_dump") else assistant_msg
                    )
                    await queue.put({"event": "done", "data": json.dumps({"message": msg_dict})})
                except Exception as exc:
                    await queue.put({"event": "error", "data": json.dumps({"error": str(exc)})})
                finally:
                    await queue.put(None)

            task = asyncio.create_task(runner())
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item
            finally:
                if not task.done():
                    task.cancel()

        return EventSourceResponse(event_generator(), sep="\n")

    # Transcription API
    @app.post("/api/meetings/{meeting_id}/transcribe")
    async def post_transcribe_route(meeting_id: str, request: Request) -> Response:
        body: dict[str, Any] = {}
        try:
            parsed = await request.json()
            if isinstance(parsed, dict):
                body = parsed
        except Exception:
            pass

        similarity_threshold = body.get("similarityThreshold")
        clustering_threshold = body.get("clusteringThreshold")
        model_id = body.get("modelId")

        try:
            cfg = load_app_config(app_config_paths)
            if similarity_threshold is None:
                similarity_threshold = cfg.localSimilarityThreshold
            if clustering_threshold is None:
                clustering_threshold = cfg.localClusteringThreshold
            if model_id is None:
                model_id = cfg.localAsrModel
        except Exception:
            pass

        wants_stream = (
            body.get("stream") is True
            or request.query_params.get("stream") == "true"
            or "text/event-stream" in (request.headers.get("accept") or "")
        )

        if wants_stream:

            async def sse_gen():
                queue: asyncio.Queue[dict[str, str] | None] = asyncio.Queue()
                stream_closed = False

                async def on_progress(update: Any) -> None:
                    if stream_closed:
                        return
                    upd_dict = (
                        update.model_dump(exclude_none=True) if hasattr(update, "model_dump") else update
                    )
                    await queue.put({"event": "progress", "data": json.dumps(upd_dict)})

                async def run_job() -> None:
                    await queue.put(
                        {
                            "event": "progress",
                            "data": json.dumps(
                                {
                                    "stage": "decoding",
                                    "percent": 2,
                                    "message": "Initializing transcription...",
                                }
                            ),
                        }
                    )
                    try:
                        res = await _call_maybe_async(
                            get_transcription_service(),
                            ("transcribe_meeting", "transcribeMeeting"),
                            meeting_id,
                            {
                                "provider": body.get("provider"),
                                "language": body.get("language"),
                                "poll": body.get("poll"),
                                "force": body.get("force"),
                                "pollIntervalMs": body.get("pollIntervalMs"),
                                "maxPollWaitMs": body.get("maxPollWaitMs"),
                                "similarityThreshold": similarity_threshold,
                                "clusteringThreshold": clustering_threshold,
                                "modelId": model_id,
                                "onProgress": on_progress,
                            },
                        )
                        status = res.get("status") if isinstance(res, dict) else getattr(res, "status", None)
                        err_str = res.get("error") if isinstance(res, dict) else getattr(res, "error", None)
                        if status == "cancelled":
                            await queue.put(
                                {
                                    "event": "cancelled",
                                    "data": json.dumps(
                                        {
                                            "stage": "cancelled",
                                            "percent": 0,
                                            "message": "Transcription was cancelled.",
                                            "result": res,
                                        }
                                    ),
                                }
                            )
                        elif status == "error":
                            await queue.put(
                                {
                                    "event": "error",
                                    "data": json.dumps(
                                        {
                                            "stage": "error",
                                            "percent": 0,
                                            "message": err_str or "Transcription failed",
                                            "error": err_str or "Transcription failed",
                                            "result": res,
                                        }
                                    ),
                                }
                            )
                        else:
                            await queue.put(
                                {
                                    "event": "result",
                                    "data": json.dumps(
                                        {
                                            "stage": "done",
                                            "percent": 100,
                                            "message": "Transcription complete!",
                                            "result": res,
                                        }
                                    ),
                                }
                            )
                    except Exception as exc:
                        is_cancelled = "cancel" in str(exc).lower()
                        if is_cancelled:
                            await queue.put(
                                {
                                    "event": "cancelled",
                                    "data": json.dumps(
                                        {
                                            "stage": "cancelled",
                                            "percent": 0,
                                            "message": "Transcription was cancelled.",
                                        }
                                    ),
                                }
                            )
                        else:
                            await queue.put(
                                {
                                    "event": "error",
                                    "data": json.dumps(
                                        {
                                            "stage": "error",
                                            "percent": 0,
                                            "message": str(exc),
                                            "error": str(exc),
                                        }
                                    ),
                                }
                            )
                    finally:
                        if not stream_closed:
                            await queue.put(None)

                asyncio.create_task(run_job())
                try:
                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        yield item
                finally:
                    stream_closed = True

            return EventSourceResponse(sse_gen(), sep="\n")

        transcribe_opts = {
            "provider": body.get("provider"),
            "language": body.get("language"),
            "poll": body.get("poll"),
            "force": body.get("force"),
            "pollIntervalMs": body.get("pollIntervalMs"),
            "maxPollWaitMs": body.get("maxPollWaitMs"),
            "similarityThreshold": similarity_threshold,
            "clusteringThreshold": clustering_threshold,
            "modelId": model_id,
        }
        wants_background = (
            body.get("background") is True
            or body.get("async") is True
            or request.query_params.get("background") == "true"
        )
        try:
            svc = get_transcription_service()
            if wants_background and (
                hasattr(svc, "start_transcription") or hasattr(svc, "startTranscription")
            ):
                started = await _call_maybe_async(
                    svc,
                    ("start_transcription", "startTranscription"),
                    meeting_id,
                    transcribe_opts,
                )
                return JSONResponse(started, status_code=202)

            result = await _call_maybe_async(
                svc,
                ("transcribe_meeting", "transcribeMeeting"),
                meeting_id,
                transcribe_opts,
            )
            status = result.get("status") if isinstance(result, dict) else getattr(result, "status", None)
            return JSONResponse(result, status_code=502 if status == "error" else 200)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/meetings/{meeting_id}/transcribe/progress")
    async def get_transcribe_progress_route(meeting_id: str, request: Request) -> Response:
        wants_stream = request.query_params.get("stream") == "true" or "text/event-stream" in (
            request.headers.get("accept") or ""
        )
        if wants_stream:

            async def progress_sse():
                queue: asyncio.Queue[dict[str, str] | None] = asyncio.Queue()
                current = _call_sync(
                    get_transcription_service(),
                    ("get_transcription_progress", "getTranscriptionProgress"),
                    meeting_id,
                )
                if current:
                    c_dict = (
                        current.model_dump(exclude_none=True) if hasattr(current, "model_dump") else current
                    )
                    c_stage = c_dict.get("stage") if isinstance(c_dict, dict) else None
                    c_evt = (
                        "result"
                        if c_stage == "done"
                        else "cancelled"
                        if c_stage == "cancelled"
                        else "error"
                        if c_stage == "error"
                        else "progress"
                    )
                    yield {"event": c_evt, "data": json.dumps(c_dict)}

                async def listener(update: Any) -> None:
                    upd_dict = (
                        update.model_dump(exclude_none=True) if hasattr(update, "model_dump") else update
                    )
                    stage = upd_dict.get("stage")
                    evt = (
                        "result"
                        if stage == "done"
                        else "cancelled"
                        if stage == "cancelled"
                        else "error"
                        if stage == "error"
                        else "progress"
                    )
                    await queue.put({"event": evt, "data": json.dumps(upd_dict)})
                    if stage in ("done", "cancelled", "error"):
                        await queue.put(None)

                unsub = _call_sync(
                    get_transcription_service(),
                    ("subscribe_transcription_progress", "subscribeTranscriptionProgress"),
                    meeting_id,
                    listener,
                )
                from .workflows import is_shutting_down

                try:
                    while not is_shutting_down():
                        try:
                            item = await asyncio.wait_for(queue.get(), timeout=1.0)
                            if item is None or is_shutting_down():
                                break
                            yield item
                        except TimeoutError:
                            if is_shutting_down():
                                break
                            yield {"event": "ping", "data": "{}"}
                finally:
                    if callable(unsub):
                        unsub()

            return EventSourceResponse(progress_sse(), sep="\n")

        progress = _call_sync(
            get_transcription_service(),
            ("get_transcription_progress", "getTranscriptionProgress"),
            meeting_id,
        )
        p_dict = progress.model_dump(exclude_none=True) if hasattr(progress, "model_dump") else progress
        return JSONResponse({"progress": p_dict})

    @app.get("/api/transcriptions/active")
    async def get_active_transcriptions_route() -> JSONResponse:
        active = _call_sync(
            get_transcription_service(),
            ("get_all_active_transcription_progress", "getAllActiveTranscriptionProgress"),
        )
        return JSONResponse({"active": active})

    @app.post("/api/meetings/{meeting_id}/transcribe/cancel")
    async def post_cancel_transcribe_route(meeting_id: str) -> JSONResponse:
        try:
            res = await _call_maybe_async(
                get_transcription_service(),
                ("cancel_transcription", "cancelTranscription"),
                meeting_id,
            )
            return JSONResponse(res, status_code=200)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/meetings/bulk-transcribe")
    async def post_bulk_transcribe_route(request: Request) -> JSONResponse:
        """Queue re-transcription for many meetings with one chosen engine.

        Body: ``{ meetingIds?, provider, force=true, language?, filters? }``
        where ``filters`` may carry ``search``/``startTime``/``endTime``/
        ``currentProvider``. Returns per-meeting queue results; progress can
        be followed via ``GET /api/transcriptions/active``.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "JSON body required"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "JSON object required"}, status_code=400)

        provider = body.get("provider")
        if isinstance(provider, str) and provider.startswith("local"):
            body = {**body, "provider": "local"}
            provider = "local"
        if provider not in ("local", "speechmatics", "assemblyai", "muse"):
            return JSONResponse(
                {"error": "provider must be one of 'local', 'speechmatics', 'assemblyai', 'muse'"},
                status_code=400,
            )

        try:
            cfg = load_app_config(app_config_paths)
            if body.get("similarityThreshold") is None:
                body["similarityThreshold"] = cfg.localSimilarityThreshold
            if body.get("clusteringThreshold") is None:
                body["clusteringThreshold"] = cfg.localClusteringThreshold
            if body.get("modelId") is None and body.get("provider") == "local":
                body["modelId"] = cfg.localAsrModel
        except Exception:
            pass

        try:
            result = await _call_maybe_async(
                get_transcription_service(),
                ("start_bulk_transcription", "startBulkTranscription"),
                body,
            )
            return JSONResponse(result, status_code=202)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # Summaries API
    @app.post("/api/meetings/{meeting_id}/summarize")
    async def post_summarize_route(meeting_id: str, request: Request) -> JSONResponse:
        body: dict[str, Any] = {}
        try:
            parsed = await request.json()
            if isinstance(parsed, dict):
                body = parsed
        except Exception:
            pass

        try:
            status = await _call_maybe_async(
                get_summary_service(),
                ("start_summary_generation", "startSummaryGeneration"),
                {
                    "meetingId": meeting_id,
                    "templateId": body.get("templateId"),
                    "provider": body.get("provider"),
                    "model": body.get("model"),
                    "thinkingLevel": body.get("thinkingLevel"),
                    "setPrimary": body.get("setPrimary"),
                },
            )
            st_dict = status.model_dump(exclude_none=True) if hasattr(status, "model_dump") else status
            return JSONResponse({"started": True, "status": st_dict}, status_code=202)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/meetings/{meeting_id}/summarize/status")
    async def get_summarize_status_route(meeting_id: str) -> JSONResponse:
        status = _call_sync(
            get_summary_service(),
            ("get_summary_generation_status", "getSummaryGenerationStatus"),
            meeting_id,
        )
        st_dict = status.model_dump(exclude_none=True) if hasattr(status, "model_dump") else status
        stage = st_dict.get("stage") if isinstance(st_dict, dict) else None
        return JSONResponse({"generating": stage == "running", "status": st_dict})

    @app.delete("/api/meetings/{meeting_id}/summaries/{artifact_id}")
    async def delete_summary_route(meeting_id: str, artifact_id: str) -> JSONResponse:
        try:
            await _call_maybe_async(
                get_summary_service(),
                ("delete_summary", "deleteSummary"),
                meeting_id,
                artifact_id,
            )
            return JSONResponse({"status": "deleted"})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/meetings/{meeting_id}/summaries/{artifact_id}/primary")
    async def post_primary_summary_route(meeting_id: str, artifact_id: str) -> JSONResponse:
        try:
            await _call_maybe_async(
                get_summary_service(),
                ("set_primary_summary", "setPrimarySummary"),
                meeting_id,
                artifact_id,
            )
            return JSONResponse({"status": "updated", "primarySummaryArtifactId": artifact_id})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    # Speechmatics webhook
    @app.post("/api/webhooks/speechmatics")
    async def post_speechmatics_webhook_route(request: Request) -> JSONResponse:
        if speechmatics_webhook_secret:
            auth_header = request.headers.get("authorization") or ""
            if auth_header != f"Bearer {speechmatics_webhook_secret}":
                return JSONResponse({"error": "Unauthorized"}, status_code=401)

        meeting_id = request.query_params.get("meetingId")
        stage_run_id = request.query_params.get("stageRunId")
        if not meeting_id or not stage_run_id:
            return JSONResponse(
                {"error": "meetingId and stageRunId query params are required"},
                status_code=400,
            )

        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON payload"}, status_code=400)

        try:
            job_id = (payload.get("job") or {}).get("id", "") if isinstance(payload, dict) else ""
            res = await _call_maybe_async(
                get_transcription_service(),
                ("complete_transcription", "completeTranscription"),
                meeting_id,
                stage_run_id,
                job_id,
                payload,
            )
            return JSONResponse(res)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # Speakers API
    # NOTE: /api/speakers/export must be registered before /api/speakers/{speaker_id}
    # or FastAPI routes "export" as a speaker id.
    @app.get("/api/speakers/export")
    async def get_speakers_export_route(request: Request) -> JSONResponse:
        q = request.query_params
        raw_ids = (q.get("speakerIds") or "").strip()
        speaker_ids = [s.strip() for s in raw_ids.split(",") if s.strip()] or None
        download = (q.get("download") or "").strip().lower() in ("1", "true", "yes")
        try:
            library = await _call_maybe_async(
                get_speaker_service(),
                ("export_speaker_library", "exportSpeakerLibrary"),
                {"speakerIds": speaker_ids} if speaker_ids else {},
            )
            if not download:
                return JSONResponse({"library": library})
            filename = q.get("filename") or "olive-voiceprints.json"
            return JSONResponse(
                {"library": library},
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/speakers/import")
    async def post_speakers_import_route(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)
        # Accept either { library, mode, dryRun } or the library object itself.
        library = body.get("library") if isinstance(body.get("library"), dict) else body
        try:
            res = await _call_maybe_async(
                get_speaker_service(),
                ("import_speaker_library", "importSpeakerLibrary"),
                {
                    "library": library,
                    "mode": body.get("mode") or "merge",
                    "dryRun": body.get("dryRun") or False,
                },
            )
            return JSONResponse(res)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/speakers")
    async def get_speakers_route() -> JSONResponse:
        # Full speaker payloads: the People view + assignment combobox read
        # voiceprint lengths and clip counts off providerIds/enrollmentClipPaths.
        speakers = await _call_maybe_async(
            get_speaker_service(),
            ("list_speakers", "listSpeakers"),
            {"includeVoiceprints": True},
        )
        return JSONResponse({"speakers": speakers})

    @app.get("/api/speakers/{speaker_id}")
    async def get_speaker_route(speaker_id: str) -> JSONResponse:
        res = await _call_maybe_async(
            get_speaker_service(),
            ("get_speaker", "getSpeaker"),
            speaker_id,
            {"includeVoiceprints": True},
        )
        if not res:
            return JSONResponse({"error": "Speaker not found"}, status_code=404)
        return JSONResponse(res)

    @app.post("/api/speakers/enroll")
    async def post_speaker_enroll_route(request: Request) -> JSONResponse:
        content_type = request.headers.get("content-type") or ""
        if "multipart/form-data" not in content_type.lower():
            return JSONResponse({"error": "Multipart form data required"}, status_code=400)

        try:
            form = await request.form()
        except Exception:
            return JSONResponse({"error": "Multipart form data required"}, status_code=400)

        name = form.get("name")
        file_field = form.get("file")
        speaker_id = form.get("speakerId")
        language = form.get("language")

        if not isinstance(name, str) or not name.strip():
            return JSONResponse({"error": "name field is required"}, status_code=400)
        if file_field is None or not hasattr(file_field, "read"):
            return JSONResponse({"error": "file field (audio blob) is required"}, status_code=400)

        try:
            audio_bytes = await file_field.read()  # type: ignore[union-attr]
            filename = getattr(file_field, "filename", None) or "audio.wav"
            mime = getattr(file_field, "content_type", None) or "audio/wav"
            speaker = await _call_maybe_async(
                get_speaker_service(),
                ("enroll_speaker", "enrollSpeaker"),
                {
                    "name": name.strip(),
                    "audioBytes": audio_bytes,
                    "mime": mime,
                    "filename": filename,
                    "speakerId": str(speaker_id) if isinstance(speaker_id, str) else None,
                    "language": str(language) if isinstance(language, str) else None,
                },
            )
            return JSONResponse({"status": "enrolled", "speaker": speaker})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/speakers/backfill")
    async def post_speakers_backfill_route(request: Request) -> JSONResponse:
        body: dict[str, Any] = {}
        try:
            parsed = await request.json()
            if isinstance(parsed, dict):
                body = parsed
        except Exception:
            pass
        try:
            res = await _call_maybe_async(
                get_speaker_service(),
                ("backfill_voiceprints", "backfillVoiceprints"),
                {"speakerId": body.get("speakerId"), "force": body.get("force")},
            )
            return JSONResponse({"status": "completed", **res})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/speakers/rebuild")
    async def post_speakers_rebuild_route(request: Request) -> JSONResponse:
        body: dict[str, Any] = {}
        try:
            parsed = await request.json()
            if isinstance(parsed, dict):
                body = parsed
        except Exception:
            pass
        try:
            res = await _call_maybe_async(
                get_speaker_service(),
                ("rebuild_speaker_profiles", "rebuildSpeakerProfiles"),
                {"speakerId": body.get("speakerId"), "force": body.get("force")},
            )
            return JSONResponse({"status": "completed", **res})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/speakers/{speaker_id}/rebuild")
    async def post_speaker_id_rebuild_route(speaker_id: str, request: Request) -> JSONResponse:
        body: dict[str, Any] = {}
        try:
            parsed = await request.json()
            if isinstance(parsed, dict):
                body = parsed
        except Exception:
            pass
        try:
            res = await _call_maybe_async(
                get_speaker_service(),
                ("rebuild_speaker_profiles", "rebuildSpeakerProfiles"),
                {"speakerId": speaker_id, "force": body.get("force")},
            )
            return JSONResponse({"status": "completed", **res})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/speakers/{speaker_id}/backfill")
    async def post_speaker_id_backfill_route(speaker_id: str, request: Request) -> JSONResponse:
        body: dict[str, Any] = {}
        try:
            parsed = await request.json()
            if isinstance(parsed, dict):
                body = parsed
        except Exception:
            pass
        try:
            res = await _call_maybe_async(
                get_speaker_service(),
                ("backfill_voiceprints", "backfillVoiceprints"),
                {"speakerId": speaker_id, "force": body.get("force")},
            )
            return JSONResponse({"status": "completed", **res})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/speakers/merge")
    async def post_speakers_merge_route(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)

        src_id = body.get("sourceSpeakerId") if isinstance(body, dict) else None
        tgt_id = body.get("targetSpeakerId") if isinstance(body, dict) else None
        if not src_id or not tgt_id:
            return JSONResponse(
                {"error": "sourceSpeakerId and targetSpeakerId are required"},
                status_code=400,
            )

        try:
            merged = await _call_maybe_async(
                get_speaker_service(),
                ("merge_speakers", "mergeSpeakers"),
                src_id,
                tgt_id,
            )
            return JSONResponse({"speaker": merged})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/meetings/{meeting_id}/speakers/rapid-enroll-candidates")
    async def get_rapid_enroll_candidates_route(meeting_id: str, limit: int = 5) -> JSONResponse:
        try:
            res = await _call_maybe_async(
                get_speaker_service(),
                ("get_rapid_enroll_candidates",),
                meeting_id,
                limit,
            )
            return JSONResponse(res)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/meetings/{meeting_id}/speakers/rapid-enroll")
    async def post_rapid_enroll_route(meeting_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)

        if (
            not isinstance(body, dict)
            or not body.get("fromLabel")
            or (not body.get("toSpeakerName") and not body.get("toSpeakerId"))
        ):
            return JSONResponse(
                {"error": "fromLabel and either toSpeakerName or toSpeakerId are required"},
                status_code=400,
            )

        try:
            res = await _call_maybe_async(
                get_speaker_service(),
                ("rapid_enroll_assign",),
                {
                    "meetingId": meeting_id,
                    "fromLabel": body.get("fromLabel"),
                    "toSpeakerName": body.get("toSpeakerName"),
                    "toSpeakerId": body.get("toSpeakerId"),
                    "segmentIndices": body.get("segmentIndices") or [],
                },
            )
            return JSONResponse(res)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/meetings/{meeting_id}/speakers/reassign")
    async def post_meeting_speaker_reassign_route(meeting_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)

        if (
            not isinstance(body, dict)
            or not body.get("fromLabel")
            or (not body.get("toSpeakerName") and not body.get("toSpeakerId"))
        ):
            return JSONResponse(
                {"error": "fromLabel and either toSpeakerName or toSpeakerId are required"},
                status_code=400,
            )

        try:
            res = await _call_maybe_async(
                get_speaker_service(),
                ("reassign_meeting_speaker", "reassignMeetingSpeaker"),
                {
                    "meetingId": meeting_id,
                    "fromLabel": body.get("fromLabel"),
                    "toSpeakerName": body.get("toSpeakerName"),
                    "toSpeakerId": body.get("toSpeakerId"),
                    "adoptVoiceprint": body.get("adoptVoiceprint"),
                    "localOnly": body.get("localOnly"),
                    "segmentIndex": body.get("segmentIndex"),
                    "scope": body.get("scope"),
                },
            )
            return JSONResponse(res)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/meetings/{meeting_id}/speakers/confirm-segment")
    async def post_confirm_segment_route(meeting_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)

        seg_idx = body.get("segmentIndex") if isinstance(body, dict) else None
        if not isinstance(seg_idx, int) or isinstance(seg_idx, bool):
            return JSONResponse(
                {"error": "segmentIndex is required and must be a number"},
                status_code=400,
            )

        try:
            res = await _call_maybe_async(
                get_speaker_service(),
                ("confirm_meeting_segment_speaker", "confirmMeetingSegmentSpeaker"),
                {
                    "meetingId": meeting_id,
                    "segmentIndex": seg_idx,
                    "speakerName": body.get("speakerName"),
                    "speakerId": body.get("speakerId"),
                },
            )
            return JSONResponse(res)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/meetings/{meeting_id}/speakers/unassign-segment")
    async def post_unassign_segment_route(meeting_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)

        seg_idx = body.get("segmentIndex") if isinstance(body, dict) else None
        if not isinstance(seg_idx, int) or isinstance(seg_idx, bool):
            return JSONResponse(
                {"error": "segmentIndex is required and must be a number"},
                status_code=400,
            )

        try:
            res = await _call_maybe_async(
                get_speaker_service(),
                ("unassign_meeting_segment_speaker", "unassignMeetingSegmentSpeaker"),
                {"meetingId": meeting_id, "segmentIndex": seg_idx},
            )
            return JSONResponse(res)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/meetings/{meeting_id}/speakers/split-segment")
    async def post_split_segment_route(meeting_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)

        seg_idx = body.get("segmentIndex") if isinstance(body, dict) else None
        if not isinstance(seg_idx, int) or isinstance(seg_idx, bool):
            return JSONResponse(
                {"error": "segmentIndex is required and must be a number"},
                status_code=400,
            )

        try:
            res = await _call_maybe_async(
                get_speaker_service(),
                ("split_meeting_segment", "splitMeetingSegment"),
                {
                    "meetingId": meeting_id,
                    "segmentIndex": seg_idx,
                    "wordIndex": body.get("wordIndex"),
                    "splitMs": body.get("splitMs"),
                    "newSpeakerName": body.get("newSpeakerName"),
                    "newSpeakerId": body.get("newSpeakerId"),
                },
            )
            return JSONResponse(res)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/meetings/{meeting_id}/speakers/merge-segments")
    async def post_merge_segments_route(meeting_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)

        seg_idx = body.get("segmentIndex") if isinstance(body, dict) else None
        if not isinstance(seg_idx, int) or isinstance(seg_idx, bool):
            return JSONResponse(
                {"error": "segmentIndex is required and must be a number"},
                status_code=400,
            )

        try:
            res = await _call_maybe_async(
                get_speaker_service(),
                ("merge_meeting_segments", "mergeMeetingSegments"),
                {"meetingId": meeting_id, "segmentIndex": seg_idx},
            )
            return JSONResponse(res)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/meetings/{meeting_id}/speakers/reidentify")
    async def post_meeting_speakers_reidentify_route(meeting_id: str) -> JSONResponse:
        try:
            res = await _call_maybe_async(
                get_transcription_service(),
                ("request_speaker_reidentification", "requestSpeakerReidentification"),
                meeting_id,
            )
            return JSONResponse(res, status_code=202)
        except ValueError as exc:
            msg = str(exc)
            status = 404 if "not found" in msg.lower() else 409
            return JSONResponse({"error": msg}, status_code=status)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/meetings/{meeting_id}/speakers/reidentify/status")
    async def get_meeting_speakers_reidentify_status_route(meeting_id: str) -> JSONResponse:
        status = _call_sync(
            get_transcription_service(),
            ("get_speaker_reidentification_status", "getSpeakerReidentificationStatus"),
            meeting_id,
        )
        st_dict = status.model_dump(exclude_none=True) if hasattr(status, "model_dump") else status
        return JSONResponse({"status": st_dict})

    @app.post("/api/meetings/{meeting_id}/speakers/reidentify/cancel")
    async def post_meeting_speakers_reidentify_cancel_route(meeting_id: str) -> JSONResponse:
        try:
            res = await _call_maybe_async(
                get_transcription_service(),
                ("cancel_speaker_reidentification", "cancelSpeakerReidentification"),
                meeting_id,
            )
            return JSONResponse(res, status_code=200)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/meetings/{meeting_id}/speakers/link")
    async def post_link_speaker_route(meeting_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)

        if not isinstance(body, dict) or not body.get("speakerId"):
            return JSONResponse({"error": "speakerId is required"}, status_code=400)

        try:
            res = await _call_maybe_async(
                get_speaker_service(),
                ("link_meeting_speaker", "linkMeetingSpeaker"),
                {
                    "meetingId": meeting_id,
                    "speakerId": body["speakerId"],
                    "speechmaticsLabel": body.get("speechmaticsLabel"),
                },
            )
            return JSONResponse(res)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.patch("/api/speakers/{speaker_id}")
    async def patch_speaker_route(speaker_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)

        try:
            updated = await _call_maybe_async(
                get_speaker_service(),
                ("update_speaker", "updateSpeaker"),
                speaker_id,
                body,
            )
            return JSONResponse({"speaker": updated})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.delete("/api/speakers/{speaker_id}")
    async def delete_speaker_route(speaker_id: str) -> JSONResponse:
        await _call_maybe_async(get_speaker_service(), ("delete_speaker", "deleteSpeaker"), speaker_id)
        return JSONResponse({"status": "deleted"})

    # Templates API
    @app.get("/api/templates")
    async def get_templates_route() -> JSONResponse:
        templates = await template_service.list_templates()
        return JSONResponse({"templates": [t.model_dump() for t in templates]})

    @app.get("/api/templates/{template_id}")
    async def get_template_route(template_id: str) -> JSONResponse:
        template = await template_service.get_template(template_id)
        if not template:
            return JSONResponse({"error": "Template not found"}, status_code=404)
        return JSONResponse({"template": template.model_dump()})

    @app.post("/api/templates")
    async def post_template_route(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)

        try:
            template = await template_service.create_template(body)
            return JSONResponse({"template": template.model_dump()}, status_code=201)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.patch("/api/templates/{template_id}")
    @app.put("/api/templates/{template_id}")
    async def update_template_route(template_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)

        try:
            template = await template_service.update_template(template_id, body)
            return JSONResponse({"template": template.model_dump()})
        except Exception as exc:
            msg = str(exc)
            return JSONResponse(
                {"error": msg},
                status_code=404 if "not found" in msg.lower() else 400,
            )

    @app.delete("/api/templates/{template_id}")
    async def delete_template_route(template_id: str) -> JSONResponse:
        try:
            await template_service.delete_template(template_id)
            return JSONResponse({"status": "deleted"})
        except Exception as exc:
            msg = str(exc)
            return JSONResponse(
                {"error": msg},
                status_code=404 if "not found" in msg.lower() else 400,
            )

    @app.post("/api/templates/{template_id}/default")
    async def post_default_template_route(template_id: str) -> JSONResponse:
        try:
            template = await template_service.set_default_template(template_id)
            return JSONResponse({"template": template.model_dump()})
        except Exception as exc:
            msg = str(exc)
            return JSONResponse(
                {"error": msg},
                status_code=404 if "not found" in msg.lower() else 400,
            )

    # LLM API
    @app.get("/api/llm/providers")
    async def get_llm_providers_route() -> JSONResponse:
        providers = await _call_maybe_async(get_llm_service(), ("list_providers", "listProviders"))
        return JSONResponse(
            {"providers": [p.model_dump() if hasattr(p, "model_dump") else p for p in providers]}
        )

    @app.get("/api/llm/models")
    async def get_llm_models_route(request: Request) -> JSONResponse:
        q = request.query_params
        models = await _call_maybe_async(
            get_llm_service(),
            ("list_models", "listModels"),
            {
                "provider": q.get("provider"),
                "search": q.get("search"),
                "limit": _numeric_query_param(q.get("limit")),
            },
        )
        return JSONResponse(
            {"models": [m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m for m in models]}
        )

    @app.post("/api/llm/refresh-catalog")
    async def post_llm_refresh_catalog_route() -> JSONResponse:
        result = await _call_maybe_async(get_llm_service(), ("refresh_catalog", "refreshCatalog"))
        return JSONResponse({"status": "refreshed", **(result or {})})

    @app.get("/api/usage/summary")
    async def get_usage_summary_route(request: Request) -> JSONResponse:
        from .services.usage import current_month_str, get_usage_summary, month_bounds_ms

        q = request.query_params
        period = (q.get("period") or "month").strip().lower()
        try:
            if period == "all":
                summary = get_usage_summary(db)
            else:
                month = (q.get("month") or current_month_str()).strip()
                start_ms, end_ms = month_bounds_ms(month)
                summary = get_usage_summary(db, start_ms=start_ms, end_ms=end_ms)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(summary)

    @app.get("/api/meetings/{meeting_id}/usage")
    async def get_meeting_usage_route(meeting_id: str) -> JSONResponse:
        from .services.usage import get_meeting_usage

        with db.connect() as conn:
            exists = conn.execute(
                text("SELECT id FROM meetings WHERE id = :id"),
                {"id": meeting_id},
            ).first()
        if not exists:
            return JSONResponse({"error": f"Meeting not found: {meeting_id}"}, status_code=404)
        return JSONResponse(get_meeting_usage(db, meeting_id))

    @app.get("/api/llm/config")
    async def get_llm_config_route() -> JSONResponse:
        settings = await _call_maybe_async(get_llm_service(), ("get_settings", "getSettings"))
        settings_dict = settings.model_dump() if hasattr(settings, "model_dump") else dict(settings)
        sanitized_providers: dict[str, Any] = {}
        for p_id, p_cfg in (settings_dict.get("providers") or {}).items():
            entry: dict[str, Any] = {"hasApiKey": bool(p_cfg.get("apiKey"))}
            if p_cfg.get("baseUrl") is not None:
                entry["baseUrl"] = p_cfg["baseUrl"]
            if p_cfg.get("headers") is not None:
                entry["headers"] = p_cfg["headers"]
            sanitized_providers[p_id] = entry
        return JSONResponse(
            {
                "defaultProvider": settings_dict.get("defaultProvider", "google"),
                "defaultModel": settings_dict.get("defaultModel", "gemini-2.5-flash"),
                "defaultThinkingLevel": settings_dict.get("defaultThinkingLevel") or "off",
                "providers": sanitized_providers,
                "customModels": settings_dict.get("customModels") or [],
            }
        )

    @app.post("/api/llm/config")
    async def post_llm_config_route(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)

        try:
            updated = await _call_maybe_async(get_llm_service(), ("update_settings", "updateSettings"), body)
            upd_dict = updated.model_dump() if hasattr(updated, "model_dump") else updated
            return JSONResponse({"status": "updated", "settings": upd_dict})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/config")
    async def get_config_route() -> JSONResponse:
        try:
            config = load_app_config(app_config_paths)
            return JSONResponse(mask_app_config(config))
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/config")
    async def post_config_route(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("JSON object required")
            current = load_app_config(app_config_paths)
            merged = {**current.model_dump(), **body}
            for secret_field in ("hfToken", "speechmaticsApiKey", "assemblyaiApiKey", "museApiKey"):
                if secret_field in body and is_masked_secret(body.get(secret_field)):
                    merged[secret_field] = getattr(current, secret_field)
            updated = save_app_config(merged, app_config_paths)
            _apply_stored_transcription_keys(updated)
            if transcription_service is not None:
                _call_sync(
                    transcription_service,
                    ("set_default_provider", "setDefaultProvider"),
                    updated.transcriptionProvider,
                )
            ingest_service.set_default_transcription_provider(updated.transcriptionProvider)
            return JSONResponse({"status": "updated", "config": mask_app_config(updated)})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/llm/generate")
    async def post_llm_generate_route(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Valid JSON body required"}, status_code=400)

        try:
            res = await _call_maybe_async(get_llm_service(), ("generate_text", "generateText"), body)
            r_dict = res.model_dump() if hasattr(res, "model_dump") else res
            return JSONResponse(r_dict)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/llm/test")
    async def post_llm_test_route(request: Request) -> JSONResponse:
        body: dict[str, Any] = {}
        try:
            parsed = await request.json()
            if isinstance(parsed, dict):
                body = parsed
        except Exception:
            pass

        try:
            res = await _call_maybe_async(
                get_llm_service(),
                ("generate_text", "generateText"),
                {
                    "provider": body.get("provider"),
                    "model": body.get("model"),
                    "prompt": "Reply with strictly the word: PONG",
                    "apiKeyOverride": body.get("apiKey"),
                    "baseUrlOverride": body.get("baseUrl"),
                    "thinkingLevel": body.get("thinkingLevel"),
                },
            )
            r_dict = res.model_dump() if hasattr(res, "model_dump") else res
            text_out = str(r_dict.get("text") or "").strip()
            return JSONResponse(
                {
                    "ok": ("pong" in text_out.lower()) or len(text_out) > 0,
                    "response": text_out,
                    "provider": r_dict.get("provider"),
                    "model": r_dict.get("model"),
                    "durationMs": r_dict.get("durationMs"),
                    "usage": r_dict.get("usage"),
                }
            )
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    # Plaud API
    @app.get("/api/plaud/status")
    async def get_plaud_status_route() -> JSONResponse:
        status = await _call_maybe_async(get_plaud_poller(), ("get_status", "getStatus"))
        return JSONResponse(status)

    @app.post("/api/plaud/sync/trigger")
    async def post_plaud_trigger_route() -> JSONResponse:
        poller = get_plaud_poller()
        start_bg = getattr(poller, "trigger_background", None) or getattr(poller, "triggerBackground", None)
        if callable(start_bg):
            started = bool(start_bg())
        else:
            asyncio.create_task(_call_maybe_async(poller, ("trigger",)))
            started = True
        return JSONResponse(
            {"status": "started", "alreadyRunning": not started},
            status_code=202,
        )

    @app.post("/api/plaud/auth/start")
    async def post_plaud_auth_start_route() -> JSONResponse:
        try:
            login = await _call_maybe_async(get_oauth_manager(), ("start_manual_login", "startManualLogin"))
            verifier = login["verifier"] if isinstance(login, dict) else login.verifier
            state = login["state"] if isinstance(login, dict) else login.state
            auth_url = login["authUrl"] if isinstance(login, dict) else login.auth_url
            session_id = get_auth_sessions().create({"verifier": verifier, "state": state})
            return JSONResponse({"authUrl": auth_url, "sessionId": session_id})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)

    @app.post("/api/plaud/auth/complete")
    async def post_plaud_auth_complete_route(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Request body must be valid JSON"}, status_code=400)

        if not isinstance(body, dict):
            return JSONResponse({"error": "Request body must be a JSON object"}, status_code=400)

        pasted = body.get("pastedUrlOrCode")
        session_id = body.get("sessionId")
        if not isinstance(pasted, str) or not pasted.strip():
            return JSONResponse({"error": "pastedUrlOrCode is required"}, status_code=400)
        if not isinstance(session_id, str) or not session_id:
            return JSONResponse({"error": "sessionId is required"}, status_code=400)

        sessions = get_auth_sessions()
        session = sessions.get(session_id)
        if not session:
            return JSONResponse({"error": "Unknown or expired Plaud auth session"}, status_code=400)

        try:
            await _call_maybe_async(
                get_oauth_manager(),
                ("complete_manual_login", "completeManualLogin"),
                pasted,
                session["verifier"],
                session["state"],
            )
            sessions.delete(session_id)
            return JSONResponse({"status": "completed"})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)

    # Backup & Restore API
    @app.post("/api/backup")
    async def post_backup_route(request: Request) -> JSONResponse:
        try:
            filename: str | None = None
            content_type = request.headers.get("content-type") or ""
            if "application/json" in content_type.lower():
                try:
                    body = await request.json()
                    if (
                        isinstance(body, dict)
                        and isinstance(body.get("filename"), str)
                        and body["filename"].strip()
                    ):
                        filename = body["filename"].strip()
                except Exception:
                    pass
            res = await _call_maybe_async(
                get_backup_service(),
                ("create_backup", "createBackup"),
                {"filename": filename},
            )
            return JSONResponse({"ok": True, "backup": res}, status_code=201)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/backup/list")
    async def get_backup_list_route() -> JSONResponse:
        try:
            backups = await _call_maybe_async(get_backup_service(), ("list_backups", "listBackups"))
            return JSONResponse({"ok": True, "backups": backups})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/backup/download/{filename}")
    async def get_backup_download_route(filename: str) -> Response:
        try:
            full_path = _call_sync(get_backup_service(), ("get_backup_path", "getBackupPath"), filename)
            return FileResponse(
                path=full_path,
                media_type="application/gzip",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    @app.get("/api/backup/export")
    async def get_backup_export_route() -> Response:
        try:
            res = await _call_maybe_async(get_backup_service(), ("create_backup", "createBackup"))
            return FileResponse(
                path=res["path"],
                media_type="application/gzip",
                headers={"Content-Disposition": f'attachment; filename="{res["filename"]}"'},
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.delete("/api/backup/{filename}")
    async def delete_backup_route(filename: str) -> JSONResponse:
        try:
            await _call_maybe_async(get_backup_service(), ("delete_backup", "deleteBackup"), filename)
            return JSONResponse({"ok": True, "message": f"Backup {filename} deleted"})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    @app.post("/api/backup/restore")
    async def post_backup_restore_route(request: Request) -> JSONResponse:
        try:
            content_type = (request.headers.get("content-type") or "").lower()
            if "multipart/form-data" in content_type:
                form = await request.form()
                file_field = form.get("file")
                if file_field is None or not hasattr(file_field, "read"):
                    return JSONResponse(
                        {"error": "file field (backup .tar.gz archive) is required"},
                        status_code=400,
                    )
                buf = await file_field.read()  # type: ignore[union-attr]
                res = await _call_maybe_async(get_backup_service(), ("restore_backup", "restoreBackup"), buf)
                return JSONResponse({"ok": True, "result": res})
            if "application/json" in content_type:
                try:
                    body = await request.json()
                except Exception:
                    body = {}
                filename = body.get("filename") if isinstance(body, dict) else None
                if not isinstance(filename, str) or not filename:
                    return JSONResponse(
                        {"error": "filename is required when restoring from stored backups"},
                        status_code=400,
                    )
                res = await _call_maybe_async(
                    get_backup_service(), ("restore_backup", "restoreBackup"), filename
                )
                return JSONResponse({"ok": True, "result": res})
            return JSONResponse(
                {"error": "Content-Type must be multipart/form-data or application/json"},
                status_code=400,
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # Logs API
    @app.get("/api/logs")
    async def get_logs_route(request: Request) -> JSONResponse:
        try:
            q = request.query_params
            min_level = (q.get("level") or "debug").lower()
            category = (q.get("category") or "").strip()
            meeting_id = (q.get("meetingId") or "").strip()
            search = (q.get("search") or "").strip()
            limit = min(max(_numeric_query_param(q.get("limit")) or 100, 1), 500)
            offset = max(_numeric_query_param(q.get("offset")) or 0, 0)

            level_hierarchy: dict[str, list[str]] = {
                "debug": ["debug", "info", "warn", "error"],
                "info": ["info", "warn", "error"],
                "warn": ["warn", "error"],
                "error": ["error"],
            }
            allowed_levels = level_hierarchy.get(min_level, level_hierarchy["debug"])
            lvl_placeholders = ", ".join(f":lvl_{i}" for i in range(len(allowed_levels)))
            params: dict[str, Any] = {f"lvl_{i}": lvl for i, lvl in enumerate(allowed_levels)}
            params["limit"] = limit
            params["offset"] = offset

            where_clauses = [f"logs.level IN ({lvl_placeholders})"]
            count_where_clauses = [f"logs.level IN ({lvl_placeholders})"]

            if category:
                where_clauses.append("logs.category = :category")
                count_where_clauses.append("logs.category = :category")
                params["category"] = category

            if meeting_id:
                where_clauses.append("logs.meeting_id = :meeting_id")
                count_where_clauses.append("logs.meeting_id = :meeting_id")
                params["meeting_id"] = meeting_id

            if search:
                params["search_pat"] = f"%{search}%"
                where_clauses.append(
                    "(logs.message LIKE :search_pat OR logs.details LIKE :search_pat "
                    "OR meetings.title LIKE :search_pat OR logs.category LIKE :search_pat)"
                )
                count_where_clauses.append(
                    "(logs.message LIKE :search_pat OR logs.details LIKE :search_pat "
                    "OR logs.category LIKE :search_pat)"
                )

            where_sql = " AND ".join(where_clauses)
            count_where_sql = " AND ".join(count_where_clauses)

            with db.connect() as conn:
                rows = (
                    conn.execute(
                        text(
                            f"""
                        SELECT logs.id, logs.level, logs.category, logs.message,
                               logs.meeting_id AS meetingId, meetings.title AS meetingTitle,
                               logs.details, logs.created_at AS createdAt
                        FROM logs
                        LEFT JOIN meetings ON meetings.id = logs.meeting_id
                        WHERE {where_sql}
                        ORDER BY logs.created_at DESC
                        LIMIT :limit OFFSET :offset
                        """
                        ),
                        params,
                    )
                    .mappings()
                    .all()
                )

                total = conn.execute(
                    text(f"SELECT COUNT(id) FROM logs WHERE {count_where_sql}"),
                    params,
                ).scalar()

                cat_rows = (
                    conn.execute(text("SELECT DISTINCT category FROM logs ORDER BY category ASC"))
                    .mappings()
                    .all()
                )

            logs = [
                LogItem(
                    id=r["id"],
                    level=r["level"],  # type: ignore[arg-type]
                    category=r["category"],
                    message=r["message"],
                    meetingId=r["meetingId"],
                    meetingTitle=r["meetingTitle"],
                    details=parse_json_field(r["details"], None),
                    createdAt=int(r["createdAt"]),
                ).model_dump()
                for r in rows
            ]
            categories = [c["category"] for c in cat_rows if c["category"]]
            return JSONResponse(
                {
                    "ok": True,
                    "logs": logs,
                    "pagination": {"total": int(total or 0), "limit": limit, "offset": offset},
                    "categories": categories,
                }
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/debug/queues")
    async def get_debug_queues_route() -> JSONResponse:
        try:
            from .workflows import get_debug_queues_snapshot

            active_map: dict[str, Any] = {}
            if transcription_service is not None:
                try:
                    active_map = _call_sync(
                        transcription_service,
                        ("get_all_active_transcription_progress", "getAllActiveTranscriptionProgress"),
                    )
                except Exception:
                    active_map = {}

            snapshot = await get_debug_queues_snapshot(
                db,
                config_dir=config_dir,
                meetings_dir=meetings_dir,
                active_transcriptions=active_map,
            )
            return JSONResponse(snapshot)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.delete("/api/logs")
    async def delete_logs_route(request: Request) -> JSONResponse:
        try:
            older_than = _numeric_query_param(request.query_params.get("olderThanMs"))
            with db.begin() as conn:
                if older_than is not None:
                    conn.execute(
                        text("DELETE FROM logs WHERE created_at < :older_than"),
                        {"older_than": older_than},
                    )
                else:
                    conn.execute(text("DELETE FROM logs"))
            return JSONResponse({"ok": True})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.get("/api/changelog")
    async def get_changelog_route() -> JSONResponse:
        try:
            changelog_path = _resolve_changelog_path()
            markdown = changelog_path.read_text(encoding="utf-8") if changelog_path else ""
            # "version" is the build identity: the CalVer release tag plus
            # commit hash when built from a tag, else date/time/commit hash.
            # See build_info.py.
            return JSONResponse({"version": get_build_display(), "markdown": markdown})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    web_root = opts.get("webRoot") or opts.get("web_root")
    if web_root:

        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str) -> Response:
            if full_path.startswith("api/"):
                return JSONResponse({"error": "Not found"}, status_code=404)
            no_cache_headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
            candidate = _safe_web_path(str(web_root), full_path)
            if candidate and candidate.is_file():
                if candidate.name == "index.html":
                    return FileResponse(candidate, headers=no_cache_headers)
                return FileResponse(candidate)
            index_path = Path(str(web_root)) / "index.html"
            if index_path.is_file():
                return FileResponse(
                    index_path,
                    media_type="text/html; charset=utf-8",
                    headers=no_cache_headers,
                )
            return JSONResponse({"error": "Not found"}, status_code=404)

    return app


def export_openapi_schema() -> dict[str, Any]:
    """Exports the complete OpenAPI 3.1 schema including all Olive domain models."""
    from . import schemas as s

    db_handle = get_db()
    app = create_app({"db": db_handle})
    spec = app.openapi()
    components = spec.setdefault("components", {})
    schemas_dict = components.setdefault("schemas", {})

    domain_models = [
        s.Meeting,
        s.Recording,
        s.Artifact,
        s.Speaker,
        s.StageRunRow,
        s.Template,
        s.ChatUsage,
        s.ChatMessage,
        s.LlmModelCost,
        s.LlmModelCatalogItem,
        s.LlmProviderSummary,
        s.MeetingSummaryItem,
        s.SummaryGenerationStatus,
        s.SpeakerReidentificationStatus,
        s.TranscriptionProgressUpdate,
        s.BulkTranscribeFilters,
        s.BulkTranscribeRequest,
        s.BulkTranscribeItemResult,
        s.BulkTranscribeResponse,
        s.MeetingDetailAggregate,
        s.BackupManifestStats,
        s.UsageEvent,
        s.UsageProviderBreakdown,
        s.UsageMeetingBreakdown,
        s.UsageRates,
        s.UsageSummary,
        s.MeetingUsageResponse,
        s.BackupManifest,
        s.BackupInfo,
        s.RestoreStats,
        s.RestoreResult,
        s.SpeakerLibrarySpeaker,
        s.SpeakerLibrary,
        s.SpeakerLibraryPlanItem,
        s.SpeakerLibraryImportStats,
        s.SpeakerLibraryImportResult,
        s.LogItem,
        s.LogPagination,
        s.LogListResponse,
        s.RapidEnrollCandidatesResponse,
        s.RapidEnrollAssignResponse,
        s.InsightsSeriesPoint,
        s.InsightsDyadPoint,
        s.InsightsOverview,
        s.InsightsPerson,
        s.InsightsMeetingSignals,
        s.InsightsMoment,
        s.InsightsDyad,
        s.InsightsDyads,
        s.InsightsFindResult,
        s.InsightsFindResponse,
        s.InsightsPlaybook,
        s.InsightsCoaching,
        s.InsightsBackfillResult,
        s.InsightsStatus,
        s.InsightsMe,
        s.TranscriptImportResult,
    ]
    for model_cls in domain_models:
        json_schema = model_cls.model_json_schema(
            ref_template="#/components/schemas/{model}",
            mode="serialization",
        )
        defs = json_schema.pop("$defs", {})
        for def_name, def_schema in defs.items():
            schemas_dict.setdefault(def_name, def_schema)

        props = json_schema.get("properties", {})
        req_set = set(json_schema.get("required", []))
        if model_cls.__name__ == "Meeting":
            req_set.update(props.keys())
        elif model_cls.__name__ == "Speaker":
            req_set.update(props.keys())
        elif model_cls.__name__ == "MeetingDetailAggregate":
            req_set.update(["meeting", "recordings", "artifacts", "speakers", "stageRuns", "summaries"])
        elif model_cls.__name__ in ("TranscriptionProgressUpdate", "SummaryGenerationStatus"):
            for k, prop_schema in list(props.items()):
                if k not in req_set and "anyOf" in prop_schema:
                    non_null = [t for t in prop_schema["anyOf"] if t.get("type") != "null"]
                    if len(non_null) == 1:
                        props[k] = {**non_null[0], "title": prop_schema.get("title", k)}

        if req_set:
            json_schema["required"] = sorted(req_set)
        schemas_dict[model_cls.__name__] = json_schema

    return spec


def _call_sync(target: Any, method_names: tuple[str, ...], *args: Any) -> Any:
    if target is None:
        return None
    for name in method_names:
        fn = getattr(target, name, None)
        if callable(fn):
            return fn(*args)
    return None


async def _call_maybe_async(target: Any, method_names: tuple[str, ...], *args: Any) -> Any:
    for name in method_names:
        fn = getattr(target, name, None)
        if callable(fn):
            res = fn(*args)
            if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
                return await res
            return res
    raise AttributeError(f"Object {type(target).__name__} has none of methods {method_names}")
