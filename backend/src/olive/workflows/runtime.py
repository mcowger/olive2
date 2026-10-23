"""DBOS SQLite runtime initialization and lifecycle management."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from pathlib import Path

from dbos import DBOS, DBOSConfig
from sse_starlette.sse import AppStatus

from ..settings import resolve_paths
from .queues import STAGE_QUEUE_SPECS

_RUNTIME_LOCK = threading.RLock()
_ACTIVE_DB_URL: str | None = None
_ACTIVE_LOOP: asyncio.AbstractEventLoop | None = None
_LAUNCHED: bool = False
_SHUTTING_DOWN = threading.Event()
_TEMP_DIRS: list[tempfile.TemporaryDirectory[str]] = []


def is_shutting_down() -> bool:
    return _SHUTTING_DOWN.is_set() or bool(getattr(AppStatus, "should_exit", False))


def mark_shutting_down() -> None:
    _SHUTTING_DOWN.set()


def is_dbos_launched() -> bool:
    return bool(_LAUNCHED)


def get_existing_dbos_sqlite_path(
    config_dir: str | Path | None = None,
    meetings_dir: str | Path | None = None,
) -> Path | None:
    if _ACTIVE_DB_URL and _ACTIVE_DB_URL.startswith("sqlite:///"):
        p = Path(_ACTIVE_DB_URL.replace("sqlite:///", "", 1))
        if p.exists():
            return p
    if config_dir is not None:
        p = Path(config_dir).resolve() / "dbos.sqlite"
        if p.exists():
            return p
    if meetings_dir is not None and str(meetings_dir).strip():
        p = Path(meetings_dir).resolve().parent / "dbos.sqlite"
        if p.exists():
            return p
    return None


def reset_shutdown_flag() -> None:
    _SHUTTING_DOWN.clear()
    AppStatus.should_exit = False


def _is_default_home_config(path_str: str) -> bool:
    default_cfg = resolve_paths(env={}).config_dir
    try:
        return Path(path_str).resolve() == Path(default_cfg).resolve()
    except Exception:
        return False


def resolve_dbos_database_url(
    config_dir: str | Path | None = None,
    meetings_dir: str | Path | None = None,
) -> str:
    env_url = os.environ.get("OLIVE_DBOS_DATABASE_URL")
    if env_url:
        return env_url

    in_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))

    if config_dir is not None:
        cfg_path = Path(config_dir).resolve()
        if not (in_pytest and _is_default_home_config(str(cfg_path))):
            cfg_path.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{cfg_path / 'dbos.sqlite'}"

    if meetings_dir is not None and str(meetings_dir).strip():
        parent_path = Path(meetings_dir).resolve().parent
        if not (in_pytest and _is_default_home_config(str(parent_path))):
            parent_path.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{parent_path / 'dbos.sqlite'}"

    if not in_pytest:
        env_cfg = os.environ.get("OLIVE_CONFIG_DIR")
        if env_cfg:
            p = Path(env_cfg).resolve()
            p.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{p / 'dbos.sqlite'}"
        if _ACTIVE_DB_URL is None:
            resolved = Path(resolve_paths().config_dir)
            resolved.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{resolved / 'dbos.sqlite'}"

    if _ACTIVE_DB_URL is not None:
        return _ACTIVE_DB_URL

    tmp = tempfile.TemporaryDirectory(prefix="olive-dbos-")
    _TEMP_DIRS.append(tmp)
    return f"sqlite:///{Path(tmp.name) / 'dbos.sqlite'}"


def _build_dbos_config(db_url: str) -> DBOSConfig:
    return {
        "name": "olive",
        "application_version": "0.2.0",
        "system_database_url": db_url,
        "notification_listener_polling_interval_sec": 0.05,
        "scheduler_polling_interval_sec": 1.0,
        "console_log_level": "ERROR",
        "max_executor_threads": resolve_max_executor_threads(),
    }


def resolve_max_executor_threads() -> int:
    """Bound DBOS's executor thread pool.

    DBOS defaults ``max_executor_threads`` to ``sys.maxsize`` — unbounded. That
    pool runs *sync* workflows and sync steps; an unbounded pool turns a queue
    backlog into unbounded thread growth (the 2026-10 transcription OOM: ~950
    parked threads). Async workflows (including every local stage) run on
    DBOS's single background event loop, so the flat stage pipeline does not
    grow this pool — the backlog-to-threads amplifier is gone by construction.

    The cap is therefore a safety net, not a limiter: sized from the sum of
    queue worker concurrencies plus headroom, it never serializes healthy work
    but a runaway (or a future sync workflow fan-out) still bottoms out here.

    Override with ``OLIVE_DBOS_MAX_EXECUTOR_THREADS`` (must be >= 1).
    """
    env_raw = (os.environ.get("OLIVE_DBOS_MAX_EXECUTOR_THREADS") or "").strip()
    if env_raw:
        try:
            parsed = int(env_raw)
        except ValueError:
            parsed = 0
        if parsed >= 1:
            return parsed

    queue_slots = sum(spec.worker_concurrency for spec in STAGE_QUEUE_SPECS)
    return max(16, queue_slots + 16)


async def ensure_dbos_launched_async(
    config_dir: str | Path | None = None,
    meetings_dir: str | Path | None = None,
) -> str:
    global _ACTIVE_DB_URL, _ACTIVE_LOOP, _LAUNCHED

    loop = asyncio.get_running_loop()
    target_url = resolve_dbos_database_url(config_dir=config_dir, meetings_dir=meetings_dir)

    with _RUNTIME_LOCK:
        _SHUTTING_DOWN.clear()
        if (
            _LAUNCHED
            and _ACTIVE_LOOP is loop
            and not loop.is_closed()
            and (_ACTIVE_DB_URL == target_url or (config_dir is None and meetings_dir is None))
        ):
            assert _ACTIVE_DB_URL is not None
            return _ACTIVE_DB_URL

        if _LAUNCHED:
            try:
                DBOS.destroy(destroy_registry=False, workflow_completion_timeout_sec=0)
            except Exception:
                pass
            _LAUNCHED = False

        DBOS(config=_build_dbos_config(target_url))
        DBOS.launch()
        _ACTIVE_DB_URL = target_url
        _ACTIVE_LOOP = loop
        _LAUNCHED = True

    for spec in STAGE_QUEUE_SPECS:
        await DBOS.register_queue_async(
            spec.name,
            worker_concurrency=spec.worker_concurrency,
            polling_interval_sec=spec.polling_interval_sec,
        )

    return target_url


def ensure_dbos_launched_sync(
    config_dir: str | Path | None = None,
    meetings_dir: str | Path | None = None,
) -> str:
    global _ACTIVE_DB_URL, _ACTIVE_LOOP, _LAUNCHED

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    target_url = resolve_dbos_database_url(config_dir=config_dir, meetings_dir=meetings_dir)

    with _RUNTIME_LOCK:
        if _LAUNCHED and _ACTIVE_DB_URL == target_url and _ACTIVE_LOOP is running_loop:
            return target_url

        if _LAUNCHED:
            try:
                DBOS.destroy(destroy_registry=False, workflow_completion_timeout_sec=0)
            except Exception:
                pass
            _LAUNCHED = False

        DBOS(config=_build_dbos_config(target_url))
        DBOS.launch()
        _ACTIVE_DB_URL = target_url
        _ACTIVE_LOOP = running_loop
        _LAUNCHED = True

        if running_loop is None:
            for spec in STAGE_QUEUE_SPECS:
                DBOS.register_queue(
                    spec.name,
                    worker_concurrency=spec.worker_concurrency,
                    polling_interval_sec=spec.polling_interval_sec,
                )

    return target_url


def shutdown_dbos() -> None:
    global _ACTIVE_DB_URL, _ACTIVE_LOOP, _LAUNCHED
    _SHUTTING_DOWN.set()
    with _RUNTIME_LOCK:
        if _LAUNCHED:
            try:
                DBOS.destroy(destroy_registry=False, workflow_completion_timeout_sec=0)
            except Exception:
                pass
        _LAUNCHED = False
        _ACTIVE_LOOP = None
        _ACTIVE_DB_URL = None
        while _TEMP_DIRS:
            tmp = _TEMP_DIRS.pop()
            try:
                tmp.cleanup()
            except Exception:
                pass


__all__ = [
    "ensure_dbos_launched_async",
    "ensure_dbos_launched_sync",
    "get_existing_dbos_sqlite_path",
    "is_dbos_launched",
    "is_shutting_down",
    "mark_shutting_down",
    "reset_shutdown_flag",
    "resolve_dbos_database_url",
    "resolve_max_executor_threads",
    "shutdown_dbos",
]
