"""Database handle management (port of packages/server/src/db.ts)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from ..settings import resolve_paths
from .migrations import run_migrations


@dataclass
class DbHandle:
    engine: Engine
    session_factory: sessionmaker[Session]
    database_path: str

    def session(self) -> Session:
        return self.session_factory()

    def raw_sqlite(self) -> sqlite3.Connection:
        raw = self.engine.raw_connection()
        return raw.driver_connection  # type: ignore[return-value]


def _configure_sqlite_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute("PRAGMA busy_timeout = 5000")
    cursor.close()


def create_db(database_path: str) -> DbHandle:
    if database_path == ":memory:":
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            # One connection keeps the in-memory database alive, but checkouts
            # must be exclusive: a concurrent reader's rollback would otherwise
            # discard another thread's uncommitted writes on that connection.
            poolclass=QueuePool,
            pool_size=1,
            max_overflow=0,
        )
    else:
        db_file = Path(database_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite:///{db_file}",
            connect_args={"check_same_thread": False, "timeout": 30.0},
        )

    event.listen(engine, "connect", _configure_sqlite_pragmas)
    run_migrations(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return DbHandle(engine=engine, session_factory=session_factory, database_path=database_path)


_active_handle: DbHandle | None = None


def get_db_handle() -> DbHandle:
    global _active_handle
    if _active_handle is None:
        _active_handle = create_db(resolve_paths().database_path)
    return _active_handle


def get_db() -> Engine:
    return get_db_handle().engine


def set_db_for_tests(handle: DbHandle | None) -> None:
    global _active_handle
    if _active_handle is not None and _active_handle is not handle:
        try:
            _active_handle.engine.dispose()
        except Exception:
            pass
    _active_handle = handle


def close_db() -> None:
    global _active_handle
    if _active_handle is None:
        return
    try:
        _active_handle.engine.dispose()
    except Exception:
        pass
    _active_handle = None


def reset_db_handle(database_path: str | None = None) -> DbHandle:
    global _active_handle
    close_db()
    handle = create_db(database_path or resolve_paths().database_path)
    _active_handle = handle
    return handle
