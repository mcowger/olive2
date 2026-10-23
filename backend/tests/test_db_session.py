"""Connection isolation for the in-memory database used by threaded tests."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

from sqlalchemy import text

from olive.db import create_db


def test_in_memory_reader_cannot_rollback_another_threads_write() -> None:
    handle = create_db(":memory:")
    inserted = Event()
    commit = Event()
    reader_started = Event()
    reader_finished = Event()

    def write() -> None:
        with handle.engine.begin() as conn:
            conn.execute(text("INSERT INTO sync_state (key, value) VALUES ('checkpoint', 'saved')"))
            inserted.set()
            assert commit.wait(5)

    def read() -> str | None:
        reader_started.set()
        try:
            with handle.engine.connect() as conn:
                return conn.execute(text("SELECT value FROM sync_state WHERE key = 'checkpoint'")).scalar()
        finally:
            reader_finished.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            writer = executor.submit(write)
            try:
                assert inserted.wait(5)
                reader = executor.submit(read)
                assert reader_started.wait(5)
                read_uncommitted = reader_finished.wait(0.2)
            finally:
                commit.set()
            writer.result(timeout=5)
            assert reader.result(timeout=5) == "saved"

        assert not read_uncommitted, "a reader checked out the writer's active SQLite connection"
        with handle.engine.connect() as conn:
            assert (
                conn.execute(text("SELECT value FROM sync_state WHERE key = 'checkpoint'")).scalar()
                == "saved"
            )
    finally:
        handle.engine.dispose()
