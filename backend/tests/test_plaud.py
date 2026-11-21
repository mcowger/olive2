"""Plaud poller, OAuth PKCE, token persistence, and rate-limit retry tests (port of plaud.test.ts)."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from fastapi.testclient import TestClient
from sqlalchemy import text

from olive.app import create_app
from olive.db import create_db, run_migrations
from olive.layout import meeting_folder_name
from olive.plaud import (
    FileTokenStore,
    NotFoundError,
    PlaudPoller,
    RateLimitError,
    alternate_plaud_file_id,
    canonical_plaud_file_id,
    clean_plaud_summary_markdown,
    compute_retry_delay_ms,
    create_plaud_client,
    duration_to_ms,
    is_rate_limit_error,
    parse_retry_after_ms,
    resolve_retry_policy,
    summary_to_markdown,
)
from olive.settings import AppConfig, clear_app_config_cache

AUDIO_BYTES = b"fake plaud audio"
FIXTURE_FILE: dict[str, Any] = {
    "id": "plaud-file-1",
    "name": "Fixture planning",
    "created_at": "2026-08-11T10:00:00.000Z",
    "start_at": "2026-08-11T10:00:00.000Z",
    "duration": 12,
    "serial_number": "fixture-device",
}


def _ready_detail() -> dict[str, Any]:
    return {
        **FIXTURE_FILE,
        "presigned_url": "https://audio.test/fixture.m4a",
        "source_list": [
            {
                "data_type": "transcript",
                "data_content": json.dumps(
                    [
                        {
                            "start_time": 0,
                            "end_time": 5000,
                            "speaker": "Alice",
                            "content": "Let's ship it.",
                        },
                        {
                            "start_time": 5000,
                            "end_time": 12000,
                            "speaker": "Speaker 1",
                            "content": "Generic label.",
                        },
                    ]
                ),
            }
        ],
        "note_list": [
            {
                "data_type": "summary",
                "data_content": json.dumps({"summary": "A fixture summary."}),
            }
        ],
    }


def _pending_detail() -> dict[str, Any]:
    return {
        **FIXTURE_FILE,
        "presigned_url": "https://audio.test/fixture.m4a",
        "source_list": [],
        "note_list": [],
    }


class _FakeOAuth:
    async def startManualLogin(self) -> dict[str, str]:
        return {
            "authUrl": "https://auth.test/authorize",
            "verifier": "server-held-verifier",
            "state": "server-held-state",
        }

    async def completeManualLogin(self, *_args: Any) -> dict[str, str]:
        return {"access_token": "fake-token"}

    async def getAccessToken(self) -> str:
        return "fake-token"


class _FakePlaudClient:
    def __init__(self, detail: dict[str, Any] | Callable[[], dict[str, Any]]) -> None:
        self._detail = detail
        self.get_file_calls = 0
        self.oauth = _FakeOAuth()

    async def getCurrentUser(self) -> dict[str, str]:
        return {"id": "fake-user"}

    async def listFilesIterator(self, _opts: Any = None) -> AsyncIterator[dict[str, Any]]:
        yield FIXTURE_FILE

    async def getFile(self, _fid: str) -> dict[str, Any]:
        self.get_file_calls += 1
        return self._detail() if callable(self._detail) else self._detail


class _FakeResponse:
    def __init__(
        self, content: bytes = AUDIO_BYTES, status: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        self.content = content
        self.status = status
        self.headers = headers or {"content-type": "audio/mp4"}

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    async def arrayBuffer(self) -> bytes:
        return self.content


def _fake_fetch():
    async def _fetch(_url: str) -> _FakeResponse:
        return _FakeResponse()

    return _fetch


async def test_ingests_ready_recording_and_is_idempotent_on_repoll(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    client = _FakePlaudClient(_ready_detail())
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "client": client,
            "fetchImpl": _fake_fetch(),
            "now": lambda: 1_000_000,
        }
    )

    first = await poller.trigger()
    second = await poller.trigger()

    assert first["discoveryCompleted"] is True
    assert first["discoveryError"] is None
    assert first["discovered"] == 1
    assert first["error"] is None
    assert second["discoveryCompleted"] is True
    assert second["discovered"] == 0

    with handle.engine.connect() as conn:
        meetings = conn.execute(text("SELECT * FROM meetings")).mappings().all()
        recordings = conn.execute(text("SELECT * FROM recordings")).mappings().all()
        artifacts = conn.execute(text("SELECT * FROM artifacts")).mappings().all()
        speakers = conn.execute(text("SELECT * FROM speakers")).mappings().all()
        ms = conn.execute(text("SELECT * FROM meeting_speakers")).mappings().all()

    assert len(meetings) == 1
    assert len(recordings) == 1
    assert len(artifacts) == 3
    assert len(speakers) == 1
    assert len(ms) == 1

    meeting = meetings[0]
    recording = recordings[0]
    assert meeting["source"] == "plaud"
    assert meeting["status"] == "ready"
    assert meeting["primary_transcript_artifact_id"]
    assert meeting["primary_summary_artifact_id"]
    assert recording["provider"] == "plaud"
    assert recording["provider_recording_id"] == FIXTURE_FILE["id"]
    assert [(a["provider"], a["kind"], a["format"]) for a in artifacts] == [
        ("plaud", "transcript", "json"),
        ("plaud", "transcript", "txt"),
        ("plaud", "summary", "md"),
    ]
    assert speakers[0]["name"] == "Alice"

    folder = Path(meetings_dir) / meeting_folder_name(
        int(meeting["start_time"]), meeting["title"], meeting["id"]
    )
    assert (folder / recording["path"]).read_bytes() == AUDIO_BYTES
    txt_art = next(a for a in artifacts if a["format"] == "txt")
    assert "Alice: Let's ship it." in (folder / txt_art["path"]).read_text(encoding="utf-8")
    handle.engine.dispose()


async def test_renaming_meeting_preserves_files_and_plaud_import_identity(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    client = _FakePlaudClient(_ready_detail())
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "client": client,
            "fetchImpl": _fake_fetch(),
            "now": lambda: 1_000_000,
        }
    )
    app = create_app({"db": handle.engine, "meetingsDir": meetings_dir, "plaudPoller": poller})
    http_client = TestClient(app)

    await poller.trigger()
    with handle.engine.connect() as conn:
        before = conn.execute(text("SELECT * FROM meetings")).mappings().one()
        rec_before = conn.execute(text("SELECT * FROM recordings")).mappings().one()

    old_folder = Path(meetings_dir) / meeting_folder_name(
        int(before["start_time"]), before["title"], before["id"]
    )
    rename_res = http_client.put(
        f"/api/meetings/{before['id']}/title",
        json={"title": "Renamed planning"},
    )
    assert rename_res.status_code == 200
    assert rename_res.json()["meeting"]["title"] == "Renamed planning"

    with handle.engine.connect() as conn:
        after = conn.execute(text("SELECT * FROM meetings")).mappings().one()

    new_folder = Path(meetings_dir) / meeting_folder_name(
        int(after["start_time"]), after["title"], after["id"]
    )
    assert not old_folder.exists()
    assert new_folder.exists()
    assert (new_folder / rec_before["path"]).read_bytes() == AUDIO_BYTES

    second = await poller.trigger()
    assert second["discovered"] == 0
    assert second["resolved"] == 0
    handle.engine.dispose()


async def test_falls_back_to_audio_only_once_pcs_deadline_passes(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    client = _FakePlaudClient(_pending_detail())
    now = 10_000
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "client": client,
            "fetchImpl": _fake_fetch(),
            "now": lambda: now,
            "pcsWaitMs": 1_000,
        }
    )

    pending = await poller.trigger()
    assert pending["discoveryCompleted"] is True
    assert pending["resolved"] == 0
    assert pending["pcsPending"] == 1

    now = 11_000
    fallback = await poller.trigger()
    repeat = await poller.trigger()

    assert fallback["resolved"] == 1
    assert fallback["pcsPending"] == 0
    assert repeat["resolved"] == 0
    assert client.get_file_calls == 2

    with handle.engine.connect() as conn:
        meeting = conn.execute(text("SELECT * FROM meetings")).mappings().one()
        state = conn.execute(text("SELECT * FROM plaud_ingest_state")).mappings().one()
    assert meeting["status"] == "ready"
    assert state["pcs_resolved"] == 1
    handle.engine.dispose()


class _FakeJob:
    def __init__(self, next_run_time: datetime | None) -> None:
        self.next_run_time = next_run_time


class _FakeScheduler:
    """Records add_job calls; exposes an optional interval job for get_job."""

    def __init__(self, interval_next_run: datetime | None = None) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self._interval = _FakeJob(interval_next_run) if interval_next_run else None

    def get_job(self, job_id: str) -> _FakeJob | None:
        return self._interval if job_id == "plaud_poll" else None

    def add_job(self, func: Any, trigger: str, **kwargs: Any) -> None:
        self.jobs[kwargs["id"]] = {"func": func, "trigger": trigger, **kwargs}


def _pending_poller(tmp_path: Path, clock: dict[str, int], detail: dict[str, Any]) -> tuple[Any, PlaudPoller]:
    handle = create_db(":memory:")
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": str(tmp_path / "meetings"),
            "client": _FakePlaudClient(detail),
            "fetchImpl": _fake_fetch(),
            "now": lambda: clock["now"],
            "pcsWaitMs": 300_000,
        }
    )
    return handle, poller


async def test_pending_recording_schedules_followup_at_pcs_deadline(tmp_path: Path) -> None:
    """A recording discovered without Plaud content is re-polled when its PCS
    wait expires instead of waiting for the next interval poll (#34)."""
    clock = {"now": 1_000_000}
    handle, poller = _pending_poller(tmp_path, clock, _pending_detail())
    scheduler = _FakeScheduler(interval_next_run=datetime.now(UTC) + timedelta(minutes=15))
    poller._scheduler = cast(Any, scheduler)

    before = datetime.now(UTC)
    first = await poller.trigger()
    assert first["discovered"] == 1
    assert first["pcsPending"] == 1

    job = scheduler.jobs["plaud_pcs_followup"]
    assert job["trigger"] == "date"
    # The timer only enqueues the durable DBOS follow-up workflow
    # (see test_plaud_pcs_followup.py); it never runs the sync itself.
    assert job["func"] == poller._enqueue_pcs_followup
    assert job["args"] == [1_000_000 + 300_000]
    delay_s = (job["run_date"] - before).total_seconds()
    assert 300 <= delay_s <= 310  # PCS wait (5 min) + slack, not the 15-min interval

    # The follow-up poll lands after the deadline and ingests audio-only.
    clock["now"] += 302_000
    followup = await poller.run_pcs_followup_poll()
    assert followup["resolved"] == 1
    assert followup["pcsPending"] == 0
    with handle.engine.connect() as conn:
        meeting = conn.execute(text("SELECT * FROM meetings")).mappings().one()
        recordings = conn.execute(text("SELECT * FROM recordings")).mappings().all()
    assert meeting["status"] == "ready"
    assert len(recordings) == 1
    handle.engine.dispose()


async def test_no_followup_when_nothing_pending_or_interval_is_sooner(tmp_path: Path) -> None:
    clock = {"now": 1_000_000}

    # Ready on discovery: resolved in the same pass, nothing to follow up.
    handle, poller = _pending_poller(tmp_path, clock, _ready_detail())
    scheduler = _FakeScheduler(interval_next_run=datetime.now(UTC) + timedelta(minutes=15))
    poller._scheduler = cast(Any, scheduler)
    res = await poller.trigger()
    assert res["resolved"] == 1
    assert scheduler.jobs == {}
    handle.engine.dispose()

    # Interval poll fires before the PCS deadline: no extra job.
    handle, poller = _pending_poller(tmp_path / "b", clock, _pending_detail())
    scheduler = _FakeScheduler(interval_next_run=datetime.now(UTC) + timedelta(minutes=2))
    poller._scheduler = cast(Any, scheduler)
    res = await poller.trigger()
    assert res["pcsPending"] == 1
    assert scheduler.jobs == {}
    handle.engine.dispose()

    # No scheduler (manual-only use): pending is fine, scheduling is a no-op.
    handle, poller = _pending_poller(tmp_path / "c", clock, _pending_detail())
    res = await poller.trigger()
    assert res["pcsPending"] == 1
    assert poller._schedule_pcs_followup() is False
    handle.engine.dispose()


async def test_followup_skips_expired_but_unresolved_recordings(tmp_path: Path) -> None:
    """A recording whose fetch keeps failing after its deadline must not
    drive repeated follow-ups; the interval poll keeps retrying it."""
    clock = {"now": 1_000_000}
    handle, poller = _pending_poller(tmp_path, clock, _pending_detail())
    await poller.trigger()
    clock["now"] += 600_000  # well past the PCS deadline, still unresolved
    assert poller._next_pcs_expiry() is None
    poller._scheduler = cast(Any, _FakeScheduler())
    assert poller._schedule_pcs_followup() is False
    handle.engine.dispose()


def _settings_poller(
    tmp_path: Path, clock: dict[str, int], settings: dict[str, Any]
) -> tuple[Any, PlaudPoller]:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    handle = create_db(":memory:")
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": str(tmp_path / "meetings"),
            "client": _FakePlaudClient(_pending_detail()),
            "fetchImpl": _fake_fetch(),
            "now": lambda: clock["now"],
            "configDir": str(config_dir),
        }
    )
    return handle, poller


async def test_pcs_wait_zero_setting_ingests_audio_on_discovery_poll(tmp_path: Path) -> None:
    """With Plaud transcription off there is nothing to wait for:
    ``plaudPcsWaitMinutes: 0`` imports audio on the poll that finds it."""
    clock = {"now": 1_000_000}
    handle, poller = _settings_poller(tmp_path, clock, {"plaudPcsWaitMinutes": 0})
    scheduler = _FakeScheduler(interval_next_run=datetime.now(UTC) + timedelta(minutes=15))
    poller._scheduler = cast(Any, scheduler)

    res = await poller.trigger()
    assert res["discovered"] == 1
    assert res["resolved"] == 1
    assert res["pcsPending"] == 0
    assert scheduler.jobs == {}
    with handle.engine.connect() as conn:
        meeting = conn.execute(text("SELECT * FROM meetings")).mappings().one()
        recordings = conn.execute(text("SELECT * FROM recordings")).mappings().all()
    assert meeting["status"] == "ready"
    assert len(recordings) == 1
    handle.engine.dispose()


async def test_pcs_wait_setting_is_read_live_and_defaults_to_five_minutes(tmp_path: Path) -> None:
    clock = {"now": 1_000_000}
    handle, poller = _settings_poller(tmp_path, clock, {})
    assert poller._resolve_pcs_wait_ms() == 300_000

    settings_path = tmp_path / "config" / "settings.json"
    settings_path.write_text(json.dumps({"plaudPcsWaitMinutes": 2}), encoding="utf-8")
    assert poller._resolve_pcs_wait_ms() == 120_000

    # A recording already waiting under the old value is released as soon as
    # the setting drops to 0, without a restart.
    settings_path.write_text(json.dumps({"plaudPcsWaitMinutes": 2}), encoding="utf-8")
    first = await poller.trigger()
    assert first["pcsPending"] == 1
    settings_path.write_text(json.dumps({"plaudPcsWaitMinutes": 0}), encoding="utf-8")
    second = await poller.trigger()
    assert second["resolved"] == 1
    assert second["pcsPending"] == 0

    settings_path.write_text(json.dumps({"plaudPcsWaitMinutes": "junk"}), encoding="utf-8")
    assert poller._resolve_pcs_wait_ms() == 300_000
    handle.engine.dispose()


async def test_constructor_pcs_wait_zero_is_honored(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": str(tmp_path / "meetings"),
            "client": _FakePlaudClient(_pending_detail()),
            "fetchImpl": _fake_fetch(),
            "now": lambda: 1_000_000,
            "pcsWaitMs": 0,
        }
    )
    assert poller._resolve_pcs_wait_ms() == 0
    res = await poller.trigger()
    assert res["resolved"] == 1
    handle.engine.dispose()


def test_config_route_saves_plaud_pcs_wait_where_the_poller_reads_it(tmp_path: Path) -> None:
    clear_app_config_cache()
    handle = create_db(":memory:")
    config_dir = tmp_path / "config"
    app = create_app(
        {"db": handle.engine, "configDir": str(config_dir), "meetingsDir": str(tmp_path / "meetings")}
    )
    client = TestClient(app)
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": str(tmp_path / "meetings"),
            "configDir": str(config_dir),
            "client": _FakePlaudClient(_pending_detail()),
        }
    )

    assert client.get("/api/config").json()["plaudPcsWaitMinutes"] == 5
    assert poller._resolve_pcs_wait_ms() == 300_000

    saved = client.post("/api/config", json={"plaudPcsWaitMinutes": 0})
    assert saved.status_code == 200
    assert saved.json()["config"]["plaudPcsWaitMinutes"] == 0
    assert poller._resolve_pcs_wait_ms() == 0

    # Saving other settings (as the Settings modal does) leaves it alone.
    assert client.post("/api/config", json={"transcriptionProvider": "local"}).status_code == 200
    assert poller._resolve_pcs_wait_ms() == 0

    assert client.post("/api/config", json={"plaudPcsWaitMinutes": 61}).status_code == 400
    assert poller._resolve_pcs_wait_ms() == 0
    clear_app_config_cache()
    handle.engine.dispose()


def test_app_config_validates_plaud_pcs_wait_minutes() -> None:
    assert AppConfig().plaudPcsWaitMinutes == 5
    assert AppConfig.model_validate({"plaudPcsWaitMinutes": 0}).plaudPcsWaitMinutes == 0
    for bad in (-1, 61):
        try:
            AppConfig.model_validate({"plaudPcsWaitMinutes": bad})
        except ValueError:
            continue
        raise AssertionError(f"plaudPcsWaitMinutes={bad} should be rejected")


async def test_permanently_skips_duplicate_audio_meeting_instead_of_redownloading(
    tmp_path: Path,
) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")

    dup_file = {
        "id": "plaud-file-duplicate",
        "name": "Re-ID'd duplicate",
        "created_at": "2026-08-11T11:00:00.000Z",
        "start_at": "2026-08-11T11:00:00.000Z",
        "duration": 12,
        "serial_number": "fixture-device",
    }
    details = {
        FIXTURE_FILE["id"]: _ready_detail(),
        dup_file["id"]: {
            **_ready_detail(),
            **dup_file,
            "presigned_url": "https://audio.test/duplicate.m4a",
        },
    }

    download_calls = 0

    class _DupClient:
        oauth = _FakeOAuth()

        async def getCurrentUser(self) -> dict[str, str]:
            return {"id": "fake-user"}

        async def listFilesIterator(self, _opts: Any = None) -> AsyncIterator[dict[str, Any]]:
            yield FIXTURE_FILE
            yield dup_file

        async def getFile(self, fid: str) -> dict[str, Any]:
            return details[fid]

    async def _counting_fetch(_url: str) -> _FakeResponse:
        nonlocal download_calls
        download_calls += 1
        return _FakeResponse()

    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "client": _DupClient(),
            "fetchImpl": _counting_fetch,
            "now": lambda: 1_000_000,
        }
    )

    first = await poller.trigger()
    assert first["discovered"] == 2
    assert download_calls == 2

    with handle.engine.connect() as conn:
        meetings = conn.execute(text("SELECT * FROM meetings ORDER BY created_at ASC")).mappings().all()
        original = next(m for m in meetings if m["title"] == FIXTURE_FILE["name"])
        duplicate = next(m for m in meetings if m["title"] == dup_file["name"])
        dup_state = (
            conn.execute(
                text("SELECT * FROM plaud_ingest_state WHERE meeting_id = :mid"),
                {"mid": duplicate["id"]},
            )
            .mappings()
            .one()
        )
        recs = conn.execute(text("SELECT * FROM recordings")).fetchall()

    assert original["status"] == "ready"
    assert duplicate["status"] == "duplicate"
    assert dup_state["pcs_resolved"] == 1
    assert dup_state["duplicate_of_meeting_id"] == original["id"]
    assert len(recs) == 1

    second = await poller.trigger()
    third = await poller.trigger()
    assert second["discovered"] == 0
    assert second["resolved"] == 0
    assert third["resolved"] == 0
    assert download_calls == 2
    handle.engine.dispose()


def test_plaud_api_status_sync_trigger_and_pkce_auth_routes(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    client = _FakePlaudClient(_pending_detail())
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "client": client,
            "fetchImpl": _fake_fetch(),
            "now": lambda: 1_000,
        }
    )
    completed: list[str] = []

    class _CustomOAuth:
        async def startManualLogin(self) -> dict[str, str]:
            return {
                "authUrl": "https://auth.test/authorize?code_challenge=abc",
                "verifier": "verifier-not-returned",
                "state": "state-not-returned",
            }

        async def completeManualLogin(
            self, inp: str, verifier: str, state: str | None = None
        ) -> dict[str, Any]:
            completed.append(f"{inp}|{verifier}|{state}")
            return {}

    app = create_app(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "plaudPoller": poller,
            "plaudClient": client,
            "oauthManager": _CustomOAuth(),
        }
    )
    http_client = TestClient(app)

    initial_status = http_client.get("/api/plaud/status")
    assert initial_status.status_code == 200
    assert initial_status.json() == {"connected": True, "lastPollAt": None, "pcsPending": 0}

    trigger_res = http_client.post("/api/plaud/sync/trigger")
    assert trigger_res.status_code == 202
    assert trigger_res.json() == {"status": "started", "alreadyRunning": False}

    deadline = time.time() + 5
    status_body: dict[str, Any] = {}
    while time.time() < deadline:
        status_body = http_client.get("/api/plaud/status").json()
        if status_body.get("lastPollAt") is not None:
            break
        time.sleep(0.05)
    assert status_body.get("lastPollAt") == 1_000
    assert status_body.get("pcsPending") == 1

    start_res = http_client.post("/api/plaud/auth/start")
    assert start_res.status_code == 200
    start_body = start_res.json()
    assert start_body["authUrl"] == "https://auth.test/authorize?code_challenge=abc"
    assert start_body["sessionId"]
    assert "verifier" not in start_body
    assert "state" not in start_body

    complete_res = http_client.post(
        "/api/plaud/auth/complete",
        json={
            "sessionId": start_body["sessionId"],
            "pastedUrlOrCode": "https://localhost/callback?code=fake",
        },
    )
    assert complete_res.status_code == 200
    assert complete_res.json() == {"status": "completed"}
    assert completed == ["https://localhost/callback?code=fake|verifier-not-returned|state-not-returned"]

    reused_res = http_client.post(
        "/api/plaud/auth/complete",
        json={"sessionId": start_body["sessionId"], "pastedUrlOrCode": "fake"},
    )
    assert reused_res.status_code == 400
    handle.engine.dispose()


async def test_token_store_duration_conversion_and_migration_repair(tmp_path: Path) -> None:
    tokens_path = str(tmp_path / "plaud-tokens.json")
    paths = {
        "configDir": str(tmp_path),
        "meetingsDir": str(tmp_path / "meetings"),
        "backupsDir": str(tmp_path / "backups"),
        "modelsDir": str(tmp_path / "models"),
        "databasePath": str(tmp_path / "olive.sqlite"),
        "settingsPath": str(tmp_path / "settings.json"),
        "plaudTokensPath": tokens_path,
    }
    client = create_plaud_client(paths)
    assert client.tokenStore.filePath == tokens_path

    store = FileTokenStore(tokens_path)
    await store.save(
        {
            "access_token": "persisted-access-token",
            "token_type": "Bearer",
            "refresh_token": "persisted-refresh-token",
            "expires_at": int(time.time() * 1000) + 3_600_000,
        }
    )
    assert Path(tokens_path).exists()
    reloaded = await FileTokenStore(tokens_path).load()
    assert reloaded is not None
    assert reloaded["access_token"] == "persisted-access-token"
    assert reloaded["refresh_token"] == "persisted-refresh-token"

    assert duration_to_ms(252_960) == 252_960
    assert duration_to_ms(462_960) == 462_960
    assert duration_to_ms(3_600_000) == 3_600_000
    assert duration_to_ms(12) == 12_000
    assert duration_to_ms(0.5) == 500
    assert duration_to_ms(None) == 0

    handle = create_db(":memory:")
    now = 1_700_000_000_000
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  'corrupted-plaud-meeting', '2026-08-14 16:26:26', :now, :end,
                  'plaud', 'processing', '[]', NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {"now": now, "end": now + 252_960_000},
        )
    run_migrations(handle.engine)
    with handle.engine.connect() as conn:
        repaired = (
            conn.execute(
                text("SELECT start_time, end_time FROM meetings WHERE id = 'corrupted-plaud-meeting'")
            )
            .mappings()
            .one()
        )
    assert int(repaired["end_time"]) - int(repaired["start_time"]) == 252_960
    handle.engine.dispose()


async def test_plaud_rate_limit_backoff_and_retry_helpers(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    get_file_calls = 0

    class _ThrottledClient(_FakePlaudClient):
        async def getFile(self, _fid: str) -> dict[str, Any]:
            nonlocal get_file_calls
            get_file_calls += 1
            if get_file_calls <= 2:
                raise RateLimitError("Rate limit exceeded.")
            return _ready_detail()

    delays: list[int] = []

    async def _rec_sleep(ms: int) -> None:
        delays.append(ms)

    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "client": _ThrottledClient(_ready_detail()),
            "fetchImpl": _fake_fetch(),
            "now": lambda: 1_000_000,
            "retryOptions": {"baseDelayMs": 1_000, "random": lambda: 1.0, "sleep": _rec_sleep},
        }
    )
    res = await poller.trigger()
    assert res["discoveryCompleted"] is True
    assert res["error"] is None
    assert res["resolved"] == 1
    assert get_file_calls == 3
    assert delays == [750, 1500]

    assert is_rate_limit_error(RateLimitError("Rate limit exceeded.")) is True
    assert is_rate_limit_error(RuntimeError("Too Many Requests")) is True
    assert is_rate_limit_error(RuntimeError("connection reset")) is False

    policy = resolve_retry_policy({"baseDelayMs": 1_000, "maxDelayMs": 5_000, "random": lambda: 0.0})
    assert compute_retry_delay_ms(RateLimitError("429", 4_000), 0, policy) == 4_000
    assert compute_retry_delay_ms(RateLimitError("429", 9_999), 0, policy) == 5_000
    assert compute_retry_delay_ms(RateLimitError("429"), 2, policy) == 4_000
    assert parse_retry_after_ms("2") == 2_000
    assert parse_retry_after_ms(None) is None
    assert parse_retry_after_ms("not-a-date") is None

    raw_note = "## Auto_sum_note\n\n---\n**Project Requirements**\n- Item 1\n- Item 2"
    cleaned = clean_plaud_summary_markdown(raw_note)
    assert "Auto_sum_note" not in cleaned
    assert not cleaned.startswith("---")
    assert cleaned.startswith("**Project Requirements**")
    converted = summary_to_markdown([{"type": "auto_sum_note", "content": raw_note}])
    assert "Auto_sum_note" not in converted
    assert converted.startswith("**Project Requirements**")
    handle.engine.dispose()


def test_canonical_plaud_file_id_normalizes_migrated_prefix() -> None:
    hex_id = "69bf84fe38ba0a1871b1e81842131f70"
    assert canonical_plaud_file_id(hex_id) == hex_id
    assert canonical_plaud_file_id(f"of_{hex_id}") == hex_id
    assert alternate_plaud_file_id(hex_id) == f"of_{hex_id}"
    assert alternate_plaud_file_id(f"of_{hex_id}") == hex_id


def _seed_legacy_plaud_state(
    handle: Any, meeting_id: str, file_id: str, now: int, title: str = "Legacy Plaud"
) -> None:
    with handle.engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO meetings (
                  id, title, start_time, end_time, source, status, tags,
                  primary_transcript_artifact_id, primary_summary_artifact_id,
                  last_error, created_at, updated_at
                ) VALUES (
                  :id, :title, :now, :end, 'plaud', 'processing', '[]',
                  NULL, NULL, NULL, :now, :now
                )
                """
            ),
            {"id": meeting_id, "title": title, "now": now, "end": now + 12_000},
        )
        conn.execute(
            text(
                """
                INSERT INTO plaud_ingest_state (
                  meeting_id, plaud_file_id, first_seen_at, pcs_deadline_at, pcs_resolved
                ) VALUES (:mid, :fid, :now, :deadline, 0)
                """
            ),
            {"mid": meeting_id, "fid": file_id, "now": now, "deadline": now + 300_000},
        )


async def test_plaud_id_migration_does_not_reimport_recording(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    hex_id = "69bf84fe38ba0a1871b1e81842131f70"
    legacy_file = {**FIXTURE_FILE, "id": hex_id}
    migrated_file = {**FIXTURE_FILE, "id": f"of_{hex_id}"}

    class _MigratingClient:
        oauth = _FakeOAuth()

        def __init__(self) -> None:
            self.current = [legacy_file]

        async def getCurrentUser(self) -> dict[str, str]:
            return {"id": "fake-user"}

        async def listFilesIterator(self, _opts: Any = None) -> AsyncIterator[dict[str, Any]]:
            for item in self.current:
                yield item

        async def getFile(self, _fid: str) -> dict[str, Any]:
            return _ready_detail()

    client = _MigratingClient()
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "client": client,
            "fetchImpl": _fake_fetch(),
            "now": lambda: 1_000_000,
        }
    )

    first = await poller.trigger()
    assert first["discovered"] == 1
    assert first["resolved"] == 1

    client.current = [migrated_file]
    second = await poller.trigger()
    assert second["discovered"] == 0
    assert second["resolved"] == 0

    with handle.engine.connect() as conn:
        meeting_count = conn.execute(text("SELECT COUNT(*) FROM meetings")).scalar()
        state_count = conn.execute(text("SELECT COUNT(*) FROM plaud_ingest_state")).scalar()
    assert meeting_count == 1
    assert state_count == 1
    handle.engine.dispose()


async def test_plaud_legacy_id_falls_back_to_prefixed_id(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    hex_id = "69bf84fe38ba0a1871b1e81842131f70"
    prefixed_id = f"of_{hex_id}"
    now = 1_000_000
    _seed_legacy_plaud_state(handle, "legacy-meeting", hex_id, now)

    calls: list[str] = []

    class _MigratedClient:
        oauth = _FakeOAuth()

        async def getCurrentUser(self) -> dict[str, str]:
            return {"id": "fake-user"}

        async def listFilesIterator(self, _opts: Any = None) -> AsyncIterator[dict[str, Any]]:
            for _ in ():
                yield _

        async def getFile(self, fid: str) -> dict[str, Any]:
            calls.append(fid)
            if fid == hex_id:
                raise NotFoundError(f"Resource not found (/open/third-party/files/{fid})")
            assert fid == prefixed_id
            return _ready_detail()

    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "client": _MigratedClient(),
            "fetchImpl": _fake_fetch(),
            "now": lambda: now,
        }
    )

    res = await poller.trigger()
    assert res["error"] is None
    assert res["resolved"] == 1
    assert calls == [hex_id, prefixed_id]

    with handle.engine.connect() as conn:
        meeting = conn.execute(text("SELECT * FROM meetings WHERE id = 'legacy-meeting'")).mappings().one()
        state = (
            conn.execute(text("SELECT * FROM plaud_ingest_state WHERE meeting_id = 'legacy-meeting'"))
            .mappings()
            .one()
        )
        recording = conn.execute(text("SELECT * FROM recordings")).mappings().one()
    assert meeting["status"] == "ready"
    assert state["pcs_resolved"] == 1
    assert recording["provider_recording_id"] == prefixed_id
    handle.engine.dispose()


async def test_plaud_permanently_missing_file_stops_retrying(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    hex_id = "c5d65a5dec71e7a78399482286596cad"
    now = 1_000_000
    _seed_legacy_plaud_state(handle, "missing-meeting", hex_id, now)

    calls = 0

    class _MissingClient:
        oauth = _FakeOAuth()

        async def getCurrentUser(self) -> dict[str, str]:
            return {"id": "fake-user"}

        async def listFilesIterator(self, _opts: Any = None) -> AsyncIterator[dict[str, Any]]:
            for _ in ():
                yield _

        async def getFile(self, fid: str) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            raise NotFoundError(f"Resource not found (/open/third-party/files/{fid})")

    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "client": _MissingClient(),
            "fetchImpl": _fake_fetch(),
            "now": lambda: now,
        }
    )

    first = await poller.trigger()
    assert first["error"] is None
    assert first["resolved"] == 0
    assert first["pcsPending"] == 0
    assert calls == 2

    with handle.engine.connect() as conn:
        meeting = conn.execute(text("SELECT * FROM meetings WHERE id = 'missing-meeting'")).mappings().one()
        state = (
            conn.execute(text("SELECT * FROM plaud_ingest_state WHERE meeting_id = 'missing-meeting'"))
            .mappings()
            .one()
        )
    assert meeting["status"] == "error"
    assert "no longer available" in meeting["last_error"]
    assert state["pcs_resolved"] == 1

    second = await poller.trigger()
    assert calls == 2
    assert second["resolved"] == 0
    handle.engine.dispose()


async def test_plaud_trigger_background_does_not_block_and_dedupes(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "client": _FakePlaudClient(_ready_detail()),
            "fetchImpl": _fake_fetch(),
            "now": lambda: 1_000_000,
        }
    )

    assert poller.trigger_background() is True
    assert poller.trigger_background() is False
    in_flight = poller._in_flight
    assert in_flight is not None
    await in_flight

    with handle.engine.connect() as conn:
        meetings = conn.execute(text("SELECT * FROM meetings")).mappings().all()
    assert len(meetings) == 1
    assert meetings[0]["status"] == "ready"
    assert poller._in_flight is None
    handle.engine.dispose()


async def test_plaud_hands_off_transcription_without_blocking(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    meetings_dir = str(tmp_path / "meetings")
    release = asyncio.Event()
    calls: list[str] = []

    class _BlockingTranscription:
        async def transcribe_meeting(self, meeting_id: str) -> dict[str, Any]:
            calls.append(meeting_id)
            await release.wait()
            return {"status": "done"}

    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": meetings_dir,
            "client": _FakePlaudClient(_ready_detail()),
            "fetchImpl": _fake_fetch(),
            "now": lambda: 1_000_000,
            "transcriptionService": _BlockingTranscription(),
            "autoTranscribeWithSpeechmatics": True,
        }
    )

    res = await poller.trigger()
    assert res["resolved"] == 1
    assert not release.is_set()

    await asyncio.sleep(0)
    assert len(calls) == 1
    assert not release.is_set()

    release.set()
    await asyncio.sleep(0)
    handle.engine.dispose()


def test_poller_start_runs_in_lifespan_with_running_loop(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    started: dict[str, Any] = {}

    class _FakePoller:
        def start(self, interval_minutes: int) -> None:
            started["interval"] = interval_minutes

        def stop(self) -> None:
            started["stopped"] = True

    app = create_app(
        {
            "db": handle.engine,
            "meetingsDir": str(tmp_path / "meetings"),
            "plaudPoller": _FakePoller(),
            "startPlaudPoller": True,
            "pollIntervalMinutes": 7,
        }
    )
    with TestClient(app):
        assert started == {"interval": 7}
    assert started.get("stopped") is True
    handle.engine.dispose()


def test_poller_does_not_start_without_flag(tmp_path: Path) -> None:
    handle = create_db(":memory:")

    class _FakePoller:
        def start(self, _interval_minutes: int) -> None:
            raise AssertionError("must not start")

        def stop(self) -> None:
            pass

    app = create_app(
        {
            "db": handle.engine,
            "meetingsDir": str(tmp_path / "meetings"),
            "plaudPoller": _FakePoller(),
        }
    )
    with TestClient(app):
        pass
    handle.engine.dispose()


def _named_detail() -> dict[str, Any]:
    return {
        **FIXTURE_FILE,
        "presigned_url": "https://audio.test/fixture.m4a",
        "source_list": [
            {
                "data_type": "transcript",
                "data_content": json.dumps(
                    [
                        {
                            "start_time": 0,
                            "end_time": 5000,
                            "speaker": "Alice",
                            "content": "Let's ship it.",
                        },
                        {
                            "start_time": 5000,
                            "end_time": 12000,
                            "speaker": "Bob",
                            "content": "Agreed.",
                        },
                    ]
                ),
            }
        ],
        "note_list": [],
    }


def _unnamed_detail() -> dict[str, Any]:
    return {
        **FIXTURE_FILE,
        "presigned_url": "https://audio.test/fixture.m4a",
        "source_list": [
            {
                "data_type": "transcript",
                "data_content": json.dumps(
                    [
                        {
                            "start_time": 0,
                            "end_time": 5000,
                            "speaker": "Speaker 1",
                            "content": "Hello.",
                        },
                        {
                            "start_time": 5000,
                            "end_time": 12000,
                            "speaker": "Speaker 2",
                            "content": "Hi there.",
                        },
                    ]
                ),
            }
        ],
        "note_list": [],
    }


def _summary_only_detail() -> dict[str, Any]:
    return {
        **FIXTURE_FILE,
        "presigned_url": "https://audio.test/fixture.m4a",
        "source_list": [],
        "note_list": [
            {
                "data_type": "summary",
                "data_content": json.dumps({"summary": "A summary with no transcript."}),
            }
        ],
    }


class _RecordingTranscription:
    """Production-shaped double: prefers start_transcription like the real service."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def start_transcription(self, meeting_id: str) -> dict[str, Any]:
        self.calls.append(meeting_id)
        return {"status": "running"}


async def test_always_retranscribe_fires_on_named_speakers_and_skips_identify(
    tmp_path: Path,
) -> None:
    handle = create_db(":memory:")
    svc = _RecordingTranscription()
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": str(tmp_path / "meetings"),
            "client": _FakePlaudClient(_named_detail()),
            "fetchImpl": _fake_fetch(),
            "now": lambda: 1_000_000,
            "transcriptionService": svc,
            "alwaysRetranscribePlaud": True,
        }
    )
    identify_calls: list[str] = []

    async def _fake_identify(meeting_id: str) -> bool:
        identify_calls.append(meeting_id)
        return True

    poller._enqueue_plaud_speaker_identification = _fake_identify  # type: ignore[method-assign]
    res = await poller.trigger()
    assert res["resolved"] == 1
    await asyncio.sleep(0)
    assert len(svc.calls) == 1
    # Replacement transcript carries its own speaker identification.
    assert identify_calls == []
    handle.engine.dispose()


async def test_unnamed_only_skips_named_but_fires_on_generic(tmp_path: Path) -> None:
    for detail_fn, expect_calls in ((_named_detail, 0), (_unnamed_detail, 1)):
        handle = create_db(":memory:")
        svc = _RecordingTranscription()
        poller = PlaudPoller(
            {
                "db": handle.engine,
                "meetingsDir": str(tmp_path / "meetings"),
                "client": _FakePlaudClient(detail_fn()),
                "fetchImpl": _fake_fetch(),
                "now": lambda: 1_000_000,
                "transcriptionService": svc,
                "retranscribePlaudWhenUnnamed": True,
            }
        )
        res = await poller.trigger()
        assert res["resolved"] == 1
        await asyncio.sleep(0)
        assert len(svc.calls) == expect_calls
        handle.engine.dispose()


async def test_summary_without_segments_hands_off_despite_flags_off(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    svc = _RecordingTranscription()
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": str(tmp_path / "meetings"),
            "client": _FakePlaudClient(_summary_only_detail()),
            "fetchImpl": _fake_fetch(),
            "now": lambda: 1_000_000,
            "transcriptionService": svc,
        }
    )
    res = await poller.trigger()
    assert res["resolved"] == 1
    await asyncio.sleep(0)
    assert len(svc.calls) == 1
    with handle.engine.connect() as conn:
        meeting = conn.execute(text("SELECT * FROM meetings")).mappings().one()
    # Plaud summary is kept as fallback; no transcript to stay authoritative on.
    assert meeting["primary_transcript_artifact_id"] is None
    assert meeting["primary_summary_artifact_id"] is not None
    handle.engine.dispose()


async def test_legacy_alias_still_triggers_retranscribe(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    svc = _RecordingTranscription()
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": str(tmp_path / "meetings"),
            "client": _FakePlaudClient(_named_detail()),
            "fetchImpl": _fake_fetch(),
            "now": lambda: 1_000_000,
            "transcriptionService": svc,
            "autoTranscribeWithSpeechmatics": True,
        }
    )
    res = await poller.trigger()
    assert res["resolved"] == 1
    await asyncio.sleep(0)
    assert len(svc.calls) == 1
    handle.engine.dispose()


async def test_retranscribe_flags_resolve_live_from_settings_without_restart(
    tmp_path: Path,
) -> None:
    handle = create_db(":memory:")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    settings_path = config_dir / "settings.json"
    settings_path.write_text(json.dumps({"alwaysRetranscribePlaud": True}), encoding="utf-8")
    svc = _RecordingTranscription()
    poller = PlaudPoller(
        {
            "db": handle.engine,
            "meetingsDir": str(tmp_path / "meetings"),
            "client": _FakePlaudClient(_named_detail()),
            "fetchImpl": _fake_fetch(),
            "now": lambda: 1_000_000,
            "transcriptionService": svc,
            "configDir": str(config_dir),
        }
    )
    # Same instance sees the toggle flip: no restart required.
    assert poller._resolve_plaud_retranscribe_flags() == (True, False)
    res = await poller.trigger()
    assert res["resolved"] == 1
    await asyncio.sleep(0)
    assert len(svc.calls) == 1
    settings_path.write_text(json.dumps({}), encoding="utf-8")
    assert poller._resolve_plaud_retranscribe_flags() == (False, False)
    handle.engine.dispose()
