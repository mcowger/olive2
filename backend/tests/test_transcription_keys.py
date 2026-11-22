"""Settings-UI transcription keys (Speechmatics / AssemblyAI / Muse)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from olive.app import create_app, refresh_transcription_client_keys
from olive.db import create_db
from olive.providers.assemblyai import AssemblyAIClient
from olive.providers.muse import MuseClient
from olive.providers.speechmatics import SpeechmaticsClient
from olive.settings import (
    AppConfig,
    clear_app_config_cache,
    get_effective_assemblyai_key,
    get_effective_muse_key,
    get_effective_speechmatics_key,
    load_app_config,
    mask_app_config,
    resolve_paths,
    save_app_config,
)

SM_KEY = "sm_live_1234567890"
AAI_KEY = "aai_live_0987654321"
MUSE_KEY = "muse_live_abcdef1234"


def _paths(tmp_path):  # type: ignore[no-untyped-def]
    clear_app_config_cache()
    return resolve_paths({"OLIVE_CONFIG_DIR": str(tmp_path)})


def test_new_key_fields_default_empty_and_masked(tmp_path, monkeypatch) -> None:
    for var in (
        "SPEECHMATICS_API_KEY",
        "ASSEMBLY_API_KEY",
        "ASSEMBLYAI_API_KEY",
        "MUSE_API_KEY",
        "MODEL_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    paths = _paths(tmp_path)
    cfg = AppConfig.model_validate(
        {
            "speechmaticsApiKey": SM_KEY,
            "assemblyaiApiKey": AAI_KEY,
            "museApiKey": MUSE_KEY,
        }
    )
    save_app_config(cfg, paths)
    loaded = load_app_config(paths)
    assert loaded.speechmaticsApiKey == SM_KEY
    assert loaded.assemblyaiApiKey == AAI_KEY
    assert loaded.museApiKey == MUSE_KEY

    masked = mask_app_config(loaded)
    assert masked["speechmaticsApiKey"] == "••••••••7890"
    assert masked["assemblyaiApiKey"] == "••••••••4321"
    assert masked["museApiKey"] == "••••••••1234"
    assert masked["hasSpeechmaticsApiKey"] is True
    assert masked["hasAssemblyaiApiKey"] is True
    assert masked["hasMuseApiKey"] is True

    fresh = mask_app_config(AppConfig())
    assert fresh["speechmaticsApiKey"] == ""
    assert fresh["hasSpeechmaticsApiKey"] is False


def test_effective_keys_prefer_env_over_stored(tmp_path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    cfg = save_app_config(
        {
            "speechmaticsApiKey": SM_KEY,
            "assemblyaiApiKey": AAI_KEY,
            "museApiKey": MUSE_KEY,
        },
        paths,
    )
    # No env -> stored keys win.
    env: dict[str, str] = {}
    assert get_effective_speechmatics_key(cfg, env) == SM_KEY
    assert get_effective_assemblyai_key(cfg, env) == AAI_KEY
    assert get_effective_muse_key(cfg, env) == MUSE_KEY

    # Env wins over stored.
    env = {
        "SPEECHMATICS_API_KEY": "sm_env",
        "ASSEMBLYAI_API_KEY": "aai_env",
        "MUSE_API_KEY": "muse_env",
    }
    assert get_effective_speechmatics_key(cfg, env) == "sm_env"
    assert get_effective_assemblyai_key(cfg, env) == "aai_env"
    assert get_effective_muse_key(cfg, env) == "muse_env"

    # Legacy alternates also win.
    assert get_effective_assemblyai_key(cfg, {"ASSEMBLY_API_KEY": "legacy"}) == "legacy"
    assert get_effective_muse_key(cfg, {"MODEL_API_KEY": "legacy"}) == "legacy"

    # Masked / blank stored values are ignored (never treated as keys).
    masked_cfg = AppConfig.model_validate(
        {
            "speechmaticsApiKey": "••••••••7890",
            "assemblyaiApiKey": "   ",
            "museApiKey": "",
        }
    )
    assert get_effective_speechmatics_key(masked_cfg, {}) is None
    assert get_effective_assemblyai_key(masked_cfg, {}) is None
    assert get_effective_muse_key(masked_cfg, {}) is None


def test_client_set_api_key_updates_inner_speechmatics_client() -> None:
    client = SpeechmaticsClient({"apiKey": "old"})
    assert client.is_configured is True
    client.set_api_key("new-key")
    assert client.api_key == "new-key"
    assert client.raw_client.api_key == "new-key"

    for client in (AssemblyAIClient({"apiKey": "old"}), MuseClient({"apiKey": "old"})):
        client.set_api_key("new-key")
        assert client.api_key == "new-key"
        assert client.is_configured is True
        client.set_api_key("")
        assert client.is_configured is False


def test_refresh_helper_pushes_effective_keys() -> None:
    sm = SpeechmaticsClient({"apiKey": "stale"})
    aai = AssemblyAIClient({"apiKey": "stale"})
    muse = MuseClient({"apiKey": "stale"})
    cfg = AppConfig.model_validate(
        {
            "speechmaticsApiKey": SM_KEY,
            "assemblyaiApiKey": AAI_KEY,
            "museApiKey": MUSE_KEY,
        }
    )
    refresh_transcription_client_keys(
        [("speechmatics", sm), ("assemblyai", aai), ("muse", muse)],
        cfg,
        {},
    )
    assert sm.api_key == SM_KEY
    assert aai.api_key == AAI_KEY
    assert muse.api_key == MUSE_KEY

    # Env overrides stored during refresh.
    refresh_transcription_client_keys(
        [("speechmatics", sm)],
        cfg,
        {"SPEECHMATICS_API_KEY": "sm_env"},
    )
    assert sm.api_key == "sm_env"

    # Duplicate objects updated once; unknown kinds / Nones skipped.
    refresh_transcription_client_keys(
        [("speechmatics", sm), ("speechmatics", sm), ("nope", object()), ("muse", None)],
        cfg,
        {},
    )
    assert sm.api_key == SM_KEY


def test_assemblyai_speech_model_config_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OLIVE_CONFIG_DIR", str(tmp_path / "unrelated"))
    clear_app_config_cache()
    handle = create_db(":memory:")
    app = create_app({"db": handle.engine, "configDir": str(tmp_path / "settings")})
    client = TestClient(app)

    assert client.get("/api/config").json()["assemblyaiSpeechModel"] == "universal-3-5-pro"
    saved = client.post("/api/config", json={"assemblyaiSpeechModel": "assemblyai-stt-3-7"})
    assert saved.status_code == 200
    assert saved.json()["config"]["assemblyaiSpeechModel"] == "assemblyai-stt-3-7"
    clear_app_config_cache()
    assert client.get("/api/config").json()["assemblyaiSpeechModel"] == "assemblyai-stt-3-7"

    invalid = client.post("/api/config", json={"assemblyaiSpeechModel": "not-a-model"})
    assert invalid.status_code == 400
    assert client.get("/api/config").json()["assemblyaiSpeechModel"] == "assemblyai-stt-3-7"
    assert client.post("/api/config", json={"assemblyaiSpeechModel": "universal-3-5-pro"}).status_code == 200
    assert client.get("/api/config").json()["assemblyaiSpeechModel"] == "universal-3-5-pro"
    handle.engine.dispose()
    clear_app_config_cache()


def test_config_routes_round_trip_keys_masked(tmp_path, monkeypatch) -> None:
    for var in (
        "SPEECHMATICS_API_KEY",
        "ASSEMBLY_API_KEY",
        "ASSEMBLYAI_API_KEY",
        "MUSE_API_KEY",
        "MODEL_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    clear_app_config_cache()
    handle = create_db(":memory:")
    app = create_app({"db": handle.engine, "configDir": str(tmp_path)})
    client = TestClient(app)

    res = client.post(
        "/api/config",
        json={
            "speechmaticsApiKey": SM_KEY,
            "assemblyaiApiKey": AAI_KEY,
            "museApiKey": MUSE_KEY,
        },
    )
    assert res.status_code == 200
    body = res.json()["config"]
    assert body["speechmaticsApiKey"] == "••••••••7890"
    assert body["hasSpeechmaticsApiKey"] is True
    assert body["hasAssemblyaiApiKey"] is True
    assert body["hasMuseApiKey"] is True

    get_res = client.get("/api/config")
    assert get_res.status_code == 200
    assert get_res.json()["speechmaticsApiKey"] == "••••••••7890"

    # Re-saving the masked values must preserve the real stored keys.
    res2 = client.post(
        "/api/config",
        json={
            "speechmaticsApiKey": body["speechmaticsApiKey"],
            "assemblyaiApiKey": body["assemblyaiApiKey"],
            "museApiKey": body["museApiKey"],
        },
    )
    assert res2.status_code == 200
    assert res2.json()["config"]["speechmaticsApiKey"] == "••••••••7890"

    get_res2 = client.get("/api/config")
    assert get_res2.json()["hasAssemblyaiApiKey"] is True

    # Clearing a key sticks and flips the has* flag.
    res3 = client.post("/api/config", json={"museApiKey": ""})
    assert res3.status_code == 200
    assert res3.json()["config"]["museApiKey"] == ""
    assert res3.json()["config"]["hasMuseApiKey"] is False

    handle.engine.dispose()
    clear_app_config_cache()
