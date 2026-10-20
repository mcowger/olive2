"""Paths and application configuration (port of paths.ts, shared/config.ts, server/config.ts)."""

from __future__ import annotations

import json
import os
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_LOCAL_ASR_MODEL = "onnx-community/cohere-transcribe-03-2026-ONNX"
MASK_PREFIX = "••••••••"


@dataclass(frozen=True)
class OlivePaths:
    config_dir: str
    meetings_dir: str
    backups_dir: str
    models_dir: str
    database_path: str
    settings_path: str
    plaud_tokens_path: str

    @property
    def configDir(self) -> str:
        return self.config_dir

    @property
    def meetingsDir(self) -> str:
        return self.meetings_dir

    @property
    def backupsDir(self) -> str:
        return self.backups_dir

    @property
    def modelsDir(self) -> str:
        return self.models_dir

    @property
    def databasePath(self) -> str:
        return self.database_path

    @property
    def settingsPath(self) -> str:
        return self.settings_path

    @property
    def plaudTokensPath(self) -> str:
        return self.plaud_tokens_path


def _default_config_dir(platform: str, home_dir: str, env: Mapping[str, str]) -> str:
    if platform == "win32":
        appdata = env.get("APPDATA") or str(Path(home_dir) / "AppData" / "Roaming")
        return str(Path(appdata) / "olive")
    if platform == "darwin":
        return str(Path(home_dir) / "Library" / "Application Support" / "olive")
    xdg = env.get("XDG_CONFIG_HOME") or str(Path(home_dir) / ".config")
    return str(Path(xdg) / "olive")


def resolve_paths(
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
    home_dir: str | None = None,
) -> OlivePaths:
    effective_env = env if env is not None else os.environ
    effective_platform = platform if platform is not None else sys.platform
    effective_home = home_dir if home_dir is not None else str(Path.home())

    config_dir = str(
        Path(
            effective_env.get("OLIVE_CONFIG_DIR")
            or _default_config_dir(effective_platform, effective_home, effective_env)
        ).resolve()
    )
    meetings_dir = str(
        Path(effective_env.get("OLIVE_MEETINGS_DIR") or Path(config_dir) / "meetings").resolve()
    )
    backups_dir = str(Path(effective_env.get("OLIVE_BACKUPS_DIR") or Path(config_dir) / "backups").resolve())
    models_dir = str(Path(effective_env.get("OLIVE_MODELS_DIR") or Path(config_dir) / "models").resolve())
    plaud_tokens_path = str(
        Path(effective_env.get("PLAUD_TOKEN_PATH") or Path(config_dir) / "plaud-tokens.json").resolve()
    )

    return OlivePaths(
        config_dir=config_dir,
        meetings_dir=meetings_dir,
        backups_dir=backups_dir,
        models_dir=models_dir,
        database_path=str(Path(config_dir) / "olive.sqlite"),
        settings_path=str(Path(config_dir) / "settings.json"),
        plaud_tokens_path=plaud_tokens_path,
    )


class AppConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    pollIntervalMinutes: int = Field(default=15, gt=0)
    retranscribePlaudWhenUnnamed: bool = Field(default=False)
    alwaysRetranscribePlaud: bool = Field(default=False)
    transcriptionProvider: Literal["speechmatics", "assemblyai", "muse", "local"] = Field(default="local")
    localAsrModel: str = Field(default=DEFAULT_LOCAL_ASR_MODEL)
    localClusteringThreshold: float = Field(default=0.50, ge=0.1, le=0.99)
    localSimilarityThreshold: float = Field(default=0.45, ge=0.1, le=0.99)
    primaryArtifactPreference: list[str] = Field(
        default_factory=lambda: ["speechmatics-with-named-speakers", "local", "plaud"]
    )
    llmProvider: Literal["openai", "anthropic"] = Field(default="openai")
    llmPrompt: str = Field(default="")
    hfToken: str = Field(default="")
    speechmaticsApiKey: str = Field(default="")
    assemblyaiApiKey: str = Field(default="")
    museApiKey: str = Field(default="")
    insightsEnabled: bool = Field(default=False)

    @field_validator("localAsrModel", mode="before")
    @classmethod
    def _coerce_qwen_model(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return DEFAULT_LOCAL_ASR_MODEL
        if "qwen" in value.lower():
            return DEFAULT_LOCAL_ASR_MODEL
        return value

    @field_validator("primaryArtifactPreference")
    @classmethod
    def _validate_preferences(cls, value: list[str]) -> list[str]:
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("primaryArtifactPreference items must be non-empty strings")
        return value


DEFAULT_CONFIG = AppConfig()


@dataclass(frozen=True)
class RuntimeConfig:
    paths: OlivePaths
    bind_host: str
    bind_port: int

    @property
    def bindHost(self) -> str:
        return self.bind_host

    @property
    def bindPort(self) -> int:
        return self.bind_port


def _parse_port(value: str | None) -> int:
    if not value:
        return 4471
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError("OLIVE_BIND_PORT must be an integer between 1 and 65535") from exc
    if port < 1 or port > 65535:
        raise ValueError("OLIVE_BIND_PORT must be an integer between 1 and 65535")
    return port


def load_runtime_config(env: Mapping[str, str] | None = None) -> RuntimeConfig:
    effective_env = env if env is not None else os.environ
    port_val = (
        effective_env.get("OLIVE_BIND_PORT") or effective_env.get("PORT") or effective_env.get("PASEO_PORT")
    )
    return RuntimeConfig(
        paths=resolve_paths(effective_env),
        bind_host=effective_env.get("OLIVE_BIND_HOST") or "127.0.0.1",
        bind_port=_parse_port(port_val),
    )


_cached_app_config: tuple[str, AppConfig] | None = None


def clear_app_config_cache() -> None:
    global _cached_app_config
    _cached_app_config = None


def save_app_config(config: Any, paths: OlivePaths | None = None) -> AppConfig:
    global _cached_app_config
    resolved = paths or resolve_paths()
    if isinstance(config, AppConfig):
        validated = AppConfig.model_validate(config.model_dump())
    else:
        validated = AppConfig.model_validate(config)

    Path(resolved.config_dir).mkdir(parents=True, exist_ok=True)
    Path(resolved.settings_path).write_text(
        json.dumps(validated.model_dump(), indent=2) + "\n",
        encoding="utf-8",
    )
    _cached_app_config = (resolved.settings_path, validated)
    return validated


def load_app_config(paths: OlivePaths | None = None) -> AppConfig:
    global _cached_app_config
    resolved = paths or resolve_paths()
    Path(resolved.config_dir).mkdir(parents=True, exist_ok=True)

    if _cached_app_config and _cached_app_config[0] == resolved.settings_path:
        return _cached_app_config[1]

    settings_file = Path(resolved.settings_path)
    if not settings_file.exists():
        return save_app_config(DEFAULT_CONFIG, resolved)

    try:
        raw = json.loads(settings_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {resolved.settings_path}") from exc

    try:
        validated = AppConfig.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid settings in {resolved.settings_path}") from exc

    _cached_app_config = (resolved.settings_path, validated)
    return validated


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return MASK_PREFIX
    return f"{MASK_PREFIX}{value[-4:]}"


def is_masked_secret(value: str | None) -> bool:
    return bool(value and value.startswith(MASK_PREFIX))


def mask_app_config(config: AppConfig) -> dict[str, Any]:
    data = config.model_dump()
    data["hfToken"] = mask_secret(config.hfToken)
    data["hasHfToken"] = bool(get_effective_hf_token(config))
    data["speechmaticsApiKey"] = mask_secret(config.speechmaticsApiKey)
    data["hasSpeechmaticsApiKey"] = bool(get_effective_speechmatics_key(config))
    data["assemblyaiApiKey"] = mask_secret(config.assemblyaiApiKey)
    data["hasAssemblyaiApiKey"] = bool(get_effective_assemblyai_key(config))
    data["museApiKey"] = mask_secret(config.museApiKey)
    data["hasMuseApiKey"] = bool(get_effective_muse_key(config))
    return data


def _read_hf_token_from_mise_local() -> str | None:
    search_dirs: list[Path] = []
    for base in (Path.cwd(), Path(__file__).resolve().parents[3]):
        cur = base.resolve()
        for d in (cur, *cur.parents):
            if d not in search_dirs:
                search_dirs.append(d)
            if (d / ".git").exists():
                break

    for directory in search_dirs:
        candidate = directory / "mise.local.toml"
        if candidate.is_file():
            try:
                data = tomllib.loads(candidate.read_text(encoding="utf-8"))
                env_sec = data.get("env")
                if isinstance(env_sec, dict):
                    tok = str(env_sec.get("HF_TOKEN") or env_sec.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
                    if tok:
                        return tok
            except Exception:
                pass
    return None


def _bootstrap_hf_token_env() -> None:
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        tok = _read_hf_token_from_mise_local()
        if tok:
            os.environ["HF_TOKEN"] = tok


_bootstrap_hf_token_env()


def get_effective_hf_token(
    config: AppConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    effective_env = env if env is not None else os.environ
    env_token = (effective_env.get("HF_TOKEN") or effective_env.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if env_token:
        return env_token
    if env is None:
        mise_token = _read_hf_token_from_mise_local()
        if mise_token:
            os.environ.setdefault("HF_TOKEN", mise_token)
            return mise_token
    cfg = config or load_app_config()
    if cfg.hfToken and cfg.hfToken.strip() and not is_masked_secret(cfg.hfToken):
        return cfg.hfToken.strip()
    return None


def _effective_stored_secret(value: str | None) -> str | None:
    if value and value.strip() and not is_masked_secret(value):
        return value.strip()
    return None


def get_effective_speechmatics_key(
    config: AppConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Env ``SPEECHMATICS_API_KEY`` wins; otherwise the Settings-UI stored key."""
    effective_env = env if env is not None else os.environ
    env_token = (effective_env.get("SPEECHMATICS_API_KEY") or "").strip()
    if env_token:
        return env_token
    cfg = config or load_app_config()
    return _effective_stored_secret(cfg.speechmaticsApiKey)


def get_effective_assemblyai_key(
    config: AppConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Env ``ASSEMBLY_API_KEY``/``ASSEMBLYAI_API_KEY`` wins; otherwise the stored key."""
    effective_env = env if env is not None else os.environ
    env_token = (
        effective_env.get("ASSEMBLY_API_KEY") or effective_env.get("ASSEMBLYAI_API_KEY") or ""
    ).strip()
    if env_token:
        return env_token
    cfg = config or load_app_config()
    return _effective_stored_secret(cfg.assemblyaiApiKey)


def get_effective_muse_key(
    config: AppConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Env ``MUSE_API_KEY``/``MODEL_API_KEY`` wins; otherwise the stored key."""
    effective_env = env if env is not None else os.environ
    env_token = (effective_env.get("MUSE_API_KEY") or effective_env.get("MODEL_API_KEY") or "").strip()
    if env_token:
        return env_token
    cfg = config or load_app_config()
    return _effective_stored_secret(cfg.museApiKey)
