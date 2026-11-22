"""AssemblyAI batch transcription client.

The default path uses the `assemblyai` SDK (imported lazily, so the package is only
required when a real transcription is actually run). The SDK's `transcribe()` uploads,
submits and polls internally and is synchronous, so it runs in a worker thread.

Injection seam: pass `transcriber=` (or `transcriber_client=`) with a method
`transcribe(audio_input, config: dict) -> dict | object`, where `audio_input` is a path
str, URL str, or bytes, and `config` is the AssemblyAI transcription config as a dict.
It replaces the SDK adapter and is used for tests.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any

from .types import AssemblyAITranscript

SPEECH_MODELS = ["universal-3-5-pro"]
TRANSCRIPT_KEYS = ("id", "status", "text", "language_code", "audio_duration", "error")
WORD_KEYS = ("text", "start", "end", "confidence", "speaker", "channel")
UTTERANCE_KEYS = ("text", "start", "end", "confidence", "speaker")


class _SdkTranscriber:
    """Adapter over the real `assemblyai` SDK. Blocking; call from a worker thread."""

    def __init__(self, api_key: str) -> None:
        import assemblyai as aai  # lazy: only needed for real transcription

        self._aai = aai
        self._api_key = api_key
        self._transcriber = aai.Transcriber(api_key=api_key)

    def transcribe(self, audio_input: str | bytes, config: dict[str, Any]) -> AssemblyAITranscript:
        aai_config = self._aai.TranscriptionConfig(**config)
        transcript = self._transcriber.transcribe(audio_input, config=aai_config)
        return _to_transcript_dict(transcript)

    def submit(self, audio_input: str | bytes, config: dict[str, Any]) -> AssemblyAITranscript:
        """Upload + create a transcript job without polling for completion."""
        aai_config = self._aai.TranscriptionConfig(**config)
        transcript = self._transcriber.submit(audio_input, config=aai_config)
        return _to_transcript_dict(transcript)

    def get_transcript(self, job_id: str) -> AssemblyAITranscript:
        """Fetch the current transcript state (status, plus words when done)."""
        import httpx

        res = httpx.get(
            f"https://api.assemblyai.com/v2/transcript/{job_id}",
            headers={"authorization": self._api_key},
            timeout=60.0,
        )
        res.raise_for_status()
        data: dict[str, Any] = res.json()
        data["status"] = _enum_value(data.get("status"))
        return data  # type: ignore[return-value]


def _read(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _to_transcript_dict(result: Any) -> AssemblyAITranscript:
    """Return the raw AssemblyAI transcript as a plain dict."""
    if isinstance(result, dict):
        data: dict[str, Any] = dict(result)
    else:
        raw_json = getattr(result, "json_response", None)
        if isinstance(raw_json, dict) and raw_json:
            data = dict(raw_json)
        else:
            data = {key: _read(result, key) for key in TRANSCRIPT_KEYS}
            data["words"] = [
                {key: _read(w, key) for key in WORD_KEYS} for w in (_read(result, "words") or [])
            ]
            data["utterances"] = [
                {
                    **{key: _read(u, key) for key in UTTERANCE_KEYS},
                    "words": [{key: _read(w, key) for key in WORD_KEYS} for w in (_read(u, "words") or [])],
                }
                for u in (_read(result, "utterances") or [])
            ]
    data["status"] = _enum_value(data.get("status"))
    return data  # type: ignore[return-value]


def _audio_input(audio: Any) -> str | bytes:
    if isinstance(audio, dict) and "url" in audio:
        return str(audio["url"])
    if isinstance(audio, dict) and "path" in audio:
        return str(audio["path"])
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)
    raise ValueError("Unsupported audio input: expected bytes, {'path': ...} or {'url': ...}")


def _keyterms(raw: Any) -> list[str]:
    terms: list[str] = []
    for item in raw or []:
        content = item.get("content") if isinstance(item, dict) else item
        if content and str(content).strip():
            terms.append(str(content).strip())
    return terms


def _build_config(options: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "speech_models": [options.get("speechModel") or SPEECH_MODELS[0]],
        "speaker_labels": True,
        "punctuate": True,
        "format_text": True,
    }
    language = str(options.get("language") or "").strip()
    if language:
        config["language_code"] = language
    else:
        config["language_detection"] = True

    keyterms = _keyterms(
        options.get("keyterms") or options.get("additionalVocab") or options.get("additional_vocab")
    )
    if keyterms:
        config["keyterms_prompt"] = keyterms
    return config


class AssemblyAIClient:
    def __init__(self, options: dict[str, Any] | None = None, **kwargs: Any) -> None:
        opts = {**(options or {}), **kwargs}
        if "apiKey" in opts or "api_key" in opts:
            raw_key = opts.get("apiKey") if "apiKey" in opts else opts.get("api_key")
            self.api_key = str(raw_key if raw_key is not None else "")
        else:
            self.api_key = os.environ.get("ASSEMBLY_API_KEY") or os.environ.get("ASSEMBLYAI_API_KEY") or ""

        st_opt = opts.get("submitTimeoutMs")
        if st_opt is None:
            st_opt = opts.get("submit_timeout_ms")
        # Bound the SDK's internal polling so a stuck job cannot run forever.
        self.submit_timeout_ms: int | None = int(st_opt) if st_opt is not None else 1_800_000

        self._transcriber: Any = opts.get("transcriber") or opts.get("transcriber_client") or None
        self._staged_results: dict[str, AssemblyAITranscript] = {}

    def set_api_key(self, api_key: str | None) -> None:
        """Updates the key at runtime (e.g. after a Settings-UI save)."""
        self.api_key = str(api_key or "")

    setApiKey = set_api_key

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip())

    @property
    def isConfigured(self) -> bool:
        return self.is_configured

    @property
    def raw_client(self) -> Any:
        return self._transcriber

    @property
    def rawClient(self) -> Any:
        return self._transcriber

    async def _run_transcriber(
        self, transcriber: Any, audio_input: str | bytes, config: dict[str, Any]
    ) -> Any:
        result = await asyncio.to_thread(transcriber.transcribe, audio_input, config)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def submit_and_transcribe(self, options: dict[str, Any]) -> AssemblyAITranscript:
        if not self.is_configured:
            raise RuntimeError("AssemblyAI API key is not configured")

        audio_input = _audio_input(options.get("audio"))
        config = _build_config(options)
        transcriber = self._transcriber or _SdkTranscriber(self.api_key)

        timeout = self.submit_timeout_ms / 1000.0 if self.submit_timeout_ms is not None else None
        try:
            result = await asyncio.wait_for(
                self._run_transcriber(transcriber, audio_input, config), timeout=timeout
            )
        except TimeoutError as exc:
            raise RuntimeError(
                f"AssemblyAI transcription timed out after {self.submit_timeout_ms}ms"
            ) from exc

        data = _to_transcript_dict(result)
        if data.get("status") == "error":
            raise RuntimeError(str(data.get("error") or "AssemblyAI transcription failed"))
        return data

    submitAndTranscribe = submit_and_transcribe

    async def _call_staged(self, transcriber: Any, method: str, *args: Any) -> Any:
        fn = getattr(transcriber, method)
        result = await asyncio.to_thread(fn, *args)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def submit_job(self, options: dict[str, Any]) -> AssemblyAITranscript:
        """Upload + submit a job; returns {id, status} without waiting.

        Injected transcribers may implement ``submit(audio, config)`` for the
        staged flow; ones with only ``transcribe`` are run to completion and
        their result is stashed for :meth:`get_transcript`.
        """
        if not self.is_configured:
            raise RuntimeError("AssemblyAI API key is not configured")
        audio_input = _audio_input(options.get("audio"))
        config = _build_config(options)
        transcriber = self._transcriber or _SdkTranscriber(self.api_key)

        if callable(getattr(transcriber, "submit", None)):
            timeout = self.submit_timeout_ms / 1000.0 if self.submit_timeout_ms is not None else None
            try:
                result = await asyncio.wait_for(
                    self._call_staged(transcriber, "submit", audio_input, config),
                    timeout=timeout,
                )
            except TimeoutError as exc:
                raise RuntimeError(f"AssemblyAI upload timed out after {self.submit_timeout_ms}ms") from exc
            data = _to_transcript_dict(result)
            if data.get("status") == "error":
                raise RuntimeError(str(data.get("error") or "AssemblyAI transcription failed"))
            return data

        data = await self.submit_and_transcribe(options)
        job_id = str(data.get("id") or "")
        if job_id:
            self._staged_results[job_id] = data
        return {**data, "status": data.get("status") or "completed"}  # type: ignore[return-value]

    submitJob = submit_job

    async def get_transcript(self, job_id: str) -> AssemblyAITranscript:
        """Fetch current transcript state for a submitted job."""
        if job_id in self._staged_results:
            return self._staged_results[job_id]
        transcriber = self._transcriber or _SdkTranscriber(self.api_key)
        if not callable(getattr(transcriber, "get_transcript", None)):
            raise RuntimeError("Transcriber does not support staged polling")
        result = await self._call_staged(transcriber, "get_transcript", job_id)
        data = _to_transcript_dict(result) if not isinstance(result, dict) else result
        data["status"] = _enum_value(data.get("status"))
        return data  # type: ignore[return-value]

    getTranscript = get_transcript
