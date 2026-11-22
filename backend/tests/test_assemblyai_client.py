"""AssemblyAI client tests using fake transcribers (no network, no SDK)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from olive.providers.assemblyai import AssemblyAIClient


class _FakeTranscriber:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def transcribe(self, audio_input: Any, config: dict[str, Any]) -> Any:
        self.calls.append((audio_input, config))
        return self.result


def _client(result: Any = None, api_key: str = "test-key") -> tuple[AssemblyAIClient, _FakeTranscriber]:
    fake = _FakeTranscriber(result if result is not None else {"id": "t1", "status": "completed", "text": "hi"})
    return AssemblyAIClient({"apiKey": api_key, "transcriber": fake}), fake


async def test_success_returns_raw_transcript_dict() -> None:
    payload = {"id": "t1", "status": "completed", "text": "Hello there.", "words": []}
    client, _ = _client(payload)
    out = await client.submit_and_transcribe({"audio": b"\x01\x02"})
    assert out == payload


async def test_success_accepts_attribute_objects() -> None:
    fake_result = SimpleNamespace(
        id="t2",
        status="completed",
        text="Hi",
        error=None,
        json_response=None,
        words=[SimpleNamespace(text="Hi", start=0, end=400, confidence=0.9, speaker="A", channel=None)],
        utterances=[
            SimpleNamespace(
                text="Hi",
                start=0,
                end=400,
                confidence=0.9,
                speaker="A",
                words=[SimpleNamespace(text="Hi", start=0, end=400, confidence=0.9, speaker="A", channel=None)],
            )
        ],
    )
    client, _ = _client(fake_result)
    out = await client.submit_and_transcribe({"audio": b"x"})
    assert out["id"] == "t2"
    assert out["status"] == "completed"
    assert out["words"][0]["text"] == "Hi"
    assert out["utterances"][0]["words"][0]["speaker"] == "A"


async def test_attribute_object_prefers_json_response() -> None:
    raw = {"id": "t3", "status": "completed", "text": "from json"}
    client, _ = _client(SimpleNamespace(json_response=raw, status="completed", error=None))
    out = await client.submit_and_transcribe({"audio": b"x"})
    assert out == raw


async def test_error_status_raises_with_message() -> None:
    client, _ = _client({"id": "t4", "status": "error", "error": "audio too short"})
    with pytest.raises(RuntimeError, match="audio too short"):
        await client.submit_and_transcribe({"audio": b"x"})


async def test_key_not_configured_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASSEMBLY_API_KEY", raising=False)
    fake = _FakeTranscriber({"status": "completed"})
    client = AssemblyAIClient({"transcriber": fake})
    assert client.is_configured is False
    with pytest.raises(RuntimeError, match="AssemblyAI API key is not configured"):
        await client.submit_and_transcribe({"audio": b"x"})
    assert fake.calls == []


async def test_env_key_is_used_when_no_option_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSEMBLY_API_KEY", "env-key")
    client = AssemblyAIClient({"transcriber": _FakeTranscriber({"status": "completed"})})
    assert client.is_configured is True
    assert client.isConfigured is True


async def test_config_includes_required_flags_and_language_code() -> None:
    client, fake = _client()
    await client.submit_and_transcribe({"audio": b"x", "language": "de"})
    audio_input, config = fake.calls[0]
    assert audio_input == b"x"
    assert config["speech_models"] == ["universal-3-5-pro"]
    assert config["speaker_labels"] is True
    assert config["punctuate"] is True
    assert config["format_text"] is True
    assert config["language_code"] == "de"
    assert "language_detection" not in config
    assert "keyterms_prompt" not in config


async def test_config_uses_language_detection_when_no_language() -> None:
    client, fake = _client()
    await client.submit_and_transcribe({"audio": b"x"})
    _, config = fake.calls[0]
    assert config["language_detection"] is True
    assert "language_code" not in config


async def test_selected_model_reaches_transcribe_fallback() -> None:
    client, fake = _client()
    await client.submit_job({"audio": b"x", "speechModel": "assemblyai-stt-3-7"})
    assert fake.calls[0][1]["speech_models"] == ["assemblyai-stt-3-7"]


def test_sdk_accepts_stt_37_model_string() -> None:
    from assemblyai import TranscriptionConfig

    config = TranscriptionConfig(speech_models=["assemblyai-stt-3-7"])
    assert config.speech_models == ["assemblyai-stt-3-7"]


async def test_keyterms_accept_strings_and_content_dicts() -> None:
    client, fake = _client()
    await client.submit_and_transcribe(
        {"audio": b"x", "additionalVocab": ["Olive", {"content": "Kubernetes"}]}
    )
    _, config = fake.calls[0]
    assert config["keyterms_prompt"] == ["Olive", "Kubernetes"]


async def test_audio_path_and_url_are_passed_through() -> None:
    client, fake = _client()
    await client.submit_and_transcribe({"audio": {"path": "/tmp/a.m4a"}})
    await client.submit_and_transcribe({"audio": {"url": "https://example.com/a.mp3"}})
    assert fake.calls[0][0] == "/tmp/a.m4a"
    assert fake.calls[1][0] == "https://example.com/a.mp3"


async def test_missing_audio_raises() -> None:
    client, _ = _client()
    with pytest.raises(ValueError):
        await client.submit_and_transcribe({})


async def test_camel_case_alias_works() -> None:
    client, _ = _client({"id": "t5", "status": "completed"})
    out = await client.submitAndTranscribe({"audio": b"x"})
    assert out["id"] == "t5"


class _StagedFakeTranscriber:
    def __init__(self) -> None:
        self.states = [
            {"id": "job-9", "status": "processing"},
            {"id": "job-9", "status": "completed", "text": "done", "words": []},
        ]

    def submit(self, audio_input: Any, config: dict[str, Any]) -> dict[str, Any]:
        return {"id": "job-9", "status": "queued"}

    def get_transcript(self, job_id: str) -> dict[str, Any]:
        assert job_id == "job-9"
        return self.states.pop(0)


async def test_staged_submit_and_poll() -> None:
    client = AssemblyAIClient({"apiKey": "test-key", "transcriber": _StagedFakeTranscriber()})
    submitted = await client.submit_job({"audio": b"x"})
    assert submitted == {"id": "job-9", "status": "queued"}
    assert (await client.get_transcript("job-9"))["status"] == "processing"
    assert (await client.get_transcript("job-9"))["status"] == "completed"


async def test_staged_submit_falls_back_to_transcribe_only_fake() -> None:
    payload = {"id": "t-legacy", "status": "completed", "text": "hi", "words": []}
    client, _ = _client(payload)
    submitted = await client.submit_job({"audio": b"x"})
    assert submitted["id"] == "t-legacy"
    assert (await client.get_transcript("t-legacy"))["text"] == "hi"
