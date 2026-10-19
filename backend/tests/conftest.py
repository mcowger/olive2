"""Pytest configuration for Olive backend test suite."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip `live`-marked tests unless explicitly opted in.

    Live tests hit real ML models (multi-GB downloads + CPU inference) or
    external APIs. Run them with ``OLIVE_RUN_LIVE_TESTS=1``.
    """
    if os.environ.get("OLIVE_RUN_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(
        reason="live test (downloads real ML models); set OLIVE_RUN_LIVE_TESTS=1 to run"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _reset_shared_transcription_pipeline():
    """Isolate the process-wide shared transcription pipeline between tests."""
    from olive.workflows.transcription_workflow import reset_shared_pipeline

    reset_shared_pipeline()
    yield
    reset_shared_pipeline()
