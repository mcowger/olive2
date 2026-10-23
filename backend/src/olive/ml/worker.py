"""Cancellation primitives shared by the local transcription pipeline.

Orchestration lives in ``olive.workflows.transcription_workflow`` (flat DBOS
stage hand-off); this module intentionally holds only the abort signal/controller
types that the request path and tests pass around.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class AbortSignal:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._listeners: list[Callable[[], Any]] = []

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def add_event_listener(self, _event_name: str, callback: Callable[[], Any]) -> None:
        if self._event.is_set():
            callback()
        else:
            self._listeners.append(callback)

    addEventListener = add_event_listener

    def _trigger(self) -> None:
        if self._event.is_set():
            return
        self._event.set()
        for cb in list(self._listeners):
            try:
                cb()
            except Exception:
                pass


class AbortController:
    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self) -> None:
        self.signal._trigger()


def _is_signal_aborted(signal: Any) -> bool:
    if signal is None:
        return False
    if hasattr(signal, "aborted"):
        return bool(signal.aborted)
    if hasattr(signal, "is_set") and callable(signal.is_set):
        return bool(signal.is_set())
    return False


__all__ = ["AbortController", "AbortSignal"]
