from __future__ import annotations

from threading import Lock
from typing import Any


_lock = Lock()
_latest_output: dict[str, Any] = {}
_latest_scene: dict[str, Any] = {}


def set_latest_output(output: dict[str, Any]) -> None:
    with _lock:
        _latest_output.clear()
        _latest_output.update(output)


def get_latest_output() -> dict[str, Any]:
    with _lock:
        return dict(_latest_output)


def set_latest_scene(scene: dict[str, Any]) -> None:
    with _lock:
        _latest_scene.clear()
        _latest_scene.update(scene)


def get_latest_scene() -> dict[str, Any]:
    with _lock:
        return dict(_latest_scene)