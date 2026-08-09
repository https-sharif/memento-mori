from __future__ import annotations

from threading import Lock
from typing import Any

_lock = Lock()
_latest_scene: dict[str, Any] = {}
_latest_patient_card: dict[str, Any] = {}
_latest_caretaker_advice: dict[str, Any] = {}


def set_latest_scene(scene: dict[str, Any]) -> None:
    with _lock:
        _latest_scene.clear()
        _latest_scene.update(scene)


def get_latest_scene() -> dict[str, Any]:
    with _lock:
        return dict(_latest_scene)


def set_latest_patient_card(card: dict[str, Any]) -> None:
    with _lock:
        _latest_patient_card.clear()
        _latest_patient_card.update(card)


def get_latest_patient_card() -> dict[str, Any]:
    with _lock:
        return dict(_latest_patient_card)


def set_latest_caretaker_advice(advice: dict[str, Any]) -> None:
    with _lock:
        _latest_caretaker_advice.clear()
        _latest_caretaker_advice.update(advice)


def get_latest_caretaker_advice() -> dict[str, Any]:
    with _lock:
        return dict(_latest_caretaker_advice)
