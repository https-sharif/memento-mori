"""Endpoint contract tests.

Everything here runs against ``mock_llm`` (``MOCK_LLM=true`` is set in
``conftest.py``), so the suite needs no API key and makes no network calls.

``TestClient`` is deliberately *not* used as a context manager: entering it would
run the app's lifespan, which starts the perception task and tries to dial the
vision service websocket. The routes under test don't need it.
"""

import pytest
from fastapi.testclient import TestClient

import main
import store

client = TestClient(main.app)

SCENE = {
    "person": {"recognized": True, "name": "Sarah", "relationship": "Daughter", "note": "", "face_detected": True},
    "objects": [{"label": "cup", "confidence": 0.9}],
    "timestamp": 1751190195,
}


@pytest.fixture(autouse=True)
def clear_ask_cache():
    main._ask_cache.clear()
    yield
    main._ask_cache.clear()


class CountingModels:
    """Wraps the mock client so a test can prove a response came from cache."""

    def __init__(self):
        self.calls = 0
        self._inner = main.client.aio.models

    async def generate_content(self, **kwargs):
        self.calls += 1
        return await self._inner.generate_content(**kwargs)


def install_counter(monkeypatch):
    counter = CountingModels()
    monkeypatch.setattr(main.client, "aio", type("Aio", (), {"models": counter})())
    return counter


def install_failure(monkeypatch, exc=RuntimeError("gemini is down")):
    class BoomModels:
        async def generate_content(self, **kwargs):
            raise exc

    monkeypatch.setattr(main.client, "aio", type("Aio", (), {"models": BoomModels()})())


def install_unparseable(monkeypatch):
    class NoneModels:
        async def generate_content(self, **kwargs):
            return type("Response", (), {"parsed": None})()

    monkeypatch.setattr(main.client, "aio", type("Aio", (), {"models": NoneModels()})())


def test_health():
    assert client.get("/health").json() == {"ok": True}


# ── GET /latest ──────────────────────────────────────────────────────────────

def test_latest_is_empty_before_any_scene():
    assert client.get("/latest").json() == {"status": "empty"}


def test_latest_returns_the_patient_card_flat():
    """The frontend reads card_title/card_body/voice_guidance off the top level."""
    store.set_latest_patient_card({"card_title": "Sarah", "card_body": "b", "voice_guidance": "v"})

    body = client.get("/latest").json()

    assert body["card_title"] == "Sarah"
    assert body["voice_guidance"] == "v"


def test_latest_caretaker_slot_is_separate():
    store.set_latest_patient_card({"card_title": "patient"})
    store.set_latest_caretaker_advice({"card_title": "caretaker"})

    assert client.get("/latest").json()["card_title"] == "patient"
    assert client.get("/latest", params={"type": "caretaker"}).json()["card_title"] == "caretaker"


def test_latest_caretaker_slot_can_be_empty_independently():
    store.set_latest_patient_card({"card_title": "patient"})

    assert client.get("/latest", params={"type": "caretaker"}).json() == {"status": "empty"}


def test_latest_unknown_type_falls_back_to_the_patient_card():
    store.set_latest_patient_card({"card_title": "patient"})

    assert client.get("/latest", params={"type": "nonsense"}).json()["card_title"] == "patient"


# ── POST /ask ────────────────────────────────────────────────────────────────

def test_ask_returns_a_memory_card():
    body = client.post("/ask", json={"question": "Is Sarah visiting today?"}).json()

    assert set(body) == {"card_title", "card_body", "voice_guidance"}
    assert all(isinstance(v, str) and v for v in body.values())


def test_ask_rejects_an_empty_question():
    assert client.post("/ask", json={"question": ""}).status_code == 400


def test_ask_rejects_a_whitespace_question():
    assert client.post("/ask", json={"question": "   "}).status_code == 400


def test_ask_rejects_a_malformed_body():
    assert client.post("/ask", json={}).status_code == 422


def test_ask_does_not_clobber_the_patient_card():
    """Regression: a caretaker question used to overwrite the patient-facing
    card, so asking anything blanked the screen the patient was looking at."""
    store.set_latest_patient_card({"card_title": "Sarah (Daughter)"})

    client.post("/ask", json={"question": "What should I do?"})

    assert store.get_latest_patient_card()["card_title"] == "Sarah (Daughter)"
    assert store.get_latest_caretaker_advice()


def test_ask_caches_repeated_questions(monkeypatch):
    counter = install_counter(monkeypatch)
    store.set_latest_scene(SCENE)

    first = client.post("/ask", json={"question": "What should I do?"}).json()
    second = client.post("/ask", json={"question": "  WHAT should I DO?  "}).json()

    assert counter.calls == 1
    assert first == second


def test_ask_cache_is_keyed_by_scene(monkeypatch):
    counter = install_counter(monkeypatch)
    store.set_latest_scene(SCENE)
    client.post("/ask", json={"question": "Who is this?"})

    store.set_latest_scene({**SCENE, "person": {**SCENE["person"], "name": "Tom"}})
    client.post("/ask", json={"question": "Who is this?"})

    assert counter.calls == 2


def test_ask_cache_ignores_the_timestamp(monkeypatch):
    """The scene ticks at 1 Hz; keying on the timestamp would defeat the cache."""
    counter = install_counter(monkeypatch)
    store.set_latest_scene(SCENE)
    client.post("/ask", json={"question": "Who is this?"})

    store.set_latest_scene({**SCENE, "timestamp": SCENE["timestamp"] + 30})
    client.post("/ask", json={"question": "Who is this?"})

    assert counter.calls == 1


def test_ask_returns_503_when_the_llm_fails(monkeypatch):
    install_failure(monkeypatch)

    res = client.post("/ask", json={"question": "What should I do?"})

    assert res.status_code == 503
    assert "unavailable" in res.json()["detail"].lower()


def test_ask_returns_503_when_the_llm_output_is_unparseable(monkeypatch):
    install_unparseable(monkeypatch)

    assert client.post("/ask", json={"question": "What should I do?"}).status_code == 503


def test_ask_does_not_cache_failures(monkeypatch):
    install_failure(monkeypatch)
    client.post("/ask", json={"question": "What should I do?"})

    monkeypatch.undo()
    counter = install_counter(monkeypatch)

    assert client.post("/ask", json={"question": "What should I do?"}).status_code == 200
    assert counter.calls == 1


def test_ask_works_with_no_scene_yet():
    """The backend may be asked a question before the camera has produced
    anything."""
    assert client.post("/ask", json={"question": "How do I keep him calm?"}).status_code == 200


# ── POST /api/caregiver/analyze ──────────────────────────────────────────────

def test_analyze_returns_the_full_schema():
    res = client.post(
        "/api/caregiver/analyze",
        json={"message": "Arthur got agitated and started pacing before dinner."},
    )

    assert res.status_code == 200
    body = res.json()
    assert set(body) == {
        "category",
        "observed_triggers",
        "clinical_rationale",
        "actionable_intervention",
        "is_crisis",
    }
    assert isinstance(body["observed_triggers"], list)
    assert isinstance(body["is_crisis"], bool)


def test_analyze_rejects_an_empty_message():
    assert client.post("/api/caregiver/analyze", json={"message": "  "}).status_code == 400


def test_analyze_rejects_a_malformed_body():
    assert client.post("/api/caregiver/analyze", json={}).status_code == 422


def test_analyze_flags_a_crisis():
    res = client.post("/api/caregiver/analyze", json={"message": "He fell and is bleeding."})

    assert res.json()["is_crisis"] is True


def test_analyze_does_not_flag_a_routine_report():
    res = client.post("/api/caregiver/analyze", json={"message": "He seemed a little confused."})

    assert res.json()["is_crisis"] is False


def test_analyze_returns_503_when_the_llm_fails(monkeypatch):
    install_failure(monkeypatch)

    assert client.post("/api/caregiver/analyze", json={"message": "He is pacing."}).status_code == 503


def test_analyze_returns_503_when_the_llm_output_is_unparseable(monkeypatch):
    install_unparseable(monkeypatch)

    assert client.post("/api/caregiver/analyze", json={"message": "He is pacing."}).status_code == 503


# ── Internals ────────────────────────────────────────────────────────────────

def test_scene_signature_survives_junk_input():
    """The signature feeds the cache key and runs on whatever the vision service
    sent, so it must never raise."""
    for junk in [None, [], "scene", {}, {"person": "nope", "objects": "nope"}, {"objects": [None, "x", {}]}]:
        assert isinstance(main._scene_signature(junk), str)


def test_scene_signature_is_stable_across_equal_scenes():
    assert main._scene_signature(dict(SCENE)) == main._scene_signature(dict(SCENE))


def test_ask_cache_evicts_the_oldest_entry(monkeypatch):
    """Unbounded, this cache grew with every (scene x question) pair for the life
    of the process."""
    monkeypatch.setattr(main, "_ASK_CACHE_MAX", 2)

    main._cache_put("a", {"n": 1})
    main._cache_put("b", {"n": 2})
    main._cache_put("c", {"n": 3})

    assert main._cache_get("a") is None
    assert main._cache_get("c") == {"n": 3}


def test_ask_cache_eviction_is_least_recently_used(monkeypatch):
    monkeypatch.setattr(main, "_ASK_CACHE_MAX", 2)

    main._cache_put("a", {"n": 1})
    main._cache_put("b", {"n": 2})
    main._cache_get("a")
    main._cache_put("c", {"n": 3})

    assert main._cache_get("a") == {"n": 1}
    assert main._cache_get("b") is None
