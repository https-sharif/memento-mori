"""Scene deduplication and orientation-cue generation.

The perception loop receives the vision service's 1 Hz stream, so the dedup check
is what stands between a demo and hundreds of LLM calls per minute. These tests
run entirely against ``mock_llm`` — no network, no API key.

``asyncio.run`` is used directly rather than pytest-asyncio to keep the CI
dependency list to pytest + httpx.
"""

import asyncio

from models import MemoryCard
from patient_memory import PatientMemoryStore
from perception import PerceptionStateTracker

SARAH = {"recognized": True, "name": "Sarah", "relationship": "Daughter", "note": "", "face_detected": True}
CUP = [{"label": "cup", "confidence": 0.9}]


def make_tracker(profile_path):
    return PerceptionStateTracker(memory=PatientMemoryStore(str(profile_path)))


def test_first_scene_always_triggers(profile_path):
    assert make_tracker(profile_path)._should_trigger_llm(SARAH, CUP) is True


def test_identical_scene_is_deduplicated(profile_path):
    tracker = make_tracker(profile_path)

    assert tracker._should_trigger_llm(SARAH, CUP) is True
    assert tracker._should_trigger_llm(SARAH, CUP) is False
    assert tracker._should_trigger_llm(dict(SARAH), list(CUP)) is False


def test_object_order_does_not_count_as_a_change(profile_path):
    """YOLO's output order is not stable; reordering the same objects must not
    burn an LLM call."""
    tracker = make_tracker(profile_path)
    objects = [{"label": "cup"}, {"label": "phone"}]

    assert tracker._should_trigger_llm(SARAH, objects) is True
    assert tracker._should_trigger_llm(SARAH, list(reversed(objects))) is False


def test_confidence_drift_does_not_count_as_a_change(profile_path):
    """Only labels matter — per-frame confidence jitter must not retrigger."""
    tracker = make_tracker(profile_path)

    assert tracker._should_trigger_llm(SARAH, [{"label": "cup", "confidence": 0.91}]) is True
    assert tracker._should_trigger_llm(SARAH, [{"label": "cup", "confidence": 0.62}]) is False


def test_new_person_triggers(profile_path):
    tracker = make_tracker(profile_path)
    tracker._should_trigger_llm(SARAH, CUP)

    tom = {**SARAH, "name": "Tom", "relationship": "Son"}
    assert tracker._should_trigger_llm(tom, CUP) is True


def test_new_object_triggers(profile_path):
    tracker = make_tracker(profile_path)
    tracker._should_trigger_llm(SARAH, CUP)

    assert tracker._should_trigger_llm(SARAH, CUP + [{"label": "phone", "confidence": 0.7}]) is True


def test_person_leaving_triggers(profile_path):
    tracker = make_tracker(profile_path)
    tracker._should_trigger_llm(SARAH, CUP)

    assert tracker._should_trigger_llm({"recognized": False, "face_detected": False}, CUP) is True


def test_recognized_person_without_a_name_does_not_raise(profile_path):
    """vision_service can report ``recognized`` with a null name; indexing that
    key directly used to raise KeyError and kill the perception loop."""
    tracker = make_tracker(profile_path)

    assert tracker._should_trigger_llm({"recognized": True}, CUP) is True
    assert tracker._should_trigger_llm({"recognized": True, "name": None}, CUP) is False


def test_object_without_a_label_does_not_raise(profile_path):
    make_tracker(profile_path)._should_trigger_llm(SARAH, [{"confidence": 0.5}])


def test_generate_orientation_cue_returns_a_memory_card(profile_path):
    tracker = make_tracker(profile_path)

    card = asyncio.run(tracker.generate_orientation_cue(SARAH, CUP))

    assert isinstance(card, MemoryCard)
    assert "Sarah" in card.card_title
    assert card.voice_guidance
    assert tracker.last_output is card


def test_generate_orientation_cue_handles_nobody_present(profile_path):
    tracker = make_tracker(profile_path)

    card = asyncio.run(tracker.generate_orientation_cue({"recognized": False}, []))

    assert isinstance(card, MemoryCard)
    assert card.card_body


def test_generate_orientation_cue_falls_back_when_the_llm_fails(profile_path, monkeypatch):
    """A Gemini outage must leave the patient with calm text, never a traceback
    or a blank card."""
    import perception

    class BoomModels:
        async def generate_content(self, **kwargs):
            raise RuntimeError("gemini is down")

    monkeypatch.setattr(perception.client, "aio", type("Aio", (), {"models": BoomModels()})())

    tracker = make_tracker(profile_path)
    card = asyncio.run(tracker.generate_orientation_cue(SARAH, CUP))

    assert card is perception._FALLBACK_CARD


def test_generate_orientation_cue_keeps_the_previous_card_on_failure(profile_path, monkeypatch):
    import perception

    tracker = make_tracker(profile_path)
    good = asyncio.run(tracker.generate_orientation_cue(SARAH, CUP))

    class BoomModels:
        async def generate_content(self, **kwargs):
            raise RuntimeError("gemini is down")

    monkeypatch.setattr(perception.client, "aio", type("Aio", (), {"models": BoomModels()})())

    assert asyncio.run(tracker.generate_orientation_cue(SARAH, CUP)) is good


def test_generate_orientation_cue_handles_unparseable_output(profile_path, monkeypatch):
    """``response.parsed`` is None on a safety block or malformed JSON."""
    import perception

    class NoneModels:
        async def generate_content(self, **kwargs):
            return type("Response", (), {"parsed": None})()

    monkeypatch.setattr(perception.client, "aio", type("Aio", (), {"models": NoneModels()})())

    card = asyncio.run(make_tracker(profile_path).generate_orientation_cue(SARAH, CUP))

    assert card is perception._FALLBACK_CARD
