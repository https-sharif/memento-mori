from __future__ import annotations

import re
from typing import Any

from models import BehavioralAnalysis, MemoryCard


class MockClient:
    """Drop-in stand-in for google.genai.Client. Used when MOCK_LLM=true so the
    backend can run without a Gemini API key or quota. Matches the subset of the
    real client's interface that this project calls: both the sync
    client.models.generate_content(...) and the async
    client.aio.models.generate_content(...), each returning an object with a
    .parsed attribute."""

    def __init__(self):
        self.models = _MockModels()
        self.aio = _MockAio()


class _MockModels:
    def generate_content(self, *, model: str, contents: str, config: Any = None) -> _MockResponse:
        schema = getattr(config, "response_schema", None)
        if schema is MemoryCard:
            parsed = _mock_memory_card(contents)
        elif schema is BehavioralAnalysis:
            parsed = _mock_behavioral_analysis(contents)
        else:
            parsed = None
        return _MockResponse(parsed)


class _MockAio:
    """Mirrors client.aio — the async surface of the real genai client."""

    def __init__(self):
        self.models = _MockAioModels()


class _MockAioModels:
    def __init__(self):
        self._sync = _MockModels()

    async def generate_content(self, *, model: str, contents: str, config: Any = None) -> _MockResponse:
        return self._sync.generate_content(model=model, contents=contents, config=config)


class _MockResponse:
    def __init__(self, parsed: Any | None):
        self.parsed = parsed


def _mock_memory_card(contents: str) -> MemoryCard:
    if "Caretaker question:" in contents:
        match = re.search(r"Caretaker question:\s*(.+)", contents)
        question = match.group(1).strip() if match else "your question"
        return MemoryCard(
            card_title="Caretaker Tip (mock)",
            card_body=f'Regarding "{question}": stay calm, use short reassuring sentences, and check the patient profile for specifics.',
            voice_guidance="Everything is okay. Take a slow breath, and offer calm, simple reassurance.",
        )

    match = re.search(r"Person detected:\s*([^(.\n]+)(?:\(([^)]*)\))?", contents)
    name = match.group(1).strip() if match else ""
    if match and name.lower() not in ("unrecognized face or no familiar person present", ""):
        relationship = (match.group(2) or "").strip()
        title = f"{name} ({relationship})" if relationship else name
        return MemoryCard(
            card_title=title,
            card_body=f"{name} is here with you. It is safe and familiar.",
            voice_guidance=f"Hello, it's {name}. You are safe, and they are happy to see you.",
        )

    return MemoryCard(
        card_title="No One Recognized (mock)",
        card_body="No familiar face is in view right now.",
        voice_guidance="You're safe. Take your time looking around.",
    )


_TRIGGER_KEYWORDS = {
    "Sundowning": ["evening", "dusk", "afternoon", "sunset", "4pm", "5pm", "before dinner"],
    "Wandering": ["wander", "walk out", "left the house", "door", "outside"],
    "Aggression": ["hit", "yell", "scream", "aggress", "angry", "throw"],
    "Confusion": ["confus", "lost", "forget", "disorient"],
}

_CRISIS_KEYWORDS = ["fell", "fall", "bleeding", "injur", "hurt", "chest pain", "emergency", "unconscious", "can't breathe"]


def _mock_behavioral_analysis(contents: str) -> BehavioralAnalysis:
    text = contents.lower()

    category = "General Behavioral Change"
    for label, keywords in _TRIGGER_KEYWORDS.items():
        if any(k in text for k in keywords):
            category = label
            break

    triggers = [k for keywords in _TRIGGER_KEYWORDS.values() for k in keywords if k in text][:3] or ["Not enough detail provided"]
    is_crisis = any(k in text for k in _CRISIS_KEYWORDS)

    return BehavioralAnalysis(
        category=category,
        observed_triggers=triggers,
        clinical_rationale=(
            "Mock response (MOCK_LLM=true, no live Gemini call). In a real run, this field explains the "
            "likely behavioral cause based on the caregiver's description."
        ),
        actionable_intervention=(
            "Approach calmly, use short reassuring sentences, remove immediate triggers if possible, "
            "and redirect to a familiar, calming activity."
        ),
        is_crisis=is_crisis,
    )
