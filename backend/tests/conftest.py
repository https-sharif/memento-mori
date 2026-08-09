"""Shared pytest setup for the backend suite.

Two things must happen before any backend module is imported:

1. ``MOCK_LLM`` must be set. ``gemini_client`` raises ``SystemExit`` at import
   time when there is no ``GEMINI_API_KEY``, and CI deliberately has none — the
   whole suite runs offline against ``mock_llm.py``.
2. ``backend/`` must be on ``sys.path``. The backend uses flat imports
   (``import store``) rather than a package, so it has to be importable by name.
"""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

os.environ["MOCK_LLM"] = "true"
# If a test ever runs the app's lifespan, keep the perception task from
# reconnecting in a tight loop against a vision service that isn't there.
os.environ.setdefault("VISION_RECONNECT_DELAY", "3600")

import pytest  # noqa: E402

import store  # noqa: E402


@pytest.fixture(autouse=True)
def clean_store():
    """``store`` is module-level global state, so reset it between tests."""
    store.set_latest_scene({})
    store.set_latest_patient_card({})
    store.set_latest_caretaker_advice({})
    yield
    store.set_latest_scene({})
    store.set_latest_patient_card({})
    store.set_latest_caretaker_advice({})


@pytest.fixture
def profile_path(tmp_path):
    """A small, fixed patient profile so retrieval tests don't depend on the
    shipped sample data."""
    import json

    path = tmp_path / "patient_profile.json"
    path.write_text(
        json.dumps(
            {
                "patient_name": "Arthur",
                "condition_stage": "moderate dementia",
                "baseline": "Needs calm, simple orientation cues.",
                "family": [
                    {"name": "Sarah", "relationship": "Daughter", "note": "Visits on weekends"},
                    {"name": "Tom", "relationship": "Son", "note": "Calls every evening"},
                ],
                "preferences": ["Prefers short sentences", "Likes gardening talk"],
                "memories": ["Worked as a carpenter for forty years"],
            }
        ),
        encoding="utf-8",
    )
    return path
