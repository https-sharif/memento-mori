"""Retrieval and person-enrichment behaviour of the keyword-overlap store."""

from pathlib import Path

from patient_memory import PatientMemoryStore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_baseline_chunk_is_always_returned(profile_path):
    """The LLM needs stable patient context even when nothing matches the query."""
    store = PatientMemoryStore(str(profile_path))

    chunks = store.retrieve("zzzz qqqq no overlap whatsoever")

    assert [c["id"] for c in chunks] == ["baseline"]
    assert "Arthur" in chunks[0]["text"]


def test_family_chunk_retrieved_for_matching_query(profile_path):
    store = PatientMemoryStore(str(profile_path))

    ids = [c["id"] for c in store.retrieve("Sarah is visiting today")]

    assert "family:sarah" in ids
    assert "baseline" in ids


def test_preference_chunk_retrieved_for_matching_query(profile_path):
    store = PatientMemoryStore(str(profile_path))

    ids = [c["id"] for c in store.retrieve("he likes talking about gardening")]

    assert any(i.startswith("preference:") for i in ids)


def test_retrieve_respects_top_k(profile_path):
    store = PatientMemoryStore(str(profile_path))

    assert len(store.retrieve("Sarah Tom gardening carpenter dementia", top_k=2)) == 2


def test_missing_profile_falls_back(tmp_path):
    """A missing profile must not crash the pipeline — it degrades to generic
    reassurance."""
    store = PatientMemoryStore(str(tmp_path / "does_not_exist.json"))

    chunks = store.retrieve("anything at all")

    assert chunks == [
        {
            "id": "fallback",
            "text": "No stored patient memory was found. Use gentle general reassurance.",
        }
    ]


def test_retrieve_text_is_prefixed_by_chunk_id(profile_path):
    store = PatientMemoryStore(str(profile_path))

    text = store.retrieve_text("Sarah")

    assert "[baseline]" in text
    assert "[family:sarah]" in text


def test_enrich_person_merges_known_family_member(profile_path):
    store = PatientMemoryStore(str(profile_path))

    enriched = store.enrich_person({"recognized": True, "name": "Sarah"})

    assert enriched["relationship"] == "Daughter"
    assert enriched["note"] == "Visits on weekends"


def test_enrich_person_is_case_insensitive(profile_path):
    store = PatientMemoryStore(str(profile_path))

    assert store.enrich_person({"recognized": True, "name": "sARAh"})["relationship"] == "Daughter"


def test_enrich_person_keeps_vision_service_values(profile_path):
    """vision_service already stores a relationship per registered face; the
    profile must not overwrite what the camera pipeline supplied."""
    store = PatientMemoryStore(str(profile_path))

    enriched = store.enrich_person(
        {"recognized": True, "name": "Sarah", "relationship": "Nurse", "note": "On shift today"}
    )

    assert enriched["relationship"] == "Nurse"
    assert enriched["note"] == "On shift today"


def test_enrich_person_passes_unrecognized_through_untouched(profile_path):
    store = PatientMemoryStore(str(profile_path))
    person = {"recognized": False, "name": None, "face_detected": True}

    assert store.enrich_person(person) == person


def test_enrich_person_passes_unknown_name_through_untouched(profile_path):
    store = PatientMemoryStore(str(profile_path))
    person = {"recognized": True, "name": "Nobody"}

    assert store.enrich_person(person) == person


def test_shipped_sample_profile_loads():
    """The sample profile in the repo root must stay valid — it is what a fresh
    clone runs against."""
    store = PatientMemoryStore(str(REPO_ROOT / "patient_profile.json"))

    assert store.chunks, "sample patient_profile.json produced no chunks"
    assert any(c["id"] == "baseline" for c in store.chunks)
