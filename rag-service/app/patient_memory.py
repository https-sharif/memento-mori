import json
import re
from pathlib import Path
from typing import Any


class PatientMemoryStore:
    """Small local RAG store backed by JSON.

    This is intentionally simple for a demo: it turns patient profile data into
    searchable memory chunks, retrieves the most relevant chunks for a live
    query/vision event, and passes only that context into the LLM.
    """

    def __init__(self, path: str = "data/patient_profile.json"):
        self.path = Path(path)
        self.profile = self._load_profile()
        self.chunks = self._build_chunks()

    def _load_profile(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _build_chunks(self) -> list[dict[str, str]]:
        if not self.profile:
            return []

        chunks: list[dict[str, str]] = []
        patient_name = self.profile.get("patient_name", "the patient")

        chunks.append({
            "id": "baseline",
            "text": (
                f"Patient name: {patient_name}. "
                f"Condition/stage: {self.profile.get('condition_stage', 'dementia')}. "
                f"Baseline needs: {self.profile.get('baseline', '')}"
            ),
        })

        for person in self.profile.get("family", []):
            chunks.append({
                "id": f"family:{person.get('name', '').lower()}",
                "text": (
                    f"{person.get('name')} is {patient_name}'s {person.get('relationship')}. "
                    f"Helpful reminder: {person.get('note', '')}"
                ),
            })

        for idx, pref in enumerate(self.profile.get("preferences", []), start=1):
            chunks.append({"id": f"preference:{idx}", "text": f"Communication preference: {pref}"})

        for idx, memory in enumerate(self.profile.get("memories", []), start=1):
            chunks.append({"id": f"memory:{idx}", "text": str(memory)})

        return chunks

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, str]]:
        """Return the most relevant memory chunks for the incoming situation."""
        if not self.chunks:
            return [{"id": "fallback", "text": "No stored patient memory was found. Use gentle general reassurance."}]

        query_tokens = self._tokens(query)
        scored: list[tuple[int, dict[str, str]]] = []
        for chunk in self.chunks:
            chunk_tokens = self._tokens(chunk["text"])
            score = len(query_tokens & chunk_tokens)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [chunk for _, chunk in scored[:top_k]]

        # Always include baseline, because the LLM needs stable patient context.
        baseline = next((c for c in self.chunks if c["id"] == "baseline"), None)
        if baseline and baseline not in selected:
            selected.insert(0, baseline)

        return selected[:top_k]

    def retrieve_text(self, query: str, top_k: int = 5) -> str:
        chunks = self.retrieve(query=query, top_k=top_k)
        return "\n".join(f"- [{chunk['id']}] {chunk['text']}" for chunk in chunks)

    def profile_text(self) -> str:
        """Backward-compatible full profile summary."""
        return self.retrieve_text("baseline family preferences", top_k=8)

    def enrich_person(self, person: dict) -> dict:
        if not person.get("recognized"):
            return person
        for family_member in self.profile.get("family", []):
            if family_member.get("name", "").lower() == person.get("name", "").lower():
                merged = dict(person)
                merged.setdefault("relationship", family_member.get("relationship", ""))
                merged["note"] = person.get("note") or family_member.get("note", "")
                return merged
        return person
