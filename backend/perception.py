from __future__ import annotations

import asyncio
import json
import logging
import os

import websockets
from google.genai import types

import store
from gemini_client import GEMINI_MODEL, client
from models import MemoryCard
from patient_memory import PatientMemoryStore

logger = logging.getLogger(__name__)

VISION_WS_URL = os.environ.get("VISION_WS_URL", "ws://localhost:8000/ws")
VISION_RECONNECT_DELAY = float(os.environ.get("VISION_RECONNECT_DELAY", "2.0"))

# Shown when Gemini is unreachable or returns nothing parseable, so the patient
# still sees something calm instead of a stale or blank card.
_FALLBACK_CARD = MemoryCard(
    card_title="You are safe",
    card_body="Take your time. Someone will be with you soon.",
    voice_guidance="You're safe. Take a slow breath, and take all the time you need.",
)


class PerceptionStateTracker:
    def __init__(self, memory: PatientMemoryStore | None = None):
        self.memory = memory or PatientMemoryStore()
        self.current_scene_signature: str | None = None
        self.last_output: MemoryCard | None = None

    def _should_trigger_llm(self, new_person: dict, new_objects: list) -> bool:
        """Determines if the scene has changed enough to warrant a new cue."""
        new_name = (new_person.get("name") or "Unknown") if new_person.get("recognized") else "Unknown"
        new_obj_labels = sorted(obj.get("label", "") for obj in new_objects)
        new_scene_signature = json.dumps(
            {
                "recognized": bool(new_person.get("recognized")),
                "face_detected": bool(new_person.get("face_detected")),
                "name": new_name,
                "relationship": new_person.get("relationship", ""),
                "note": new_person.get("note", ""),
                "objects": new_obj_labels,
            },
            sort_keys=True,
        )

        if new_scene_signature != self.current_scene_signature:
            self.current_scene_signature = new_scene_signature
            return True
        return False

    async def generate_orientation_cue(self, person: dict, objects: list) -> MemoryCard:
        """Generates the structured memory card data using Gemini."""
        person = self.memory.enrich_person(person)
        if person.get("recognized"):
            vision_context = f"Person detected: {person['name']} ({person.get('relationship', '')}). Note: {person.get('note', '')}.\n"
        else:
            vision_context = "Person detected: Unrecognized face or no familiar person present.\n"

        if objects:
            vision_context += "Objects currently visible in frame: " + ", ".join(
                o.get("label", "") for o in objects
            )

        retrieved_context = self.memory.retrieve_text(vision_context, top_k=5)

        try:
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"Patient Memory Context:\n{retrieved_context}\nLive Camera Feed Data:\n{vision_context}",
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a compassionate, real-time memory assistant for a person with dementia. "
                        "Your job is to read environmental data from their camera and output a clear, visual "
                        "memory card specification and spoken audio script to orient them gently. "
                        "Rule 1: Keep UI text extremely simple and large. "
                        "Rule 2: Voice guidance must be spoken to the patient, calm, conversational, and slow. "
                        "Never say 'Based on the camera feed' or look clinical. Act like a loving, unseen companion."
                    ),
                    response_mime_type="application/json",
                    response_schema=MemoryCard,
                    temperature=0.3,
                ),
            )
        except Exception:
            logger.exception("Gemini call failed while generating an orientation cue")
            return self.last_output or _FALLBACK_CARD

        # .parsed is None when the model is safety-blocked or returns unparseable JSON.
        if response.parsed is None:
            logger.warning("Gemini returned no parseable content for the orientation cue.")
            return self.last_output or _FALLBACK_CARD

        result = response.parsed
        self.last_output = result
        return result

    async def listen_and_process(self) -> None:
        logger.info("Connecting to live vision websocket stream at %s ...", VISION_WS_URL)

        while True:
            try:
                async with websockets.connect(VISION_WS_URL) as ws:
                    logger.info("Connected to vision service.")
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                            store.set_latest_scene(data)
                            person = data.get("person") or {
                                "recognized": False, "name": "Unknown", "relationship": "", "note": "",
                            }
                            objects = data.get("objects") or []

                            # Dedup the 1Hz stream to protect LLM calls
                            if self._should_trigger_llm(person, objects):
                                logger.info("Scene shift detected (timestamp=%s).", data.get("timestamp"))
                                output = await self.generate_orientation_cue(person, objects)
                                store.set_latest_patient_card(output.model_dump())
                            elif self.last_output is not None:
                                # Scene is stable, keep the previous LLM card.
                                store.set_latest_patient_card(self.last_output.model_dump())

                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception("Error handling a vision frame — skipping it.")
            except asyncio.CancelledError:
                logger.info("Perception loop cancelled.")
                raise
            except Exception as e:
                logger.warning(
                    "Vision connection lost (%s). Reconnecting in %.1fs...", e, VISION_RECONNECT_DELAY
                )
                await asyncio.sleep(VISION_RECONNECT_DELAY)
