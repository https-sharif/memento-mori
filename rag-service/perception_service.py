import os
import json
import asyncio
from typing import Optional

import httpx
import websockets
from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.models import MemoryCardOutput
from app.patient_memory import PatientMemoryStore

load_dotenv()


class PerceptionStateTracker:
    def __init__(self):
        self.client = genai.Client()
        self.model = "gemini-2.5-flash"
        self.memory = PatientMemoryStore()
        self.current_person_name: Optional[str] = None
        self.current_objects: list[str] = []
        self.frontend_api_url = os.getenv("FRONTEND_API_URL", "http://localhost:8080/api/cue")

    def _should_trigger_llm(self, new_person: dict, new_objects: list) -> bool:
        new_name = new_person["name"] if new_person.get("recognized") else "Unknown"
        new_obj_labels = sorted([obj["label"] for obj in new_objects])
        if new_name != self.current_person_name or new_obj_labels != self.current_objects:
            self.current_person_name = new_name
            self.current_objects = new_obj_labels
            return True
        return False

    async def generate_orientation_cue(self, person: dict, objects: list) -> MemoryCardOutput:
        person = self.memory.enrich_person(person)
        if person.get("recognized"):
            vision_context = f"Person detected: {person['name']} ({person.get('relationship', '')}). Note: {person.get('note', '')}.\n"
        else:
            vision_context = "Person detected: Unrecognized face or no familiar person present.\n"

        if objects:
            vision_context += "Objects visible: " + ", ".join(o["label"] for o in objects)

        response = self.client.models.generate_content(
            model=self.model,
            contents=f"Patient Profile / Memory Context: {self.memory.profile_text()}\nLive Camera Feed Data:\n{vision_context}",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a compassionate, real-time memory assistant for a person with dementia. "
                    "Output a clear visual memory card and a spoken audio script. "
                    "Keep the card extremely simple and reassuring. Voice guidance must be calm, slow, direct, and non-clinical. "
                    "Never say 'camera feed', 'detected', or 'analysis' to the patient."
                ),
                response_mime_type="application/json",
                response_schema=MemoryCardOutput,
                temperature=0.3,
            ),
        )
        return response.parsed

    async def send_to_frontend(self, output: MemoryCardOutput):
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(self.frontend_api_url, json=output.model_dump())

    async def listen_and_process(self):
        uri = os.getenv("VISION_WS_URL", "ws://localhost:8000/ws")
        print(f"Connecting to live vision websocket stream at {uri}...")
        async with websockets.connect(uri) as ws:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                    person = data.get("person", {"recognized": False, "name": "Unknown", "relationship": "", "note": ""})
                    objects = data.get("objects", [])

                    if self._should_trigger_llm(person, objects):
                        print(f"\n[Scene Shift Detected] timestamp={data.get('timestamp')}")
                        output = await self.generate_orientation_cue(person, objects)
                        print(json.dumps(output.model_dump(), indent=2))
                        await self.send_to_frontend(output)
                except Exception as exc:
                    print(f"Error handling frame: {exc}")


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY before running.")
        raise SystemExit(1)
    asyncio.run(PerceptionStateTracker().listen_and_process())
