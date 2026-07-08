import os
import json
import asyncio
import websockets
from typing import Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types
from store import set_latest_output, set_latest_scene


load_dotenv()

VISION_WS_URL = os.environ.get("VISION_WS_URL", "ws://localhost:8000/ws")
VISION_RECONNECT_DELAY = float(os.environ.get("VISION_RECONNECT_DELAY", "2.0"))

# 1. Structured output for the Frontend
class MemoryCardOutput(BaseModel):
    card_title: str = Field(description="Large, simple title text for the frontend display (e.g., 'Sarah (Daughter)').")
    card_body: str = Field(description="Max 2 short sentences reinforcing context (e.g., 'She visits every weekend. She loves you.').")
    voice_guidance: str = Field(description="The exact text to convert to audio. Must be short, exceptionally gentle, comforting, and spoken directly to the patient.")

class PerceptionStateTracker:
    def __init__(self):
        self.client = genai.Client()
        self.model = "gemini-2.5-flash"
        
        # State tracking variables to prevent redundant LLM fires
        self.current_person_name: Optional[str] = None
        self.current_objects: List[str] = []
        self.current_scene_signature: Optional[str] = None
        self.last_output: Optional[MemoryCardOutput] = None

    def _should_trigger_llm(self, new_person: dict, new_objects: list) -> bool:
        """Determines if the scene has changed enough to warrant a new cue."""
        new_name = new_person["name"] if new_person.get("recognized") else "Unknown"
        new_obj_labels = sorted([obj["label"] for obj in new_objects])
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
            self.current_person_name = new_name
            self.current_objects = new_obj_labels
            self.current_scene_signature = new_scene_signature
            return True
        return False

    async def generate_orientation_cue(self, person: dict, objects: list) -> MemoryCardOutput:
        """Generates the structured memory card data using Gemini."""
        # Compile vision context dynamically
        vision_context = f"Person detected: {person['name']} ({person['relationship']}). Note: {person['note']}.\n" if person['recognized'] else "Person detected: Unrecognized face.\n"
        if objects:
            vision_context += "Objects currently visible in frame: " + ", ".join(o["label"] for o in objects)

        # Patient Baseline/RAG context injection point
        # (Kept flat here, but this is where your static memory books attach)
        patient_profile = "Patient is Arthur, advanced Alzheimer's. Gets easily startled by sudden changes. Needs reassurance."

        response = self.client.models.generate_content(
            model=self.model,
            contents=f"Patient Profile: {patient_profile}\nLive Camera Feed Data:\n{vision_context}",
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
                response_schema=MemoryCardOutput,
                temperature=0.3,
            )
        )
        result = response.parsed
        self.last_output = result
        return result

    async def listen_and_process(self):
        print(f"Connecting to live vision websocket stream at {VISION_WS_URL}...")

        while True:
            try:
                async with websockets.connect(VISION_WS_URL) as ws:
                    print("Connected to vision service.")
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                            set_latest_scene(data)
                            person = data.get("person", {"recognized": False, "name": "Unknown", "relationship": "", "note": ""})
                            objects = data.get("objects", [])

                            # Dedup the 1Hz stream to protect LLM calls
                            if self._should_trigger_llm(person, objects):
                                print(f"\n[Scene Shift Detected] Processing telemetry timestamp: {data.get('timestamp')}")

                                # Call Gemini to compute memory parameters
                                output: MemoryCardOutput = await self.generate_orientation_cue(person, objects)

                                # --- ROUTE TO FRONTEND ---
                                # Here you push this clean payload down to your mobile app or frontend display
                                print("\n>>> BROADCASTING TO FRONTEND:")
                                output_data = output.model_dump()
                                print(json.dumps(output_data, indent=2))
                                set_latest_output(output_data)
                                # e.g., await self.frontend_websocket.send(output.model_dump_json())
                            else:
                                # Scene is stable, keep the previous LLM card.
                                if self.last_output is not None:
                                    set_latest_output(self.last_output.model_dump())

                        except Exception as e:
                            print(f"Error handling frame: {e}")
            except Exception as e:
                print(f"Vision connection lost ({e}). Reconnecting in {VISION_RECONNECT_DELAY:.1f}s...")
                await asyncio.sleep(VISION_RECONNECT_DELAY)

if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY before running.")
        exit(1)
        
    tracker = PerceptionStateTracker()
    asyncio.run(tracker.listen_and_process())