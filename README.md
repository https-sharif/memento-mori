# Memento Mori — A Dementia Companion

An ambient, camera-based assistant that helps people with dementia recognize
familiar faces and objects, hear a gentle spoken reminder of who's in the
room, and lets caregivers ask for real-time, practical guidance — all without
the patient having to operate anything.

---

## Why this exists

People with moderate-to-advanced dementia frequently lose the ability to
recognize even their closest family members, and forget routine things like
whether they've taken their medication. That moment of not-recognizing is
frightening for the patient and exhausting for the caregiver, who often has
to repeat the same reassurance dozens of times a day.

Most "AI for dementia" demos stop at a chatbot. We wanted something that
works passively, in the background, through a camera that's already pointed
at the room — so the patient never has to type, tap, or ask.

The result is a three-stage pipeline:

```
Webcam / Photo
      │
      ▼
Vision Service   →  { person, objects }        (face + object recognition)
      │
      ├──► LLM / RAG   →  memory card + voice   (turns detections into comfort)
      │
      └──► Frontend    →  shows card, plays audio, caregiver workspace
```

Each stage is a separate, independently runnable service, which let us split
work cleanly and swap pieces (e.g. mock vision → real webcam) without
touching the rest of the system.

---

## What it actually does

- **Recognizes people** the patient knows, and silently tracks who is
  currently in frame.
- **Recognizes care-relevant objects** — medicine bottles, glasses, a cane,
  keys — distinguishing them from visually similar everyday objects.
- **Generates a memory card** in real time: a short title, a couple of
  reassuring sentences, and a spoken line — written to sound like a warm,
  unseen companion, never like a system reporting a detection.
- **Speaks that line aloud** to the patient via the browser's text-to-speech,
  so no reading is required.
- **Gives caregivers a separate workspace**: a live text Q&A ("he's anxious
  and asking where his daughter is — what do I do?") that returns a
  structured, non-clinical action plan, and flags anything that looks like a
  genuine safety risk.
- **Grounds every response** in a small, editable patient profile (family
  members, preferences, personal notes) instead of generic advice, via a
  lightweight retrieval step.

---

## Architecture & why each part exists

| Component | What it does | Why we built it this way |
|---|---|---|
| **Vision Service** (`vision_service/`) | Offline face recognition (InsightFace/ArcFace) + object detection (YOLOv8), served over a FastAPI HTTP + WebSocket API | Runs **entirely on-device**. This matters for two reasons: latency (we need ~1 update/second, a round trip to a cloud vision API would be too slow) and privacy (raw camera frames of the patient never leave the machine — only a small JSON summary like `person: Sarah, daughter` goes downstream). |
| ↳ Face matching | Two-pass cosine similarity: first against each person's *averaged* embedding (fast), falling back to comparing every stored photo individually only if there's no confident match | Keeps recognition fast as the registered-people list grows, while still catching awkward angles that pull a face away from its own average. |
| ↳ Object detection | Base YOLOv8-nano (general COCO objects) + an optional custom-trained model layered on top, with the custom model's detections taking priority on overlap | A generic model can tell you "bottle" but not "medicine bottle vs. water bottle" — the distinction that actually matters for dementia care. Stacking models let us keep broad household context *and* get fine-grained, demo-specific detail without retraining from scratch. |
| **LLM / RAG Service** (`rag-service/`, `llm-service/`) | Consumes the vision stream, deduplicates scene changes, retrieves relevant patient-memory context, and calls Gemini with a forced JSON schema (Pydantic) to produce the memory card and voice line | Raw detections mean nothing to someone with dementia — "Sarah, confidence 0.96" isn't comforting. Structured outputs mean the frontend never parses free text; it just plugs values straight into the UI. Retrieval over a small patient profile means the same "Sarah is here" event becomes personal ("she visits every weekend and loves you") instead of generic. |
| ↳ Scene-change dedup | Only calls the LLM when the detected person/objects actually change | Stops the card from flickering with reworded versions of the same message every second, and keeps API cost and latency down. |
| ↳ Caregiver analysis endpoint | A second, separate LLM call/schema focused on behavioral triggers, rationale, and one concrete non-pharmacological action — flagged separately if it detects a crisis | Patients and caregivers need fundamentally different tones and content. One audience needs to be soothed; the other needs to be informed and given something actionable. Mixing the two would serve neither well. |
| **Frontend** (`frontend/`, `rag-service/static/`) | Displays the live memory card, speaks it via Web Speech API, shows a live camera preview, and gives caregivers a question box | Kept deliberately simple (plain HTML/JS, no build step) so it's trivial to run anywhere and easy to swap for a native app later. |

---

## Project structure

```
├── frontend/            lightweight standalone UI (talks to vision + llm-service)
├── llm-service/          minimal HTTP server variant of the LLM layer
├── rag-service/          fuller FastAPI app: RAG + caregiver analysis + bundled frontend
└── vision_service/       face + object recognition, fully offline
    ├── models/           InsightFace + YOLOv8 wrappers
    ├── storage/          JSON-backed face embedding store
    ├── api/               /recognize, /register, /people, /ws
    └── train/            tools + config for fine-tuning a custom object model
```

---

## Running it

Three terminals — vision service first, then whichever LLM path you're using.

**1. Vision Service** (required for everything else)
```bash
cd vision_service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
# → http://localhost:8000
# → registration UI: http://localhost:8000/static/register.html
```

**2a. Full RAG app (recommended — includes caregiver analysis + bundled frontend)**
```bash
cd rag-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8080
# → http://localhost:8080
```

**2b. Minimal LLM service + standalone frontend**
```bash
cd llm-service
export GEMINI_API_KEY=...
export VISION_WS_URL=ws://localhost:8000/ws
python main.py
# then open frontend/index.html in a browser
```

No webcam handy? Run `mock_vision.py` in either service folder to simulate a
person walking in and out of frame — useful for demos and development.

See `vision_service/README.md` and `rag-service/README.md` for full endpoint
references and configuration options.

---

## Key API shapes

**Vision Service → everyone else**
```json
{
  "person": {
    "recognized": true,
    "name": "Sarah",
    "relationship": "Daughter",
    "note": "Visits on weekends",
    "confidence": 0.96,
    "face_detected": true
  },
  "objects": [{ "label": "Medicine Bottle", "confidence": 0.91 }],
  "timestamp": 1751190195
}
```

**LLM output → frontend**
```json
{
  "card_title": "Sarah, your daughter",
  "card_body": "Sarah is here with you. She visits often and cares about you very much.",
  "voice_guidance": "Hi Arthur, Sarah is here with you. You are safe, and she is happy to see you."
}
```
