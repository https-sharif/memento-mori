# Dementia Memory Assistant

This completes the pipeline:

```text
Webcam / Photo
  ↓
Vision Service → JSON { person, objects }
  ↓
LLM / RAG → memory card text + voice answer
  ↓
Frontend → shows card, plays audio
```

## Files

- `mock_vision.py` — local simulated vision websocket service.
- `perception_service.py` — consumes vision frames, deduplicates scene changes, calls Gemini, and posts cues to the frontend API.
- `app/main.py` — FastAPI app with frontend websocket broadcasting and caregiver-analysis endpoint.
- `app/models.py` — shared Pydantic schemas.
- `app/patient_memory.py` — simple JSON-backed RAG placeholder.
- `data/patient_profile.json` — editable patient profile and family memory.
- `static/*` — browser frontend. It displays the card and uses Web Speech API for audio playback.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set GEMINI_API_KEY
```

## Run in three terminals

Terminal 1:
```bash
python mock_vision.py
```

Terminal 2:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Terminal 3:
```bash
python perception_service.py
```

Open:

```text
http://localhost:8080
```

## Real camera integration later

Replace `mock_vision.py` with a real service that emits the same JSON shape:

```json
{
  "person": {
    "recognized": true,
    "name": "Sarah",
    "relationship": "Daughter",
    "note": "Visits on weekends"
  },
  "objects": [{ "label": "Medicine Bottle" }],
  "timestamp": 1710000000
}
```
