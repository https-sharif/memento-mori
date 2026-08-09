# backend

Single FastAPI service (port 8001). Consumes the `vision_service` websocket, turns scenes into memory cards via Gemini, answers caregiver questions, and serves both to `frontend/`.

See [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the design rationale behind scene dedup, the two output slots, and the retrieval approach.

```text
Vision Service (ws://localhost:8000/ws)
  ↓
backend (this service, port 8001)
  ↓
frontend/  (polls GET /latest, calls POST /ask)
```

## Files

- `main.py` — FastAPI app, CORS, `/health`, `/latest`, `/ask`, `/api/caregiver/analyze`. Starts the perception background task on startup.
- `perception.py` — connects to `VISION_WS_URL` as a client, dedups scene changes, calls Gemini, writes the patient card to the store.
- `patient_memory.py` — reads the single shared `../patient_profile.json`, retrieves relevant memory chunks for a query (keyword overlap, no embeddings — good enough for a demo).
- `models.py` — `MemoryCard` (shared shape for both the patient card and caretaker advice), `BehavioralAnalysis`, request bodies.
- `gemini_client.py` — the one `genai.Client()` instance, shared by every call site. Set `MOCK_LLM=true` to swap it for `mock_llm.py`'s canned responses instead (no API key or quota needed).
- `mock_llm.py` — rule-based stand-in for the Gemini client, used only when `MOCK_LLM=true`.
- `store.py` — in-memory state: latest scene, latest patient card, latest caretaker advice (two separate slots so a caretaker question can't clobber the patient-facing card).
- `mock_vision.py` — local test double for `vision_service`, serves at `/ws` only.
- `tests/` — offline pytest suite; runs against `mock_llm.py` with no API key, camera, or network.

Every Gemini call uses the async client surface (`client.aio.models.generate_content`). The synchronous one blocks the whole event loop for the duration of the round trip, which stalls the vision websocket and every concurrent request.

## Run

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GEMINI_API_KEY is read from the repo-root .env
uvicorn main:app --host 127.0.0.1 --port 8001
```

To run without a Gemini key or quota, use the canned responses instead:

```bash
MOCK_LLM=true uvicorn main:app --host 127.0.0.1 --port 8001
```

To test without a real camera:

```bash
python mock_vision.py    # serves ws://localhost:8000/ws
```

## Test

```bash
pip install -r requirements-dev.txt
pytest tests
```

## Endpoints

- `GET /health` → `{"ok": true}`
- `GET /latest` → the current patient-facing `MemoryCard` (or `{"status": "empty"}` before the first scene). Add `?type=caretaker` to get the latest caregiver advice instead.
- `POST /ask` `{"question": "..."}` → a `MemoryCard`-shaped caregiver answer, grounded in the current scene + patient memory, cached by scene + question in a bounded LRU. `400` on an empty question, `503` if Gemini is unreachable or returns nothing parseable.
- `POST /api/caregiver/analyze` `{"message": "..."}` → a `BehavioralAnalysis` for a caregiver's free-text situation report. Same `400`/`503` behaviour.

`BehavioralAnalysis.clinical_rationale` is language-model output, not a clinical assessment — see [../DISCLAIMER.md](../DISCLAIMER.md).

