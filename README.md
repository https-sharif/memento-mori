# Dementia Memory Assistant

[![CI](https://github.com/https-sharif/memento-mori/actions/workflows/ci.yml/badge.svg)](https://github.com/https-sharif/memento-mori/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A prototype memory-support system for people living with dementia. A camera
recognises familiar faces and everyday objects; a language model turns what it
sees into a short, calm memory card — *"Sarah (Daughter). She visits every
weekend."* — while a caregiver can ask questions about the same scene and get
practical, grounded suggestions.

> [!WARNING]
> **This is a research and demonstration project. It is not a medical device and
> must not be used to make care decisions.** It processes face biometrics and is
> not clinically validated. Read [DISCLAIMER.md](DISCLAIMER.md) and
> [PRIVACY.md](PRIVACY.md) before running it.

---

## How it fits together

```
  webcam
    │
    ▼
┌───────────────────┐   WS /ws, 1 Hz    ┌───────────────────┐
│  vision_service   │ ────────────────► │      backend      │
│     port 8000     │                   │     port 8001     │
│                   │                   │                   │
│ InsightFace +     │                   │ dedup → retrieve  │
│ YOLOv8n, offline  │                   │ → Gemini → card   │
└───────────────────┘                   └───────────────────┘
          ▲                                ▲            ▲
          │  GET /frame, POST /register    │ GET /latest│ POST /ask
          │                                │            │
          │                       ┌────────┴────────────┴───┐
          └───────────────────────│        frontend         │
                                  │      static HTML        │
                                  └─────────────────────────┘
```

- **`vision_service/`** (port 8000) — face recognition and object detection.
  Runs entirely locally; no image ever leaves the machine.
- **`backend/`** (port 8001) — subscribes to the vision stream, retrieves
  relevant facts from `patient_profile.json`, and calls Gemini to produce memory
  cards and caregiver answers. Only text is sent to the API.
- **`frontend/`** — a static page, no build step. Open it directly.
- **`offline-chatbot/`** — a separate, fully offline Streamlit prototype
  (Ollama + Chroma). Not connected to the pipeline above.

Full detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Design notes

The decisions that were not obvious, and the reasoning behind them.

**LLM calls are async, and that is load-bearing.** `google-genai` exposes both a
sync and an async surface. The sync one, called from inside `async def`, blocks
the entire event loop for the length of the API round trip — the vision
websocket stops draining, frames back up, and `/health` stops answering. Every
call goes through `client.aio.models.generate_content`, and `mock_llm` mirrors
both surfaces so the offline path exercises the same code path as production.

**The vision stream is deduplicated before it reaches the model.** `/ws` pushes a
frame every second — roughly 86,000 model calls a day if forwarded naively, most
of them re-narrating a room that has not changed. The backend hashes a *scene
signature*: who is recognised, their relationship, and the sorted object labels.
Confidence scores and timestamps are deliberately excluded, because they jitter
on every frame and would make each signature unique. The model is called only
when that signature actually changes.

**Two output slots, not one.** The patient-facing card and the caregiver's answer
live in separate slots in `store.py`, because they have different lifecycles: the
card refreshes as the room changes, while an answer must survive until it is
read. Collapsing them means a 2-second poll wipes an answer the caregiver is
still reading — which is precisely the bug this repo shipped with.

**Failure degrades to calm, not to blank.** `response.parsed` is `None` on a
safety block, quota exhaustion, or malformed JSON. For someone with dementia, a
blank screen or a stack trace is worse than a slightly stale message, so the
perception loop falls back to the last good card or a fixed reassuring one. The
caregiver endpoints — whose reader can interpret an error — return a 503 with a
readable message instead. Different users, different failure modes.

**Retrieval is keyword overlap, not embeddings.** The patient profile is a few
dozen facts. A vector store would add a service dependency, an index build, and
startup latency to beat a linear scan over data that fits on one screen. The
retrieval interface is narrow enough to swap if a profile ever outgrows it.

**The whole stack runs with no API key and no camera.** `MOCK_LLM=true` and
`mock_vision.py` serve the real contracts with rule-based responses. That is what
makes the project reviewable by a stranger in two minutes, and what lets CI
verify endpoint behaviour on every push without a secret or a device.

---

## Quickstart

Requires **Python 3.10+** (the code uses `X | None` type syntax throughout).

You can try the whole thing **without an API key and without a camera** — start
here if you just want to see it work.

### 1. Backend, in mock mode

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
MOCK_LLM=true uvicorn main:app --port 8001
```

`MOCK_LLM=true` swaps Gemini for canned, rule-based responses. No key, no
network calls, no rate limit. Card titles are labelled `(mock)` so the mode is
never ambiguous.

### 2. A fake camera, in a second terminal

```bash
cd backend && source .venv/bin/activate
python mock_vision.py        # serves the real ws://localhost:8000/ws contract
```

### 3. Open the page

Open `frontend/index.html` in a browser. A memory card appears within a couple
of seconds, and the caregiver question box works.

On Windows, use `.venv\Scripts\activate` in place of `source .venv/bin/activate`.

---

## Running the real thing

### Vision service

```bash
cd vision_service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Serves on <http://localhost:8000>. The first run downloads InsightFace and YOLO
weights (a few hundred MB); after that it is fully offline. A webcam is required.

**Register a face** at <http://localhost:8000/static/register.html> — enter a
name and relationship, capture 3–5 photos, submit. Recognition only matches
people enrolled on this machine, so this step is not optional.

> Registering someone stores a **face embedding — biometric data** — on your
> disk. Only enrol people who have knowingly agreed. See
> [PRIVACY.md](PRIVACY.md) for what is stored and how to delete it.

Check the roster with `curl http://localhost:8000/people`.

### Backend with Gemini

Copy `.env.example` to `.env` at the repository root and set your key from
<https://aistudio.google.com/apikey>:

```
GEMINI_API_KEY=your_key_here
```

Then:

```bash
cd backend && source .venv/bin/activate
uvicorn main:app --port 8001
```

Both services read that root `.env` automatically. It is gitignored — keep real
keys out of source. The free Gemini tier allows roughly 5 requests per minute;
`MOCK_LLM=true` avoids it entirely.

---

## API

### vision_service — port 8000

`GET /recognize` and `WS /ws` return the same payload. This is the stable
contract:

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
  "objects": [
    { "label": "cup", "confidence": 0.91 }
  ],
  "timestamp": 1751190195
}
```

`/ws` pushes once per second. Also available: `GET /health`, `GET /people`,
`POST /register`, `POST /register/capture`, `DELETE /people/{name}`,
`GET /frame` (JPEG still).

### backend — port 8001

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness |
| `GET /latest` | Current patient memory card |
| `GET /latest?type=caretaker` | Latest caregiver answer (separate slot, so neither clobbers the other) |
| `POST /ask` | `{"question": "..."}` → advice grounded in the live scene and the patient profile |
| `POST /api/caregiver/analyze` | `{"message": "..."}` → structured behavioural analysis |

```bash
curl -s -X POST http://localhost:8001/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How do I keep him calm at dinner?"}'
```

Subscribing to the vision stream directly, if you want to build something else
on top:

```python
import asyncio, json, websockets

async def listen():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        async for msg in ws:
            scene = json.loads(msg)   # ~1 per second
```

---

## The patient profile

`patient_profile.json` at the repository root is the single source of patient
facts — name, condition, family, communication preferences. The backend
retrieves matching lines from it and passes only those to the model. Edit this
file, not code, to change who the patient is.

**The file shipped here is sample data.** "Arthur" is fictional, invented for
demonstration. The repository contains no real patient information, and none
should be added to it — see [DISCLAIMER.md](DISCLAIMER.md).

---

## Security

This is a prototype and its security posture reflects that. **Neither service
authenticates anything**, and CORS is fully open on both. Anyone who can reach
port 8000 can pull a live camera still from `GET /frame`, enrol a face, or delete
a registered person.

Both services therefore bind `127.0.0.1` by default. Set `VISION_HOST=0.0.0.0`
only on a network you trust, and do not deploy this as-is. Details and the full
limitation list are in [PRIVACY.md](PRIVACY.md).

---

## Development

```bash
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
pytest backend/tests
ruff check .
```

The test suite runs fully offline against `backend/mock_llm.py` — no API key, no
camera, no network. CI runs it on Python 3.10, 3.11, and 3.12.

`vision_service` is deliberately excluded from CI: its InsightFace/YOLO/OpenCV
stack is heavy and its endpoints need a physical camera.

---

## Known gaps

What a reviewer should know is missing, and where this would go next.

- **Recognition is single-machine.** Embeddings live in a JSON file on the host
  that captured them. Anything multi-device needs a real store plus an enrolment
  flow with revocation.
- **Nothing is authenticated.** Acceptable on localhost, disqualifying anywhere
  else. Auth belongs in front of both services before the bind address widens.
- **`vision_service` has no automated tests.** Its stack needs a camera, so CI
  skips it and correctness there rests on manual runs. Separating frame
  processing from the capture loop would make most of it testable without
  hardware.
- **Retrieval will not scale with the profile.** Keyword overlap is the right
  call at a few dozen facts and the wrong one at a thousand.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — services, contracts, and design decisions
- [docs/DEMO.md](docs/DEMO.md) — running a walkthrough
- [DISCLAIMER.md](DISCLAIMER.md) — what this is not
- [PRIVACY.md](PRIVACY.md) — biometric data handling and deletion
- [vision_service/README.md](vision_service/README.md) — vision service internals
- [backend/README.md](backend/README.md) — backend internals
- [offline-chatbot/README.md](offline-chatbot/README.md) — the separate offline prototype

## License

MIT — see [LICENSE](LICENSE). Provided with no warranty of any kind.
