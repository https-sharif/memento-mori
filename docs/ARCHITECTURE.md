# Architecture

Three independent processes. Nothing shares memory or a database; they talk over
HTTP and one websocket.

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
    ▲         ▲                            ▲            ▲
    │         │ GET /frame                 │ GET /latest│ POST /ask
    │         │ (JPEG still)               │ (poll 2 s) │
    │         └──────────────┬─────────────┴────────────┘
    │  POST /register        │
    │                 ┌──────────────┐         ┌──────────────────────┐
    └─────────────────│  frontend    │         │ patient_profile.json │
                      │ static HTML  │         │  (retrieval source)  │
                      └──────────────┘         └──────────────────────┘
```

`offline-chatbot/` is a fourth, entirely separate prototype. It shares no code,
no ports, and no data with the pipeline above.

---

## vision_service (port 8000)

Face recognition and object detection. Fully offline — no image or embedding
ever leaves the machine.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness |
| `POST /recognize` | One-shot analysis of the current frame |
| `WS /ws` | Same payload, broadcast once per second |
| `GET /frame` | Current camera frame as JPEG |
| `GET /people` | Registered roster |
| `POST /register` | Enrol a face from uploaded images |
| `POST /register/capture` | Enrol from the live camera |
| `DELETE /people/{name}` | Remove a person and their embeddings |
| `GET /static/register.html` | Registration UI |

**`/recognize` and `/ws` return the same shape.** Treat it as the stable
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

When no face is found, `recognized` and `face_detected` are `false` and `name`
is `null`. An unfamiliar face gives `face_detected: true` with
`recognized: false`.

Implementation notes:

- Face matching uses InsightFace `buffalo_sc` embeddings with cosine similarity
  above a `0.45` threshold (`config.py`).
- Object detection is YOLOv8n at a `0.50` confidence floor, filtered to a
  27-class care-relevant subset (`object_detector.py`).
- `/ws` broadcasts at `ws_interval = 1.0` s, only while a client is connected.
- Storage is a flat JSON file, `storage/data/embeddings.json`, rewritten on
  every change. There is no database.
- `config.frame_skip` is declared but never read; the real cadence is
  `ws_interval` plus a separate display tick.
- The `/ws` docstring mentions a 30 s keepalive ping that is not implemented.
  Harmless unless a proxy with an idle timeout sits in front of it.

---

## backend (port 8001)

A single FastAPI process — one event loop, no threads, no worker pool. It turns
scenes into patient-facing memory cards and answers caregiver questions.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness |
| `GET /latest` | Latest patient memory card |
| `GET /latest?type=caretaker` | Latest caregiver answer |
| `POST /ask` | `{"question": "..."}` → a memory card for the caregiver |
| `POST /api/caregiver/analyze` | `{"message": "..."}` → structured behavioural analysis |

### The perception loop

`perception.py` runs as an `asyncio.Task` started by FastAPI's `lifespan`. It
dials `VISION_WS_URL`, and reconnects with a fixed backoff whenever the socket
drops.

The vision stream arrives at 1 Hz, which would mean 3,600 LLM calls an hour.
`_should_trigger_llm` collapses it: each frame is reduced to a **scene
signature** over `recognized`, `face_detected`, `name`, `relationship`, `note`,
and the sorted set of object labels. Confidence values and timestamps are
excluded deliberately — they jitter every frame. Gemini is called only when that
signature changes; otherwise the previous card is re-published unchanged.

If Gemini fails or returns nothing parseable, the loop keeps the last good card,
or falls back to fixed reassurance text. It never surfaces an error to the
patient.

### Retrieval

`patient_memory.py` (`PatientMemoryStore`) reads the repo-root
`patient_profile.json` and flattens it into id-tagged chunks: one `baseline`
chunk, one `family:<name>` chunk per relative, and one `preference:<n>` chunk
per preference.

Retrieval is **keyword overlap, not vector search** — the query is tokenised,
scored against each chunk by set intersection, and the top *k* are returned. The
`baseline` chunk is always included so the model never loses core patient
context. A paraphrased question that shares no tokens with a chunk will miss it;
that is a known limitation of the demo-grade approach.

`enrich_person` merges profile data into a recognized person, without
overwriting anything the vision service already supplied.

### Two output slots

`store.py` keeps `latest_patient_card` and `latest_caretaker_advice` separately.
This matters: a caregiver's question must never overwrite the card the patient is
looking at. `GET /latest` reads the first, `?type=caretaker` the second, and
`POST /ask` writes only the second.

Responses are flat JSON — `{card_title, card_body, voice_guidance}` — or
`{"status": "empty"}` before anything has been generated.

### The /ask cache

`/ask` is cached on `(scene signature, normalised question)`, so re-asking the
same thing about an unchanged scene costs nothing. The cache is a bounded LRU
(256 entries) because that key space is effectively unbounded over a long run.
Failures are not cached.

### Gemini

One shared `genai.Client` in `gemini_client.py`, constructed at import. Every
call site uses the **async** surface (`client.aio.models.generate_content`) —
the synchronous one blocks the entire event loop for the duration of the round
trip, which stalls the websocket and every other request.

Structured output is enforced with Pydantic response schemas (`models.py`), so
`response.parsed` is a validated `MemoryCard` or `BehavioralAnalysis`. It is
`None` on a safety block, a quota error, or malformed JSON; every call site
checks for that.

Setting `MOCK_LLM=true` swaps the client for `mock_llm.py`, which implements the
same sync and async surfaces with canned, rule-based responses.

**Only text is sent to Gemini** — scene descriptions, retrieved profile lines,
and caregiver questions. Images never leave the machine.

---

## frontend

Static HTML and JavaScript, no build step. `index.html` opens directly from
disk or from any static server.

- Subscribes to `ws://localhost:8000/ws` for the live scene.
- Polls `http://localhost:8001/latest` every 2 s for the patient card.
- Posts to `http://localhost:8001/ask` for caregiver questions.
- Pulls `http://localhost:8000/frame` for the optional camera preview.

The patient card and the caregiver answer render into separate elements, mirroring
the backend's two slots.

`register.html` exists in two copies — `frontend/register.html` and
`vision_service/static/register.html`, served at
`http://localhost:8000/static/register.html`. The second is the one the home page
links to, because it is same-origin with the API it calls.

---

## offline-chatbot

A self-contained Streamlit prototype, unconnected to everything above. Ollama
`llama3.2` for generation, `all-MiniLM-L6-v2` for embeddings, Chroma for the
vector store. Two apps share a data directory: a patient chat UI on 8501 and a
caregiver admin UI on 8502.

It solves a different shape of the same problem — offline document Q&A rather
than live scene orientation. Its Chroma store is a plausible replacement for
`patient_memory.py`'s keyword retrieval, but nothing wires the two together
today.

See [offline-chatbot/README.md](../offline-chatbot/README.md).

---

## Configuration

All of it lives in a repo-root `.env`, loaded with `python-dotenv`. See
[.env.example](../.env.example).

| Variable | Default | Used by |
|---|---|---|
| `GEMINI_API_KEY` | — | backend (required unless `MOCK_LLM=true`) |
| `MOCK_LLM` | `false` | backend |
| `VISION_WS_URL` | `ws://localhost:8000/ws` | backend |
| `VISION_RECONNECT_DELAY` | `2.0` | backend |
| `VISION_HOST` | `127.0.0.1` | vision_service |
| `VISION_PORT` | `8000` | vision_service |

The backend's own host and port are uvicorn CLI flags, not environment
variables.

---

## Known limitations

- **No authentication anywhere.** Both services trust every caller. They bind
  localhost by default for that reason.
- **CORS is fully open** on both services.
- **No durable state in the backend.** Cards, answers, and analyses live in
  process memory and vanish on restart. Only `embeddings.json` persists.
- **Retrieval is keyword overlap**, so paraphrases can miss.
- **No retention policy** on stored biometrics. See [PRIVACY.md](../PRIVACY.md).
