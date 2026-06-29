# Vision Service

Real-time face recognition + object detection for the dementia assistant hackathon.  
Runs fully offline on a MacBook (CPU; Apple Silicon MPS/CoreML used automatically when available).

---

## Quick Start

```bash
# 1. Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies (models auto-download on first run)
pip install -r requirements.txt

# 3. Start the server
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- **Registration UI**: http://localhost:8000/static/register.html  
- **API docs** (Swagger): http://localhost:8000/docs  
- **WebSocket stream**: `ws://localhost:8000/ws`

---

## API Reference

### `POST /recognize`
Recognize a face and detect objects in a single frame.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image`   | file (optional) | JPEG/PNG file. Omit to capture from webcam. |

**Response:**
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
    { "label": "Water/Medicine Bottle", "confidence": 0.91 }
  ],
  "timestamp": 1234567890
}
```

---

### `POST /register`
Register a new person from uploaded images (multipart/form-data).

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Person's name |
| `relationship` | string | e.g. "Daughter", "Nurse" |
| `note` | string (optional) | Free-text note |
| `images` | file[] | 3–5 JPEG/PNG face images |

---

### `POST /register/capture`
Register from the live webcam (no file upload needed).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | — | Person's name |
| `relationship` | string | — | Relationship |
| `note` | string | `""` | Optional note |
| `count` | int | `5` | Number of frames to capture |
| `interval` | float | `0.6` | Seconds between captures |

---

### `GET /people`
List all registered people.

### `DELETE /people/{name}`
Remove a person and their embeddings.

### `GET /health`
Liveness probe.

### `WS /ws`
WebSocket endpoint. Streams recognition results as JSON at ~1 s intervals (same schema as `/recognize`). Sends the cached last result between inference frames so clients always have fresh data.

**JavaScript example:**
```js
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (e) => {
  const result = JSON.parse(e.data);
  console.log(result.person.name, result.objects);
};
```

---

## Architecture

```
vision_service/
├── main.py                  # FastAPI app, lifespan, background vision loop
├── config.py                # All tunable settings (model paths, thresholds, …)
├── models/
│   ├── face_recognizer.py   # InsightFace / ArcFace wrapper
│   └── object_detector.py   # YOLOv8n wrapper with care-object mapping
├── storage/
│   ├── embedding_store.py   # JSON-backed in-memory embedding store
│   └── data/                # embeddings.json stored here (auto-created)
├── utils/
│   ├── camera.py            # Threaded webcam capture
│   └── image.py             # Decode / resize helpers
├── api/
│   ├── recognize.py         # /recognize, /people, /health
│   ├── register.py          # /register, /register/capture, DELETE /people/{name}
│   └── websocket_handler.py # /ws + ConnectionManager
├── static/
│   └── register.html        # Browser registration UI
└── requirements.txt
```

### Vision loop

```
Camera thread (30 fps, daemon)
     │  latest frame (shared, locked)
     ▼
Vision loop (asyncio, every ws_interval seconds)
     │  run_in_executor (thread pool, non-blocking)
     ├─► FaceRecognizer.recognize()   InsightFace ArcFace + cosine sim
     └─► ObjectDetector.detect()      YOLOv8n → care-object filter
          │
          ▼
     cache latest_result
          │
          ▼
     ConnectionManager.broadcast()  → all /ws clients
```

---

## Configuration (`config.py`)

| Setting | Default | Notes |
|---------|---------|-------|
| `camera_index` | `0` | Change if you have multiple cameras |
| `face_model` | `buffalo_sc` | Swap to `buffalo_l` for higher accuracy |
| `face_threshold` | `0.45` | Raise to reduce false positives |
| `yolo_model` | `yolov8n.pt` | `yolov8s.pt` for better detection |
| `object_confidence` | `0.50` | Lower to catch more objects |
| `frame_skip` | `3` | Inference on every 3rd frame |
| `ws_interval` | `1.0` | Seconds between WS broadcasts |

---

## Performance notes

- **Apple Silicon**: CoreML execution provider is automatically selected for InsightFace; YOLOv8 uses MPS via PyTorch. Expect ~20–50 ms/frame total.
- **Intel Mac / CPU-only**: ~100–200 ms/frame. Raise `ws_interval` to `2.0` and `frame_skip` to `5` if needed.
- Embeddings are pre-normalised on write so cosine similarity is a single dot product.
- `frame_skip` decouples the camera rate from inference rate — the camera always reads at full speed; inference only runs every Nth frame.

## Limitations

- COCO (YOLOv8n) does not include "glasses" or "walking cane/stick" classes. A fine-tuned model would be needed for those specific objects.
- Face recognition works best on frontal, well-lit faces. Register 5+ images for best results.
- One face per frame is currently recognised (the largest detected face).
