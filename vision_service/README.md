# Vision Service — Internals

Face recognition and object detection backend. Runs fully offline on a MacBook.

---

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

---

## Folder structure

```
vision_service/
├── main.py                  # app entry point, background vision loop
├── config.py                # all settings in one place
├── models/
│   ├── face_recognizer.py   # InsightFace / ArcFace wrapper
│   └── object_detector.py   # YOLOv8 wrapper, two-model design
├── storage/
│   ├── embedding_store.py   # in-memory store backed by embeddings.json
│   └── data/                # embeddings.json lives here
├── utils/
│   ├── camera.py            # threaded webcam capture
│   └── image.py             # decode / resize helpers
├── api/
│   ├── recognize.py         # POST /recognize, GET /people, GET /health
│   ├── register.py          # POST /register, POST /register/capture, DELETE /people/{name}
│   └── websocket_handler.py # WS /ws + connection manager
├── static/
│   └── register.html        # face registration UI
└── train/                   # custom model training (see Fine-Tune.md inside)
```

---

## How it works

**Face recognition**
1. InsightFace (ArcFace, `buffalo_sc` model) detects faces and extracts a 512-float embedding per face
2. On recognition, embedding is compared against the store using cosine similarity (dot product of normalised vectors)
3. Two-pass lookup: averaged embedding first (fast), then individual embeddings if no match (thorough)
4. Match threshold is `face_threshold` in `config.py` (default 0.45)

**Object detection**
- Base model: `yolov8n.pt` pretrained on COCO — filters to a care-relevant subset (bottles, cups, clocks, etc.)
- Custom model (optional): fine-tuned on demo-specific classes (medicine bottle vs water bottle, glasses, cane)
- When both are loaded, custom detections take priority; base model fills in everything else

**Vision loop**
- Camera thread reads frames at full speed (~30 fps) into a shared buffer
- Async loop samples the latest frame every 100 ms for display, runs inference every `ws_interval` seconds (default 1 s)
- Inference runs in a thread pool so it never blocks the async event loop
- Result is broadcast to all connected WebSocket clients after each inference

---

## Config (`config.py`)

```python
face_model        = "buffalo_sc"   # swap to "buffalo_l" for higher accuracy
face_threshold    = 0.45           # raise to reduce false positives
yolo_model        = "yolov8n.pt"   # swap to "yolov8s.pt" for better detection
custom_yolo_model = ""             # path to fine-tuned weights (see train/)
object_confidence = 0.50
ws_interval       = 1.0            # seconds between WebSocket pushes
enable_ui         = True           # OpenCV window (press q to close)
enable_terminal_log = True         # print results to terminal
```

---

## Embeddings storage

Stored in `storage/data/embeddings.json`. Plain JSON — human-readable, no database needed.

Structure: per person, a list of raw embeddings (preserved for fallback matching) and a pre-computed average embedding (used for fast lookup). Average is recomputed from the raw list on every write and on startup — it's never persisted separately.

To wipe all registered faces: `rm storage/data/embeddings.json`

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/recognize` | Image file or live webcam → recognition result |
| `WS` | `/ws` | Live stream of results |
| `POST` | `/register` | Register from uploaded images |
| `POST` | `/register/capture` | Register from webcam (server-side) |
| `GET` | `/people` | List registered people |
| `DELETE` | `/people/{name}` | Remove a person |
| `GET` | `/health` | Service + camera status |
| `GET` | `/docs` | Swagger UI |
