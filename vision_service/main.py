"""
Vision Service — entry point.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Or via the helper script:
    python main.py

Toggle UI / terminal logging in config.py:
    enable_ui           = True / False
    enable_terminal_log = True / False
"""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from models.face_recognizer import FaceRecognizer
from models.object_detector import ObjectDetector
from storage.embedding_store import EmbeddingStore
from utils.camera import CameraCapture
from utils.image import resize_if_larger

import api.recognize as rec_api
import api.register as reg_api
from api.websocket_handler import manager, router as ws_router
from api.recognize import router as rec_router
from api.register import router as reg_router

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
_vlog  = logging.getLogger("VISION")   # dedicated logger for recognition results

# Thread pool for CPU-bound inference
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision")

# Singletons — populated in lifespan
face_recognizer: FaceRecognizer | None = None
object_detector: ObjectDetector | None = None
embedding_store: EmbeddingStore | None = None
camera: CameraCapture | None = None

# Latest raw result (includes bboxes) — used by the display
_last_raw: dict = {}


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def _run_inference(frame: np.ndarray) -> dict:
    """Runs in a thread-pool worker. Returns full result including bboxes."""
    person = face_recognizer.recognize(frame, threshold=settings.face_threshold)
    objects_raw = object_detector.detect(frame)   # already carries bbox per item
    return {
        "person": person,
        "objects": objects_raw,
        "timestamp": int(time.time()),
    }


def _clean_result(raw: dict) -> dict:
    """Strip internal fields (face_bbox, coco_class, bbox) before sending externally."""
    person = {k: v for k, v in raw["person"].items() if k != "face_bbox"}
    objects = [
        {"label": o["label"], "confidence": o["confidence"]}
        for o in raw["objects"]
    ]
    return {"person": person, "objects": objects, "timestamp": raw["timestamp"]}


# ------------------------------------------------------------------
# Terminal logging
# ------------------------------------------------------------------

def _log_result(result: dict) -> None:
    person  = result.get("person", {})
    objects = result.get("objects", [])

    if person.get("face_detected"):
        name = person.get("name") or "Unknown"
        rel  = person.get("relationship") or ""
        conf = person.get("confidence", 0.0)
        if person.get("recognized"):
            p_str = f"{name} ({rel})  [{conf:.2f}]" if rel else f"{name}  [{conf:.2f}]"
        else:
            p_str = f"Unknown  [{conf:.2f}]"
    else:
        p_str = "—  (no face)"

    o_str = "  ·  ".join(o["label"] for o in objects) if objects else "none"
    _vlog.info("%-42s | %s", p_str, o_str)


# ------------------------------------------------------------------
# OpenCV annotation
# ------------------------------------------------------------------

_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_GREEN  = (50, 205,  50)    # BGR — recognised person
_ORANGE = (30, 145, 255)    # BGR — unknown person
_CYAN   = (200, 210,  30)   # BGR — objects
_WHITE  = (235, 235, 235)
_DARK   = ( 20,  20,  20)
_GRAY   = (130, 130, 130)


def _annotate_frame(frame: np.ndarray, result: dict) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    person  = result.get("person", {})
    objects = result.get("objects", [])

    # ── Face bounding box ──────────────────────────────────────────
    bbox = person.get("face_bbox")
    if bbox:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        color = _GREEN if person.get("recognized") else _ORANGE
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = person.get("name") or "Unknown"
        (tw, th), _ = cv2.getTextSize(name, _FONT, 0.55, 1)
        ty = max(th + 8, y1 - 2)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty), color, -1)
        cv2.putText(out, name, (x1 + 4, ty - 4), _FONT, 0.55, _DARK, 1)

    # ── Object bounding boxes ──────────────────────────────────────
    for obj in objects:
        ob = obj.get("bbox")
        if not ob:
            continue
        x1, y1, x2, y2 = (int(v) for v in ob)
        cv2.rectangle(out, (x1, y1), (x2, y2), _CYAN, 2)
        cv2.putText(out, obj["label"], (x1 + 2, max(14, y1 - 5)), _FONT, 0.42, _CYAN, 1)

    # ── Top status bar ─────────────────────────────────────────────
    cv2.rectangle(out, (0, 0), (w, 50), _DARK, -1)
    if person.get("face_detected"):
        name  = person.get("name") or "Unknown"
        rel   = person.get("relationship") or ""
        conf  = person.get("confidence", 0.0)
        pclr  = _GREEN if person.get("recognized") else _ORANGE
        ptxt  = f"{name}"
        if rel:
            ptxt += f"  |  {rel}"
        ptxt += f"  ({conf:.2f})"
    else:
        ptxt, pclr = "No face detected", _GRAY
    cv2.putText(out, ptxt, (12, 33), _FONT, 0.7, pclr, 2)

    # ── Bottom object bar ──────────────────────────────────────────
    if objects:
        cv2.rectangle(out, (0, h - 34), (w, h), _DARK, -1)
        otxt = "  ·  ".join(o["label"] for o in objects[:6])
        cv2.putText(out, otxt, (10, h - 11), _FONT, 0.45, _WHITE, 1)

    return out


# ------------------------------------------------------------------
# Background vision loop
# ------------------------------------------------------------------

_DISPLAY_TICK = 0.10   # seconds between UI refreshes (~10 fps)


async def _vision_loop() -> None:
    """
    Runs on the asyncio event loop (= main thread on macOS), which is the
    only thread allowed to call cv2.imshow on macOS.

    Display refresh:  every _DISPLAY_TICK  (~10 fps, smooth video)
    Inference:        every settings.ws_interval  (default 1 s)
    """
    loop = asyncio.get_event_loop()
    last_infer_t: float = 0.0

    logger.info(
        "Vision loop started  (display=%.0f fps, inference=%.1f s interval).",
        1 / _DISPLAY_TICK,
        settings.ws_interval,
    )

    while True:
        try:
            await asyncio.sleep(_DISPLAY_TICK)

            if camera is None or not camera.is_available:
                if settings.enable_ui:
                    cv2.waitKey(1)
                continue

            frame = camera.read()
            if frame is None:
                if settings.enable_ui:
                    cv2.waitKey(1)
                continue

            # ── Inference (time-gated) ─────────────────────────────
            now = time.monotonic()
            if now - last_infer_t >= settings.ws_interval:
                resized = resize_if_larger(frame, max_dim=640)
                raw = await loop.run_in_executor(_executor, _run_inference, resized)
                _last_raw.clear()
                _last_raw.update(raw)
                last_infer_t = now

                clean = _clean_result(raw)
                if settings.enable_terminal_log:
                    _log_result(clean)

                rec_api.update_latest(clean)
                if manager.num_clients > 0:
                    await manager.broadcast(clean)

            # ── Display (every tick) ───────────────────────────────
            if settings.enable_ui and _last_raw:
                annotated = _annotate_frame(frame, _last_raw)
                cv2.imshow("Vision Service", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    settings.enable_ui = False
                    cv2.destroyAllWindows()
                    logger.info("UI closed — set enable_ui=True in config.py to re-open.")

        except asyncio.CancelledError:
            logger.info("Vision loop cancelled.")
            break
        except Exception:
            logger.exception("Error in vision loop — continuing.")

    if settings.enable_ui:
        cv2.destroyAllWindows()


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_recognizer, object_detector, embedding_store, camera

    logger.info("=== Vision Service starting up ===")

    # Storage
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    embedding_store = EmbeddingStore(settings.embeddings_file)

    # Models
    face_recognizer = FaceRecognizer(
        model_name=settings.face_model,
        det_size=settings.face_det_size,
        providers=settings.onnx_providers(),
    )
    face_recognizer.set_store(embedding_store)

    object_detector = ObjectDetector(
        base_model_path=settings.yolo_model,
        custom_model_path=settings.custom_yolo_model,
        confidence=settings.object_confidence,
    )

    # Camera
    camera = CameraCapture(
        index=settings.camera_index,
        width=settings.frame_width,
        height=settings.frame_height,
    ).start()

    # Inject into routers
    rec_api.init(face_recognizer, object_detector, camera, embedding_store)
    reg_api.init(face_recognizer, camera, embedding_store)

    # Background loop
    _loop_task = asyncio.create_task(_vision_loop())

    logger.info("=== Vision Service ready — http://localhost:8000 ===")
    logger.info("    Registration UI:  http://localhost:8000/static/register.html")
    logger.info("    API docs:          http://localhost:8000/docs")
    logger.info("    UI window:         %s", "ON  (press q to close)" if settings.enable_ui else "OFF")
    logger.info("    Terminal logging:  %s", "ON" if settings.enable_terminal_log else "OFF")

    yield

    _loop_task.cancel()
    camera.stop()
    _executor.shutdown(wait=False)
    logger.info("=== Vision Service shut down ===")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="Vision Service",
    description=(
        "Real-time face recognition and object detection for dementia assistants. "
        "POST /recognize for one-shot inference; connect to /ws for live streaming."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rec_router, tags=["Recognition"])
app.include_router(reg_router, tags=["Registration"])
app.include_router(ws_router, tags=["WebSocket"])
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
