"""
POST /recognize  — one-shot inference on an uploaded image or a live webcam frame.
GET  /people     — list all registered faces.
GET  /health     — liveness probe.
"""
import logging
import time
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from utils.image import bytes_to_bgr, resize_if_larger

logger = logging.getLogger(__name__)
router = APIRouter()

# These are injected by main.py after models are loaded.
_face_recognizer = None
_object_detector = None
_camera = None
_embedding_store = None
_latest_result: dict = {}


def init(face_rec, obj_det, cam, store):
    global _face_recognizer, _object_detector, _camera, _embedding_store
    _face_recognizer = face_rec
    _object_detector = obj_det
    _camera = cam
    _embedding_store = store


def update_latest(result: dict) -> None:
    global _latest_result
    _latest_result = result


def get_latest() -> dict:
    return _latest_result


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.get("/health")
async def health():
    return {
        "status": "ok",
        "registered_people": len(_embedding_store) if _embedding_store else 0,
        "camera_available": _camera.is_available if _camera else False,
    }


@router.post("/recognize")
async def recognize(image: Optional[UploadFile] = File(None)):
    """
    Recognize faces and objects in an image.

    - Supply an image file via `image` (multipart/form-data).
    - Omit `image` to capture a frame from the live webcam instead.

    Returns:
    ```json
    {
      "person": { "recognized": true, "name": "Sarah", "relationship": "Daughter",
                  "confidence": 0.96 },
      "objects": [{ "label": "Medicine Bottle", "confidence": 0.91 }],
      "timestamp": 1234567890
    }
    ```
    """
    if image is not None:
        data = await image.read()
        frame = bytes_to_bgr(data)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
    else:
        if _camera is None or not _camera.is_available:
            raise HTTPException(status_code=503, detail="Camera not available.")
        frame = _camera.read()
        if frame is None:
            raise HTTPException(status_code=503, detail="No frame available from camera.")

    frame = resize_if_larger(frame, max_dim=640)

    person_result = _face_recognizer.recognize(frame) if _face_recognizer else {}
    object_results = _object_detector.detect(frame) if _object_detector else []

    # Strip internal fields (face_bbox, coco_class, bbox) from public response
    person_clean = {k: v for k, v in person_result.items() if k != "face_bbox"}
    clean_objects = [
        {"label": o["label"], "confidence": o["confidence"]} for o in object_results
    ]

    result = {
        "person": person_clean,
        "objects": clean_objects,
        "timestamp": int(time.time()),
    }
    update_latest(result)
    return JSONResponse(content=result)


@router.get("/people")
async def list_people():
    """List all registered people and their metadata."""
    if _embedding_store is None:
        raise HTTPException(status_code=503, detail="Store not ready.")
    return {"people": _embedding_store.list_people()}
