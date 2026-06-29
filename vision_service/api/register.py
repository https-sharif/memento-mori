"""
POST /register          — register a new person from uploaded images.
POST /register/capture  — register from live webcam (server-side capture).
DELETE /people/{name}   — remove a person from the store.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from utils.image import bytes_to_bgr, resize_if_larger

logger = logging.getLogger(__name__)
router = APIRouter()

_face_recognizer = None
_camera = None
_embedding_store = None


def init(face_rec, cam, store):
    global _face_recognizer, _camera, _embedding_store
    _face_recognizer = face_rec
    _camera = cam
    _embedding_store = store


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("/register")
async def register_from_upload(
    name: str = Form(...),
    relationship: str = Form(...),
    note: str = Form(""),
    images: List[UploadFile] = File(...),
):
    """
    Register a new person from one or more uploaded images.

    Each image should clearly show a frontal face.
    Recommended: 3–5 images from slightly different angles.
    """
    _require_ready()

    embeddings = []
    failed = 0
    for upload in images:
        raw = await upload.read()
        frame = bytes_to_bgr(raw)
        if frame is None:
            failed += 1
            continue
        frame = resize_if_larger(frame, max_dim=640)
        emb = _face_recognizer.extract_embedding(frame)
        if emb is None:
            failed += 1
            logger.warning("No face detected in one of the uploaded images for '%s'.", name)
            continue
        embeddings.append(emb)

    if not embeddings:
        raise HTTPException(
            status_code=422,
            detail=f"No face detected in any of the {len(images)} image(s). "
                   "Please use clear, well-lit frontal face photos.",
        )

    _embedding_store.add_embeddings(name, relationship, embeddings, note=note)

    return {
        "success": True,
        "name": name,
        "relationship": relationship,
        "embeddings_stored": len(embeddings),
        "images_failed": failed,
    }


@router.post("/register/capture")
async def register_from_webcam(
    name: str = Form(...),
    relationship: str = Form(...),
    note: str = Form(""),
    count: int = Form(5),
    interval: float = Form(0.6),
):
    """
    Register a new person by capturing frames from the live webcam.

    `count`    — number of frames to capture (default 5).
    `interval` — seconds between captures (default 0.6 s).
    """
    _require_ready()

    if _camera is None or not _camera.is_available:
        raise HTTPException(status_code=503, detail="Camera not available.")

    count = max(1, min(count, 10))   # clamp to [1, 10]

    logger.info("Capturing %d frames for '%s' …", count, name)
    frames = _camera.capture_n_frames(n=count, interval=interval)

    embeddings = []
    for frame in frames:
        frame = resize_if_larger(frame, max_dim=640)
        emb = _face_recognizer.extract_embedding(frame)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        raise HTTPException(
            status_code=422,
            detail="No face detected in any captured frame. "
                   "Ensure the face is clearly visible and well-lit.",
        )

    _embedding_store.add_embeddings(name, relationship, embeddings, note=note)

    return {
        "success": True,
        "name": name,
        "relationship": relationship,
        "frames_captured": len(frames),
        "embeddings_stored": len(embeddings),
    }


@router.delete("/people/{name}")
async def delete_person(name: str):
    """Remove a registered person and all their embeddings."""
    _require_ready()
    deleted = _embedding_store.delete_person(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No person named '{name}' found.")
    return {"success": True, "deleted": name}


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------

def _require_ready():
    if _face_recognizer is None or _embedding_store is None:
        raise HTTPException(status_code=503, detail="Service not ready.")
