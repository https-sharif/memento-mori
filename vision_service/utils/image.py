"""
Lightweight image helpers — keep this thin; do preprocessing close to the model.
"""
import io
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def bytes_to_bgr(data: bytes) -> Optional[np.ndarray]:
    """Decode image bytes (JPEG/PNG/etc.) to a BGR numpy array."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img  # None if decode failed


def bgr_to_bytes(frame: np.ndarray, ext: str = ".jpg", quality: int = 85) -> bytes:
    """Encode a BGR frame to JPEG (or other) bytes."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    _, buf = cv2.imencode(ext, frame, params)
    return buf.tobytes()


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def resize_if_larger(frame: np.ndarray, max_dim: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))
