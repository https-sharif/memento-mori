"""
Threaded webcam capture that always serves the latest frame.
Decouples capture rate from inference rate so inference never blocks reading.
"""
import logging
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "CameraCapture":
        self._cap = cv2.VideoCapture(self._index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise latency

        if not self._cap.isOpened():
            logger.warning("Camera index %d could not be opened.", self._index)
        else:
            logger.info(
                "Camera %d opened at %dx%d", self._index, self._width, self._height
            )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        while self._running:
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame
            else:
                threading.Event().wait(0.1)

    def read(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent frame, or None if not yet available."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def capture_n_frames(self, n: int = 5, interval: float = 0.5) -> list[np.ndarray]:
        """Block and collect n frames spaced ~interval seconds apart."""
        import time
        frames: list[np.ndarray] = []
        for _ in range(n):
            frame = self.read()
            if frame is not None:
                frames.append(frame)
            time.sleep(interval)
        return frames

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("Camera stopped.")

    @property
    def is_available(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
