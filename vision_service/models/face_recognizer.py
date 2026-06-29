"""
Face recognition using InsightFace (ArcFace embeddings) + cosine similarity.

Model choice: "buffalo_sc" — small, fast, good accuracy on frontal faces.
Swap to "buffalo_l" in config.py for better accuracy at the cost of speed.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FaceRecognizer:
    def __init__(
        self,
        model_name: str = "buffalo_sc",
        det_size: tuple = (320, 320),
        providers: Optional[List[str]] = None,
    ) -> None:
        if providers is None:
            providers = ["CPUExecutionProvider"]

        logger.info("Loading InsightFace model '%s' …", model_name)
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(name=model_name, providers=providers)
        self._app.prepare(ctx_id=0, det_size=det_size)
        self._store = None          # injected after construction
        logger.info("InsightFace ready.")

    # ------------------------------------------------------------------
    # Store injection (avoids circular imports in main.py)
    # ------------------------------------------------------------------

    def set_store(self, store) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def extract_embedding(self, bgr_image: np.ndarray) -> Optional[np.ndarray]:
        """Return ArcFace embedding for the largest detected face, or None."""
        faces = self._app.get(bgr_image)
        if not faces:
            return None
        face = max(faces, key=lambda f: _bbox_area(f.bbox))
        emb = face.embedding.astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-6)   # pre-normalise

    def extract_embeddings_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Extract embeddings from multiple images, skipping frames with no face."""
        results = []
        for img in images:
            emb = self.extract_embedding(img)
            if emb is not None:
                results.append(emb)
        return results

    def recognize(
        self, bgr_image: np.ndarray, threshold: float = 0.45
    ) -> Dict[str, Any]:
        """
        Run face detection + recognition on a single frame.

        Returns a dict compatible with the /recognize response schema.
        """
        faces = self._app.get(bgr_image)

        if not faces:
            return _no_face_result()

        face = max(faces, key=lambda f: _bbox_area(f.bbox))
        face_bbox = [round(float(x), 1) for x in face.bbox]   # [x1, y1, x2, y2]

        emb = face.embedding.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-6)

        if self._store is None or len(self._store) == 0:
            return _unknown_result(confidence=0.0, face_bbox=face_bbox)

        all_people = self._store.get_all().values()

        # ── Pass 1: averaged embedding — O(n_people) ──────────────────
        best_score: float = -1.0
        best_data: Optional[dict] = None

        for person_data in all_people:
            avg_emb = person_data.get("avg_embedding")
            if avg_emb is None:
                continue
            score = float(np.dot(emb, avg_emb))
            if score > best_score:
                best_score = score
                best_data = person_data

        if best_score >= threshold and best_data is not None:
            return _recognized(best_data, best_score, face_bbox)

        # ── Pass 2: individual embeddings — O(n_people × n_embeddings) ─
        # Only reached when no avg match — catches edge-case poses that
        # pulled the centroid away from the query.
        for person_data in all_people:
            for stored_emb in person_data["embeddings"]:
                se = stored_emb.astype(np.float32)
                se = se / (np.linalg.norm(se) + 1e-6)
                score = float(np.dot(emb, se))
                if score > best_score:
                    best_score = score
                    best_data = person_data

        if best_score >= threshold and best_data is not None:
            return _recognized(best_data, best_score, face_bbox)

        return _unknown_result(confidence=round(best_score, 4), face_bbox=face_bbox)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _bbox_area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _recognized(person_data: dict, score: float, face_bbox) -> Dict[str, Any]:
    return {
        "recognized": True,
        "name": person_data["name"],
        "relationship": person_data.get("relationship", ""),
        "note": person_data.get("note", ""),
        "confidence": round(score, 4),
        "face_detected": True,
        "face_bbox": face_bbox,
    }


def _no_face_result() -> Dict[str, Any]:
    return {
        "recognized": False,
        "name": None,
        "relationship": None,
        "note": None,
        "confidence": 0.0,
        "face_detected": False,
        "face_bbox": None,
    }


def _unknown_result(confidence: float, face_bbox=None) -> Dict[str, Any]:
    return {
        "recognized": False,
        "name": "Unknown",
        "relationship": None,
        "note": None,
        "confidence": confidence,
        "face_detected": True,
        "face_bbox": face_bbox,
    }
