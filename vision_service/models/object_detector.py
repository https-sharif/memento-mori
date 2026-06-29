"""
Object detection using YOLOv8.

Two-model design:
  base model  — yolov8n/s pretrained on COCO, filtered to care-relevant classes
  custom model — fine-tuned on demo-specific classes (optional, set in config.py)

When the custom model is loaded, its detections take priority: any base model
detection that overlaps a custom detection by ≥40% IoU is suppressed.
This means you keep all general COCO objects (chair, bed, tv …) while getting
accurate, demo-specific detection for medicine bottles, glasses, etc.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# COCO class name → care-friendly display label.
# Only classes in this dict are reported from the BASE model.
# COCO lacks "glasses" and "walking_cane" — those come from the custom model.
CARE_OBJECTS: Dict[str, str] = {
    "bottle": "Bottle",          # medicine OR water — custom model will refine this
    "cup": "Cup or Mug",
    "bowl": "Bowl",
    "book": "Book",
    "clock": "Clock",
    "cell phone": "Phone",
    "remote": "TV Remote",
    "scissors": "Scissors",
    "toothbrush": "Toothbrush",
    "vase": "Vase",
    "handbag": "Bag or Handbag",
    "backpack": "Backpack",
    "umbrella": "Umbrella",
    "laptop": "Laptop",
    "keyboard": "Keyboard",
    "mouse": "Computer Mouse",
    "tv": "Television",
    "bed": "Bed",
    "chair": "Chair",
    "couch": "Couch or Sofa",
    "dining table": "Dining Table",
    "potted plant": "Plant",
    "banana": "Fruit (Banana)",
    "apple": "Fruit (Apple)",
    "orange": "Fruit (Orange)",
    "pizza": "Food",
    "sandwich": "Food",
}

# Pretty-print transform for custom model class names.
# Keys are the class names you used in dataset.yaml — values are display labels.
# If a name isn't listed here, it's auto-formatted: "medicine_bottle" → "Medicine Bottle".
CUSTOM_LABELS: Dict[str, str] = {
    "medicine_bottle": "Medicine Bottle",
    "water_bottle": "Water Bottle",
    "glasses": "Eyeglasses",
    "walking_cane": "Walking Cane",
    "keys": "Keys",
    "book": "Book",
    "phone": "Phone",
    "laptop": "Laptop",
    "cup": "Cup or Mug",
}


class ObjectDetector:
    def __init__(
        self,
        base_model_path: str = "yolov8n.pt",
        custom_model_path: str = "",
        confidence: float = 0.50,
    ) -> None:
        from ultralytics import YOLO

        logger.info("Loading YOLO base model '%s' …", base_model_path)
        self._base = YOLO(base_model_path)
        self._conf = confidence

        self._custom: Optional[Any] = None
        if custom_model_path and Path(custom_model_path).exists():
            logger.info("Loading custom YOLO model '%s' …", custom_model_path)
            self._custom = YOLO(custom_model_path)
            logger.info("Custom model ready — will override base for demo classes.")
        elif custom_model_path:
            logger.warning(
                "custom_yolo_model path '%s' does not exist — using base model only.",
                custom_model_path,
            )

        logger.info("Object detector ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, bgr_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection and return care-relevant objects sorted by confidence.

        Each item: { "label": str, "confidence": float, "bbox": [x1,y1,x2,y2],
                     "coco_class": str }
        """
        device = "mps" if _mps_available() else "cpu"

        base_dets = self._detect_base(bgr_image, device)

        if self._custom is None:
            return base_dets

        custom_dets = self._detect_custom(bgr_image, device)
        return _merge(base_dets, custom_dets, iou_threshold=0.40)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_base(self, image: np.ndarray, device: str) -> List[Dict[str, Any]]:
        results = self._base(image, verbose=False, conf=self._conf, device=device)
        dets: List[Dict[str, Any]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                coco_label = r.names[int(box.cls[0])]
                display = CARE_OBJECTS.get(coco_label)
                if display is None:
                    continue
                dets.append({
                    "label": display,
                    "coco_class": coco_label,
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                    "source": "base",
                })
        return _dedup(dets)

    def _detect_custom(self, image: np.ndarray, device: str) -> List[Dict[str, Any]]:
        results = self._custom(image, verbose=False, conf=self._conf, device=device)
        dets: List[Dict[str, Any]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                raw_name = r.names[int(box.cls[0])]
                display = CUSTOM_LABELS.get(raw_name) or raw_name.replace("_", " ").title()
                dets.append({
                    "label": display,
                    "coco_class": raw_name,
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                    "source": "custom",
                })
        return _dedup(dets)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _dedup(dets: List[Dict]) -> List[Dict]:
    """Keep only the highest-confidence detection per display label."""
    seen: Dict[str, Dict] = {}
    for d in sorted(dets, key=lambda x: x["confidence"], reverse=True):
        if d["label"] not in seen:
            seen[d["label"]] = d
    return list(seen.values())


def _merge(
    base: List[Dict], custom: List[Dict], iou_threshold: float = 0.40
) -> List[Dict]:
    """
    Custom detections take priority.
    Suppress any base detection whose box overlaps a custom detection ≥ iou_threshold.
    """
    if not custom:
        return base

    kept_base = [
        b for b in base
        if not any(_iou(b["bbox"], c["bbox"]) >= iou_threshold for c in custom)
    ]
    return custom + kept_base


def _iou(b1: List[float], b2: List[float]) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False
