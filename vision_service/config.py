from pathlib import Path
from dataclasses import dataclass, field

_BASE = Path(__file__).parent


@dataclass
class Settings:
    # Camera
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # Face recognition
    face_model: str = "buffalo_sc"       # small+fast; swap to "buffalo_l" for accuracy
    face_det_size: tuple = field(default_factory=lambda: (320, 320))
    face_threshold: float = 0.45         # cosine similarity cutoff

    # Object detection
    yolo_model: str = "yolov8n.pt"       # base COCO model — general objects
    object_confidence: float = 0.50
    # After running train/train.py, set this to the output weights path.
    # The custom model overlays on top of the base model for the specific
    # demo classes; everything else still comes from the base model.
    custom_yolo_model: str = ""          # e.g. "train/runs/demo_objects/weights/best.pt"

    # Performance
    frame_skip: int = 3                  # run inference on every Nth camera frame
    ws_interval: float = 1.0            # seconds between WebSocket broadcasts

    # Storage
    data_dir: Path = field(default_factory=lambda: _BASE / "storage" / "data")
    embeddings_file: Path = field(
        default_factory=lambda: _BASE / "storage" / "data" / "embeddings.json"
    )

    # Logging
    log_level: str = "INFO"

    # ── UI / Monitoring toggles ────────────────────────────────────────
    # Set either to False here to disable without touching the server code.
    enable_ui: bool = True           # OpenCV window — live annotated camera feed
    enable_terminal_log: bool = True  # print recognition results to the terminal

    @staticmethod
    def onnx_providers() -> list[str]:
        # CoreML is incompatible with InsightFace SCRFD's dynamic output shapes
        # (shape-rank mismatch at runtime). CPU is reliable; YOLO still uses MPS
        # via PyTorch separately, so Apple Silicon acceleration is not lost.
        return ["CPUExecutionProvider"]


settings = Settings()
