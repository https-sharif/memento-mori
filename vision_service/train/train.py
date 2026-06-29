"""
Fine-tune YOLOv8 nano on the custom demo dataset.

Run from vision_service/train/:
    python train.py

Or from vision_service/:
    python train/train.py

Output:  train/runs/<run_name>/weights/best.pt
Then set in config.py:
    custom_yolo_model: str = "train/runs/<run_name>/weights/best.pt"
"""
import sys
from pathlib import Path

# ── Settings ──────────────────────────────────────────────────────────────────
BASE_WEIGHTS  = "yolov8n.pt"   # starting checkpoint (downloads if missing)
DATASET_YAML  = Path(__file__).parent / "dataset.yaml"
RUNS_DIR      = Path(__file__).parent / "runs"
RUN_NAME      = "demo_objects"

EPOCHS        = 60             # enough for a small dataset; raise to 100 if time permits
IMAGE_SIZE    = 640
BATCH_SIZE    = 16             # lower to 8 if you get OOM on CPU-only
PATIENCE      = 15             # early-stop if no improvement for N epochs

# ── Sanity checks ─────────────────────────────────────────────────────────────
def _check_data():
    train_img = DATASET_YAML.parent / "data" / "images" / "train"
    val_img   = DATASET_YAML.parent / "data" / "images" / "val"
    train_lbl = DATASET_YAML.parent / "data" / "labels" / "train"
    val_lbl   = DATASET_YAML.parent / "data" / "labels" / "val"

    errors = []
    for p in (train_img, val_img, train_lbl, val_lbl):
        imgs = list(p.glob("*.jpg")) + list(p.glob("*.png")) if "image" in str(p) \
               else list(p.glob("*.txt"))
        if not imgs:
            errors.append(f"  EMPTY: {p}")

    if errors:
        print("\n[ERROR] Missing or empty data directories:")
        for e in errors:
            print(e)
        print("\nRun collect_images.py first, label the images, then come back.")
        sys.exit(1)

    n_train = len(list(train_img.glob("*.jpg")) + list(train_img.glob("*.png")))
    n_val   = len(list(val_img.glob("*.jpg"))   + list(val_img.glob("*.png")))
    print(f"  Training images : {n_train}")
    print(f"  Validation images: {n_val}")
    if n_train < 20:
        print("[WARN] Very few training images — aim for 40+ per class for reliable results.")


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Vision Service — Custom Object Detector Training")
    print("=" * 60)

    _check_data()

    from ultralytics import YOLO
    import torch

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device  : {device}")
    print(f"  Weights : {BASE_WEIGHTS}")
    print(f"  Dataset : {DATASET_YAML}")
    print(f"  Epochs  : {EPOCHS}  (patience={PATIENCE})")
    print()

    model = YOLO(BASE_WEIGHTS)

    results = model.train(
        data=str(DATASET_YAML.resolve()),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=device,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,        # overwrite previous run with same name
        verbose=True,
        # Augmentation tweaks for small datasets
        degrees=15.0,         # random rotation ±15°
        flipud=0.0,           # don't flip upside down (objects have fixed orientation)
        fliplr=0.5,           # random horizontal flip
        mosaic=0.8,           # mosaic augmentation (helps small datasets)
        mixup=0.1,
    )

    best = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best weights: {best}")
    print()
    print("  Next step — update config.py:")
    print(f'    custom_yolo_model: str = "train/runs/{RUN_NAME}/weights/best.pt"')
    print("=" * 60)


if __name__ == "__main__":
    train()
