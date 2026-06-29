"""
Webcam image collection tool for building training data.

Usage:
    python collect_images.py medicine_bottle          # capture for one class
    python collect_images.py medicine_bottle --count 60

Controls (OpenCV window):
    SPACE  — save current frame
    Q      — quit / next class

Images are saved to:
    train/data/images/train/<class_name>/<class_name>_0001.jpg ...
"""
import argparse
import sys
from pathlib import Path

import cv2

SAVE_ROOT = Path(__file__).parent / "data" / "images" / "train"


def collect(class_name: str, target: int = 60, cam_index: int = 0) -> None:
    save_dir = SAVE_ROOT / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Count existing images so we don't overwrite
    existing = len(list(save_dir.glob("*.jpg")))
    captured = 0

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_index}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\nCollecting '{class_name}'  (already have {existing}, target {target} new)")
    print("  SPACE = capture   Q = quit\n")

    while captured < target:
        ret, frame = cap.read()
        if not ret:
            continue

        remaining = target - captured
        display = frame.copy()

        # HUD
        cv2.rectangle(display, (0, 0), (640, 48), (20, 20, 20), -1)
        cv2.putText(display, f"Class: {class_name}   {captured}/{target}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 80), 2)
        cv2.putText(display, "SPACE=save  Q=quit",
                    (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Collect Images", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            idx = existing + captured + 1
            fname = save_dir / f"{class_name}_{idx:04d}.jpg"
            cv2.imwrite(str(fname), frame)
            captured += 1
            print(f"  Saved {fname.name}  ({captured}/{target})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {captured} new images saved to {save_dir}")
    print(f"Total in folder: {existing + captured}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam image collection tool")
    parser.add_argument("class_name", help="Class name (must match dataset.yaml)")
    parser.add_argument("--count", type=int, default=60, help="Target number of images (default 60)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()
    collect(args.class_name, target=args.count, cam_index=args.camera)
