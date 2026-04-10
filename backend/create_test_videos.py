"""
create_test_videos.py  (fixed)
------------------------------
Rebuilds both test videos from CORRECT image folders:
  phishing_test.mp4   ← from data/images/train/phishing/
  legitimate_test.mp4 ← from data/images/train/legitimate/

Run:
    python create_test_videos.py
"""

import cv2
import os
import numpy as np
from PIL import Image

IMAGE_SIZE        = (640, 480)
FPS               = 2
SECONDS_PER_IMAGE = 2
OUTPUT_DIR        = "data/test_videos"
MAX_IMAGES        = 15

SOURCES = {
    "phishing":   "data/images/train/phishing",
    "legitimate": "data/images/train/legitimate",
}


def build_video(image_folder, output_path, label):

    if not os.path.isdir(image_folder):
        print(f"  ERROR: Folder not found: {image_folder}")
        return False

    images = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:MAX_IMAGES]

    if not images:
        print(f"  ERROR: No images in {image_folder}")
        return False

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        IMAGE_SIZE
    )

    color = (0, 0, 255) if label == "phishing" else (0, 200, 0)

    for fname in images:
        try:
            img   = Image.open(os.path.join(image_folder, fname)).convert("RGB")
            img   = img.resize(IMAGE_SIZE, Image.LANCZOS)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f"{label.upper()} - {fname}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            for _ in range(FPS * SECONDS_PER_IMAGE):
                writer.write(frame)
        except Exception as e:
            print(f"  Skipped {fname}: {e}")

    writer.release()
    print(f"  Saved: {output_path}  ({len(images)} images)")
    return True


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Rebuilding test videos from correct folders...\n")
    for label, folder in SOURCES.items():
        print(f"Building {label}_test.mp4 from: {folder}")
        build_video(
            folder,
            os.path.join(OUTPUT_DIR, f"{label}_test.mp4"),
            label
        )
    print("\nDone. Run: python evaluate.py")