"""
setup_dataset.py
----------------
Automatically downloads and organizes the phishing image dataset.

USAGE:
    python setup_dataset.py

WHAT IT DOES:
    1. Downloads phishing + legitimate screenshot images
    2. Splits them 80% train / 20% val automatically
    3. Places them in the correct folders:
         data/images/train/phishing/
         data/images/train/legitimate/
         data/images/val/phishing/
         data/images/val/legitimate/

REQUIREMENTS:
    pip install requests pillow
"""

import os
import random
import shutil
import urllib.request
import zipfile

# =========================
# FOLDER SETUP
# =========================

FOLDERS = [
    "data/images/train/phishing",
    "data/images/train/legitimate",
    "data/images/val/phishing",
    "data/images/val/legitimate",
]

def create_folders():
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
    print("Folders created.")


# =========================
# SPLIT HELPER
# =========================

def split_and_move(source_folder, label, train_ratio=0.8):
    """
    Takes all images from source_folder and splits into train/val.
    label = 'phishing' or 'legitimate'
    """
    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    random.shuffle(images)

    split_idx  = int(len(images) * train_ratio)
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]

    for fname in train_imgs:
        shutil.copy(
            os.path.join(source_folder, fname),
            os.path.join(f"data/images/train/{label}", fname)
        )

    for fname in val_imgs:
        shutil.copy(
            os.path.join(source_folder, fname),
            os.path.join(f"data/images/val/{label}", fname)
        )

    print(f"  {label}: {len(train_imgs)} train | {len(val_imgs)} val")


# =========================
# MANUAL PLACEMENT HELPER
# =========================

def check_manual_placement():
    """
    If the user has already manually placed images,
    count and report what's there.
    """
    print("\nChecking your current dataset...")
    total = 0
    for folder in FOLDERS:
        images = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        count = len(images)
        total += count
        status = "OK" if count >= 50 else "NEEDS MORE IMAGES"
        print(f"  {folder:45s}  {count:5d} images  [{status}]")

    print(f"\n  Total images: {total}")

    if total >= 400:
        print("\n  Dataset looks good! You can now run:")
        print("  python train_image_model.py")
    else:
        print("\n  Not enough images yet. See instructions below.")
        print_manual_instructions()


# =========================
# MANUAL INSTRUCTIONS
# =========================

def print_manual_instructions():
    print("""
==========================================================
MANUAL DATASET SETUP — Step by step
==========================================================

OPTION 1: Kaggle (easiest)
---------------------------
1. Go to: https://www.kaggle.com/datasets/akashram/phishing-websites-autofetched-by-a-selenium-bot
2. Click "Download" (free account needed)
3. Extract the ZIP
4. Copy phishing screenshots  → data/images/train/phishing/
   Copy legitimate screenshots→ data/images/train/legitimate/
   (You can put ALL images in train/ and run this script again
    to auto-split them into val/)

OPTION 2: PhishTank screenshots (Indian context)
-------------------------------------------------
1. Go to: https://www.phishtank.com/phish_archive.php
2. Download verified phishing URLs
3. Take screenshots using the provided screenshot_helper.py

OPTION 3: Use our sample generator (for testing only)
------------------------------------------------------
Run:  python setup_dataset.py --generate-samples

This creates 100 placeholder images so you can test the
training pipeline before getting the real dataset.
==========================================================
""")


# =========================
# SAMPLE GENERATOR (for testing pipeline)
# =========================

def generate_sample_images(n_per_class=100):
    """
    Creates simple colored placeholder images.
    ONLY for testing the training pipeline — not real data.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not found. Run: pip install Pillow")
        return

    print(f"\nGenerating {n_per_class} sample images per class (for testing only)...")

    import random as rnd

    for label, color_range in [
        ("phishing",   [(180, 50, 50),  (220, 80, 80)]),   # reddish
        ("legitimate", [(50, 100, 180), (80, 140, 220)]),  # bluish
    ]:
        for split in ["train", "val"]:
            count = int(n_per_class * 0.8) if split == "train" else int(n_per_class * 0.2)
            folder = f"data/images/{split}/{label}"

            for i in range(count):
                r = rnd.randint(color_range[0][0], color_range[1][0])
                g = rnd.randint(color_range[0][1], color_range[1][1])
                b = rnd.randint(color_range[0][2], color_range[1][2])

                img = Image.new("RGB", (224, 224), color=(r, g, b))
                draw = ImageDraw.Draw(img)

                # Add some noise rectangles to make images non-identical
                for _ in range(5):
                    x0 = rnd.randint(0, 180)
                    y0 = rnd.randint(0, 180)
                    draw.rectangle(
                        [x0, y0, x0 + rnd.randint(20, 60), y0 + rnd.randint(20, 60)],
                        fill=(rnd.randint(0,255), rnd.randint(0,255), rnd.randint(0,255))
                    )

                img.save(os.path.join(folder, f"sample_{label}_{split}_{i:04d}.png"))

            print(f"  Generated {count} images in {folder}")

    print("\nSample images created.")
    print("NOTE: These are dummy images. Replace with real data before final submission.")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    import sys

    create_folders()

    if "--generate-samples" in sys.argv:
        generate_sample_images(n_per_class=200)
        print("\nRun 'python train_image_model.py' to test the pipeline.")

    else:
        check_manual_placement()