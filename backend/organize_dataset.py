import os
import shutil
import random

# ── CONFIG ──────────────────────────────────────────────────────────────
KAGGLE_PHISHING = r"C:\Users\Dell\Downloads\archive (4)\dataset\phishing ss"
KAGGLE_LEGIT    = r"C:\Users\Dell\Downloads\archive (4)\dataset\legit ss"

# Destination folders
TRAIN_PHISHING  = r"data\images\train\phishing"
TRAIN_LEGIT     = r"data\images\train\legitimate"
VAL_PHISHING    = r"data\images\val\phishing"
VAL_LEGIT       = r"data\images\val\legitimate"

TRAIN_SPLIT     = 0.8
IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGES      = 200
# ────────────────────────────────────────────────────────────────────────


def clear_folder(folder):
    """Delete old images so dataset is fresh"""
    if os.path.exists(folder):
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            if os.path.isfile(path):
                os.remove(path)


def collect_images(folder):
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                images.append(os.path.join(root, f))
    return images


def copy_split(images, train_dst, val_dst, label):
    random.shuffle(images)
    images = images[:MAX_IMAGES]

    split = int(len(images) * TRAIN_SPLIT)
    train_imgs = images[:split]
    val_imgs   = images[split:]

    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(val_dst, exist_ok=True)

    # 🔥 clear old data
    clear_folder(train_dst)
    clear_folder(val_dst)

    # Copy train
    for i, src in enumerate(train_imgs):
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(train_dst, f"{label}_{i:04d}{ext}")
        shutil.copy2(src, dst)

    # Copy val
    for i, src in enumerate(val_imgs):
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(val_dst, f"{label}_{i:04d}{ext}")
        shutil.copy2(src, dst)

    print(f"  [{label.upper()}] {len(train_imgs)} → train | {len(val_imgs)} → val")


def main():
    random.seed(42)

    print("\n🔍 Scanning source folders...")

    phishing_imgs = collect_images(KAGGLE_PHISHING)
    legit_imgs    = collect_images(KAGGLE_LEGIT)

    print(f"  Found {len(phishing_imgs)} phishing images")
    print(f"  Found {len(legit_imgs)} legitimate images")

    if not phishing_imgs:
        print("❌ No phishing images found — check KAGGLE_PHISHING path")
        return

    if not legit_imgs:
        print("❌ No legit images found — check KAGGLE_LEGIT path")
        return

    print("\n📁 Copying and splitting dataset...")

    copy_split(phishing_imgs, TRAIN_PHISHING, VAL_PHISHING, "phishing")
    copy_split(legit_imgs,    TRAIN_LEGIT,    VAL_LEGIT,    "legitimate")

    print("\n✅ Dataset prepared successfully!\n")

    for folder in [TRAIN_PHISHING, TRAIN_LEGIT, VAL_PHISHING, VAL_LEGIT]:
        count = len(os.listdir(folder)) if os.path.exists(folder) else 0
        print(f"  {folder} → {count} images")

    print("\n▶ Next step: python train_image_model.py")


if __name__ == "__main__":
    main()