"""
train_image_model.py
--------------------
Fine-tunes MobileNetV2 on phishing vs. legitimate website screenshots.

Expected folder layout:
    data/images/train/phishing/     ← phishing screenshots
    data/images/train/legitimate/   ← real site screenshots
    data/images/val/phishing/
    data/images/val/legitimate/

Run:
    python train_image_model.py

Output:
    model/image_phishing_model.pth
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# =========================
# CONFIG
# =========================

DATA_DIR   = "data/images"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "image_phishing_model.pth")

IMAGE_SIZE  = 224
BATCH_SIZE  = 16       # keep small for CPU
EPOCHS      = 3
LR          = 1e-3
NUM_WORKERS = 0        # 0 = main thread (safe on all OS)
DEVICE      = torch.device("cpu")


# =========================
# DATA TRANSFORMS
# =========================

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# LOAD DATASETS
# =========================

def load_datasets():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir   = os.path.join(DATA_DIR, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training data not found at '{train_dir}'. "
            "See README for dataset setup instructions."
        )

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_transforms)

    print(f"Classes: {train_ds.classes}")
    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, train_ds.classes


# =========================
# BUILD MODEL
# =========================

def build_model(num_classes=2):
    """
    MobileNetV2 with ImageNet pre-trained weights.
    Freeze the backbone; only train the classifier head initially.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all backbone layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )

    return model.to(DEVICE)


# =========================
# TRAINING LOOP
# =========================

def train(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds      = outputs.max(1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


# =========================
# VALIDATION LOOP
# =========================

def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds      = outputs.max(1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, correct / total


# =========================
# MAIN
# =========================

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 50)
    print("PhishGuard — image model training")
    print("=" * 50)

    train_loader, val_loader, classes = load_datasets()

    model     = build_model(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    # LR scheduler — halve LR if val loss stalls
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}  acc {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  --> saved best model (val acc {val_acc:.4f})")

    print("\nTraining complete.")
    print(f"Best val accuracy : {best_val_acc:.4f}")
    print(f"Model saved to    : {MODEL_PATH}")


if __name__ == "__main__":
    main()