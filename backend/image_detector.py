import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


# =========================
# CONFIG
# =========================

MODEL_PATH = "model/image_phishing_model.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

LABELS = ["legitimate", "phishing"]


# =========================
# MODEL DEFINITION
# =========================

def build_model():
    """
    MobileNetV2 with a custom 2-class head.
    Lightweight and fast on CPU (~14MB).
    """
    model = models.mobilenet_v2(weights=None)

    # Replace classifier head for binary classification
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 2)
    )

    return model


# =========================
# IMAGE PREPROCESSING
# =========================

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# LOAD MODEL (singleton)
# =========================

_model_cache = None

def load_model():
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Image model not found at '{MODEL_PATH}'. "
            "Please run train_image_model.py first."
        )

    model = build_model()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.eval()
    _model_cache = model
    return model


# =========================
# MAIN DETECTION FUNCTION
# =========================

def detect_phishing_image(image_path: str) -> dict:
    """
    Analyse an image (screenshot) for phishing indicators.

    Args:
        image_path: Path to .jpg / .png file

    Returns:
        dict with keys:
          - prediction: "phishing" | "legitimate"
          - confidence: float (0.0 – 1.0)
          - scores: {"phishing": float, "legitimate": float}
          - threat: "High" | "Medium" | "Low"
          - explanation: human-readable string
    """

    # --- Load and preprocess ---
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image: {e}")

    tensor = preprocess(img).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

    # --- Inference ---
    model = load_model()

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    phish_score = probs[1].item()   # index 1 = phishing
    legit_score = probs[0].item()   # index 0 = legitimate

    prediction = LABELS[probs.argmax().item()]

    # --- Threat level ---
    if phish_score >= 0.75:
        threat = "High"
    elif phish_score >= 0.45:
        threat = "Medium"
    else:
        threat = "Low"

    # --- Explanation ---
    if prediction == "phishing":
        if phish_score >= 0.75:
            explanation = (
                "This image strongly resembles known phishing pages. "
                "Common indicators include fake login forms, spoofed brand logos, "
                "or urgency overlays designed to steal credentials."
            )
        else:
            explanation = (
                "This image shows some visual patterns associated with phishing pages, "
                "such as suspicious form layouts or inconsistent branding. "
                "Manual review is recommended."
            )
    else:
        explanation = (
            "This image does not match known phishing page patterns. "
            "It appears to be a legitimate website screenshot."
        )

    return {
        "prediction":  prediction,
        "confidence":  round(phish_score, 4),
        "scores": {
            "phishing":   round(phish_score, 4),
            "legitimate": round(legit_score, 4)
        },
        "threat":      threat,
        "explanation": explanation
    }