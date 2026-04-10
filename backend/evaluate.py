"""
evaluate.py  (fully fixed)
--------------------------
Fixes:
  1. Text  — installs pyspellchecker automatically if missing
  2. Voice — handles new dict return format from voice_detector
  3. Video — works correctly with existing test videos
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from tabulate import tabulate

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def header(title):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def ok(msg):   print(f"  {GREEN}OK{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}!!{RESET}  {msg}")
def err(msg):  print(f"  {RED}XX{RESET}  {msg}")

def compute_metrics(y_true, y_pred, label=""):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n  {BOLD}Results — {label}{RESET}")
    rows = [
        ["Accuracy",  f"{acc:.4f}", f"{acc*100:.1f}%"],
        ["Precision", f"{prec:.4f}", f"{prec*100:.1f}%"],
        ["Recall",    f"{rec:.4f}", f"{rec*100:.1f}%"],
        ["F1 Score",  f"{f1:.4f}", f"{f1*100:.1f}%"],
    ]
    print(tabulate(rows, headers=["Metric","Score","Percent"], tablefmt="rounded_outline"))
    print(f"\n  Confusion Matrix:")
    print(f"                Pred-Legit  Pred-Phish")
    if cm.shape == (2,2):
        print(f"  Actual-Legit   {cm[0][0]:6d}      {cm[0][1]:6d}")
        print(f"  Actual-Phish   {cm[1][0]:6d}      {cm[1][1]:6d}")
    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"cm":cm.tolist()}


# =========================
# 1. TEXT
# =========================

def evaluate_text():
    header("1 / 4  —  Text Phishing Detection  [Real Dataset: spam.csv]")

    if not os.path.exists("data/spam.csv"):
        err("spam.csv not found at data/spam.csv")
        return None

    try:
        df = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1","v2"]]
        df.columns = ["label","text"]
        df["label"] = df["label"].map({"ham":0,"spam":1})
        df = df.dropna()
    except Exception as e:
        err(f"Could not load spam.csv: {e}")
        return None

    legit   = df[df.label==0].sample(min(100, len(df[df.label==0])), random_state=42)
    phish   = df[df.label==1].sample(min(100, len(df[df.label==1])), random_state=42)
    test_df = pd.concat([legit, phish]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  Testing on {len(test_df)} real SMS messages")
    print(f"  ({len(phish)} spam / {len(legit)} legitimate)\n")

    # Install missing package automatically
    try:
        from text_analyzer import analyze_message
    except ModuleNotFoundError:
        warn("pyspellchecker not found — installing now...")
        os.system("pip install pyspellchecker -q")
        try:
            from text_analyzer import analyze_message
        except Exception as e2:
            err(f"Import still failed: {e2}")
            return None
    except Exception as e:
        err(f"Could not import text_analyzer: {e}")
        return None

    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        try:
            phishing, threat, words, explanation, patterns = analyze_message(str(row["text"]))
            y_true.append(int(row["label"]))
            y_pred.append(1 if phishing else 0)
        except Exception:
            pass

    if not y_true:
        err("No predictions made.")
        return None

    print("  Sample predictions (first 6):")
    for i in range(min(6, len(test_df))):
        actual = "SPAM " if y_true[i]==1 else "LEGIT"
        pred   = "SPAM " if y_pred[i]==1 else "LEGIT"
        match  = "OK" if y_true[i]==y_pred[i] else "!!"
        print(f"  [{match}] Actual:{actual} Pred:{pred} | {str(test_df.iloc[i]['text'])[:55]}...")

    return compute_metrics(y_true, y_pred, "Text Detection")


# =========================
# 2. VOICE — handles both old list format and new dict format
# =========================

def evaluate_voice():
    header("2 / 4  —  AI Voice Detection  [Real Dataset: test_audio/]")

    PHISH_DIR = "data/test_audio/phishing"
    LEGIT_DIR = "data/test_audio/legitimate"
    EXTS      = {".wav",".mp3",".ogg",".flac",".m4a"}

    samples = []
    for d, label in [(PHISH_DIR,1),(LEGIT_DIR,0)]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if os.path.splitext(f)[1].lower() in EXTS:
                    samples.append((os.path.join(d,f), label))
        else:
            warn(f"Folder not found: {d}")

    if not samples:
        warn("No audio files found. Skipping voice evaluation.")
        return None

    try:
        from voice_detector import detect_ai_voice
    except Exception as e:
        err(f"Could not import voice_detector: {e}")
        return None

    y_true, y_pred = [], []
    print(f"\n  Testing on {len(samples)} audio files\n")

    for path, label in samples:
        try:
            result = detect_ai_voice(path)

            # NEW format: dict with "prediction" key
            if isinstance(result, dict):
                prediction = result.get("prediction", "legitimate")
                confidence = result.get("confidence", 0.0)
                pred = 1 if prediction == "phishing" else 0

            # OLD format: list like [{"label":"fake","score":0.7}, ...]
            elif isinstance(result, list):
                top  = max(result, key=lambda x: x["score"])
                pred = 1 if top["label"] == "fake" else 0
                confidence = top["score"]
                prediction = "phishing" if pred == 1 else "legitimate"

            else:
                err(f"Unknown format from voice_detector: {type(result)}")
                continue

            y_true.append(label)
            y_pred.append(pred)
            match  = "OK" if pred==label else "!!"
            actual = "AI  " if label==1 else "REAL"
            print(f"  [{match}] Actual:{actual} Pred:{prediction} ({confidence:.0%}) | {os.path.basename(path)}")

        except Exception as e:
            err(f"Error on {os.path.basename(path)}: {e}")

    if not y_true:
        err("No voice predictions made.")
        return None

    return compute_metrics(y_true, y_pred, "Voice Detection")


# =========================
# 3. IMAGE
# =========================

def evaluate_image():
    header("3 / 4  —  Image Phishing Detection  [Real Dataset: images/val/]")

    PHISH_DIR = "data/images/val/phishing"
    LEGIT_DIR = "data/images/val/legitimate"
    EXTS      = {".png",".jpg",".jpeg",".webp"}
    MAX_EACH  = 50

    samples = []
    for d, label in [(PHISH_DIR,1),(LEGIT_DIR,0)]:
        if os.path.isdir(d):
            files = [f for f in os.listdir(d)
                     if os.path.splitext(f)[1].lower() in EXTS][:MAX_EACH]
            for f in files:
                samples.append((os.path.join(d,f), label))
        else:
            warn(f"Folder not found: {d}")

    if not samples:
        warn("No images found in data/images/val/")
        return None

    try:
        from image_detector import detect_phishing_image
    except Exception as e:
        err(f"Could not import image_detector: {e}")
        return None

    y_true, y_pred = [], []
    print(f"\n  Testing on {len(samples)} images (max {MAX_EACH} per class)...")

    for i,(path,label) in enumerate(samples):
        try:
            result = detect_phishing_image(path)
            pred   = 1 if result["prediction"]=="phishing" else 0
            y_true.append(label)
            y_pred.append(pred)
            if (i+1) % 10 == 0:
                print(f"  ... {i+1}/{len(samples)} done")
        except Exception as e:
            err(f"Skipped {os.path.basename(path)}: {e}")

    if not y_true:
        return None

    return compute_metrics(y_true, y_pred, "Image Detection")


# =========================
# 4. VIDEO
# =========================

def evaluate_video():
    header("4 / 4  —  Video Phishing Detection  [Real Dataset: test_videos/]")

    VIDEO_DIR = "data/test_videos"
    EXTS      = {".mp4",".avi",".mov",".mkv",".webm"}
    samples   = []

    if os.path.isdir(VIDEO_DIR):
        for f in os.listdir(VIDEO_DIR):
            if os.path.splitext(f)[1].lower() in EXTS:
                name = f.lower()
                if "phish" in name:
                    samples.append((os.path.join(VIDEO_DIR,f), 1))
                elif "legit" in name or "real" in name or "safe" in name:
                    samples.append((os.path.join(VIDEO_DIR,f), 0))
    else:
        warn(f"Folder not found: {VIDEO_DIR}")

    if not samples:
        warn("No labeled videos found.")
        return None

    try:
        from video_detector import detect_phishing_video
    except Exception as e:
        err(f"Could not import video_detector: {e}")
        return None

    y_true, y_pred = [], []
    print(f"\n  Testing on {len(samples)} video files\n")

    for path, label in samples:
        try:
            result = detect_phishing_video(path)
            pred   = 1 if result["prediction"]=="phishing" else 0
            y_true.append(label)
            y_pred.append(pred)
            match = "OK" if pred==label else "!!"
            print(
                f"  [{match}] Actual:{'PHISH' if label else 'LEGIT'} "
                f"Pred:{result['prediction']} ({result['confidence']:.0%}) "
                f"| {result['phishing_frames']}/{result['frames_analysed']} frames "
                f"| {os.path.basename(path)}"
            )
        except Exception as e:
            err(f"Error on {os.path.basename(path)}: {e}")

    if not y_true:
        return None

    return compute_metrics(y_true, y_pred, "Video Detection")


# =========================
# SUMMARY
# =========================

def print_summary(results):
    header("OVERALL SUMMARY — PhishGuard v2  |  All 4 Modalities")

    rows = []
    for name, m in results.items():
        if m:
            rows.append([name,
                f"{m['accuracy']*100:.1f}%",
                f"{m['precision']*100:.1f}%",
                f"{m['recall']*100:.1f}%",
                f"{m['f1']*100:.1f}%"])
        else:
            rows.append([name,"skipped","—","—","—"])

    print()
    print(tabulate(rows,
        headers=["Modality","Accuracy","Precision","Recall","F1 Score"],
        tablefmt="rounded_outline"))

    out = {k: {"accuracy":round(v["accuracy"],4),
               "precision":round(v["precision"],4),
               "recall":round(v["recall"],4),
               "f1":round(v["f1"],4)}
           for k,v in results.items() if v}

    with open("evaluation_results.json","w") as f:
        json.dump(out, f, indent=2)

    ok("Results saved to evaluation_results.json")
    print()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print(f"\n{BOLD}PhishGuard — Real Dataset Evaluation{RESET}")
    print("Testing all 4 modalities on real data...\n")

    results = {}
    results["Text"]  = evaluate_text()
    results["Voice"] = evaluate_voice()
    results["Image"] = evaluate_image()
    results["Video"] = evaluate_video()

    print_summary(results)