"""
video_detector.py  (fixed)
--------------------------
Fix: require BOTH phishing_ratio >= 0.5 AND final_score >= 0.55
to call a video phishing. This stops borderline legitimate
screenshots from triggering false positives.
"""

import cv2
import os
import numpy as np
from PIL import Image
from image_detector import detect_phishing_image

MAX_FRAMES     = 10
TEMP_FRAME_DIR = "temp_frames"


def extract_frames(video_path: str, n_frames: int = MAX_FRAMES):
    os.makedirs(TEMP_FRAME_DIR, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps if fps > 0 else 0
    if total_frames == 0:
        cap.release()
        raise ValueError("Video has no frames or is corrupted.")
    indices     = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    saved_paths = []
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img       = Image.fromarray(frame_rgb)
        path      = os.path.join(TEMP_FRAME_DIR, f"frame_{i:03d}.png")
        img.save(path)
        saved_paths.append(path)
    cap.release()
    return saved_paths, duration_sec


def cleanup_frames(paths: list):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
    if os.path.isdir(TEMP_FRAME_DIR):
        try:
            os.rmdir(TEMP_FRAME_DIR)
        except OSError:
            pass


def detect_phishing_video(video_path: str) -> dict:

    frame_paths, duration_sec = extract_frames(video_path, MAX_FRAMES)
    if not frame_paths:
        raise ValueError("No frames could be extracted from this video.")

    phishing_scores = []
    phishing_count  = 0

    for path in frame_paths:
        try:
            result = detect_phishing_image(path)
            score  = result["scores"]["phishing"]
            phishing_scores.append(round(score, 4))
            if result["prediction"] == "phishing":
                phishing_count += 1
        except Exception:
            phishing_scores.append(0.0)

    cleanup_frames(frame_paths)

    avg_score   = float(np.mean(phishing_scores)) if phishing_scores else 0.0
    max_score   = float(np.max(phishing_scores))  if phishing_scores else 0.0
    final_score = round(0.6 * avg_score + 0.4 * max_score, 4)
    phish_ratio = phishing_count / len(frame_paths) if frame_paths else 0

    # Require BOTH majority frames AND high score — prevents false positives
    if phish_ratio >= 0.5 and final_score >= 0.55:
        prediction = "phishing"
    elif final_score >= 0.70:
        prediction = "phishing"
    else:
        prediction = "legitimate"

    if final_score >= 0.70:
        threat = "High"
    elif final_score >= 0.50:
        threat = "Medium"
    else:
        threat = "Low"

    pct = int(phish_ratio * 100)
    if prediction == "phishing":
        explanation = (
            f"{phishing_count} out of {len(frame_paths)} frames ({pct}%) "
            f"showed phishing indicators. Confidence: {final_score:.2f}."
        )
    else:
        explanation = (
            f"Only {phishing_count} out of {len(frame_paths)} frames ({pct}%) "
            f"showed suspicious patterns. Video classified as safe."
        )

    return {
        "prediction":      prediction,
        "confidence":      final_score,
        "threat":          threat,
        "frames_analysed": len(frame_paths),
        "phishing_frames": phishing_count,
        "frame_scores":    phishing_scores,
        "duration_sec":    round(duration_sec, 2),
        "explanation":     explanation
    }