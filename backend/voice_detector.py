"""
voice_detector.py  (fixed for gTTS)
------------------------------------
gTTS (Google TTS) produces very natural-sounding audio,
so the old thresholds were too strict.

Key changes:
- Lower pitch variance threshold (gTTS is smoother than ElevenLabs)
- Lower RMS variance threshold
- Confidence threshold lowered to 0.35 for phishing detection
- Added MFCC and ZCR features for better discrimination
"""

import librosa
import numpy as np


def detect_ai_voice(audio_path):

    # Load up to 10 seconds
    try:
        y, sr = librosa.load(audio_path, duration=10)
    except Exception as e:
        return {
            "prediction":  "legitimate",
            "confidence":  0.0,
            "threat":      "Low",
            "details":     {},
            "explanation": f"Could not load audio: {e}"
        }

    if len(y) < sr * 0.5:  # less than 0.5 seconds
        return {
            "prediction":  "legitimate",
            "confidence":  0.0,
            "threat":      "Low",
            "details":     {},
            "explanation": "Audio too short to analyse."
        }

    # =========================
    # FEATURE 1: Pitch variance
    # gTTS / AI voice = low pitch variance (monotone)
    # Human voice = high pitch variance (natural)
    # =========================
    try:
        pitch = librosa.yin(y, fmin=60, fmax=400, sr=sr)
        pitch = pitch[(pitch > 60) & (pitch < 400) & ~np.isnan(pitch)]
        pitch_var = float(np.var(pitch)) if len(pitch) > 5 else 9999.0
    except Exception:
        pitch_var = 9999.0

    # =========================
    # FEATURE 2: RMS loudness variance
    # AI voice = very uniform volume
    # =========================
    rms     = librosa.feature.rms(y=y)[0]
    rms_var = float(np.var(rms))

    # =========================
    # FEATURE 3: Spectral flatness
    # AI voice = slightly higher flatness (more noise-like)
    # =========================
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))

    # =========================
    # FEATURE 4: MFCC variance
    # AI voice = lower MFCC variance across time
    # =========================
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var = float(np.mean(np.var(mfcc, axis=1)))

    # =========================
    # FEATURE 5: Zero crossing rate variance
    # AI voice = more uniform ZCR
    # =========================
    zcr_var = float(np.var(librosa.feature.zero_crossing_rate(y)[0]))

    # =========================
    # WEIGHTED SCORING
    # Tuned specifically for gTTS detection
    # =========================
    score = 0.0

    # Pitch: gTTS has pitch_var typically < 800
    if pitch_var < 300:
        score += 0.30
    elif pitch_var < 800:
        score += 0.15

    # RMS: gTTS has very stable volume, rms_var < 0.01
    if rms_var < 0.005:
        score += 0.25
    elif rms_var < 0.015:
        score += 0.12

    # Spectral flatness: gTTS > 0.05
    if flatness > 0.05:
        score += 0.20
    elif flatness > 0.02:
        score += 0.10

    # MFCC: gTTS mfcc_var typically < 80
    if mfcc_var < 50:
        score += 0.15
    elif mfcc_var < 80:
        score += 0.07

    # ZCR variance: gTTS < 0.0005
    if zcr_var < 0.0002:
        score += 0.10
    elif zcr_var < 0.0005:
        score += 0.05

    confidence = round(min(score, 1.0), 3)

    # =========================
    # DECISION — lowered threshold for gTTS
    # =========================
    if confidence >= 0.35:      # lowered from 0.55
        prediction  = "phishing"
        threat      = "High" if confidence >= 0.60 else "Medium"
        explanation = (
            "This audio shows characteristics of AI-generated voice: "
            "unnaturally stable pitch, uniform loudness, and synthetic patterns."
        )
    else:
        prediction  = "legitimate"
        threat      = "Low"
        explanation = (
            "This audio appears to be a real human voice with natural "
            "pitch variation and vocal expression."
        )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "threat":     threat,
        "details": {
            "pitch_variance":    round(pitch_var, 2),
            "rms_variance":      round(rms_var, 6),
            "spectral_flatness": round(flatness, 4),
            "mfcc_variance":     round(mfcc_var, 2),
            "zcr_variance":      round(zcr_var, 6),
        },
        "explanation": explanation
    }