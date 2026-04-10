"""
generate_ai_voice.py
--------------------
Automatically generates AI voice audio samples using Google TTS (gTTS).
Saves them directly into data/test_audio/phishing/

These will be used to test and evaluate the voice phishing detector.

INSTALL:
    pip install gtts

RUN:
    python generate_ai_voice.py
"""

import os
from gtts import gTTS


# =========================
# OUTPUT FOLDER
# =========================

OUTPUT_DIR = "data/test_audio/phishing"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# SCAM MESSAGES TO CONVERT
# These are typical AI-generated phishing voice call scripts
# =========================

SCAM_MESSAGES = [
    "Your SBI account will be blocked today. Please verify your details immediately by calling us back.",
    "This is an urgent message from the Income Tax Department. Your PAN card has been suspended. Press 1 to speak to an officer.",
    "Congratulations! You have won ten lakh rupees in the KBC lucky draw. Call us now to claim your prize.",
    "Dear customer, your UPI account shows suspicious activity. Please share your OTP to reverse the transaction.",
    "Your Aadhaar card has been deactivated due to verification failure. Update your KYC immediately to avoid penalty.",
    "This is TRAI. Your mobile number will be disconnected in two hours due to illegal activity. Press 1 to speak to officer.",
    "Your electricity connection will be cut off today. Pay your outstanding bill immediately to avoid disconnection.",
    "Dear user, your credit card has been blocked. Please call our helpline immediately to unblock your card.",
    "You have been selected for a government scheme. You will receive fifty thousand rupees. Share your bank details to proceed.",
    "This is an automated call from your bank. We have detected a fraudulent transaction. Press 1 to cancel it now.",
]


# =========================
# GENERATE AND SAVE
# =========================

print("Generating AI voice samples...\n")

for i, message in enumerate(SCAM_MESSAGES):
    filename = f"ai_scam_{i+1:02d}.mp3"
    filepath = os.path.join(OUTPUT_DIR, filename)

    try:
        tts = gTTS(text=message, lang="en", slow=False)
        tts.save(filepath)
        print(f"  Saved: {filename}")
    except Exception as e:
        print(f"  Failed: {filename} — {e}")

print(f"\nDone! {len(SCAM_MESSAGES)} AI voice files saved to: {OUTPUT_DIR}")
print("\nNext: record your own voice for data/test_audio/legitimate/")
print("Then run: python evaluate.py")