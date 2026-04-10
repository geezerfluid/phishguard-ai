"""
text_analyzer.py
----------------
Standalone text phishing analysis module.
Used by both app.py (live detection) and evaluate.py (testing).
"""

import numpy as np
import joblib
from spellchecker import SpellChecker

model      = joblib.load("model/phishguard_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

DOMAIN_WHITELIST = {
    "upi", "kyc", "sbi", "irctc", "aadhaar", "hdfc", "icici",
    "pan", "sms", "otp", "bank", "rs", "paytm", "phonepe",
    "uidai", "neft", "rtgs", "imps", "ifsc", "tds", "gst"
}

spell = SpellChecker()
spell.word_frequency.load_words(DOMAIN_WHITELIST)

HIGH_RISK_KEYWORDS = [
    "blocked", "verify", "urgent", "account", "suspended",
    "upi", "kyc", "refund", "deactivated", "credit", "card",
    "immediately", "click", "login", "update", "expire",
    "winner", "prize", "won", "claim", "reward",
    "aadhaar", "pan", "otp", "password", "locked"
]

GRAMMAR_PHRASES = [
    "has been block", "will be blocked today",
    "verify immediately", "avoid block",
    "account will be block", "do the needful",
    "kindly verify", "please update immediately",
    "your account suspended", "click here to verify"
]


def analyze_message(message):
    vec  = vectorizer.transform([message])
    prob = model.predict_proba(vec)[0][1]

    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_scores  = vec.toarray()[0]
    message_words = set(message.lower().split())

    candidate_words = feature_names[tfidf_scores.argsort()[-15:][::-1]]
    top_words = [
        w for w in candidate_words
        if w in message_words and len(w) > 2
    ][:5]

    message_lower = message.lower()

    words      = [w for w in message_lower.split() if w.isalpha()]
    misspelled = {w for w in spell.unknown(words) if w not in DOMAIN_WHITELIST}
    spell_rate = len(misspelled) / max(len(words), 1)
    linguistic_risk = spell_rate > 0.2

    grammar_flags = 0
    for phrase in GRAMMAR_PHRASES:
        if phrase in message_lower:
            grammar_flags += 1
    if message.isupper():                                           grammar_flags += 1
    if message.count("!") >= 2:                                    grammar_flags += 1
    if "kindly" in message_lower and "verify" in message_lower:    grammar_flags += 1
    if "please" in message_lower and (
        "blocked" in message_lower or "suspended" in message_lower
    ):                                                              grammar_flags += 1
    if len(message_lower.split()) < 8 and (
        "blocked" in message_lower or "verify" in message_lower
    ):                                                              grammar_flags += 1

    grammar_risk = grammar_flags >= 1
    hits         = [w for w in HIGH_RISK_KEYWORDS if w in message_lower]

    attack_patterns = []
    if any(w in message_lower for w in ["card","upi","refund","debit","credit","payment","rs"]):
        attack_patterns.append("Financial Threat")
    if any(w in message_lower for w in ["verify","login","password","otp","update","kyc","aadhaar","pan"]):
        attack_patterns.append("Credential Harvesting")
    if any(w in message_lower for w in ["blocked","suspended","deactivated","locked","expire"]):
        attack_patterns.append("Account Impersonation")
    if any(w in message_lower for w in ["urgent","immediately","today","now","hours","warning","last"]):
        attack_patterns.append("Urgency / Fear-based Scam")
    if any(w in message_lower for w in ["won","winner","prize","lottery","congratulations","reward","claim"]):
        attack_patterns.append("Lottery / Prize Scam")
    if not attack_patterns:
        attack_patterns.append("General Message")

    keyword_score = len(hits) / len(HIGH_RISK_KEYWORDS)
    combined      = 0.6 * prob + 0.3 * keyword_score + 0.1 * (1 if linguistic_risk or grammar_risk else 0)

    if combined >= 0.55 or prob >= 0.60:
        threat = "High"
    elif combined >= 0.30 or hits or linguistic_risk or grammar_risk:
        threat = "Medium"
    else:
        threat = "Low"

    phishing = prob >= 0.35 or len(hits) >= 2 or linguistic_risk or grammar_risk

    parts = []
    if prob >= 0.35:        parts.append(f"ML confidence {prob:.0%}")
    if hits:                parts.append("keywords: " + ", ".join(hits[:4]))
    if misspelled:          parts.append("spelling: " + ", ".join(list(misspelled)[:3]))
    if grammar_risk:        parts.append("suspicious grammar")

    explanation = (
        "Flagged — " + " | ".join(parts) + "."
        if parts else
        "No significant phishing indicators detected."
    )

    return phishing, threat, top_words, explanation, attack_patterns