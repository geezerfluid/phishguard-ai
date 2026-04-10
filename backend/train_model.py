"""
train_model.py  (fixed)
-----------------------
Correctly loads ALL rows from spam.csv (5572 messages).
The previous version only got 43 rows because spam.csv
has 5 columns and needs special handling.

Run:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =========================
# LOAD spam.csv CORRECTLY
# spam.csv has 5 columns: v1, v2, Unnamed:2, Unnamed:3, Unnamed:4
# We only need v1 (label) and v2 (text)
# =========================

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1",
    usecols=[0, 1],          # only first 2 columns
    names=["label", "text"], # rename them
    header=0                 # skip the original header row
)

# Clean
df["label"] = df["label"].str.strip().str.lower()
df["text"]  = df["text"].astype(str).str.strip()
df = df[df["label"].isin(["ham", "spam"])]
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df = df.dropna()

print(f"Loaded from spam.csv: {len(df)} rows")
print(f"  Spam : {df['label'].sum()}")
print(f"  Ham  : {(df['label']==0).sum()}")


# =========================
# ADD INDIAN PHISHING EXAMPLES
# =========================

indian = [
    # Phishing
    ("spam", "Your SBI account will be blocked today. Verify immediately."),
    ("spam", "Dear SBI customer, your net banking is suspended. Click to reactivate."),
    ("spam", "HDFC Bank Alert: Account deactivated. Update KYC within 24 hours."),
    ("spam", "Your bank account blocked. Verify PAN to restore access immediately."),
    ("spam", "Axis Bank: Unusual login detected. Confirm identity or account locked."),
    ("spam", "UPI Alert: Rs.9999 debited from account. Not you? Click to reverse."),
    ("spam", "Your UPI ID will be blocked. Complete verification immediately."),
    ("spam", "Paytm: Wallet on hold. Submit Aadhaar to continue transactions."),
    ("spam", "PhonePe: Rs.4999 sent to unknown. Tap to cancel transaction now."),
    ("spam", "Aadhaar suspended due to suspicious activity. Update KYC now."),
    ("spam", "Your KYC is incomplete. Account will close in 24 hours. Verify."),
    ("spam", "UIDAI: Your Aadhaar is deactivated. Link mobile number immediately."),
    ("spam", "Income Tax Department: Refund of Rs.15000 pending. Submit PAN now."),
    ("spam", "IT Dept: TDS refund approved. Login to claim before expiry today."),
    ("spam", "Congratulations! You won Rs.25,00,000 in KBC lucky draw. Claim now."),
    ("spam", "Amazon Lucky Winner: You won iPhone 15. Pay Rs.99 shipping."),
    ("spam", "IRCTC account locked. Confirm details immediately to restore access."),
    ("spam", "URGENT: Your SIM card will be deactivated in 2 hours. Verify now."),
    ("spam", "Last warning: Complete KYC today or lose access permanently."),
    ("spam", "Your credit card blocked. Call us immediately to unblock it now."),
    ("spam", "Free recharge Rs.500 click link to claim your reward immediately."),
    ("spam", "Loan approved Rs.500000 click to claim before offer expires today."),
    # Legitimate
    ("ham", "Your OTP for SBI login is 345621. Valid 10 minutes. Do not share."),
    ("ham", "Your order has been shipped. Expected delivery by Thursday."),
    ("ham", "Thank you for recharging Rs.299. Plan active for 28 days."),
    ("ham", "Your appointment confirmed for 15th April at 10:30 AM."),
    ("ham", "Salary of Rs.45000 credited to your account on 1st April."),
    ("ham", "Your electricity bill of Rs.1240 is due on 20th April."),
    ("ham", "PNR 2345678901 confirmed. Train departs at 06:15 from Mumbai."),
    ("ham", "Your Aadhaar update request has been successfully submitted."),
    ("ham", "EMI of Rs.8500 auto-debited for your home loan this month."),
    ("ham", "Meeting scheduled for tomorrow at 3 PM. Please join on time."),
    ("ham", "Your package will be delivered today between 2 PM and 6 PM."),
    ("ham", "Happy Birthday! Wishing you a wonderful day from all of us."),
    ("ham", "Your leave application has been approved by your manager."),
]

extra = pd.DataFrame(indian, columns=["label", "text"])
extra["label"] = extra["label"].map({"ham": 0, "spam": 1})
df = pd.concat([df, extra], ignore_index=True)

print(f"\nAfter adding Indian examples: {len(df)} total rows")
print(f"  Spam : {df['label'].sum()}")
print(f"  Ham  : {(df['label']==0).sum()}")


# =========================
# TRAIN
# =========================

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=1.0,
    solver="lbfgs"
)

model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred,
      target_names=["Legitimate", "Phishing"]))


# =========================
# SAVE
# =========================

os.makedirs("model", exist_ok=True)
joblib.dump(model,      "model/phishguard_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("Saved: model/phishguard_model.pkl  and  model/vectorizer.pkl")