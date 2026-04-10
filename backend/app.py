import os
import re
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import pickle
import librosa
import tempfile

# IMAGE + VIDEO
from image_detector import detect_phishing_image
from video_detector import detect_phishing_video

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────

# TEXT MODEL — try both possible save locations
text_model = None
for path in ['models/text_model.pkl', 'model/phishguard_model.pkl']:
    try:
        with open(path, 'rb') as f:
            text_model = pickle.load(f)
        print(f"[✓] Text model loaded from {path}")
        break
    except Exception:
        pass

# Also try loading vectorizer separately (used by older train_model.py)
text_vectorizer = None
for path in ['models/vectorizer.pkl', 'model/vectorizer.pkl']:
    try:
        with open(path, 'rb') as f:
            text_vectorizer = pickle.load(f)
        print(f"[✓] Text vectorizer loaded from {path}")
        break
    except Exception:
        pass

if text_model is None:
    print("[✗] Text model not found — run train_model.py first")

# VOICE MODEL — try both possible save locations
voice_scaler = None
voice_clf    = None
for path in ['models/voice_model.pkl', 'model/voice_model.pkl']:
    try:
        voice_data   = pickle.load(open(path, 'rb'))
        voice_scaler = voice_data['scaler']
        voice_clf    = voice_data['model']
        print(f"[✓] Voice model loaded from {path}")
        break
    except Exception:
        pass

if voice_clf is None:
    try:
        from voice_detector import detect_ai_voice
        print("[✓] Using heuristic voice detector (voice_detector.py)")
        USE_HEURISTIC_VOICE = True
    except Exception as e:
        print(f"[✗] Voice detector not found: {e}")
        USE_HEURISTIC_VOICE = False
else:
    USE_HEURISTIC_VOICE = False

# TEXT ANALYZER — try to load the standalone text_analyzer module
try:
    from text_analyzer import analyze_message
    print("[✓] text_analyzer.py loaded")
    USE_TEXT_ANALYZER = True
except Exception:
    USE_TEXT_ANALYZER = False
    print("[~] text_analyzer.py not found — using built-in text logic")

# ─────────────────────────────────────────
# HELPER — extract confidence % from explanation string
# e.g. "Flagged — ML confidence 86%." → 0.86
# ─────────────────────────────────────────
def _parse_conf(explanation_str):
    try:
        m = re.search(r'(\d+(?:\.\d+)?)\s*%', str(explanation_str))
        if m:
            return round(float(m.group(1)) / 100, 4)
    except Exception:
        pass
    return None

# ─────────────────────────────────────────
# HOME — serves the dashboard UI
# ─────────────────────────────────────────

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception:
        return send_file('templates/index.html')

# ─────────────────────────────────────────
# TEXT DETECTION
# ─────────────────────────────────────────

@app.route('/detect-text', methods=['POST'])
def detect_text():
    try:
        # Accept both JSON and form data
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '').strip()
        else:
            text = request.form.get('text', request.form.get('message', '')).strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # ── Option 1: use text_analyzer.py
        if USE_TEXT_ANALYZER:
            try:
                result = analyze_message(text)

                # text_analyzer returns a TUPLE:
                # (is_phishing, threat, keywords, explanation, patterns)
                # e.g. (np.True_, 'High', ['87121','cup',...], 'Flagged — ML confidence 86%.', ['General Message'])
                if isinstance(result, tuple):
                    is_phishing = bool(result[0])
                    threat      = str(result[1]) if len(result) > 1 else ('High' if is_phishing else 'Low')
                    keywords    = list(result[2]) if len(result) > 2 else []
                    explanation = str(result[3]) if len(result) > 3 else ''
                    pattern     = result[4][0] if len(result) > 4 and result[4] else ''

                    # Try to extract confidence from explanation string
                    conf = _parse_conf(explanation)
                    if conf is None:
                        conf = 0.86 if is_phishing else 0.92  # sensible defaults

                else:
                    # dict response (future-proof)
                    is_phishing = bool(
                        result.get('is_phishing') == True or
                        result.get('prediction') in ['phishing', 'spam'] or
                        result.get('label') in ['phishing', 'spam']
                    )
                    conf        = float(result.get('confidence', result.get('probability', 0.5)))
                    threat      = result.get('threat', result.get('threat_level',
                                    'High' if (is_phishing and conf > 0.7) else
                                    'Medium' if is_phishing else 'Low'))
                    keywords    = result.get('keywords', [])
                    explanation = result.get('explanation', '')
                    pattern     = result.get('pattern', result.get('attack_pattern', ''))

                return jsonify({
                    'is_phishing': is_phishing,
                    'prediction':  'phishing' if is_phishing else 'legitimate',
                    'label':       'phishing' if is_phishing else 'legitimate',
                    'confidence':  conf,
                    'threat':      threat,
                    'explanation': explanation,
                    'keywords':    keywords,
                    'pattern':     pattern
                })

            except Exception as e:
                traceback.print_exc()
                print(f"[!] text_analyzer failed: {e}, falling back to sklearn model")

        # ── Option 2: sklearn pipeline model directly
        if text_model is not None:
            try:
                if text_vectorizer is not None:
                    X          = text_vectorizer.transform([text])
                    prediction = text_model.predict(X)[0]
                    proba      = text_model.predict_proba(X)[0]
                else:
                    prediction = text_model.predict([text])[0]
                    proba      = text_model.predict_proba([text])[0]

                is_phishing = bool(
                    prediction == 1 or
                    str(prediction).lower() in ['phishing', 'spam', '1']
                )
                confidence = float(max(proba))

                phishing_keywords = [
                    'click', 'verify', 'urgent', 'suspended', 'congratulations',
                    'winner', 'prize', 'claim', 'free', 'password', 'login',
                    'account', 'expire', 'immediately', 'limited', 'offer',
                    'confirm', 'blocked', 'kyc', 'otp', 'recharge', 'refund',
                    'aadhaar', 'pan', 'upi', 'bank', 'reward', 'lucky'
                ]
                lower    = text.lower()
                keywords = [kw for kw in phishing_keywords if kw in lower][:6] if is_phishing else []

                return jsonify({
                    'is_phishing': is_phishing,
                    'prediction':  'phishing' if is_phishing else 'legitimate',
                    'label':       'phishing' if is_phishing else 'legitimate',
                    'confidence':  confidence,
                    'threat':      'High' if (is_phishing and confidence > 0.7) else 'Medium' if is_phishing else 'Low',
                    'keywords':    keywords,
                    'pattern':     '',
                    'explanation': (
                        f"Message contains {len(keywords)} phishing indicator(s). "
                        f"Model confidence: {confidence * 100:.1f}%."
                        if is_phishing else
                        f"No significant phishing patterns detected. "
                        f"Model confidence: {confidence * 100:.1f}%."
                    )
                })
            except Exception as e:
                traceback.print_exc()
                return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500

        return jsonify({'error': 'Text model not loaded. Run train_model.py first.'}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# /analyze-text alias — works with either endpoint name
@app.route('/analyze-text', methods=['POST'])
def analyze_text_alias():
    return detect_text()

# ─────────────────────────────────────────
# VOICE DETECTION
# ─────────────────────────────────────────

@app.route('/detect-voice', methods=['POST'])
def detect_voice():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided. Use key "audio".'}), 400

        file   = request.files['audio']
        suffix = os.path.splitext(file.filename)[1] or '.wav'

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            if USE_HEURISTIC_VOICE:
                from voice_detector import detect_ai_voice
                result = detect_ai_voice(tmp_path)
                return jsonify(result)

            if voice_clf is not None:
                y, sr    = librosa.load(tmp_path, duration=30)
                mfccs    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                features = np.mean(mfccs.T, axis=0).reshape(1, -1)

                features_scaled = voice_scaler.transform(features)
                prediction      = voice_clf.predict(features_scaled)[0]
                proba           = voice_clf.predict_proba(features_scaled)[0]

                is_phishing = bool(prediction == 1 or str(prediction).lower() in ['phishing', 'fake', '1'])
                confidence  = float(max(proba))

                return jsonify({
                    'is_phishing': is_phishing,
                    'prediction':  'phishing' if is_phishing else 'legitimate',
                    'label':       'phishing' if is_phishing else 'legitimate',
                    'confidence':  confidence,
                    'threat':      'High' if (is_phishing and confidence > 0.7) else 'Medium' if is_phishing else 'Low',
                    'explanation': (
                        f"Audio shows robotic/scripted speech patterns typical of AI-generated phishing calls. "
                        f"Confidence: {confidence * 100:.1f}%."
                        if is_phishing else
                        f"Audio appears to be natural human speech. No phishing patterns detected. "
                        f"Confidence: {confidence * 100:.1f}%."
                    )
                })

            return jsonify({'error': 'Voice model not loaded.'}), 500

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────
# IMAGE DETECTION
# ─────────────────────────────────────────

@app.route('/detect-image', methods=['POST'])
def detect_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided. Use key "image".'}), 400

        file   = request.files['image']
        suffix = os.path.splitext(file.filename)[1] or '.jpg'

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = detect_phishing_image(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────
# VIDEO DETECTION
# ─────────────────────────────────────────

@app.route('/detect-video', methods=['POST'])
def detect_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided. Use key "video".'}), 400

        file   = request.files['video']
        suffix = os.path.splitext(file.filename)[1] or '.mp4'

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = detect_phishing_video(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)