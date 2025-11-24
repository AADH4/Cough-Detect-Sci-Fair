import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import google.generativeai as genai
import os

# -----------------------------
# Page Config (NEW)
# -----------------------------
st.set_page_config(
    page_title="CoughDetect",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Custom CSS (Makes UI modern)
# -----------------------------
st.markdown("""
<style>

html {
    background-color: #f7f9fc;
}

body {
    background-color: #f7f9fc;
}

.main {
    background-color: #f7f9fc;
}

section[data-testid="stSidebar"] {
    background-color: #e9eef5;
}

h1 {
    font-weight: 800;
    text-align: center;
}

.upload-box {
    padding: 25px;
    border-radius: 15px;
    background: white;
    border: 2px dashed #8ab4f8;
    text-align: center;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    background: #e8f7ec;
    border-left: 6px solid #37a867;
}

.advice-box {
    padding: 20px;
    border-radius: 12px;
    background: white;
    border-left: 6px solid #ff5f5f;
    border-right: 6px solid #ff5f5f;
}

.footer {
    text-align:center;
    margin-top:40px;
    color:#555;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header with icon + description
# -----------------------------
st.markdown("### <div style='text-align:center;'>🩺</div>", unsafe_allow_html=True)
st.markdown("<h1>Welcome to <span style='color:#2b78e4;'>CoughDetect</span>!</h1>", unsafe_allow_html=True)

st.write("""
CoughDetect helps determine whether your cough sound is **Healthy** or **Abnormal**  
and provides friendly AI-powered health insights.

Simply upload a `.wav` file to begin.
""")

# -----------------------------
# Configure Gemini
# -----------------------------
genai.configure(api_key="AIzaSyDloja-gMt9Ix8VqdmQVqMLodZzKnqDRYg")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lung_sound_classifier.keras")

model = load_model()

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_audio(file_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=target_sr)
    if len(y) > 1024:
        y = y[:1024]
    else:
        y = np.pad(y, (0, max(0, 1024 - len(y))))
    X = np.expand_dims(y, axis=0).astype(np.float32)
    return X

# -----------------------------
# Gemini Advice Generator
# -----------------------------
def get_gemini_advice(label, confidence):
    prompt = f"""
    You are an AI health assistant. A lung sound classifier analyzed a user's cough recording.
    Classification: {label}

    Provide 2–3 friendly sentences of general advice.
    Avoid medical claims. Suggest doctor visits if appropriate.
    If healthy, provide reassurance and basic wellness tips.
    If abnormal, give cautious but useful next-step suggestions.
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# -----------------------------
# Upload Box (NEW)
# -----------------------------
st.markdown("<div class='upload-box'>🔊 **Upload Your Cough (.wav) File**</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["wav"])

# -----------------------------
# Prediction Flow
# -----------------------------
if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        X = preprocess_audio("temp.wav")
        preds = model.predict(X)

        abnormal_prob = float(preds[0][0])
        healthy_prob = float(preds[0][1])
        threshold = 0.5

        if healthy_prob >= threshold:
            label = "Abnormal"
            confidence = healthy_prob
        else:
            label = "Healthy"
            confidence = abnormal_prob

        st.audio(uploaded_file, format="audio/wav")

        # Result box
        st.markdown(f"""
        <div class="result-box">
            <b>Prediction:</b> {label}<br>
        </div>
        """, unsafe_allow_html=True)

        # Advice
        with st.spinner("🧠 Generating personalized AI health tips..."):
            advice = get_gemini_advice(label, confidence)

        st.markdown("<h3>🧠 Gemini AI Health Advice</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='advice-box'>{advice}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")

# Footer
st.markdown("<div class='footer'>© 2025 CoughDetect • AI-Powered Lung Health Tool</div>", unsafe_allow_html=True)
