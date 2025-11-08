import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import google.generativeai as genai
import os

st.title("🩺 Lung Sound Classifier")
st.write("Upload a `.wav` file to classify it as **Healthy** or **Abnormal**, then get health tips.")

# -----------------------------
# Configure Gemini
# -----------------------------
genai.configure(api_key="AIzaSyD4NkDZ6QZUDZvD_KOgLRpin1NXjp-6LlI")

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lung_sound_classifier.keras")

model = load_model()

# -----------------------------
# Preprocess audio
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
# Gemini advice generator
# -----------------------------
def get_gemini_advice(label, confidence):
    prompt = f"""
    You are an AI health assistant. A lung sound classifier analyzed a user's cough recording.

    Result:
    - Classification: {label}
    - Confidence: {confidence:.2f}

    Give 2-3 sentences of general, advice appropriate for that result.
    Keep it professional but friendly.
    Avoid medical claims. Encourage doctor visits if needed.
    Provide good reccomendations for the scenario and what the user should do, make sure to emphasize that this is not fully diagnostic but provides a prediction. 
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# -----------------------------
# File upload + prediction
# -----------------------------
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

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
        st.success(f"Prediction: **{label}** ({confidence:.2f})")

        # -----------------------------
        # Gemini-generated recommendation
        # -----------------------------
        with st.spinner("Generating personalized advice..."):
            advice = get_gemini_advice(label, confidence)

        st.subheader("🧠 Gemini AI Health Advice")
        st.write(advice)

    except Exception as e:
        st.error(f"Error processing file: {e}")
