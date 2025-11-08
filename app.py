import streamlit as st
import tensorflow as tf
import numpy as np
import librosa

st.title("🩺 Lung Sound Classifier")
st.write("Upload a `.wav` file to classify it as **Healthy** or **Abnormal**.")

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lung_sound_classifier.keras")

model = load_model()

# -----------------------------
# Preprocess audio correctly
# -----------------------------
def preprocess_audio(file_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=target_sr)

    # If longer than 1024 samples, take first 1024
    # If shorter, pad with zeros
    if len(y) > 1024:
        y = y[:1024]
    else:
        y = np.pad(y, (0, max(0, 1024 - len(y))))

    # Reshape for model: (1, 1024)
    X = np.expand_dims(y, axis=0).astype(np.float32)
    return X

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

        # -----------------------------
        # ✅ Step 1: Threshold-based classification
        # -----------------------------
        abnormal_prob = float(preds[0][0])  # assuming output = [abnormal, healthy]
        healthy_prob = float(preds[0][1])

        threshold = 0.5

        if healthy_prob >= threshold:
            label = "Healthy"
            confidence = healthy_prob
        else:
            label = "Abnormal"
            confidence = abnormal_prob

        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: {confidence:.2f}")
        st.write(f"(Abnormal={abnormal_prob:.2f}, Healthy={healthy_prob:.2f})")

        st.audio(uploaded_file, format="audio/wav")

    except Exception as e:
        st.error(f"Error processing file: {e}")
