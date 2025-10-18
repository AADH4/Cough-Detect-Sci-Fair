import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import os

st.set_page_config(page_title="Lung Sound Classifier", layout="centered")

# ------------------------------
# 1️⃣ Load YAMNet model (for feature extraction)
# ------------------------------
@st.cache_resource
def load_yamnet():
    return hub.load("https://tfhub.dev/google/yamnet/1")

yamnet_model = load_yamnet()

# ------------------------------
# 2️⃣ Load your trained classifier
# ------------------------------
@st.cache_resource
def load_classifier():
    model = tf.keras.models.load_model("lung_sound_classifier.keras", compile=False)
    return model

model = load_classifier()

# ------------------------------
# 3️⃣ Audio preprocessing — use YAMNet embeddings
# ------------------------------
def preprocess_audio(file_path):
    try:
        waveform, sr = librosa.load(file_path, sr=16000)
        waveform = waveform.astype(np.float32)
        # Extract YAMNet embeddings
        scores, embeddings, spectrogram = yamnet_model(waveform)
        # Average embeddings into a single 1024-length vector
        features = np.mean(embeddings.numpy(), axis=0)
        return np.expand_dims(features, axis=0)  # Shape (1, 1024)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# ------------------------------
# 4️⃣ Streamlit interface
# ------------------------------
st.title("🩺 Lung Sound Classifier")
st.write("Upload a `.wav` file to detect whether the lung sound is **Healthy** or **Abnormal**.")

uploaded_file = st.file_uploader("Upload your lung sound (.wav)", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format="audio/wav")

    st.write("🔍 Processing audio...")
    X = preprocess_audio("temp.wav")

    if X is not None:
        preds = model.predict(X)
        pred_class = np.argmax(preds)
        confidence = float(np.max(preds))

        st.success(f"Prediction: **{'Healthy' if pred_class == 1 else 'Abnormal'}**")
        st.write(f"Confidence: {confidence:.2f}")

        # Optional: show raw probabilities
        st.write("Raw model output:", preds)

# ------------------------------
# 5️⃣ Footer
# ------------------------------
st.caption("Built with TensorFlow + Streamlit + YAMNet | © 2025")
