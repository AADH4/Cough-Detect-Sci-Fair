import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf

# === Load Model ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("lung_sound_classifier.keras")  # or "model.keras"
    return model

model = load_model()

# === Audio Preprocessing ===
def preprocess_audio(file_path, target_sr=16000, duration=3):
    y, sr = librosa.load(file_path, sr=target_sr)
    # Trim/pad to fixed length (3 seconds)
    max_len = target_sr * duration
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))
    # Convert to Mel-spectrogram (or whatever preprocessing you trained with)
    mel = librosa.feature.melspectrogram(y, sr=target_sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db[np.newaxis, ..., np.newaxis]  # shape (1, 64, T, 1)
    return mel_db

# === Streamlit UI ===
st.title("🩺 Heart Sound Classifier")
st.write("Upload a **.wav** file to check if it’s **Healthy** or **Abnormal**.")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    # Save temp file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp.wav", format="audio/wav")

    st.write("Processing audio...")
    X = preprocess_audio("temp.wav")

    preds = model.predict(X)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    label_map = {0: "Abnormal", 1: "Healthy"}  # adjust if reversed
    st.subheader(f"Prediction: **{label_map[pred_class]}**")
    st.write(f"Confidence: {confidence:.2f}")

    if label_map[pred_class] == "Healthy":
        st.success("✅ The heart sound appears healthy.")
    else:
        st.error("⚠️ The heart sound may be abnormal.")
