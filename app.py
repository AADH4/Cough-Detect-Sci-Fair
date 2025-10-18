import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import os

# === Load Model ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")  # or "model.keras"
    return model

model = load_model()

# === Safe Audio Preprocessing ===
def preprocess_audio(file_path, target_sr=16000, duration=3):
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception:
        # fallback using soundfile if librosa fails
        y, sr = sf.read(file_path)
        if len(y.shape) > 1:  # convert to mono if stereo
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Ensure numeric & finite values
    y = np.nan_to_num(y)

    # Trim/pad to fixed length
    max_len = target_sr * duration
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    # Compute Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=target_sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Add batch and channel dimensions
    mel_db = mel_db[np.newaxis, ..., np.newaxis]  # shape: (1, 64, T, 1)
    return mel_db

# === Streamlit UI ===
st.title("🩺 Cough Sound Classifier")
st.write("Upload a **.wav** file to check if it’s **Healthy** or **Abnormal**.")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp.wav", format="audio/wav")

    try:
        st.write("Processing audio...")
        X = preprocess_audio("temp.wav")

        preds = model.predict(X)
        pred_class = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))

        label_map = {0: "Abnormal", 1: "Healthy"}  # adjust if needed
        st.subheader(f"Prediction: **{label_map[pred_class]}**")
        st.write(f"Confidence: {confidence:.2f}")

        if label_map[pred_class] == "Healthy":
            st.success("✅ The heart sound appears healthy.")
        else:
            st.error("⚠️ The heart sound may be abnormal.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
    finally:
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")
