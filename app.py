import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# ----------------------------
# STREAMLIT PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Cough Classifier",
    layout="wide",
    page_icon="🤖"
)

# ----------------------------
# GLOBAL STYLING (BACKGROUND, IMAGES, LAYOUT)
# ----------------------------
st.markdown("""
    <style>
        /* Make the main app wider */
        .main .block-container {
            max-width: 95%;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Soft light background */
        .stApp {
            background-color: #f4f6fa;
            background-image: radial-gradient(circle at 20% 20%, #ffffff 0%, #f4f6fa 70%);
        }

        /* Decorative left image */
        .left-img {
            position: fixed;
            top: 20%;
            left: 0;
            width: 220px;
            opacity: 0.18;
            z-index: -1;
        }

        /* Decorative right image */
        .right-img {
            position: fixed;
            top: 20%;
            right: 0;
            width: 220px;
            opacity: 0.18;
            z-index: -1;
        }
    </style>

    <img src="/mnt/data/a6380c32-c2fe-4ce8-9761-6c4d7d0dcc5f.png" class="left-img">
    <img src="/mnt/data/df7e5018-e460-48b8-a5dd-351fa64f29bb.png" class="right-img">
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR INFORMATION
# ----------------------------
st.sidebar.title("About This App")
st.sidebar.write("""
This cough classifier uses a machine learning model trained on
lung sound data to distinguish **Healthy** vs **Abnormal** cough sounds.

### How to Use:
1. Upload a `.wav` audio file (1–3 seconds recommended)
2. The app preprocesses the audio  
3. The model predicts the health label  
4. You'll see the prediction + confidence score

### Notes:
- This is **not medical advice**
- Model accuracy depends on the training dataset
- More data = better predictions in the future
""")

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lung_sound_classifier.keras")

model = load_model()

# ----------------------------
# AUDIO PREPROCESSING
# (kept exactly like you had it)
# ----------------------------
def preprocess_audio(path):
    y, sr = librosa.load(path, sr=None)
    target_sr = 16000

    # Resample if needed
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    mel = librosa.feature.melspectrogram(y, sr=target_sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)

    # Resize to consistent model input size
    mel_db_resized = librosa.util.fix_length(mel_db, size=94, axis=1)

    mel_db_resized = np.expand_dims(mel_db_resized, axis=-1)
    mel_db_resized = np.expand_dims(mel_db_resized, axis=0)

    return mel_db_resized

# ----------------------------
# MAIN UI CARD
# ----------------------------
st.markdown("""
<div style="
    background: white;
    padding: 35px;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    width: 85%;
    margin: 20px auto;
">
""", unsafe_allow_html=True)

st.title("🤖 AI Cough Classifier")
st.write("Upload a cough audio file (`.wav`) and the model will classify it.")

# ----------------------------
# FILE UPLOADER
# ----------------------------
audio_file = st.file_uploader("Upload your .wav file", type=["wav"])

if audio_file:
    with open("temp.wav", "wb") as f:
        f.write(audio_file.read())

    st.audio("temp.wav")

    try:
        X = preprocess_audio("temp.wav")
        pred = model.predict(X)[0][0]

        label = "Abnormal" if pred > 0.5 else "Healthy"
        confidence = pred if pred > 0.5 else (1 - pred)

        st.subheader(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}**")

    except Exception as e:
        st.error(f"Error processing file: {e}")

st.markdown("</div>", unsafe_allow_html=True)
