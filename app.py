import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import google.generativeai as genai
import os
# ----------------------------
# STREAMLIT PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="CoughGPT",
    layout="wide",
    page_icon="🩺"
)

# ----------------------------
# GLOBAL STYLING (BACKGROUND + SIDE IMAGES)
# ----------------------------
st.markdown("""
    <style>
        /* Main content width */
        .main .block-container {
            max-width: 95%;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Soft gradient background */
        .stApp {
            background-color: #f4f6fa;
            background-image: radial-gradient(circle at 20% 20%, #ffffff 0%, #f4f6fa 70%);
        }

        /* Decorative left image */
        .left-img {
            position: fixed;
            top: 20%;
            left: 0;
            width: 250px;
            opacity: 0.18;
            z-index: -1;
        }

        /* Decorative right image */
        .right-img {
            position: fixed;
            top: 20%;
            right: 0;
            width: 250px;
            opacity: 0.18;
            z-index: -1;
        }

        /* Header images row */
        .header-img-row {
            display: flex;
            justify-content: center;
            gap: 25px;
            margin-bottom: 20px;
        }
        .header-img-row img {
            width: 160px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        }

    </style>

    <!-- LEFT + RIGHT side images -->
    <img src="/mnt/data/a6380c32-c2fe-4ce8-9761-6c4d7d0dcc5f.png" class="left-img">
    <img src="/mnt/data/df7e5018-e460-48b8-a5dd-351fa64f29bb.png" class="right-img">
""", unsafe_allow_html=True)

# ----------------------------
# HEADER IMAGE ROW (top banner)
# Replace these URLs with any images you want
# ----------------------------
st.markdown("""
<div class="header-img-row">
    <img src="data:image/jpeg;base64,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">
    <img src="data:image/jpeg;base64,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">
    <img src="data:image/jpeg;base64,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">
</div>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR EXPLANATION
# ----------------------------
st.sidebar.title("ℹ️ About CoughGPT")
st.sidebar.write("""
CoughGPT analyzes **cough audio** using a deep learning model to provide you with early respiratory disease detection and provides health advice.

### How It Works:
1. Upload a `.wav` file  
2. Audio is preprocessed  
3. Model predicts: **Healthy** or **Abnormal**  

### Notes:
- Not medical advice  
- Best with clean, 1–2 second cough recordings  
""")
genai.configure(api_key="AIzaSyDloja-gMt9Ix8VqdmQVqMLodZzKnqDRYg")
# ----------------------------
# LOAD MODEL (Your original logic)
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lung_sound_classifier.keras")

model = load_model()

# ----------------------------
# ORIGINAL PREPROCESSING (unchanged)
# ----------------------------
def preprocess_audio(file_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=target_sr)
    if len(y) > 1024:
        y = y[:1024]
    else:
        y = np.pad(y, (0, max(0, 1024 - len(y))))
    X = np.expand_dims(y, axis=0).astype(np.float32)
    return X

# ----------------------------
# MAIN CARD CONTAINER
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

st.title("🩺 Welcome to CoughGPT!")
st.write("Upload a `.wav` file to check whether your cough is **Healthy** or **Abnormal**.")
def get_gemini_advice(label, confidence): 
    prompt = f""" You are an AI health assistant. A lung sound classifier analyzed a user's cough recording. Result: - Classification: {label} - Confidence: {confidence:.2f} 
    Give 2-3 sentences of general, advice appropriate for that result. Keep it professional but friendly.
    Avoid medical claims. Encourage doctor visits if needed. Provide good reccomendations for the scenario and what the user should do. 
    Don't sound super indecisive, for example if the model predicts abnormality, suggest the user should go to a doctor or take over-the-counter medication, while if it predicts healthy, just provide standard cold recovery steps or none at all. 
    Don't mention the confidence, but if you belive that it may not be a serious disease cough, but still a small common cold, provide adequate future steps. 
    Based on the confidence add further ideas, so if the model says it's normal but slightly low confidence it could be a cold so mention that and appropriate steps and if it's abnormal and with very high confidence it could be a serious disease perhaps""" 
    model = genai.GenerativeModel("gemini-2.5-flash") 
    response = model.generate_content(prompt) 
    return response.text.strip()
# ----------------------------
# FILE UPLOAD
# ----------------------------

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
        st.success(f"Prediction: **{label}**")
        with st.spinner("Generating personalized advice..."): 
            advice = get_gemini_advice(label, confidence) 
        st.subheader("🧠 Gemini AI Health Advice")
        st.write(advice) 
    except Exception as e: 
        st.error(f"Error processing file: {e}")

st.markdown("</div>", unsafe_allow_html=True)
