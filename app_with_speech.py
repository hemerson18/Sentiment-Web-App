import streamlit as st
import tempfile
import torch
import whisper
import os
import re
import string
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
from scipy.io.wavfile import write as write_wav
from st_audiorec import st_audiorec

# --- Styling ---
st.set_page_config(layout='wide')
st.markdown("""
    <style>
    .mic-button {
        background-color: red !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load models and tokenizers ---
@st.cache_resource
def load_resource():
    import requests

    # Load tokenizers
    with open("tokenizer_imdb.pkl", "rb") as f:
        tokenizer_imdb = pickle.load(f)
    with open("tokenizer_go.pkl", "rb") as f:
        tokenizer_go = pickle.load(f)

    # Load IMDB model
    model_imdb = tf.keras.models.load_model("best_imdb_model.keras")

    # Define model URL for GoEmotions
    model_url = "https://github.com/hemerson18/Sentiment-Web-App/releases/download/v1.0/best_goemotions_model.keras"
    model_path = "best_goemotions_model.keras"

    # Download if not already present
    if not os.path.exists(model_path):
        with st.spinner("Downloading GoEmotions model..."):
            response = requests.get(model_url, stream=True)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                raise Exception(f"Failed to download model: {response.status_code} - {response.reason}")

    model_go = tf.keras.models.load_model(model_path, compile=False)
    return tokenizer_imdb, tokenizer_go, model_imdb, model_go

tokenizer_imdb, tokenizer_go, model_imdb, model_go = load_resource()
whisper_model = whisper.load_model("base")

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# --- Helper Functions ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def classify_sentiment(text):
    max_len = 200
    seq = tokenizer_imdb.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model_imdb.predict(padded)[0][0]
    label = "Positive" if pred > 0.5 else "Negative"
    confidence = round(pred if pred > 0.5 else 1 - pred, 2)
    return label, confidence

def classify_emotion(text):
    max_len = 20
    threshold = 0.3
    seq = tokenizer_go.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    probs = model_go.predict(padded)[0]
    binary = (probs >= threshold).astype(int)
    return [emotion_labels[i] for i, val in enumerate(binary) if val == 1]

# --- UI Layout ---
st.title("🎙️ Speech & Text Sentiment & Emotion Classifier")
col1, col2 = st.columns(2)

# ---- COLUMN 1: TEXT INPUT
with col1:
    st.subheader("Text Input")
    user_input = st.text_area("Enter your text here")
    if st.button("Analyze Text"):
        if user_input:
            sentiment, confidence = classify_sentiment(user_input)
            emotions = classify_emotion(user_input)
            st.success(f"Sentiment: **{sentiment}** ({confidence*100:.1f}% confidence)")
            st.info(f"Emotions detected: {', '.join(emotions) if emotions else 'None'}")
        else:
            st.warning("Please enter text.")

# ---- COLUMN 2: SPEECH INPUT
with col2:
    st.subheader("🎤 Upload Speech (WAV/MP3)")

    uploaded_audio = st.file_uploader("Upload a recorded audio clip", type=["wav", "mp3", "m4a"])

    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_audio.read())
            audio_path = tmp.name

        try:
            st.info("Transcribing with Whisper...")
            result = whisper_model.transcribe(audio_path)
            text = result["text"].strip()
            st.text_area("Transcribed Text", value=text, height=100)

            sentiment, confidence = classify_sentiment(text)
            emotions = classify_emotion(text)
            st.success(f"Sentiment: **{sentiment}** ({confidence*100:.1f}% confidence)")
            st.info(f"Emotions detected: {', '.join(emotions) if emotions else 'None'}")

        except Exception as e:
            st.error(f"Transcription failed: {e}")
        finally:
            os.remove(audio_path)
