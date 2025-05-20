import streamlit as st
import tempfile
import torch
from faster_whisper import WhisperModel
import os
import re
import string
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
from streamlit_mic_recorder import mic_recorder

# --- Styling ---
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #0b3d91 !important;
        color: white !important;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stTextInput, .stTextArea {
        color: white !important;
    }
    .stTextArea textarea {
        font-size: 16px;
        color: white !important;
    }
    .stButton > button {
        font-size: 16px;
        padding: 0.4em 1.2em;
        border-radius: 8px;
        background-color: #0077cc;
        color: white;
        border: none;
    }
    .stButton > button:hover {
        background-color: #005fa3;
        color: white;
    }
    .mic-button > button {
        background-color: red !important;
        border-radius: 50% !important;
        height: 60px !important;
        width: 60px !important;
        font-size: 26px !important;
        font-weight: bold !important;
        border: none !important;
    }
    .stMarkdown {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load models and tokenizers ---
@st.cache_resource
def load_resource():
    # Load tokenizers
    with open("tokenizer_imdb.pkl", "rb") as f:
        tokenizer_imdb = pickle.load(f)
    with open("tokenizer_go.pkl", "rb") as f:
        tokenizer_go = pickle.load(f)

    # Load IMDB model
    model_imdb = tf.keras.models.load_model("best_imdb_model.keras")

    # GoEmotions model URL and download logic
    model_url = "https://github.com/hemerson18/Sentiment-Web-App/releases/download/v1.0/best_goemotions_model.keras"
    model_path = "best_goemotions_model.keras"

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

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# --- Helper functions ---
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

# -- Column 1: Text Input
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

# -- Column 2: Aud and Transcription
with col2:
    st.subheader("🎙️ Speech Input")

    audio = mic_recorder(
        start_prompt="🎙️Start recording",
        stop_prompt="⏹️Stop recording",
        just_once=True,
        use_container_width=True,
        key="speech"
    )

    if audio:
        # Save the audio bytes to a temporary file
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, "temp_audio.wav")
        with open(audio_path, "wb") as f:
            f.write(audio["bytes"])

        # Transcribe using Whisper
        model = WhisperModel("base", device="cpu")
        segments, _ = model.transcribe(audio_path)
        transcription = " ".join([seg.text for seg in segments])
        st.success("Transcription complete!")
        st.write(f"**Transcribed text:** {transciption}")

        # Analyze the transcribed text
        sentiment, conf= classify_sentiment(transcription)
        emotions = classify_emotion(transcription)
        st.success(f"**Sentiment:** {sentiment} ({conf * 100:.1f}%)")
        st.info(f"**Emotions:** {', '.join(emotions) if emotions else 'None detected'}")
