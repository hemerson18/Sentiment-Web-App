import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string, re
import sounddevice as sd
import soundfile as sf
import tempfile
import whisper

# --- Styling ----
st.set_page_config(layout='wide')
st.markdown(
    """
    <style>
    body {
    background-colour: #0b1e3f;
    colour: white;
    }
    .stButton>button {
        font-size: 16px;
        padding: 0.5em 1em;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
    }
    .stTextArea>div>textarea {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load models and tokenizers ---
@st.cache_resource
def load_resource():
    with open("saved_models/tokenizer_imdb.pkl", "rb") as f:
        tokenizer_imdb = pickle.load(f)
    with open("saved_models/tokenizer_go.pkl", "rb") as f:
        tokenizer_go = pickle.load(f)
    model_imdb = tf.keras.models.load_model("saved_models/best_imdb_model.keras")
    import os
    import urllib.request
    model_url = "https://github.com/your-username/your-repo/releases/download/v1.0/best_goemotions_model.keras"
    model_path = "best_goemotions_model.keras"

    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(model_url, model_path)

    model_go = tf.keras.models.load_model(model_path, compile=False)
    return tokenizer_imdb, tokenizer_go, model_imdb, model_go 

tokenizer_imdb, tokenizer_go, model_imdb, model_go = load_resource()

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# --- Helper function ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text 

def classify_sentiment(text):
    max_len1=200
    seq =tokenizer_imdb.texts_to_sequences([text])
    padded= pad_sequences(seq, maxlen=max_len1, padding='post')
    pred = model_imdb.predict(padded)[0][0]
    label ="Positive" if pred > 0.5 else "Negative"
    confidence = round(pred if pred > 0.5 else 1-pred, 2)
    return label, confidence

def classify_emotion(text):
    max_len2 = 20
    threshold = 0.3
    seq = tokenizer_imdb.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len2, padding='post', truncating='post')
    probs = model_go.predict(padded)[0]
    binary = (probs >= threshold).astype(int)
    return [emotion_labels[i] fot i, val in enumerate(binary) if val == 1]

def record_audio(duration=10, fs=44100):
    st.info("Recording... Speak now.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp_file.name, audio, fs)
    return tmp_file.name

def transcribe_audio(path):
    model = whisper.load_model("base")
    result = model.transcribe(path)
    return result["text"]

#--- UI layout ---
with col1:
    st.header("📝Text Input")
    user_input = st.text_area("Enter your text here", height=150)
    if st.button("Analyse Text"):
        cleaned = preprocess_text(user_input)
        sentiment, confidence = classify_sentiment(cleaned)
        emotions = classify_emotion(cleaned)
        st.subheader("Sentiment Result")
        st.success(f"{sentiment} (Confidence : {confidence})")
        st.subheader("Detected Emotions")
        st.info(", ".join(emotions) if emotions else "No emotions detected.")

# --- Audio Input Section --- 
with col2: 
    st.header("🎙️Speech Input")
    if st.button("🔴 Start Recording (Max 10 sec)", key="record"):
        audio_path = record_audio(duration=10)
        transcript = transcribe_audio(audio_path)
        cleaned = preprocess_text(transcript)
        st.subheader("Transcribed Text")
        st.code(transcript, language="text")

        sentiment, confidence = classify_sentiment(cleaned)
        emotions = classify_emotion(cleaned)

        st.subheader("Sentiment Result")
        st.success(f"{sentiment} (Confidence: {confidence})")
        st.subheader("Detected Emotions")
        st.info(", ".join(emotions) if emotions else "No emotions detected.")
