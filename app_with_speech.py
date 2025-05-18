import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import whisper
import numpy as np
import os
import tempfile
from scipy.io.wavfile import write as write_wav
import torch
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page config
st.set_page_config(page_title="Sentiment & Emotion Classifier", layout="wide")

# Load models
@st.cache_resource
def load_sentiment_model():
    return load_model("models/imdb_cnn_lstm_glove.h5")

@st.cache_resource
def load_emotion_model():
    return load_model("models/goemotions_cnn_bilstm_glove.h5")

@st.cache_resource
def load_tokenizer():
    return joblib.load("models/tokenizer.pkl")

@st.cache_resource
def load_label_encoder():
    return joblib.load("models/label_encoder.pkl")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

sentiment_model = load_sentiment_model()
emotion_model = load_emotion_model()
tokenizer = load_tokenizer()
label_encoder = load_label_encoder()
whisper_model = load_whisper_model()

MAXLEN = 100  # max length for padding

# Helper functions
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAXLEN)
    return padded

def classify_sentiment(text):
    processed = preprocess_text(text)
    prediction = sentiment_model.predict(processed, verbose=0)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return sentiment, prediction if sentiment == "Positive" else 1 - prediction

def classify_emotion(text):
    processed = preprocess_text(text)
    prediction = emotion_model.predict(processed, verbose=0)[0]
    threshold = 0.3
    emotions = [label_encoder.classes_[i] for i, p in enumerate(prediction) if p >= threshold]
    return emotions

# App UI
st.title("🎙️ Sentiment & Emotion Classifier")
st.markdown("Enter text or speak to classify **sentiment** and **emotions**.")
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["📝 Text Input", "🎤 Speech Input"])

# ---- TEXT INPUT TAB ----
with tab1:
    user_input = st.text_area("Enter your text here:", height=150)

    if st.button("Analyze Text"):
        if user_input.strip():
            sentiment, confidence = classify_sentiment(user_input)
            emotions = classify_emotion(user_input)
            st.success(f"Sentiment: **{sentiment}** ({confidence*100:.1f}% confidence)")
            st.info(f"Emotions detected: {', '.join(emotions) if emotions else 'None'}")
        else:
            st.warning("Please enter some text to analyze.")

# ---- SPEECH INPUT TAB ----
with tab2:
    class AudioProcessor:
        def __init__(self):
            self.frames = []

        def recv(self, frame: av.AudioFrame):
            self.frames.append(frame)
            return frame

    audio_processor = AudioProcessor()

    webrtc_ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True
            },
            "video": False
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        audio_receiver_size=1024,
        async_processing=True,
        audio_processor_factory=lambda: audio_processor
    )

    if webrtc_ctx.state.playing:
        st.warning("🎙️ Recording... Speak into your mic.")
    elif not webrtc_ctx.state.playing and hasattr(audio_processor, 'frames') and audio_processor.frames:
        with st.spinner("Transcribing your speech..."):
            audio_bytes = b''.join([f.to_ndarray().tobytes() for f in audio_processor.frames])
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                write_wav(tmpfile.name, 16000, np.frombuffer(audio_bytes, dtype=np.int16))
                tmp_path = tmpfile.name

            try:
                result = whisper_model.transcribe(tmp_path)
                text = result["text"].strip()
                st.text_area("Transcribed Text", value=text, height=100)

                sentiment, confidence = classify_sentiment(text)
                emotions = classify_emotion(text)
                st.success(f"Sentiment: **{sentiment}** ({confidence*100:.1f}% confidence)")
                st.info(f"Emotions detected: {', '.join(emotions) if emotions else 'None'}")
            except Exception as e:
                st.error(f"Error during transcription: {e}")
            finally:
                os.remove(tmp_path)
                audio_processor.frames.clear()
