# 🎙️ Sentiment & Emotion Analyzer Web App

Welcome to the **Sentiment & Emotion Analyzer**, a Streamlit web app that lets users classify **sentiment** (positive/negative) and **emotions** (multi-label from 28 emotion categories) from both **text and speech inputs**.

🔗 **Live Demo:** [https://sentiment-web-app.streamlit.app/](https://sentiment-web-app.streamlit.app/)

---

## 🚀 Features

- 📃 **Text Input**: Enter any text to analyze sentiment and emotions.
- 🎤 **Speech Input**: Use your microphone to record up to 10 seconds of audio.
- 🔊 **Whisper Transcription**: Converts speech to text using OpenAI's Whisper model.
- 🧠 **Deep Learning Models**: Uses pre-trained CNN + LSTM Keras models for prediction.
- 🎨 **Blue and White Theme**: A clean, accessible interface with a professional design.

---

## 📦 Files in This Repository

```bash
.
├── app_with_speech.py          # Main Streamlit app
├── requirements.txt            # Python dependencies for deployment
├── tokenizer_imdb.pkl          # Tokenizer for IMDb sentiment model
├── tokenizer_go.pkl            # Tokenizer for GoEmotions emotion model
├── best_imdb_model.keras       # Pre-trained sentiment classification model
└── (GoEmotions model downloaded from GitHub release)
```

---

## 🧪 Models Used

- **IMDb (Sentiment Analysis)**: Binary classification (positive/negative)
- **GoEmotions (Emotion Classification)**: Multi-label classification (28 emotion categories)
- Models are pre-trained and loaded on app startup.

---

## 🧰 How to Run Locally

> Python 3.9+ is recommended.

### 1. Clone the repository

```bash
git clone https://github.com/hemerson18/Sentiment-Web-App.git
cd Sentiment-Web-App
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app_with_speech.py
```

Then open `http://localhost:8501` in your browser.

---

## 🌐 Deployment on Streamlit Cloud

The app is deployed at:

👉 [https://sentiment-web-app.streamlit.app](https://sentiment-web-app.streamlit.app)

No installation needed. Simply visit the link and:

- Use the **left text box** to type text and click **Analyze Text**.
- Use the **right red button** to **record speech**, which will be automatically transcribed and analyzed.

---

## 🔧 Troubleshooting & Notes

### Audio Button Not Working?

- ✔ Use **Chrome** or **Edge** — Firefox and Safari may block microphone access.
- ✔ Make sure your browser has **permission to access your mic**.
- ❌ iPhones and some mobile browsers may restrict WebRTC-based recording.

### Model Download Error?

- The `GoEmotions` model is dynamically downloaded (~25MB) from GitHub Releases:
  [Download Link](https://github.com/hemerson18/Sentiment-Web-App/releases)

---

## 🙋 Authors

- **Harry James Emerson**  
- **Amir Eid**  
- **Gabriel Luciano Magnago**

Advanced Predictive Analytics — Católica Lisbon / FGV EBAPE  
Spring 2025

---

## 📚 References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [GoEmotions Dataset](https://github.com/google-research/google-research/tree/master/goemotions)
- [Streamlit Docs](https://docs.streamlit.io/)
- [LangChain](https://python.langchain.com/)
