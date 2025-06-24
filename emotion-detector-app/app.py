import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import wave
from audio_recorder_streamlit import audio_recorder  # New audio recorder

# Load model and label encoder
model = tf.keras.models.load_model("emotion_model.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Feature extraction
def extract_features(file_path, sr=22050, n_mfcc=40):
    y, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

# Emotion prediction
def predict_emotion(file_path):
    features = extract_features(file_path)
    padded = tf.keras.preprocessing.sequence.pad_sequences([features], padding='post', dtype='float32')
    prediction = model.predict(padded)
    predicted_index = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_index])[0]

# UI
st.title("🎙️ Emotion Detection from Voice")
option = st.radio("Choose input:", ["Upload WAV file", "Record using Microphone"])

# Upload method
if option == "Upload WAV file":
    uploaded_file = st.file_uploader("Upload a .wav file", type="wav")
    if uploaded_file:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())
        st.audio("temp.wav")
        emotion = predict_emotion("temp.wav")
        st.success(f"Predicted Emotion: **{emotion}**")

# New microphone recording implementation
elif option == "Record using Microphone":
    st.info("🎤 Click the microphone to start recording. Stop automatically after 3 seconds of silence.")
    
    # Audio recorder with automatic silence detection
    audio_bytes = audio_recorder(
        energy_threshold=(-1.0, 1.0),  # Dual threshold for start/stop
        pause_threshold=3.0,            # Stop after 3 seconds of silence
        sample_rate=22050,               # Match librosa's expected sample rate
        text="",
        recording_color="#e8b62c",
        neutral_color="#6aa36f"
    )
    
    if audio_bytes:
        # Save recording to WAV file
        with open("recorded.wav", "wb") as f:
            f.write(audio_bytes)
        
        # Playback and prediction
        st.audio(audio_bytes, format="audio/wav")
        emotion = predict_emotion("recorded.wav")
        st.success(f"Predicted Emotion: **{emotion}**")
