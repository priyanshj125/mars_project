from pipeline.data_loader import get_ravdess_files
from pipeline.feature_extractor import extract_features
from pipeline.model import build_lstm_model
from pipeline.trainer import train_model
from pipeline.evaluator import evaluate_model, plot_history

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np

# Configuration
SPEECH_DIR = "data/Audio_Speech_Actors_01-24"
SONG_DIR = "data/Audio_Song_Actors_01-24"
SELECTED_EMOTIONS = ["happy", "sad", "angry", "calm", "fearful"]

# 🔹 1. Load files
file_paths_speech, labels_speech = get_ravdess_files(SPEECH_DIR, SELECTED_EMOTIONS)
file_paths_song, labels_song = get_ravdess_files(SONG_DIR, SELECTED_EMOTIONS)

file_paths = file_paths_speech + file_paths_song
labels = labels_speech + labels_song

# 🔹 2. Extract LSTM-compatible MFCC features
X = [extract_features(f) for f in file_paths]

# 🔹 3. Pad sequences to same time length
X = pad_sequences(X, padding='post', dtype='float32')  # shape: (samples, time_steps, 40)

# 🔹 4. Encode emotion labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
y_onehot = tf.keras.utils.to_categorical(y_encoded)

# 🔹 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 🔹 6. Build & train LSTM model
input_shape = X_train.shape[1:]  # (time_steps, 40)
model = build_lstm_model(input_shape, len(SELECTED_EMOTIONS))
history = train_model(model, X_train, y_train)

# 🔹 7. Evaluate
evaluate_model(model, X_test, y_test, label_encoder.classes_)
plot_history(history)
