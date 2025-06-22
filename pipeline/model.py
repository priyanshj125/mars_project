from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

def build_lstm_model(input_shape, num_classes):
    model = Sequential()

    # Mask padding (so model ignores padded timesteps)
    model.add(Masking(mask_value=0.0, input_shape=input_shape))

    # First LSTM layer
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))

    # Second LSTM layer (deeper)
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))

    # Fully connected layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
