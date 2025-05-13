import os
import numpy as np
import joblib  # nuevo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from app.utils.preprocess import load_and_process_data
from tensorflow.keras.losses import MeanSquaredError

X, y, scaler = load_and_process_data("data/raw/earthquakes.csv")

model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(loss=MeanSquaredError(), optimizer='adam', metrics=['mae'])

early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X, y, epochs=50, batch_size=32, callbacks=[early_stop])

os.makedirs("saved_models", exist_ok=True)
model.save("saved_models/lstm_model.h5")
joblib.dump(scaler, "saved_models/scaler.pkl")


print("✅ Modelo LSTM y scaler guardados correctamente")
