import numpy as np
import joblib
from tensorflow.keras.models import load_model

class EarthquakePredictor:
    def __init__(self, model_path: str):
        self.model = load_model(model_path, compile=False)
        self.scaler = joblib.load("saved_models/scaler.pkl") 

    def predict_magnitude(self, sequence: list) -> float:
        sequence = np.array(sequence).reshape(1, len(sequence), len(sequence[0]))
        pred = self.model.predict(sequence)

        # Desnormalizar solo la magnitud (posición 0)
        descaled = self.scaler.inverse_transform([[pred[0][0], 0, 0, 0]])
        real_magnitude = descaled[0][0]
        return float(real_magnitude)
