from fastapi import FastAPI
from app.schemas.request import SeismicSequence
from app.models.predictor import EarthquakePredictor

app = FastAPI()


predictor = EarthquakePredictor("saved_models/lstm_model.h5")

@app.get("/")
def root():
    return {"message": "API de Predicción Sismica con LSTM lista "}

@app.post("/predict/")
def predict(seq: SeismicSequence):
    prediction = predictor.predict_magnitude(seq.sequence)
    return {"predicted_magnitude": prediction}