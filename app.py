
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "day2" / "sensor_model.pkl"

FEATURES = ["temperature", "pressure", "vibration", "current", "rpm"]

model = joblib.load(MODEL_PATH)
app = FastAPI(title="Predictive Maintenance ML API")


class SensorReading(BaseModel):
    temperature: float
    pressure: float
    vibration: float
    current: float
    rpm: int


@app.get("/")
def home():
    return {"message": "Predictive Maintenance ML API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(reading: SensorReading):
    values = reading.model_dump()
    X = pd.DataFrame([values], columns=FEATURES)
    prediction = str(model.predict(X)[0]).strip().upper()

    actions = {
        "NORMAL": "Machine continues running.",
        "WARNING": "Schedule machine inspection.",
        "CRITICAL": "Stop machine and alert the operator."
    }

    return {
        "sensor_readings": values,
        "prediction": prediction,
        "recommended_action": actions.get(
            prediction, "Inspect the machine."
        )
    }
