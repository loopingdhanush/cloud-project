
import joblib
import pandas as pd
import random
import time
from datetime import datetime
from pathlib import Path

# File paths relative to this Python script
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "sensor_model.pkl"
LOG_FILE = BASE_DIR / "decision_log.txt"

# Load the trained ML model
model = joblib.load(MODEL_PATH)

FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "current",
    "rpm"
]


def generate_sensor_data():
    """Generate one set of simulated sensor readings."""
    return {
        "temperature": round(random.uniform(40, 100), 2),
        "pressure": round(random.uniform(1, 12), 2),
        "vibration": round(random.uniform(0, 6), 2),
        "current": round(random.uniform(0.5, 25), 2),
        "rpm": random.randint(500, 3500)
    }


def predict_state(reading):
    """Predict machine condition using the trained ML model."""
    X = pd.DataFrame([reading], columns=FEATURES)
    prediction = model.predict(X)[0]
    return str(prediction).strip().lower()


def take_action(prediction):
    """Recommend an action based on the ML prediction."""
    actions = {
        "normal": "All readings are normal. Machine continues running.",
        "warning": "Abnormal reading detected. Schedule machine inspection.",
        "critical": "Critical condition detected. Stop machine and alert operator."
    }

    return actions.get(
        prediction,
        f"Unknown predicted state: {prediction}. Inspect the machine."
    )


def log_decision(reading, prediction, action):
    """Append the prediction and action to the log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"{datetime.now().isoformat(timespec='seconds')} | "
            f"Reading: {reading} | "
            f"Prediction: {prediction} | "
            f"Action: {action}\n"
        )


def main():
    print("ML-Based Predictive Maintenance Started")
    print("-" * 55)

    for tick in range(1, 11):
        reading = generate_sensor_data()

        # ML model predicts the state
        prediction = predict_state(reading)

        # Display an action based on the prediction
        outcome = take_action(prediction)

        print(f"[Tick {tick:02d}] SENSOR DATA: {reading}")
        print(f"           ML PREDICTION: {prediction.upper()}")
        print(f"           ACTION: {outcome}\n")

        log_decision(reading, prediction, outcome)
        time.sleep(0.2)

    print(f"Completed. Decision log: {LOG_FILE}")


if __name__ == "__main__":
    main()
