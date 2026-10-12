import os
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

PROJECT_ID = os.environ.get(
    "GOOGLE_CLOUD_PROJECT", "gen-lang-client-0756336844"
)
REGION = os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-south1")

ML_API_URL = (
    "https://predictive-maintenance-api-119540237309.asia-south1.run.app"
)

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=REGION,
)

app = FastAPI(title="Agentic Predictive Maintenance API")


class AgentRequest(BaseModel):
    question: str
    sensor_readings: dict[str, float] | None = None


def predict_machine_condition(
    temperature: float,
    pressure: float,
    vibration: float,
    current: float,
    rpm: int,
) -> dict:
    """Call the trained ML API to predict a machine's condition from its five sensor readings."""
    readings = {
        "temperature": temperature,
        "pressure": pressure,
        "vibration": vibration,
        "current": current,
        "rpm": rpm,
    }

    response = requests.post(
        f"{ML_API_URL}/predict",
        json=readings,
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def inspect_sensor_readings(
    temperature: float,
    pressure: float,
    vibration: float,
    current: float,
    rpm: int,
) -> dict:
    """Inspect supplied sensor values and identify values worth investigating using preliminary demo thresholds."""
    checks = {
        "temperature": (temperature, 80, 90),
        "pressure": (pressure, 8, 10),
        "vibration": (vibration, 4, 5),
        "current": (current, 18, 22),
        "rpm": (rpm, 3000, 3300),
    }

    findings = []
    for name, (value, warning, critical) in checks.items():
        if value >= critical:
            findings.append(
                {"sensor": name, "value": value, "level": "high concern"}
            )
        elif value >= warning:
            findings.append(
                {"sensor": name, "value": value, "level": "investigate"}
            )

    return {
        "findings": findings,
        "note": (
            "These are illustrative demo thresholds, not validated "
            "industrial safety limits. Interpret them with the ML result "
            "and site-specific engineering guidance."
        ),
    }


def lookup_maintenance_guidance(symptom: str) -> dict:
    """Retrieve general troubleshooting guidance for a machine symptom."""
    guides = {
        "temperature": (
            "Check cooling airflow, coolant level, lubrication, load, "
            "and temperature sensor calibration."
        ),
        "vibration": (
            "Inspect bearing condition, alignment, imbalance, mounting, "
            "and looseness. Compare against manufacturer limits."
        ),
        "pressure": (
            "Inspect for leaks, blocked filters, regulator issues, and "
            "pressure sensor calibration."
        ),
        "current": (
            "Compare current with the motor nameplate and operating load. "
            "Inspect for overload, friction, and electrical faults."
        ),
        "rpm": (
            "Compare RPM with the intended operating range. Check load, "
            "drive settings, and speed sensor accuracy."
        ),
    }

    text = symptom.lower()
    matched = [
        {"symptom": key, "guidance": value}
        for key, value in guides.items()
        if key in text
    ]

    return {
        "guidance": matched or [
            {
                "symptom": "general",
                "guidance": (
                    "Review recent readings, machine operating limits, "
                    "maintenance records, and the manufacturer's manual."
                ),
            }
        ],
        "note": "Guidance is advisory; confirm findings with a qualified operator.",
    }


SYSTEM_INSTRUCTION = """
You are an agentic AI assistant for predictive maintenance.

You can call tools to obtain an ML prediction, inspect sensor readings,
and retrieve general troubleshooting guidance.

Use tools when they help answer the operator's question. For diagnosis,
combine evidence from the tools and explain what supports your conclusion.
Do not claim that a possible cause is confirmed without evidence.
Do not invent sensor values, machine history, or maintenance events.
If readings are missing, ask the operator to provide them.
Clearly distinguish the ML model's prediction from your own diagnostic
interpretation. The sensor thresholds are illustrative, not certified
industrial safety limits.
Never claim to have stopped, repaired, or controlled a real machine.
For critical-looking cases, recommend following the site's approved
safety procedure and contacting the responsible operator.
"""


@app.get("/")
def home():
    return {"message": "Agentic Predictive Maintenance API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "agent": "Gemini with tool calling"}


@app.post("/agent/ask")
def ask_agent(request: AgentRequest):
    prompt = f"Operator question: {request.question}"

    if request.sensor_readings:
        prompt += (
            "\nProvided sensor readings (use only these supplied values): "
            f"{request.sensor_readings}"
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[
                    predict_machine_condition,
                    inspect_sensor_readings,
                    lookup_maintenance_guidance,
                ],
                temperature=0.2,
            ),
        )

        return {
            "question": request.question,
            "sensor_readings": request.sensor_readings,
            "agent_response": response.text,
        }

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Agent request failed: {str(error)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
