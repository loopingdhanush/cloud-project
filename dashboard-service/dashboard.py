import requests
import streamlit as st
import pandas as pd

ML_API_URL = "https://predictive-maintenance-api-119540237309.asia-south1.run.app"
AGENT_API_URL = "https://maintenance-agent-119540237309.asia-south1.run.app"

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 AI Predictive Maintenance System")
st.caption("Machine learning predictions + Agentic AI investigation")

st.sidebar.header("Machine Sensor Readings")

temperature = st.sidebar.slider("Temperature (°C)", 40.0, 100.0, 75.0)
pressure = st.sidebar.slider("Pressure (bar)", 1.0, 12.0, 5.5)
vibration = st.sidebar.slider("Vibration (mm/s)", 0.0, 6.0, 2.0)
current = st.sidebar.slider("Current (A)", 0.5, 25.0, 10.0)
rpm = st.sidebar.slider("RPM", 500, 3500, 1800)

readings = {
    "temperature": temperature,
    "pressure": pressure,
    "vibration": vibration,
    "current": current,
    "rpm": rpm,
}

st.subheader("Current Sensor Readings")
cols = st.columns(5)
for col, (name, value) in zip(cols, readings.items()):
    col.metric(name.replace("_", " ").title(), value)

st.divider()

# -------------------------------------------------
# Section 1: ML prediction
# -------------------------------------------------
st.header("1. Machine Learning Prediction")
st.write("Use the trained Decision Tree model to predict machine condition.")

if st.button("Predict Machine Condition", type="primary"):
    try:
        response = requests.post(
            f"{ML_API_URL}/predict",
            json=readings,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        prediction = str(result["prediction"]).upper()

        st.session_state["last_prediction"] = result

        if prediction == "NORMAL":
            st.success(f"Predicted state: {prediction}")
        elif prediction in ("WARNING", "HIGH_TEMP"):
            st.warning(f"Predicted state: {prediction}")
        else:
            st.error(f"Predicted state: {prediction}")

        st.write("**Recommended action:**", result["recommended_action"])

    except requests.RequestException as error:
        st.error(f"Could not reach the ML API: {error}")

st.divider()

# -------------------------------------------------
# Section 2: Agentic AI assistant
# -------------------------------------------------
st.header("2. 🤖 AI Maintenance Assistant")
st.write(
    "Ask the AI agent to investigate the sensor readings, "
    "use diagnostic tools, and recommend what to inspect."
)

with st.form("maintenance_agent_form"):
    question = st.text_area(
        "Ask a maintenance question",
        value=(
            "Investigate this machine. Identify the readings that need "
            "attention, explain possible causes, and recommend what "
            "the operator should inspect first."
        ),
        height=110,
    )

    ask_agent = st.form_submit_button(
        "Investigate with AI Agent",
        type="primary",
    )

if ask_agent:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        payload = {
            "question": question,
            "sensor_readings": readings,
        }

        with st.spinner(
            "The AI agent is investigating the machine and using its tools..."
        ):
            try:
                response = requests.post(
                    f"{AGENT_API_URL}/agent/ask",
                    json=payload,
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()

                st.session_state["last_agent_response"] = result.get(
                    "agent_response",
                    "The agent returned no response.",
                )

            except requests.RequestException as error:
                st.error(f"Could not reach the AI agent: {error}")

if st.session_state.get("last_agent_response"):
    st.subheader("AI Investigation Report")
    st.markdown(st.session_state["last_agent_response"])

st.divider()
st.caption(
    "Demo only: predictions and diagnostic guidance are not certified "
    "industrial safety decisions. Follow approved site procedures."
)
