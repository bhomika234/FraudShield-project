import time
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="FraudShield Live Dashboard", page_icon="🛡️", layout="wide"
)

st.title("🛡️ FraudShield: Real-Time Transaction Monitoring Dashboard")
st.markdown(
    "Monitoring incoming credit card transactions and detecting fraud in"
    " real-time using FastAPI & Tuned XGBoost."
)


# Load test data to stream transactions from it
@st.cache_data
def load_test_data():
  df = pd.read_csv("data/creditcard.csv")
  # Let's take a sample of transactions (e.g., last 100 rows)
  return df.tail(100)


test_data = load_test_data()

# Sidebar controls
st.sidebar.header("Control Panel")
stream_speed = st.sidebar.slider(
    "Streaming Speed (seconds per transaction)", 0.5, 3.0, 1.0
)
start_button = st.sidebar.button("▶ Start Live Stream")
stop_button = st.sidebar.button("⏹ Stop Stream")

# Metrics placeholders
col1, col2, col3 = st.columns(3)
total_metric = col1.empty()
fraud_metric = col2.empty()
latency_metric = col3.empty()

total_tx = 0
fraud_tx = 0

st.subheader("Live Transaction Feed")
table_placeholder = st.empty()

# Stream simulation logic
if "streaming" not in st.session_state:
  st.session_state.streaming = False

if start_button:
  st.session_state.streaming = True

if stop_button:
  st.session_state.streaming = False

# Live streaming loop
if st.session_state.streaming:
  stream_data = []
  for index, row in test_data.iterrows():
    if not st.session_state.streaming:
      break

    # Prepare safe payload for API by converting numpy types to standard python floats
    input_data = row.drop("Class").to_dict()
    payload = {k: float(v) for k, v in input_data.items()}

    start_time = time.time()
    try:
      response = requests.post("http://127.0.0.1:8000/predict", json=payload)
      latency = (time.time() - start_time) * 1000  # converted to milliseconds

      if response.status_code == 200:
        res_json = response.json()
        decision = res_json["decision"]
        prob = res_json["fraud_probability"]
      else:
        decision = f"ERR_{response.status_code}"
        prob = 0.0
        latency = 0
    except Exception as e:
      decision = "API_ERROR"
      prob = 0.0
      latency = 0

    total_tx += 1
    if decision == "FRAUD":
      fraud_tx += 1

    # Append to live feed
    row_data = {
        "Time": row["Time"],
        "Amount": row["Amount"],
        "Decision": decision,
        "Fraud Probability": prob,
        "Latency (ms)": round(latency, 2),
    }
    stream_data.insert(0, row_data)  # Add latest on top

    # Update metrics
    total_metric.metric("Total Transactions Scanned", total_tx)
    fraud_metric.metric("Frauds Flagged", fraud_tx)
    latency_metric.metric("Avg Latency", f"{round(latency, 2)} ms")

    # Update table with modern width parameter
    df_display = pd.DataFrame(stream_data)
    table_placeholder.dataframe(df_display, width="stretch")

    time.sleep(stream_speed)
else:
  st.info(
      "👈 Click on 'Start Live Stream' in the sidebar to start live"
      " monitoring!"
  )