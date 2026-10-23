
import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

API_BASE_URL = os.getenv(
    "FRAUDSHIELD_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

PREDICT_URL = f"{API_BASE_URL}/predict"
TRANSACTION_LIMIT = 100

st.set_page_config(
    page_title="FraudShield Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FraudShield — Transaction Monitoring")
st.caption(
    "Credit card fraud monitoring powered by FastAPI and tuned XGBoost."
)

# --------------------------------------------------
# Load sample transactions
# --------------------------------------------------
@st.cache_data
def load_test_data(path, modified_time):
    df = pd.read_csv(path)

    required_columns = {
        "Time",
        "Amount",
        "Class",
        *[f"V{i}" for i in range(1, 29)],
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    return df.tail(TRANSACTION_LIMIT).reset_index(drop=True)


if not DATA_PATH.exists():
    st.error(
        f"Dataset not found: {DATA_PATH}\n\n"
        "Place creditcard.csv inside the project's data folder."
    )
    st.stop()

try:
    test_data = load_test_data(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Could not load transaction data: {exc}")
    st.stop()

if test_data.empty:
    st.warning("The dataset contains no transactions.")
    st.stop()

# --------------------------------------------------
# Session state
# --------------------------------------------------
defaults = {
    "streaming": False,
    "next_index": 0,
    "results": [],
    "fraud_count": 0,
    "latencies": [],
    "api_errors": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Control Panel")

stream_speed = st.sidebar.slider(
    "Interval between transactions (seconds)",
    min_value=1,
    max_value=5,
    value=1,
)

st.sidebar.caption(f"API endpoint: {PREDICT_URL}")
st.sidebar.caption(f"Transactions available: {len(test_data)}")

if st.sidebar.button("▶ Start / Restart Monitoring", use_container_width=True):
    st.session_state.streaming = True
    st.session_state.next_index = 0
    st.session_state.results = []
    st.session_state.fraud_count = 0
    st.session_state.latencies = []
    st.session_state.api_errors = 0
    st.rerun()

if st.sidebar.button("⏹ Stop Monitoring", use_container_width=True):
    st.session_state.streaming = False
    st.rerun()

if st.sidebar.button("🗑 Clear Results", use_container_width=True):
    st.session_state.streaming = False
    st.session_state.next_index = 0
    st.session_state.results = []
    st.session_state.fraud_count = 0
    st.session_state.latencies = []
    st.session_state.api_errors = 0
    st.rerun()

# --------------------------------------------------
# Dashboard metrics
# --------------------------------------------------
results = st.session_state.results

total_col, fraud_col, latency_col, errors_col = st.columns(4)

total_col.metric("Transactions Scanned", len(results))
fraud_col.metric("Frauds Flagged", st.session_state.fraud_count)

latencies = st.session_state.latencies
average_latency = sum(latencies) / len(latencies) if latencies else 0.0

latency_col.metric(
    "Average API Latency",
    f"{average_latency:.2f} ms",
)

errors_col.metric("API Errors", st.session_state.api_errors)

st.subheader("Transaction Feed")

# --------------------------------------------------
# Process one transaction per fragment rerun
# --------------------------------------------------
@st.fragment(
    run_every=stream_speed if st.session_state.streaming else None
)
def transaction_monitor():
    if not st.session_state.streaming:
        if st.session_state.next_index >= len(test_data):
            st.success("All sample transactions have been processed.")
        else:
            st.info("Start monitoring from the sidebar to process transactions.")

        if st.session_state.results:
            display_results()
        return

    index = st.session_state.next_index

    if index >= len(test_data):
        st.session_state.streaming = False
        st.success("Monitoring completed for the available sample.")
        display_results()
        return

    row = test_data.iloc[index]
    payload = {
        feature: float(row[feature])
        for feature in test_data.columns
        if feature != "Class"
    }

    start_time = time.perf_counter()

    try:
        response = requests.post(
            PREDICT_URL,
            json=payload,
            timeout=10,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        response.raise_for_status()
        result = response.json()

        decision = result.get("decision", "UNKNOWN")
        probability = float(result.get("fraud_probability", 0.0))

        if decision == "FRAUD":
            st.session_state.fraud_count += 1

        st.session_state.latencies.append(latency_ms)

        record = {
            "Time": float(row["Time"]),
            "Amount": float(row["Amount"]),
            "Decision": decision,
            "Fraud Probability": round(probability, 4),
            "Latency (ms)": round(latency_ms, 2),
            "API Status": "OK",
        }

    except (requests.RequestException, ValueError, KeyError) as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000
        st.session_state.api_errors += 1

        record = {
            "Time": float(row["Time"]),
            "Amount": float(row["Amount"]),
            "Decision": "API_ERROR",
            "Fraud Probability": None,
            "Latency (ms)": round(latency_ms, 2),
            "API Status": str(exc)[:120],
        }

    st.session_state.results.insert(0, record)
    st.session_state.next_index += 1

    display_results()

    if st.session_state.next_index >= len(test_data):
        st.session_state.streaming = False
        st.success("Monitoring completed for the available sample.")


def display_results():
    if not st.session_state.results:
        st.info("No transactions processed yet.")
        return

    result_df = pd.DataFrame(st.session_state.results)

    def highlight_decision(row):
        if row["Decision"] == "FRAUD":
            return [
                "background-color: #7f1d1d; color: white"
                for _ in row
            ]
        if row["Decision"] == "API_ERROR":
            return [
                "background-color: #78350f; color: white"
                for _ in row
            ]
        return ["" for _ in row]

    styled_df = result_df.style.apply(
        highlight_decision,
        axis=1,
    )

    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
    )


transaction_monitor()
