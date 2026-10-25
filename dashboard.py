
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
).strip().rstrip("/")

PREDICT_URL = f"{API_BASE_URL}/predict"
TRANSACTION_LIMIT = 100

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]
REQUIRED_COLUMNS = FEATURE_COLUMNS + ["Class"]

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
# Dataset loading
# --------------------------------------------------
@st.cache_data
def load_test_data(path, modified_time):
    latest_rows = pd.DataFrame()

    for chunk in pd.read_csv(
        path,
        usecols=REQUIRED_COLUMNS,
        chunksize=10_000,
        low_memory=True,
    ):
        latest_rows = pd.concat(
            [latest_rows, chunk],
            ignore_index=True,
        ).tail(TRANSACTION_LIMIT)

    if latest_rows.empty:
        raise ValueError("The dataset is empty.")

    return latest_rows.reset_index(drop=True)


def create_demo_data(rows=TRANSACTION_LIMIT):
    return pd.DataFrame({
        "Time": [float(i * 60) for i in range(rows)],
        **{
            f"V{i}": [0.0] * rows
            for i in range(1, 29)
        },
        "Amount": [
            float(10 + (i * 17) % 500)
            for i in range(rows)
        ],
        "Class": [0] * rows,
    })


if DATA_PATH.exists():
    try:
        test_data = load_test_data(
            str(DATA_PATH),
            DATA_PATH.stat().st_mtime,
        )
        data_mode = "Local credit-card dataset"
    except Exception as exc:
        st.warning(
            f"Could not load CSV: {exc}. Using synthetic demo data."
        )
        test_data = create_demo_data()
        data_mode = "Synthetic demo data"
else:
    test_data = create_demo_data()
    data_mode = "Synthetic demo data"

st.info(
    f"Data source: **{data_mode}**. "
    "Synthetic demo data is not real customer transaction data."
)

if test_data.empty:
    st.error("No transactions are available.")
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
# Sidebar
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

if st.sidebar.button(
    "▶ Start / Restart Monitoring",
    use_container_width=True,
):
    st.session_state.streaming = True
    st.session_state.next_index = 0
    st.session_state.results = []
    st.session_state.fraud_count = 0
    st.session_state.latencies = []
    st.session_state.api_errors = 0
    st.rerun()

if st.sidebar.button(
    "⏹ Stop Monitoring",
    use_container_width=True,
):
    st.session_state.streaming = False
    st.rerun()

if st.sidebar.button(
    "🗑 Clear Results",
    use_container_width=True,
):
    st.session_state.streaming = False
    st.session_state.next_index = 0
    st.session_state.results = []
    st.session_state.fraud_count = 0
    st.session_state.latencies = []
    st.session_state.api_errors = 0
    st.rerun()

# --------------------------------------------------
# Results display
# --------------------------------------------------
def display_results():
    if not st.session_state.results:
        st.info("No transactions processed yet.")
        return

    result_df = pd.DataFrame(st.session_state.results)

    def highlight_decision(row):
        if row["Decision"] == "FRAUD":
            return [
                "background-color: #7f1d1d; color: white"
            ] * len(row)

        if row["Decision"] == "API_ERROR":
            return [
                "background-color: #78350f; color: white"
            ] * len(row)

        return [""] * len(row)

    st.dataframe(
        result_df.style.apply(highlight_decision, axis=1),
        width="stretch",
        hide_index=True,
    )

# --------------------------------------------------
# Live monitor
# Metrics and feed refresh together in this fragment.
# --------------------------------------------------
@st.fragment(
    run_every=stream_speed if st.session_state.streaming else None
)
def transaction_monitor():

    if st.session_state.streaming:
        index = st.session_state.next_index

        if index >= len(test_data):
            st.session_state.streaming = False
            st.success("Monitoring completed for all sample transactions.")
        else:
            row = test_data.iloc[index]

            payload = {
                f"V{i}": float(row[f"V{i}"])
                for i in range(1, 29)
            }
            payload["Time"] = float(row["Time"])
            payload["Amount"] = float(row["Amount"])

            start_time = time.perf_counter()

            try:
                response = requests.post(
                    PREDICT_URL,
                    json=payload,
                    timeout=10,
                )
                latency_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                response.raise_for_status()
                result = response.json()

                decision = str(result["decision"]).upper()
                probability = float(result["fraud_probability"])

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

            except (
                requests.RequestException,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                latency_ms = (
                    time.perf_counter() - start_time
                ) * 1000

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

            if st.session_state.next_index >= len(test_data):
                st.session_state.streaming = False

    # Always render metrics from the current session state.
    results = st.session_state.results
    latencies = st.session_state.latencies

    average_latency = (
        sum(latencies) / len(latencies)
        if latencies else 0.0
    )

    total_col, fraud_col, latency_col, errors_col = st.columns(4)

    total_col.metric(
        "Transactions Scanned",
        len(results),
    )
    fraud_col.metric(
        "Frauds Flagged",
        st.session_state.fraud_count,
    )
    latency_col.metric(
        "Average API Latency",
        f"{average_latency:.2f} ms",
    )
    errors_col.metric(
        "API Errors",
        st.session_state.api_errors,
    )

    st.subheader("Transaction Feed")
    display_results()

    if not st.session_state.streaming:
        if st.session_state.next_index >= len(test_data):
            st.success("All available sample transactions have been processed.")
        elif not results:
            st.info("Click Start / Restart Monitoring to begin.")


transaction_monitor()
