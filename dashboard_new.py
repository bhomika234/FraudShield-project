
import json
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================
# 1. PROJECT CONFIGURATION
# =========================
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "dashboard_sample.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model_tuned.pkl"
THRESHOLD_PATH = BASE_DIR / "reports" / "operating_point.json"

TRANSACTION_LIMIT = 100

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

REQUIRED_COLUMNS = FEATURE_COLUMNS + ["Class"]


# =========================
# 2. STREAMLIT PAGE SETTINGS
# =========================
st.set_page_config(
    page_title="FraudShield Dashboard",
    page_icon="🛡️",
    layout="wide",
)


# =========================
# 3. DASHBOARD STYLING
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #081225 0%, #101e36 100%);
        color: #e5edf8;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background: #0b1426;
        border-right: 1px solid #243651;
    }

    h1, h2, h3 {
        color: #f1f5f9 !important;
    }

    p, label, .stCaption {
        color: #b8c7dc;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #14243d, #101b30);
        border: 1px solid #294362;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
    }

    div[data-testid="stMetricLabel"] {
        color: #9fb4d0 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #60a5fa !important;
        font-weight: 700;
    }

    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #0ea5e9);
        color: white;
        border: none;
        border-radius: 9px;
        padding: 0.55rem 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        border: 1px solid #93c5fd;
        color: white;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #294362;
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        background: #13243c;
        border: 1px solid #294362;
        border-radius: 10px;
    }

    hr {
        border-color: #294362;
    }

    a {
        color: #60a5fa !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# 4. DASHBOARD HEADER
# =========================
st.markdown(
    """
    <div style="
        background: linear-gradient(110deg, #132746, #174b78);
        border: 1px solid #2d5b88;
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 24px;
    ">
        <div style="
            color:#60a5fa;
            font-size:12px;
            font-weight:700;
            letter-spacing:3px;
        ">
            AI-POWERED SECURITY
        </div>

        <h1 style="color:#ffffff !important; margin:8px 0;">
            🛡️ FraudShield
        </h1>

        <p style="color:#cbdcf1; font-size:16px; margin-bottom:0;">
            Real-time Credit Card Fraud Detection & Transaction Monitoring
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# 5. LOAD TRAINED MODEL
# =========================
@st.cache_resource
def load_model(model_path, modified_time):
    return joblib.load(model_path)


@st.cache_data
def load_threshold(threshold_path, modified_time):
    with open(threshold_path, "r", encoding="utf-8") as file:
        threshold = float(json.load(file)["threshold"])

    if not 0.0 < threshold < 1.0:
        raise ValueError("Fraud threshold must be between 0 and 1.")

    return threshold


try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_PATH}"
        )

    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )

    FRAUD_THRESHOLD = load_threshold(
        str(THRESHOLD_PATH),
        THRESHOLD_PATH.stat().st_mtime,
    )

    if not hasattr(model, "predict_proba"):
        raise TypeError("The loaded model does not support predict_proba().")

except Exception as exc:
    st.error(f"Could not load fraud model or threshold: {exc}")
    st.stop()


# =========================
# 6. LOAD TEST DATA
# =========================
@st.cache_data
def load_test_data(path, modified_time):
    data = pd.read_csv(
        path,
        usecols=REQUIRED_COLUMNS,
        nrows=TRANSACTION_LIMIT,
    )

    if data.empty:
        raise ValueError("The dataset is empty.")

    return data.reset_index(drop=True)


def create_demo_data(rows=TRANSACTION_LIMIT):
    data = {
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
    }

    return pd.DataFrame(data)



if DATA_PATH.exists():
    try:
        test_data = load_test_data(
            str(DATA_PATH),
            DATA_PATH.stat().st_mtime,
        )
        data_mode = "Local curated demo sample"

    except Exception as exc:
        st.warning(
            f"Could not load CSV: {exc}. Using synthetic demo data."
        )
        test_data = create_demo_data()
        data_mode = "Synthetic demo data"

else:
    test_data = create_demo_data()
    data_mode = "Synthetic demo data"


if data_mode == "Local curated demo sample":
    st.info(
        "Data source: Local curated demo sample. "
        "This sample contains 80 legitimate and 20 fraud-labeled "
        "transactions for demonstration. It does not represent "
        "real-world fraud prevalence. Predictions are made using "
        "transaction features, not the Class label."
    )
else:
    st.info(
        "Data source: Synthetic demo data. "
        "These are generated sample transactions, not real customer data."
    )


if test_data.empty:
    st.error("No transactions are available.")
    st.stop()


# =========================
# 7. SESSION STATE
# =========================
defaults = {
    "streaming": False,
    "next_index": 0,
    "results": [],
    "fraud_count": 0,
    "latencies": [],
    "prediction_errors": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================
# 8. SIDEBAR CONTROLS
# =========================
st.sidebar.header("Control Panel")

st.sidebar.caption("Prediction mode: Direct model inference")
st.sidebar.caption(f"Fraud threshold: {FRAUD_THRESHOLD:.6f}")
st.sidebar.caption(f"Transactions available: {len(test_data)}")

stream_speed = st.sidebar.slider(
    "Interval between transactions (seconds)",
    min_value=1,
    max_value=5,
    value=1,
)


if st.sidebar.button(
    "▶ Start / Restart Monitoring",
    use_container_width=True,
):
    st.session_state.streaming = True
    st.session_state.next_index = 0
    st.session_state.results = []
    st.session_state.fraud_count = 0
    st.session_state.latencies = []
    st.session_state.prediction_errors = 0
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
    st.session_state.prediction_errors = 0
    st.rerun()


# =========================
# 9. DISPLAY TRANSACTION RESULTS
# =========================
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

        if row["Decision"] == "PREDICTION_ERROR":
            return [
                "background-color: #78350f; color: white"
            ] * len(row)

        return [""] * len(row)

    styled_df = result_df.style.apply(
        highlight_decision,
        axis=1,
    )

    st.dataframe(
        styled_df,
        width="stretch",
        hide_index=True,
    )


# =========================
# 10. LIVE TRANSACTION MONITOR
# =========================
@st.fragment(
    run_every=stream_speed if st.session_state.streaming else None
)
def transaction_monitor():

    # Process one transaction per monitoring interval
    if st.session_state.streaming:
        index = st.session_state.next_index

        if index >= len(test_data):
            st.session_state.streaming = False

        else:
            row = test_data.iloc[index]
            start_time = time.perf_counter()

            try:
                input_df = pd.DataFrame(
                    [[float(row[feature]) for feature in FEATURE_COLUMNS]],
                    columns=FEATURE_COLUMNS,
                )

                probability = float(
                    model.predict_proba(input_df)[0][1]
                )

                decision = (
                    "FRAUD"
                    if probability >= FRAUD_THRESHOLD
                    else "LEGITIMATE"
                )

                latency_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                if decision == "FRAUD":
                    st.session_state.fraud_count += 1

                st.session_state.latencies.append(latency_ms)

                record = {
                    "Time": float(row["Time"]),
                    "Amount": float(row["Amount"]),
                    "Decision": decision,
                    "Fraud Probability": round(probability, 4),
                    "Threshold": round(FRAUD_THRESHOLD, 6),
                    "Latency (ms)": round(latency_ms, 2),
                    "Status": "OK",
                }

            except Exception as exc:
                latency_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                st.session_state.prediction_errors += 1

                record = {
                    "Time": float(row["Time"]),
                    "Amount": float(row["Amount"]),
                    "Decision": "PREDICTION_ERROR",
                    "Fraud Probability": None,
                    "Threshold": round(FRAUD_THRESHOLD, 6),
                    "Latency (ms)": round(latency_ms, 2),
                    "Status": str(exc)[:120],
                }

            st.session_state.results.insert(0, record)
            st.session_state.next_index += 1

            if st.session_state.next_index >= len(test_data):
                st.session_state.streaming = False

    # Calculate metrics
    results = st.session_state.results
    latencies = st.session_state.latencies

    average_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0.0
    )

    # Dashboard metrics
    st.markdown("### Monitoring Overview")

    total_col, fraud_col, latency_col, errors_col = st.columns(4)

    with total_col:
        st.metric(
            label="📊 TRANSACTIONS SCANNED",
            value=f"{len(results):,}",
        )

    with fraud_col:
        st.metric(
            label="🚨 FRAUDS FLAGGED",
            value=f"{st.session_state.fraud_count:,}",
        )

    with latency_col:
        st.metric(
            label="⚡ AVG PREDICTION LATENCY",
            value=f"{average_latency:.2f} ms",
        )

    with errors_col:
        st.metric(
            label="⚠️ PREDICTION ERRORS",
            value=f"{st.session_state.prediction_errors:,}",
        )

    # Transaction feed
    st.markdown("---")
    st.subheader("Transaction Feed")
    st.caption("Latest model predictions appear first.")

    display_results()

    if not st.session_state.streaming:
        if st.session_state.next_index >= len(test_data):
            st.success(
                "All available sample transactions have been processed."
            )
        elif not results:
            st.info(
                "Click Start / Restart Monitoring to begin."
            )


# Render the monitoring component
transaction_monitor()
