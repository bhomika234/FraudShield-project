
import json
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model_tuned.pkl"
THRESHOLD_PATH = BASE_DIR / "reports" / "operating_point.json"
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

EXPECTED_FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

st.set_page_config(
    page_title="FraudShield | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FraudShield")
st.caption("Credit Card Fraud Detection — XGBoost")
st.divider()


# --------------------------------------------------
# Load model and threshold
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_threshold():
    if not THRESHOLD_PATH.exists():
        return 0.5

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    if isinstance(config, dict):
        for key in (
            "threshold",
            "fraud_threshold",
            "operating_threshold",
            "best_threshold",
        ):
            if key in config:
                return float(config[key])

    if isinstance(config, (float, int)):
        return float(config)

    return 0.5


try:
    model = load_model()
    FRAUD_THRESHOLD = load_threshold()
    st.success("Model loaded successfully")
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()


# --------------------------------------------------
# Load transactions
# --------------------------------------------------

def create_demo_data(rows=100):
    demo = pd.DataFrame(
        0.0,
        index=range(rows),
        columns=EXPECTED_FEATURES,
    )
    demo["Time"] = [i * 60 for i in range(rows)]
    demo["Amount"] = [
        float((i * 37) % 500 + 10)
        for i in range(rows)
    ]
    return demo


def load_transactions():
    if DATA_PATH.exists():
        try:
            data = pd.read_csv(DATA_PATH)

            missing = [
                col for col in EXPECTED_FEATURES
                if col not in data.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing required features: {missing}"
                )

            data = data[EXPECTED_FEATURES].copy()
            data = data.apply(pd.to_numeric, errors="raise")
            data = data.replace(
                [float("inf"), float("-inf")],
                float("nan"),
            )
            data = data.dropna()

            if not data.empty:
                return (
                    data.reset_index(drop=True),
                    "Local credit-card dataset",
                )

        except Exception as exc:
            st.warning(
                f"Could not load CSV: {exc}. Using demo data."
            )

    return create_demo_data(), "Synthetic demo data"


test_data, data_mode = load_transactions()

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
    "prediction_errors": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------

st.sidebar.header("Control Panel")
st.sidebar.caption("Prediction mode: Direct model inference")
st.sidebar.caption(
    f"Fraud threshold: {FRAUD_THRESHOLD:.6f}"
)

max_transactions = len(test_data)

scan_limit = st.sidebar.slider(
    "Transactions to scan",
    min_value=1,
    max_value=max_transactions,
    value=min(100, max_transactions),
)

start_button = st.sidebar.button(
    "▶ Start Monitoring",
    use_container_width=True,
)

stop_button = st.sidebar.button(
    "⏹ Stop Monitoring",
    use_container_width=True,
)

reset_button = st.sidebar.button(
    "🔄 Reset Results",
    use_container_width=True,
)


# --------------------------------------------------
# Handle controls
# --------------------------------------------------

if reset_button:
    st.session_state.streaming = False
    st.session_state.next_index = 0
    st.session_state.results = []
    st.session_state.prediction_errors = 0
    st.rerun()

if start_button:
    # Start a new scan if the previous scan has finished.
    if st.session_state.next_index >= scan_limit:
        st.session_state.next_index = 0
        st.session_state.results = []
        st.session_state.prediction_errors = 0

    st.session_state.streaming = True

if stop_button:
    st.session_state.streaming = False


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_transaction(transaction):
    start_time = time.perf_counter()

    input_df = pd.DataFrame(
        [transaction],
        columns=EXPECTED_FEATURES,
    )

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    prediction = (
        "FRAUD"
        if probability >= FRAUD_THRESHOLD
        else "LEGITIMATE"
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return prediction, probability, latency_ms


# --------------------------------------------------
# Run monitoring
# One transaction per rerun keeps Stop responsive.
# --------------------------------------------------

if st.session_state.streaming:
    if st.session_state.next_index >= scan_limit:
        st.session_state.streaming = False
        st.success("Monitoring completed.")
    else:
        progress = st.progress(0)
        status_text = st.empty()

        index = st.session_state.next_index

        status_text.write(
            f"Processing transaction {index + 1} "
            f"of {scan_limit}..."
        )

        transaction = (
            test_data.iloc[index][EXPECTED_FEATURES].tolist()
        )
        transaction_dict = dict(
            zip(EXPECTED_FEATURES, transaction)
        )

        try:
            prediction, probability, latency_ms = (
                predict_transaction(transaction_dict)
            )

            st.session_state.results.append({
                "Transaction ID": index + 1,
                "Prediction": prediction,
                "Fraud Probability": round(probability, 6),
                "Latency (ms)": round(latency_ms, 2),
                "Status": "OK",
            })

        except Exception as exc:
            st.session_state.prediction_errors += 1

            st.session_state.results.append({
                "Transaction ID": index + 1,
                "Prediction": "PREDICTION_ERROR",
                "Fraud Probability": None,
                "Latency (ms)": None,
                "Status": str(exc),
            })

        st.session_state.next_index += 1

        progress.progress(
            min(
                st.session_state.next_index / scan_limit,
                1.0,
            )
        )

        st.rerun()


# --------------------------------------------------
# Metrics
# --------------------------------------------------

results = st.session_state.results
display_df = pd.DataFrame(results)

if not display_df.empty:
    fraud_count = int(
        (display_df["Prediction"] == "FRAUD").sum()
    )

    legitimate_count = int(
        (display_df["Prediction"] == "LEGITIMATE").sum()
    )

    error_count = int(
        (display_df["Prediction"] == "PREDICTION_ERROR").sum()
    )

    valid_latencies = display_df["Latency (ms)"].dropna()

    avg_latency = (
        float(valid_latencies.mean())
        if not valid_latencies.empty
        else 0.0
    )
else:
    fraud_count = 0
    legitimate_count = 0
    error_count = 0
    avg_latency = 0.0


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Transactions Scanned", len(results))
col2.metric("Fraud Detected", fraud_count)
col3.metric("Legitimate", legitimate_count)
col4.metric(
    "Avg Prediction Latency",
    f"{avg_latency:.2f} ms",
)
col5.metric("Prediction Errors", error_count)

st.divider()


# --------------------------------------------------
# Results table and separate tabs
# --------------------------------------------------

st.subheader("Transaction Monitoring Results")

if not display_df.empty:
    columns_to_show = [
        "Transaction ID",
        "Prediction",
        "Fraud Probability",
        "Latency (ms)",
        "Status",
    ]

    all_tab, legitimate_tab, fraud_tab = st.tabs([
        "All Transactions",
        "Legitimate Transactions",
        "Fraud Transactions",
    ])

    with all_tab:
        st.dataframe(
            display_df[columns_to_show],
            use_container_width=True,
            hide_index=True,
        )

    with legitimate_tab:
        legitimate_df = display_df[
            display_df["Prediction"] == "LEGITIMATE"
        ]

        st.metric(
            "Legitimate Transactions",
            len(legitimate_df),
        )

        if legitimate_df.empty:
            st.info(
                "No legitimate transactions in the current results."
            )
        else:
            st.dataframe(
                legitimate_df[columns_to_show],
                use_container_width=True,
                hide_index=True,
            )

    with fraud_tab:
        fraud_df = display_df[
            display_df["Prediction"] == "FRAUD"
        ]

        st.metric(
            "Fraud Transactions",
            len(fraud_df),
        )

        if fraud_df.empty:
            st.info(
                "No fraud transactions in the current results."
            )
        else:
            st.dataframe(
                fraud_df[columns_to_show],
                use_container_width=True,
                hide_index=True,
            )

    st.download_button(
        "⬇️ Download Results CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="fraudshield_results.csv",
        mime="text/csv",
    )

    st.subheader("Prediction Summary")

    summary = pd.DataFrame({
        "Category": [
            "Fraud",
            "Legitimate",
            "Prediction Error",
        ],
        "Count": [
            fraud_count,
            legitimate_count,
            error_count,
        ],
    })

    st.bar_chart(summary.set_index("Category"))

else:
    st.info(
        "No predictions yet. Click Start Monitoring "
        "in the sidebar to begin."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "FraudShield is a machine-learning demonstration. "
    "Predictions are not a guarantee of fraud."
)
