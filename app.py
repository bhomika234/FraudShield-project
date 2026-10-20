from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

app = FastAPI(
    title="FraudShield API",
    description="Real-Time Fraud Detection API",
    version="2.1",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model_tuned.pkl"
THRESHOLD_PATH = BASE_DIR / "reports" / "operating_point.json"

# Load the threshold selected using validation data.
DEFAULT_THRESHOLD = 0.5

try:
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        operating_point = json.load(file)

    FRAUD_THRESHOLD = float(operating_point["threshold"])

    if not 0.0 < FRAUD_THRESHOLD < 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    print(f"Operating threshold loaded: {FRAUD_THRESHOLD:.6f}")

except (OSError, ValueError, KeyError, TypeError) as exc:
    FRAUD_THRESHOLD = DEFAULT_THRESHOLD
    print(
        f"Warning: could not load operating threshold: {exc}. "
        f"Using default threshold {DEFAULT_THRESHOLD}."
    )

# Keep the feature order consistent with the trained model.
EXPECTED_FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

model = None
model_load_error = None

try:
    model = joblib.load(MODEL_PATH)
    print("Tuned XGBoost model loaded successfully!")
except Exception as exc:
    model_load_error = str(exc)
    print(f"Model load error: {model_load_error}")


class TransactionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


def check_model():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Fraud detection model unavailable: {model_load_error}",
        )


def predict_one(transaction: TransactionInput):
    check_model()

    data = transaction.model_dump()
    df = pd.DataFrame(
        [[data[feature] for feature in EXPECTED_FEATURES]],
        columns=EXPECTED_FEATURES,
    )

    try:
        probability = float(model.predict_proba(df)[0][1])

        # Apply the validation-selected operating threshold.
        prediction = int(probability >= FRAUD_THRESHOLD)

        return {
            "prediction": prediction,
            "fraud_probability": round(probability, 4),
            "threshold": round(FRAUD_THRESHOLD, 6),
            "decision": "FRAUD" if prediction == 1 else "LEGITIMATE",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc


@app.get("/")
def home():
    return {
        "message": "FraudShield API is running successfully!",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "model_loaded": False,
                "error": model_load_error,
            },
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "fraud_threshold": FRAUD_THRESHOLD,
    }


@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    return predict_one(transaction)


@app.post("/predict/batch")
def predict_batch(transactions: list[TransactionInput]):
    if not transactions:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one transaction.",
        )

    check_model()

    return {
        "count": len(transactions),
        "predictions": [
            predict_one(transaction) for transaction in transactions
        ],
    }