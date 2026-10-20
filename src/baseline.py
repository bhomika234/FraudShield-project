
from pathlib import Path
import json

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# Use the same MLflow experiment as XGBoost
mlflow.set_experiment("FraudShield")

print("1. Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Keep the same chronological 80/20 split as XGBoost
df = df.sort_values("Time", kind="stable").reset_index(drop=True)
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df.drop(columns=["Class"])
X_test = test_df.drop(columns=["Class"])
y_train = train_df["Class"]
y_test = test_df["Class"]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training fraud cases:", int(y_train.sum()))
print("Testing fraud cases:", int(y_test.sum()))

print("\n2. Training Isolation Forest...")

contamination = 0.0017

model = IsolationForest(
    contamination=contamination,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train)

print("3. Evaluating model...")

# Isolation Forest predicts -1 for anomalies and 1 for normal cases
raw_predictions = model.predict(X_test)
predictions = np.where(raw_predictions == -1, 1, 0)

# Higher anomaly scores indicate more suspicious transactions
anomaly_scores = -model.score_samples(X_test)

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

metrics = {
    "pr_auc": float(average_precision_score(y_test, anomaly_scores)),
    "roc_auc": float(roc_auc_score(y_test, anomaly_scores)),
    "fraud_precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "fraud_recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "fraud_f1": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
}

print("\n--- Isolation Forest Results ---")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion Matrix:")
print(np.array([[tn, fp], [fn, tp]]))

for name, value in metrics.items():
    print(f"{name}: {value:.6f}")

# Save results locally
report = {
    "model": "Isolation Forest",
    "split_method": "chronological 80/20 by Time",
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "training_fraud_cases": int(y_train.sum()),
    "testing_fraud_cases": int(y_test.sum()),
    "contamination": contamination,
    **metrics,
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
}

report_path = REPORT_DIR / "metrics_isolation_forest.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

# Log the run in the shared FraudShield MLflow experiment
with mlflow.start_run(run_name="IsolationForest_Baseline"):
    mlflow.log_params({
        "model": "IsolationForest",
        "contamination": contamination,
        "random_state": 42,
        "split": "chronological_80_20",
    })
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(str(report_path))

print("\nMetrics saved:", report_path)
print("Isolation Forest metrics logged to MLflow.")
