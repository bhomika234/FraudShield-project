
from pathlib import Path
import json

import mlflow
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Set up MLflow experiment
mlflow.set_experiment("FraudShield")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.sort_values("Time", kind="stable").reset_index(drop=True)

# Chronological 80/20 split
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

# Calculate imbalance ratio using training data only
fraud_count = int(y_train.sum())
normal_count = int((y_train == 0).sum())
scale_pos_weight = normal_count / fraud_count

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training fraud cases:", fraud_count)
print("Testing fraud cases:", int(y_test.sum()))
print("scale_pos_weight:", round(scale_pos_weight, 2))

# Create and train XGBoost
print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

pr_auc = average_precision_score(y_test, probabilities)
roc_auc = roc_auc_score(y_test, probabilities)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)

print("PR-AUC:", round(pr_auc, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("Fraud Precision:", round(precision, 4))
print("Fraud Recall:", round(recall, 4))

# Save model
model_path = MODEL_DIR / "xgboost_fraud_model.pkl"
metrics_path = MODEL_DIR / "metrics_xgboost.json"

joblib.dump(model, model_path)

# Save evaluation metrics to JSON
metrics = {
    "model": "XGBoost",
    "split_method": "chronological 80/20 by Time",
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "training_fraud_cases": fraud_count,
    "testing_fraud_cases": int(y_test.sum()),
    "scale_pos_weight": float(scale_pos_weight),
    "pr_auc": float(pr_auc),
    "roc_auc": float(roc_auc),
    "fraud_precision": float(precision),
    "fraud_recall": float(recall),
}

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# Log experiment parameters and metrics to MLflow
with mlflow.start_run(run_name="XGBoost_Baseline"):
    mlflow.log_params({
        "model": "XGBoost",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(scale_pos_weight),
        "split": "chronological_80_20",
    })

    mlflow.log_metrics({
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "fraud_precision": float(precision),
        "fraud_recall": float(recall),
    })

print("\nXGBoost model saved:", model_path)
print("Metrics saved:", metrics_path)
print("MLflow metrics logged successfully!")
print("\nXGBoost training completed successfully!")
