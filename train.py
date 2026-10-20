
from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Keep transactions in chronological order.
df = df.sort_values("Time", kind="stable").reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Total fraud cases:", int(df["Class"].sum()))

# First 80% for training; final 20% for testing.
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Training fraud cases:", int(y_train.sum()))
print("Testing fraud cases:", int(y_test.sum()))
print("Last training Time:", train_df["Time"].max())
print("First testing Time:", test_df["Time"].min())

# Compute imbalance ratio using training data only.
fraud_count = int(y_train.sum())
normal_count = int((y_train == 0).sum())
scale_pos_weight = normal_count / max(fraud_count, 1)

print("\nTraining class imbalance ratio:", round(scale_pos_weight, 2))

print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

pr_auc = average_precision_score(y_test, probabilities)
roc_auc = roc_auc_score(y_test, probabilities)

print("PR-AUC:", round(pr_auc, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("Fraud Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
print("Fraud Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))

# Save model and evaluation results.
joblib.dump(model, MODEL_DIR / "fraud_model.pkl")

metrics = {
    "split_method": "chronological 80/20 by Time",
    "training_rows": int(len(train_df)),
    "testing_rows": int(len(test_df)),
    "training_fraud_cases": fraud_count,
    "testing_fraud_cases": int(y_test.sum()),
    "pr_auc": float(pr_auc),
    "roc_auc": float(roc_auc),
    "fraud_precision": float(precision_score(y_test, predictions, zero_division=0)),
    "fraud_recall": float(recall_score(y_test, predictions, zero_division=0)),
    "scale_pos_weight": float(scale_pos_weight),
}

with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved:", MODEL_DIR / "fraud_model.pkl")
print("Metrics saved:", MODEL_DIR / "metrics.json")
print("\nChronological training completed successfully!")
