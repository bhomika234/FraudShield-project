
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model_tuned.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics_xgboost_tuned.json"
REPORT_PATH = BASE_DIR / "reports" / "operating_point.json"

TARGET_FPR = 0.01

print("Loading dataset with memory-efficient data types...", flush=True)

features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
dtypes = {col: "float32" for col in features}
dtypes["Class"] = "int8"

df = pd.read_csv(DATA_PATH, dtype=dtypes)
df.sort_values("Time", kind="stable", inplace=True)
df.reset_index(drop=True, inplace=True)

# Chronological 80/20 train/test split.
split_index = int(len(df) * 0.80)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# Split the training portion into fit and validation sets.
inner_split = int(len(train_df) * 0.80)
fit_df = train_df.iloc[:inner_split]
val_df = train_df.iloc[inner_split:]

X_fit = fit_df[features]
y_fit = fit_df["Class"].astype("int8")
X_val = val_df[features]
y_val = val_df["Class"].astype("int8")

print("Loading tuned hyperparameters...", flush=True)
with open(METRICS_PATH, encoding="utf-8") as file:
    metrics = json.load(file)

params = metrics["best_parameters"].copy()
params.update({
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "scale_pos_weight": float(
        (y_fit == 0).sum() / max((y_fit == 1).sum(), 1)
    ),
})

print("Training a validation-only model...", flush=True)
validation_model = XGBClassifier(**params)
validation_model.fit(X_fit, y_fit)

print("Calculating validation probabilities...", flush=True)
val_prob = validation_model.predict_proba(X_val)[:, 1]

# Select the threshold with maximum recall while keeping FPR <= 1%.
fpr_values, recall_values, thresholds = roc_curve(y_val, val_prob)

eligible = np.flatnonzero(
    (fpr_values <= TARGET_FPR) & np.isfinite(thresholds)
)

if len(eligible) == 0:
    raise RuntimeError("No finite threshold satisfies the FPR target.")

best_idx = eligible[np.argmax(recall_values[eligible])]
best_threshold = float(thresholds[best_idx])
validation_fpr = float(fpr_values[best_idx])
validation_recall = float(recall_values[best_idx])

del validation_model, X_fit, y_fit, X_val, y_val
del fit_df, val_df, val_prob, fpr_values, recall_values, thresholds

print("Loading final tuned model...", flush=True)
model = joblib.load(MODEL_PATH)

print("Evaluating threshold on untouched test data...", flush=True)
X_test = test_df[features]
y_test = test_df["Class"].astype("int8")

test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype("int8")

tn, fp, fn, tp = confusion_matrix(
    y_test, test_pred, labels=[0, 1]
).ravel()

test_fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
test_recall = float(tp / (tp + fn)) if (tp + fn) else 0.0

results = {
    "model": "Optuna-tuned XGBoost",
    "target_fpr": TARGET_FPR,
    "threshold": best_threshold,
    "validation_fpr": validation_fpr,
    "validation_recall": validation_recall,
    "test_fpr": test_fpr,
    "test_recall_at_selected_threshold": test_recall,
    "test_true_negatives": int(tn),
    "test_false_positives": int(fp),
    "test_false_negatives": int(fn),
    "test_true_positives": int(tp),
    "note": (
        "Threshold selected using a separate chronological validation model. "
        "Test data was not used to select the threshold. Test FPR may differ "
        "from the 1% validation target."
    ),
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(
    json.dumps(results, indent=4), encoding="utf-8"
)

print("\nOperating-point results:", flush=True)
print(json.dumps(results, indent=4), flush=True)
print(f"\nReport saved to: {REPORT_PATH}", flush=True)
