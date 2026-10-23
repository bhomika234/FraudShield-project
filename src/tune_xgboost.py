
from pathlib import Path
import json

import joblib
import mlflow
import optuna
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

# --------------------------------------------------
# Project configuration
# --------------------------------------------------
SEED = 42
N_TRIALS = 50

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place creditcard.csv inside the project's data folder."
    )

# --------------------------------------------------
# Load and validate dataset
# --------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

required_columns = {
    "Time",
    "Amount",
    "Class",
    *[f"V{i}" for i in range(1, 29)],
}

missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {sorted(missing_columns)}"
    )

if df["Class"].isna().any() or not set(df["Class"].unique()).issubset({0, 1}):
    raise ValueError("Class must contain only non-null binary labels 0 and 1.")

df = df.sort_values("Time", kind="stable").reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Total fraud cases:", int(df["Class"].sum()))

# --------------------------------------------------
# Chronological train / test split
# Test set remains untouched during tuning.
# --------------------------------------------------
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

# Time-ordered validation split inside training data
inner_split = int(len(train_df) * 0.80)

fit_df = train_df.iloc[:inner_split]
val_df = train_df.iloc[inner_split:]

X_fit = fit_df.drop(columns=["Class"])
y_fit = fit_df["Class"]

X_val = val_df.drop(columns=["Class"])
y_val = val_df["Class"]

X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

if y_fit.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Training, validation, and test partitions must each contain "
        "both classes for reliable evaluation."
    )

scale_pos_weight = (
    int((y_fit == 0).sum()) / max(int(y_fit.sum()), 1)
)

print("Model fitting rows:", len(X_fit))
print("Validation rows:", len(X_val))
print("Final training rows:", len(X_train))
print("Untouched test rows:", len(X_test))
print("Training fraud cases:", int(y_train.sum()))
print("Test fraud cases:", int(y_test.sum()))
print("scale_pos_weight:", round(scale_pos_weight, 2))

# --------------------------------------------------
# MLflow configuration
# --------------------------------------------------
mlflow.set_experiment("FraudShield")

# --------------------------------------------------
# Optuna objective
# --------------------------------------------------
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.7, 1.0
        ),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 0.001, 10.0, log=True
        ),
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
    }

    model = XGBClassifier(**params)
    model.fit(X_fit, y_fit)

    val_probabilities = model.predict_proba(X_val)[:, 1]
    score = average_precision_score(y_val, val_probabilities)

    # Trial run is nested inside the parent MLflow run.
    with mlflow.start_run(
        run_name=f"Optuna_Trial_{trial.number + 1}",
        nested=True,
    ):
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number + 1)
        mlflow.log_metric("validation_pr_auc", float(score))

    print(
        f"Trial {trial.number + 1}/{N_TRIALS} | "
        f"Validation PR-AUC: {score:.4f}"
    )

    return float(score)


# --------------------------------------------------
# Reproducible Optuna search
# --------------------------------------------------
print(f"\nStarting {N_TRIALS} Optuna trials...")

sampler = optuna.samplers.TPESampler(seed=SEED)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
)

with mlflow.start_run(run_name="XGBoost_Optuna_Search"):
    mlflow.log_params({
        "model": "XGBoost",
        "optuna_trials": N_TRIALS,
        "sampler": "TPESampler",
        "sampler_seed": SEED,
        "random_state": SEED,
        "validation_metric": "average_precision",
        "split_method": "chronological_80_20_with_time_ordered_validation",
    })

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=1,
    )

    mlflow.log_metric(
        "best_validation_pr_auc",
        float(study.best_value),
    )

print("\nBest validation PR-AUC:", round(study.best_value, 4))
print("Best parameters:", study.best_params)

# --------------------------------------------------
# Refit best parameters on the complete training split
# --------------------------------------------------
best_params = study.best_params.copy()

best_params.update({
    "scale_pos_weight": (
        int((y_train == 0).sum()) / max(int(y_train.sum()), 1)
    ),
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
})

best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

# --------------------------------------------------
# Final evaluation on untouched test set
# --------------------------------------------------
test_probabilities = best_model.predict_proba(X_test)[:, 1]

# 0.5 is used only for these conventional classification metrics.
# The API uses reports/operating_point.json for its operating threshold.
evaluation_threshold = 0.5
test_predictions = (
    test_probabilities >= evaluation_threshold
).astype(int)

test_pr_auc = average_precision_score(y_test, test_probabilities)
test_roc_auc = roc_auc_score(y_test, test_probabilities)
test_precision = precision_score(
    y_test, test_predictions, zero_division=0
)
test_recall = recall_score(
    y_test, test_predictions, zero_division=0
)

metrics = {
    "model": "Optuna-tuned XGBoost",
    "random_state": SEED,
    "sampler": "TPESampler",
    "sampler_seed": SEED,
    "split_method": "chronological 80/20 with time-ordered validation",
    "training_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "testing_rows": int(len(X_test)),
    "training_fraud_cases": int(y_train.sum()),
    "validation_fraud_cases": int(y_val.sum()),
    "testing_fraud_cases": int(y_test.sum()),
    "optuna_trials": N_TRIALS,
    "best_validation_pr_auc": float(study.best_value),
    "evaluation_threshold": evaluation_threshold,
    "test_pr_auc": float(test_pr_auc),
    "test_roc_auc": float(test_roc_auc),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "best_parameters": study.best_params,
}

model_path = MODEL_DIR / "xgboost_fraud_model_tuned.pkl"
metrics_path = MODEL_DIR / "metrics_xgboost_tuned.json"

# Save the final model and its evaluation report.
joblib.dump(best_model, model_path)

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

# --------------------------------------------------
# Log final model run to MLflow
# --------------------------------------------------
with mlflow.start_run(run_name="XGBoost_Tuned_Final"):
    mlflow.log_params(best_params)
    mlflow.log_params({
        "optuna_trials": N_TRIALS,
        "sampler_seed": SEED,
        "evaluation_threshold": evaluation_threshold,
    })

    mlflow.log_metrics({
        "best_validation_pr_auc": float(study.best_value),
        "test_pr_auc": float(test_pr_auc),
        "test_roc_auc": float(test_roc_auc),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
    })

    mlflow.log_artifact(str(metrics_path))
    mlflow.log_artifact(str(model_path))

print("\n--- Final Test Results (threshold 0.5) ---")
print("PR-AUC:", round(test_pr_auc, 4))
print("ROC-AUC:", round(test_roc_auc, 4))
print("Precision:", round(test_precision, 4))
print("Recall:", round(test_recall, 4))
print("\nModel saved:", model_path)
print("Metrics saved:", metrics_path)
print("Optuna tuning completed successfully!")
