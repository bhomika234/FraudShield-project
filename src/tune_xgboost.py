
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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.sort_values("Time", kind="stable").reset_index(drop=True)

# Final chronological 80/20 split
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

scale_pos_weight = int((y_fit == 0).sum()) / max(int(y_fit.sum()), 1)

mlflow.set_experiment("FraudShield")

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
        "random_state": 42,
        "n_jobs": -1,
    }

    model = XGBClassifier(**params)
    model.fit(X_fit, y_fit)

    val_probabilities = model.predict_proba(X_val)[:, 1]
    score = average_precision_score(y_val, val_probabilities)

    with mlflow.start_run(
        run_name=f"Optuna_Trial_{trial.number}", nested=True
    ):
        mlflow.log_params(params)
        mlflow.log_metric("validation_pr_auc", float(score))
        mlflow.log_param("trial_number", trial.number)

    print(f"Trial {trial.number + 1}/50 | Validation PR-AUC: {score:.4f}")
    return score


print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Untouched test rows:", len(X_test))
print("Starting 50 Optuna trials...")

study = optuna.create_study(direction="maximize")

with mlflow.start_run(run_name="XGBoost_Optuna_50_Trials"):
    study.optimize(objective, n_trials=50)

print("\nBest validation PR-AUC:", round(study.best_value, 4))
print("Best parameters:", study.best_params)

# Refit selected model on all training data
best_params = study.best_params.copy()
best_params.update({
    "scale_pos_weight": int((y_train == 0).sum()) / max(int(y_train.sum()), 1),
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
})

best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

# Evaluate only after tuning is complete
test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= 0.5).astype(int)

metrics = {
    "model": "Optuna-tuned XGBoost",
    "split_method": "Chronological 80/20 with time-ordered validation",
    "optuna_trials": 50,
    "best_validation_pr_auc": float(study.best_value),
    "test_pr_auc": float(average_precision_score(y_test, test_probabilities)),
    "test_roc_auc": float(roc_auc_score(y_test, test_probabilities)),
    "test_precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "test_recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "best_parameters": study.best_params,
}

model_path = MODEL_DIR / "xgboost_fraud_model_tuned.pkl"
metrics_path = MODEL_DIR / "metrics_xgboost_tuned.json"

joblib.dump(best_model, model_path)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

with mlflow.start_run(run_name="XGBoost_Tuned_Final"):
    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "best_validation_pr_auc": metrics["best_validation_pr_auc"],
        "test_pr_auc": metrics["test_pr_auc"],
        "test_roc_auc": metrics["test_roc_auc"],
        "test_precision": metrics["test_precision"],
        "test_recall": metrics["test_recall"],
    })
    mlflow.log_param("optuna_trials", 50)

print("\nFinal test PR-AUC:", round(metrics["test_pr_auc"], 4))
print("Final test ROC-AUC:", round(metrics["test_roc_auc"], 4))
print("Final test precision:", round(metrics["test_precision"], 4))
print("Final test recall:", round(metrics["test_recall"], 4))
print("\nModel saved:", model_path)
print("Metrics saved:", metrics_path)
print("Optuna tuning completed successfully!")
