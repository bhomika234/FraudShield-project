
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

FILES = {
    "Baseline XGBoost": BASE_DIR / "models" / "metrics_xgboost.json",
    "Tuned XGBoost": BASE_DIR / "models" / "metrics_xgboost_tuned.json",
    "Isolation Forest": REPORT_DIR / "metrics_isolation_forest.json",
    "Operating Point": REPORT_DIR / "operating_point.json",
}


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )
    with path.open(encoding="utf-8") as file:
        return json.load(file)


baseline = load_json(FILES["Baseline XGBoost"])
tuned = load_json(FILES["Tuned XGBoost"])
iso = load_json(FILES["Isolation Forest"])
operating = load_json(FILES["Operating Point"])

comparison = {
    "evaluation": {
        "split_method": "Chronological 80/20 by transaction time",
        "testing_rows": iso["testing_rows"],
        "testing_fraud_cases": iso["testing_fraud_cases"],
        "note": (
            "Metrics are read from existing reports. Confirm that saved "
            "test metrics use the same test split and evaluation procedure. "
            "Precision and recall may use different decision thresholds."
        ),
    },
    "Baseline XGBoost": {
        "PR-AUC": baseline["pr_auc"],
        "ROC-AUC": baseline["roc_auc"],
        "Fraud Precision": baseline["fraud_precision"],
        "Fraud Recall": baseline["fraud_recall"],
    },
    "Tuned XGBoost": {
        "PR-AUC": tuned["test_pr_auc"],
        "ROC-AUC": tuned["test_roc_auc"],
        "Fraud Precision": tuned["test_precision"],
        "Fraud Recall": tuned["test_recall"],
    },
    "Isolation Forest": {
        "PR-AUC": iso["pr_auc"],
        "ROC-AUC": iso["roc_auc"],
        "Fraud Precision": iso["fraud_precision"],
        "Fraud Recall": iso["fraud_recall"],
    },
    "Tuned XGBoost Operating Point": {
        "threshold": operating["threshold"],
        "target_validation_fpr": operating["target_fpr"],
        "validation_fpr": operating["validation_fpr"],
        "validation_recall": operating["validation_recall"],
        "test_fpr": operating["test_fpr"],
        "test_recall": operating["test_recall_at_selected_threshold"],
        "test_true_positives": operating["test_true_positives"],
        "test_false_negatives": operating["test_false_negatives"],
        "test_false_positives": operating["test_false_positives"],
    },
}

json_path = REPORT_DIR / "model_comparison.json"
json_path.write_text(
    json.dumps(comparison, indent=4),
    encoding="utf-8",
)

models = [
    "Baseline XGBoost",
    "Tuned XGBoost",
    "Isolation Forest",
]

lines = [
    "# FraudShield - Model Comparison",
    "",
    "## Evaluation setup",
    "",
    f"- Split: {comparison['evaluation']['split_method']}",
    f"- Test transactions: {iso['testing_rows']:,}",
    f"- Test fraud cases: {iso['testing_fraud_cases']}",
    "",
    "| Metric | Baseline XGBoost | Tuned XGBoost | Isolation Forest |",
    "|---|---:|---:|---:|",
]

for metric in [
    "PR-AUC",
    "ROC-AUC",
    "Fraud Precision",
    "Fraud Recall",
]:
    values = [comparison[name][metric] for name in models]
    lines.append(
        f"| {metric} | "
        + " | ".join(f"{value:.4f}" for value in values)
        + " |"
    )

lines.extend([
    "",
    "## Tuned XGBoost operating point",
    "",
    f"- Selected threshold: {operating['threshold']:.8f}",
    f"- Target validation FPR: {operating['target_fpr']:.2%}",
    f"- Validation FPR: {operating['validation_fpr']:.2%}",
    f"- Validation recall: {operating['validation_recall']:.2%}",
    f"- Test FPR: {operating['test_fpr']:.2%}",
    f"- Test recall at selected threshold: "
    f"{operating['test_recall_at_selected_threshold']:.2%}",
    f"- Fraud cases detected: {operating['test_true_positives']}",
    f"- Fraud cases missed: {operating['test_false_negatives']}",
    f"- Legitimate transactions flagged: "
    f"{operating['test_false_positives']}",
    "",
    "## Interpretation",
    "",
    "- PR-AUC and ROC-AUC describe ranking performance in the saved metrics.",
    "- Baseline and tuned XGBoost are reported separately.",
    "- Isolation Forest's saved decision rule detected "
    f"{iso['true_positives']} of {iso['testing_fraud_cases']} fraud cases.",
    "- The tuned XGBoost threshold was selected using validation data. "
    "The test FPR can differ from the validation target.",
    "- Precision and recall use each model's saved decision rule; "
    "this report does not claim a same-threshold comparison.",
    "",
    "## Reproducibility",
    "",
    "This script reads existing JSON reports. It does not train models "
    "or recalculate predictions. Verify that model metrics were computed "
    "using the same test split before making strict model comparisons.",
])

md_path = REPORT_DIR / "model_comparison.md"
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("Model comparison reports updated successfully.")
print("JSON report:", json_path)
print("Markdown report:", md_path)
print("\n".join(lines))
