
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent

xgb_path = BASE_DIR / "models" / "metrics_xgboost_tuned.json"
iso_path = BASE_DIR / "reports" / "metrics_isolation_forest.json"
report_dir = BASE_DIR / "reports"
report_dir.mkdir(exist_ok=True)

with open(xgb_path, encoding="utf-8") as f:
    xgb = json.load(f)

with open(iso_path, encoding="utf-8") as f:
    iso = json.load(f)

comparison = {
    "XGBoost": {
        "PR-AUC": xgb["test_pr_auc"],
        "ROC-AUC": xgb["test_roc_auc"],
        "Fraud Precision": xgb["test_precision"],
        "Fraud Recall": xgb["test_recall"],
    },
    "Isolation Forest": {
        "PR-AUC": iso["pr_auc"],
        "ROC-AUC": iso["roc_auc"],
        "Fraud Precision": iso["fraud_precision"],
        "Fraud Recall": iso["fraud_recall"],
    },
}

json_path = report_dir / "model_comparison.json"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=4)

# Create a readable Markdown report
lines = [
    "# FraudShield — Model Comparison",
    "",
    "Both models use a chronological 80/20 split by transaction time.",
    "",
    "| Metric | Tuned XGBoost | Isolation Forest |",
    "|---|---:|---:|",
]

for metric in [
    "PR-AUC",
    "ROC-AUC",
    "Fraud Precision",
    "Fraud Recall",
]:
    xgb_value = comparison["XGBoost"][metric]
    iso_value = comparison["Isolation Forest"][metric]

    lines.append(
        f"| {metric} | {xgb_value:.4f} | {iso_value:.4f} |"
    )

lines.extend([
    "",
    "## Conclusion",
    "",
    "Tuned XGBoost achieved higher PR-AUC and ROC-AUC and "
    "detected fraud cases at the reported operating threshold. "
    "Isolation Forest detected no fraud cases with its current "
    "contamination-based predictions. XGBoost is the stronger "
    "candidate based on these reported metrics.",
    "",
    "Note: XGBoost metrics are from the tuned model report; "
    "Isolation Forest metrics use its current prediction rule.",
])

md_path = report_dir / "model_comparison.md"
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("Model comparison completed.")
print("JSON report:", json_path)
print("Markdown report:", md_path)
print("\nComparison:")
print("\n".join(lines))
