# FraudShield — Model Comparison

Both models use a chronological 80/20 split by transaction time.

| Metric | Tuned XGBoost | Isolation Forest |
|---|---:|---:|
| PR-AUC | 0.7908 | 0.0404 |
| ROC-AUC | 0.9860 | 0.9501 |
| Fraud Precision | 0.7600 | 0.0000 |
| Fraud Recall | 0.7600 | 0.0000 |

## Conclusion

Tuned XGBoost achieved higher PR-AUC and ROC-AUC and detected fraud cases at the reported operating threshold. Isolation Forest detected no fraud cases with its current contamination-based predictions. XGBoost is the stronger candidate based on these reported metrics.

Note: XGBoost metrics are from the tuned model report; Isolation Forest metrics use its current prediction rule.
