# FraudShield - Model Comparison

## Evaluation setup

- Split: Chronological 80/20 by transaction time
- Test transactions: 56,962
- Test fraud cases: 75

| Metric | Baseline XGBoost | Tuned XGBoost | Isolation Forest |
|---|---:|---:|---:|
| PR-AUC | 0.7969 | 0.7908 | 0.0404 |
| ROC-AUC | 0.9881 | 0.9860 | 0.9501 |
| Fraud Precision | 0.7215 | 0.7600 | 0.0000 |
| Fraud Recall | 0.7600 | 0.7600 | 0.0000 |

## Tuned XGBoost operating point

- Selected threshold: 0.01017185
- Target validation FPR: 1.00%
- Validation FPR: 0.47%
- Validation recall: 86.54%
- Test FPR: 1.49%
- Test recall at selected threshold: 86.67%
- Fraud cases detected: 65
- Fraud cases missed: 10
- Legitimate transactions flagged: 845

## Interpretation

- PR-AUC and ROC-AUC describe ranking performance in the saved metrics.
- Baseline and tuned XGBoost are reported separately.
- Isolation Forest's saved decision rule detected 0 of 75 fraud cases.
- The tuned XGBoost threshold was selected using validation data. The test FPR can differ from the validation target.
- Precision and recall use each model's saved decision rule; this report does not claim a same-threshold comparison.

## Reproducibility

This script reads existing JSON reports. It does not train models or recalculate predictions. Verify that model metrics were computed using the same test split before making strict model comparisons.
