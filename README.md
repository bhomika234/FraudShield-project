# FraudShield — Credit Card Fraud Detection

FraudShield is a machine learning project for identifying potentially fraudulent credit card transactions. It combines supervised learning, anomaly detection, a FastAPI prediction service, and a Streamlit dashboard.

## Features

- Optuna-tuned XGBoost fraud detection
- Baseline XGBoost and Isolation Forest comparison
- Single and batch transaction prediction API
- Fraud probability and classification
- Configurable operating threshold
- API health checks and input validation
- Exploratory Data Analysis (EDA)
- Model evaluation reports
- Streamlit monitoring dashboard
- API performance benchmark
- Automated API tests
- Dockerfile for containerizing the API

## Tech Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn, XGBoost, Optuna
- **Visualization:** Matplotlib, Seaborn
- **API:** FastAPI, Pydantic, Uvicorn
- **Dashboard:** Streamlit
- **Experiment tracking:** MLflow
- **Testing:** Pytest
- **Containerization:** Docker

## Project Structure

```text
FraudShield/
├── app.py
├── dashboard.py
├── benchmark.py
├── train.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
├── data/
│   └── creditcard.csv
├── models/
│   ├── metrics.json
│   ├── metrics_xgboost.json
│   ├── metrics_xgboost_tuned.json
│   └── xgboost_fraud_model_tuned.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modelling.ipynb
├── reports/
│   ├── benchmark_results.json
│   ├── metrics_isolation_forest.json
│   ├── model_comparison.json
│   ├── model_comparison.md
│   └── operating_point.json
├── src/
│   ├── baseline.py
│   ├── compare_models.py
│   ├── eda.py
│   ├── operating_point.py
│   ├── train_xgboost.py
│   └── tune_xgboost.py
└── tests/
    └── test_api.py
```

*This is a summary of the main project files. Some generated files may be omitted from version control.*

## Dataset

The project uses the Credit Card Fraud Detection dataset.

- **Transactions:** 284,807
- **Columns:** 31, including the `Class` target
- `Class = 0`: legitimate transaction
- `Class = 1`: fraudulent transaction
- Missing values reported during EDA: 0
- Duplicate rows reported during EDA: 1,081
- Fraudulent transactions: approximately 0.1727%

The dataset is highly imbalanced, so accuracy alone is not sufficient for evaluating model performance.

Place the dataset at `data/creditcard.csv`. The dataset is not included in the repository by default.

## Installation

Run these commands from the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current terminal and retry activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## Run the API

Start the API from the project root:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open these URLs in your browser:

- Swagger API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Keep the API process running while using the dashboard or sending prediction requests.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome response |
| GET | `/health` | Service and model health |
| POST | `/predict` | Predict one transaction |
| POST | `/predict/batch` | Predict multiple transactions |

Prediction inputs must include `Time`, `Amount`, and `V1` through `V28`. Do not send the `Class` target column to the prediction API.

Example response:

```json
{
  "prediction": 0,
  "fraud_probability": 0.0014,
  "decision": "LEGITIMATE"
}
```

This is an illustrative response; actual values depend on the input transaction and the loaded model.

## Run the Streamlit Dashboard

First start the API. Then open a second PowerShell terminal, activate the virtual environment, and run:

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
python -m streamlit run dashboard.py
```

Open http://localhost:8501.

The dashboard sends prediction requests to the API and uses the local dataset. Ensure `data/creditcard.csv` is available before starting it.

## Exploratory Data Analysis

Open `notebooks/01_eda.ipynb` in Jupyter or VS Code.

EDA covers dataset structure, data quality, missing values, duplicates, class distribution, transaction amounts, and visualizations. The corresponding script is `src/eda.py`.

Generated visualizations are saved in the `reports/` directory.

## Model Evaluation

Open `notebooks/02_modelling.ipynb` to inspect modelling work and saved metrics.

The project includes baseline XGBoost, Optuna-tuned XGBoost, and an Isolation Forest unsupervised baseline. The saved comparison report is available at:

- `reports/model_comparison.md`
- `reports/model_comparison.json`

### Saved test metrics

| Metric | Baseline XGBoost | Tuned XGBoost | Isolation Forest |
|---|---:|---:|---:|
| PR-AUC | 0.7969 | 0.7908 | 0.0404 |
| ROC-AUC | 0.9881 | 0.9860 | 0.9501 |
| Fraud precision | 0.7215 | 0.7600 | 0.0000 |
| Fraud recall | 0.7600 | 0.7600 | 0.0000 |

These are the values from the saved metrics reports. Precision and recall use each model's existing decision rule; they are not a same-threshold operational comparison.

### Tuned XGBoost operating point

The saved operating-point report uses a threshold selected using validation data, targeting a validation false-positive rate (FPR) of 1%.

- Selected threshold: approximately `0.01017185`
- Validation FPR: 0.47%
- Validation recall: 86.54%
- Test FPR: 1.49%
- Test recall at the selected threshold: 86.67%
- Fraud cases detected: 65 of 75
- Fraud cases missed: 10
- Legitimate transactions flagged: 845

The test FPR can differ from the validation target. The threshold is a model operating setting, not a guarantee of real-world fraud detection performance.

## API Benchmark

The saved benchmark output is `reports/benchmark_results.json`.

The previously recorded benchmark reported 100 successful requests out of 100, with an average latency of approximately 28.87 ms. These are results from that recorded run; performance may vary with hardware and system load.

To rerun the benchmark, first start the API in another terminal, then inspect the command-line options in `benchmark.py` before executing it.

## Automated Tests

Run the API tests from the project root:

```powershell
python -m pytest -v
```

The previously recorded test run passed six tests. Rerun the tests after code or dependency changes to verify the current project state.

## MLflow

MLflow is included in the project dependencies for experiment tracking. Existing experiments and runs depend on the local tracking configuration and available tracking files.

## Docker

A `Dockerfile` and `.dockerignore` are included for API containerization. Docker must be installed separately before Docker commands can be run. Local API and dashboard development do not require Docker.

## Reproducibility and Limitations

- Use the same dataset version and feature order when reproducing results.
- Keep training, validation, and test data separate during model selection and threshold tuning.
- Review the saved reports before making model comparisons.
- Transaction fraud scores are model estimates, not definitive proof of fraud.
- Benchmark latency and predictive performance can differ in other environments.




