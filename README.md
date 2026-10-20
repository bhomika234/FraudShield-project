 FraudShield - Credit Card Fraud Detection API

FraudShield is a machine learning project that detects potentially fraudulent credit card transactions using a tuned XGBoost classification model. It provides REST API endpoints for individual and batch predictions using FastAPI.

## Features

- Tuned XGBoost fraud detection model
- Single transaction prediction
- Batch transaction prediction
- Fraud probability and classification response
- Health check endpoint
- Input validation using Pydantic
- API performance benchmarking
- Interactive API documentation with Swagger UI

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Pydantic
- Uvicorn
- HTTPX

## Project Structure

```text
FraudShield/
├── app.py
├── benchmark.py
├── dashboard.py
├── train.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── reports/
│   └── benchmark_results.json
├── src/
└── tests/
```

## Run Locally

1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the API:

```bash
python -m uvicorn app:app --reload
```

5. Open the interactive API documentation:

`http://127.0.0.1:8000/docs`

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Model health check |
| POST | `/predict` | Predict one transaction |
| POST | `/predict/batch` | Predict multiple transactions |

## Prediction Response

A successful prediction returns:

```json
{
  "prediction": 0,
  "fraud_probability": 0.0014,
  "decision": "LEGITIMATE"
}
```

Here, `0` means the model classified the transaction as legitimate, while `1` means it classified the transaction as potentially fraudulent. The probability is the model's estimated fraud probability.

## Performance Benchmark

Local benchmark results from 100 sequential API requests:

| Metric | Result |
|---|---:|
| Total requests | 100 |
| Successful requests | 100 |
| Average response time | 28.87 ms |
| Median response time | 28.69 ms |
| Fastest response | 23.14 ms |
| Slowest response | 43.35 ms |

These results are from a local development environment and are not a guarantee of production performance.

## Important Notes

- Predictions are model estimates, not proof that a transaction is safe or fraudulent.
- The project is for learning and demonstration; production use requires additional testing, security controls, monitoring, and validation.
- Do not commit sensitive data, credentials, or unnecessary large datasets to a public repository.

## Author

Bhoomika Karmani
