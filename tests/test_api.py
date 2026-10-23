from fastapi.testclient import TestClient
import app as fraud_app

client = TestClient(fraud_app.app)


def sample_transaction():
    return {
        "Time": 0,
        **{f"V{i}": 0 for i in range(1, 29)},
        "Amount": 10,
    }


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_rejects_missing_features():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_prediction_rejects_extra_fields():
    payload = sample_transaction()
    payload["unexpected_field"] = 123

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_rejects_empty_list():
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 422


def test_valid_transaction_prediction():
    response = client.post("/predict", json=sample_transaction())

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in (0, 1)
    assert 0 <= result["fraud_probability"] <= 1
    assert 0 < result["threshold"] < 1
    assert result["decision"] in ("FRAUD", "LEGITIMATE")


def test_batch_prediction():
    transactions = [sample_transaction(), sample_transaction()]

    response = client.post("/predict/batch", json=transactions)

    assert response.status_code == 200
    result = response.json()
    assert result["count"] == 2
    assert len(result["predictions"]) == 2
    for prediction in result["predictions"]:
        assert prediction["prediction"] in (0, 1)
        assert prediction["decision"] in ("FRAUD", "LEGITIMATE")
