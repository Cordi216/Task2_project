import pytest
from fastapi.testclient import TestClient
from api_sentiment import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Sentiment Analysis API" in response.json()["message"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_positive():
    response = client.post(
        "/predict",
        json={"text": "Япония отличная страна"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["positive", "neutral", "negative"]
    assert 0 <= data["score"] <= 1


def test_predict_negative():
    response = client.post(
        "/predict",
        json={"text": "Это ужасно, не рекомендую"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["negative", "neutral"]


def test_batch():
    response = client.post(
        "/batch",
        json={"texts": ["хорошо", "плохо", "нормально"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
    for result in data["results"]:
        assert "label" in result
        assert "score" in result


def test_stats():
    response = client.post(
        "/stats",
        json={"texts": ["отлично", "ужасно", "нормально"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert "positive_percent" in data
    assert "negative_percent" in data
    assert "neutral_percent" in data
    assert len(data["details"]) == 3


def test_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response.status_code == 422


def test_invalid_json():
    response = client.post(
        "/predict",
        json={"wrong_field": "text"}
    )
    assert response.status_code == 422