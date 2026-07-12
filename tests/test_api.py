"""End-to-end tests for the FastAPI service."""
from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src import config
from src.data_preprocessing import make_pipeline


@pytest.fixture(scope="module")
def client(xy, tmp_path_factory) -> TestClient:
    """Train a tiny LR model, patch the API's model, and return a TestClient."""
    X, y = xy
    pipe = make_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X, y)

    tmp_dir: Path = tmp_path_factory.mktemp("api_model")
    model_path = tmp_dir / "heart_model.joblib"
    joblib.dump(pipe, model_path)

    from src.api import main as api_main

    api_main.reload_model(model_path)
    with TestClient(api_main.app) as tc:
        yield tc


SAMPLE_PATIENT = {
    "age": 63,
    "sex": "Male",
    "dataset": "Cleveland",
    "cp": "typical angina",
    "trestbps": 145,
    "chol": 233,
    "fbs": True,
    "restecg": "lv hypertrophy",
    "thalch": 150,
    "exang": False,
    "oldpeak": 2.3,
    "slope": "downsloping",
    "ca": 0,
    "thal": "fixed defect",
}


def test_root(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == config.API_TITLE
    assert body["version"] == config.API_VERSION


def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_happy_path(client: TestClient):
    r = client.post("/predict", json=SAMPLE_PATIENT)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("disease", "no_disease")
    assert 0.0 <= body["probability"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"]


def test_predict_missing_optional_fields(client: TestClient):
    """API must impute when optional fields are omitted."""
    minimal = {
        "age": 55,
        "sex": "Female",
        "dataset": "Cleveland",
        "cp": "asymptomatic",
    }
    r = client.post("/predict", json=minimal)
    assert r.status_code == 200, r.text
    assert r.json()["label"] in ("disease", "no_disease")


def test_predict_rejects_bad_input(client: TestClient):
    bad = {**SAMPLE_PATIENT, "sex": "Alien"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_batch_predict(client: TestClient):
    r = client.post("/predict/batch", json=[SAMPLE_PATIENT, SAMPLE_PATIENT])
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_metrics_endpoint_prometheus(client: TestClient):
    # Trigger at least one prediction so counters exist
    client.post("/predict", json=SAMPLE_PATIENT)
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "heart_predictions_total" in body
    assert "http_requests_total" in body or "http_request_duration_seconds" in body
