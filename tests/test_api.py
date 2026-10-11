"""Pruebas automáticas. Ejecutar con:  pytest -v"""
from fastapi.testclient import TestClient
from app.main import app

VALID = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["modelo_cargado"] is True


def test_predict_setosa():
    with TestClient(app) as c:
        r = c.post("/predict", json=VALID)
        assert r.status_code == 200
        assert r.json()["prediction"] == "setosa"


def test_predict_virginica():
    with TestClient(app) as c:
        r = c.post("/predict", json={"sepal_length": 6.7, "sepal_width": 3.0,
                                     "petal_length": 5.2, "petal_width": 2.3})
        assert r.json()["prediction"] == "virginica"


def test_batch():
    with TestClient(app) as c:
        r = c.post("/predict/batch", json={"samples": [VALID, VALID]})
        assert r.status_code == 200 and len(r.json()["predictions"]) == 2


def test_invalid_value_returns_422():
    with TestClient(app) as c:
        r = c.post("/predict", json={**VALID, "sepal_length": -1})
        assert r.status_code == 422


def test_missing_field_returns_422():
    with TestClient(app) as c:
        r = c.post("/predict", json={"sepal_length": 5.1})
        assert r.status_code == 422
