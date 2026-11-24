"""Single API smoke test (the only unit-level test; data quality lives in dbt)."""
from fastapi.testclient import TestClient

from thermalguard.serving.app import SENSORS, app


def test_predict_returns_rul():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    readings = [{s: 500.0 for s in SENSORS} for _ in range(5)]
    r = client.post("/predict", json={"asset_id": "demo", "readings": readings})
    assert r.status_code == 200
    assert 0 <= r.json()["rul_cycles"] <= 125
