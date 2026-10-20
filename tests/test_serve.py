import pytest
from fastapi.testclient import TestClient

from ghtriage.demo_data import make_demo_data
from ghtriage.serve import create_app
from ghtriage.train import train


@pytest.fixture
def client(seeded_cfg):
    train(seeded_cfg)
    with TestClient(create_app(seeded_cfg["model"]["registry_dir"])) as c:
        yield c


def _record():
    row = make_demo_data(n_rows=1).iloc[0]
    return {f"f{i}": float(row[f"f{i}"]) for i in range(8)}


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith("v")


def test_predict_returns_probabilities(client):
    resp = client.post("/predict", json={"records": [_record(), _record()]})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 2
    assert all(0.0 <= p["probability"] <= 1.0 and p["label"] in (0, 1) for p in preds)


def test_predict_rejects_missing_feature(client):
    bad = _record()
    bad.pop("f3")
    resp = client.post("/predict", json={"records": [bad]})
    assert resp.status_code == 422
    assert "f3" in resp.text


def test_predict_rejects_empty_batch(client):
    assert client.post("/predict", json={"records": []}).status_code == 422


def test_model_info_exposes_card(client):
    assert client.get("/model-info").json()["features"][0] == "f0"


def test_health_503_without_model(tmp_path):
    with TestClient(create_app(str(tmp_path / "empty"))) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json={"records": [{"f0": 1.0}]}).status_code == 503
