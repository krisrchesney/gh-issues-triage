import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("shap")

from ghtriage.explain import explain_record, global_importance, make_explainer, save_importance_plot  # noqa: E402
from ghtriage.registry import load_current, load_reference  # noqa: E402
from ghtriage.serve import create_app  # noqa: E402
from ghtriage.train import train  # noqa: E402


@pytest.fixture
def trained(seeded_cfg):
    train(seeded_cfg)
    model, card, _ = load_current(seeded_cfg["model"]["registry_dir"])
    reference = load_reference(seeded_cfg["model"]["registry_dir"])
    return seeded_cfg, model, card, reference


def test_global_importance_covers_all_features(trained):
    _, model, card, reference = trained
    explainer = make_explainer(model, reference, card["features"])
    importance = global_importance(explainer, reference, card["features"], max_rows=40)
    assert set(importance) == set(card["features"])
    values = list(importance.values())
    assert all(v >= 0 for v in values) and values == sorted(values, reverse=True)


def test_contributions_add_up_to_the_prediction(trained):
    _, model, card, reference = trained
    explainer = make_explainer(model, reference, card["features"])
    record = {f: float(reference[f].iloc[3]) for f in card["features"]}
    result = explain_record(explainer, record, card["features"])
    predicted = float(model.predict_proba(pd.DataFrame([record]))[0, 1])
    assert result["base_value"] + sum(result["contributions"].values()) == pytest.approx(predicted, abs=1e-3)


def test_plot_is_written(tmp_path):
    path = save_importance_plot({"a": 0.3, "b": 0.1}, tmp_path / "plot.png")
    assert path.exists() and path.stat().st_size > 1000


def test_explain_endpoint(trained):
    cfg, _, card, reference = trained
    record = {f: float(reference[f].iloc[0]) for f in card["features"]}
    with TestClient(create_app(cfg["model"]["registry_dir"])) as client:
        body = client.post("/explain", json={"record": record}).json()
        assert set(body["contributions"]) == set(card["features"])
        assert 0.0 <= body["probability"] <= 1.0
        record.pop("f2")
        assert client.post("/explain", json={"record": record}).status_code == 422


def test_candidates_include_lightgbm(trained):
    _, _, card, _ = trained
    assert "lightgbm" in card["metrics"]["all_candidates"]
    assert np.isfinite(card["metrics"]["all_candidates"]["lightgbm"]["roc_auc"])
