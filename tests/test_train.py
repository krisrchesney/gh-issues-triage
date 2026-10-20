import json
from pathlib import Path

import numpy as np
import pytest

from ghtriage.demo_data import make_demo_data
from ghtriage.train import expected_calibration_error, select_features, time_split, train


def test_time_split_never_leaks_future():
    df = make_demo_data(n_rows=500).sample(frac=1.0, random_state=0)  # shuffled input
    tr, te = time_split(df, "event_ts", 0.2)
    assert tr["event_ts"].max() <= te["event_ts"].min()
    assert len(te) == 100


def test_time_split_rejects_tiny_data():
    with pytest.raises(ValueError):
        time_split(make_demo_data(n_rows=40), "event_ts", 0.2)


def test_feature_selection_excludes_ids_target_and_timestamp(cfg):
    feats = select_features(make_demo_data(n_rows=200), cfg)
    assert "row_id" not in feats
    assert "label" not in feats
    assert "event_ts" not in feats
    assert len(feats) == 8


def test_ece_is_zero_for_perfectly_calibrated_extremes():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, y.astype(float)) == pytest.approx(0.0)


def test_train_writes_registry_and_card(seeded_cfg):
    card = train(seeded_cfg)
    root = Path(seeded_cfg["model"]["registry_dir"])
    assert json.loads((root / "current.json").read_text())["version"] == card["version"]
    for name in ("model.joblib", "model_card.json", "reference.parquet"):
        assert (root / card["version"] / name).exists()
    assert card["champion"] in {"logistic_baseline", "hist_gbm", "lightgbm"}
    assert 0.5 < card["metrics"]["champion_auc"] <= 1.0
    assert card["selection_reason"]


def test_baseline_kept_when_challenger_gain_is_below_threshold(seeded_cfg):
    seeded_cfg["model"]["min_auc_improvement"] = 0.9  # impossible to clear
    assert train(seeded_cfg)["champion"] == "logistic_baseline"


def test_recall_at_top_fraction():
    from ghtriage.train import recall_at_top_fraction

    y = np.array([0] * 90 + [1] * 10)
    perfect = np.concatenate([np.linspace(0, 0.4, 90), np.linspace(0.6, 1, 10)])
    assert recall_at_top_fraction(y, perfect, 0.10) == 1.0
    assert recall_at_top_fraction(y, perfect, 0.05) == pytest.approx(0.5)
    assert np.isnan(recall_at_top_fraction(np.zeros(10), np.linspace(0, 1, 10)))


def test_selection_metric_is_configurable_and_validated(seeded_cfg):
    seeded_cfg["model"]["selection_metric"] = "avg_precision"
    card = train(seeded_cfg)
    assert card["selection_metric"] == "avg_precision"
    seeded_cfg["model"]["selection_metric"] = "nonsense"
    with pytest.raises(ValueError, match="selection_metric"):
        train(seeded_cfg)


def test_features_with_no_values_fail_with_a_clear_message(seeded_cfg):
    from ghtriage.storage import connect, read_table, write_frame

    con = connect(seeded_cfg["data"]["db_path"])
    df = read_table(con, seeded_cfg["data"]["table"])
    df["dead"] = np.nan
    write_frame(con, seeded_cfg["data"]["table"], df)
    con.close()
    seeded_cfg["model"]["features"] = [f"f{i}" for i in range(8)] + ["dead"]
    with pytest.raises(ValueError, match=r"no values at all.*dead"):
        train(seeded_cfg)
