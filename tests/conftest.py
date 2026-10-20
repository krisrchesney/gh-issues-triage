import pytest

from ghtriage.demo_data import make_demo_data
from ghtriage.storage import connect, write_frame


@pytest.fixture
def cfg(tmp_path):
    return {
        "lake": {"root": str(tmp_path / "raw")},
        "data": {
            "db_path": str(tmp_path / "wh.duckdb"),
            "table": "training_data",
            "source": "demo",
            "timestamp_col": "event_ts",
        },
        "model": {
            "target": "label",
            "features": [],
            "exclude": ["row_id"],
            "candidates": ["logistic_baseline", "hist_gbm", "lightgbm"],
            "test_fraction": 0.2,
            "random_state": 42,
            "registry_dir": str(tmp_path / "models"),
            "min_auc_improvement": 0.01,
        },
        "mlflow": {
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "experiment": "test",
        },
        "drift": {"bins": 10, "psi_warn": 0.1, "psi_alert": 0.25, "auc_drop_alert": 0.05},
    }


@pytest.fixture
def seeded_cfg(cfg):
    con = connect(cfg["data"]["db_path"])
    write_frame(con, cfg["data"]["table"], make_demo_data(n_rows=2000))
    con.close()
    return cfg
