"""End to end with the project's real config.yaml on synthetic issues (no network)."""

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from ghtriage.cli import main
from ghtriage.features import FEATURES
from ghtriage.github_issues import GitHubIssuesCollector
from ghtriage.lake import upsert_partitioned
from ghtriage.registry import load_current
from ghtriage.serve import create_app
from ghtriage.storage import connect, read_table
from tests.synth import make_raw_issues


@pytest.fixture
def project(tmp_path):
    cfg = yaml.safe_load(open("config/config.yaml"))
    cfg["lake"]["root"] = str(tmp_path / "raw")
    cfg["data"]["db_path"] = str(tmp_path / "wh.duckdb")
    cfg["model"]["registry_dir"] = str(tmp_path / "models")
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg["github"]["coverage_start"] = "2026-06-01"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    raw = make_raw_issues(n_per_repo=500, days=150, seed=7)
    upsert_partitioned(
        cfg["lake"]["root"],
        "gh_issues",
        raw,
        key_cols=["repo", "number"],
        partition_col="created_at",
        version_col="updated_at",
        keep_first_cols=GitHubIssuesCollector.keep_first_cols,
    )
    return cfg, str(path), raw


def test_build_train_serve_and_monitor(project, tmp_path):
    cfg, path, _ = project
    assert main(["--config", path, "build"]) == 0
    con = connect(cfg["data"]["db_path"])
    table = read_table(con, cfg["data"]["table"])
    con.close()
    assert set(FEATURES) <= set(table.columns) and {"label", "event_ts"} <= set(table.columns)
    assert 0.1 < table["label"].mean() < 0.9

    assert main(["--config", path, "train"]) == 0
    _, card, _ = load_current(cfg["model"]["registry_dir"])
    assert card["features"] == FEATURES
    assert card["metrics"]["champion_auc"] > 0.6  # the synthetic signal is learnable
    assert "14 days" in card["label_definition"] and card["description"]
    assert any("Closed means closed" in item for item in card["limitations"])

    row = table.iloc[-1]
    record = {f: float(row[f]) if pd.notna(row[f]) else 0.0 for f in FEATURES}
    with TestClient(create_app(cfg["model"]["registry_dir"])) as client:
        assert client.get("/health").json()["status"] == "ok"
        body = client.post("/predict", json={"records": [record]}).json()
        assert 0.0 <= body["predictions"][0]["probability"] <= 1.0

    out = tmp_path / "drift.json"
    assert main(["--config", path, "drift", "--out", str(out)]) in (0, 2)
    assert out.exists()


def test_build_fails_clearly_on_an_empty_lake(project, tmp_path):
    cfg, path, _ = project
    cfg["lake"]["root"] = str(tmp_path / "nothing_here")
    other = tmp_path / "empty.yaml"
    other.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="lake is empty"):
        main(["--config", str(other), "build"])
