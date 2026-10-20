import pandas as pd

from ghtriage.cli import MIN_MONITOR_ROWS, _current_window
from ghtriage.registry import load_current
from ghtriage.train import train


def test_monitoring_window_only_contains_rows_after_the_training_cutoff(seeded_cfg):
    train(seeded_cfg)
    _, card, _ = load_current(seeded_cfg["model"]["registry_dir"])
    window = _current_window(seeded_cfg, 10_000, card["data"]["train_end"])
    assert len(window) >= MIN_MONITOR_ROWS
    assert (window["event_ts"] > pd.Timestamp(card["data"]["train_end"])).all()
    assert len(window) == 400  # the 20% test split of the 2000 seeded rows


def test_window_warns_and_falls_back_when_too_little_new_data(seeded_cfg, capsys):
    window = _current_window(seeded_cfg, 500, after="2099-01-01")
    assert len(window) == 500
    assert "overlap the training data" in capsys.readouterr().err
