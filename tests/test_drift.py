import numpy as np
import pandas as pd
import pytest

from ghtriage.drift import drift_report, psi

DRIFT_CFG = {"bins": 10, "psi_warn": 0.1, "psi_alert": 0.25, "auc_drop_alert": 0.05}


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.05


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(loc=1.5, size=5000)) > 0.25


def test_psi_constant_feature_returns_zero():
    assert psi(np.ones(100), np.ones(100)) == 0.0


def test_psi_ignores_nans():
    rng = np.random.default_rng(0)
    base = rng.normal(size=2000)
    with_nans = np.concatenate([base, [np.nan] * 50])
    assert psi(base, with_nans) < 0.05


def test_report_needs_enough_rows():
    df = pd.DataFrame({"a": np.arange(10.0)})
    with pytest.raises(ValueError):
        drift_report(df, df, ["a"], DRIFT_CFG)


def test_report_flags_retrain_on_major_shift():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(size=3000)})
    cur = pd.DataFrame({"a": rng.normal(loc=2.0, size=500)})
    report = drift_report(ref, cur, ["a"], DRIFT_CFG)
    assert report["overall_status"] == "alert"
    assert report["retrain_recommended"] is True


def test_report_quiet_when_stable():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"a": rng.normal(size=3000)})
    cur = pd.DataFrame({"a": rng.normal(size=1000)})
    report = drift_report(ref, cur, ["a"], DRIFT_CFG)
    assert report["retrain_recommended"] is False


def test_report_flags_performance_drop():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({"a": rng.normal(size=3000)})
    cur = pd.DataFrame({"a": rng.normal(size=1000)})
    report = drift_report(ref, cur, ["a"], DRIFT_CFG, live_auc=0.70, baseline_auc=0.85)
    assert report["retrain_recommended"] is True
    assert report["performance"]["auc_drop"] == pytest.approx(0.15)


def test_psi_detects_shift_in_a_binary_flag():
    rng = np.random.default_rng(0)
    reference = (rng.random(4000) < 0.10).astype(float)  # 10% ones
    assert psi(reference, (rng.random(1000) < 0.10).astype(float)) < 0.05
    assert psi(reference, (rng.random(1000) < 0.40).astype(float)) > 0.25


def test_psi_flags_a_value_never_seen_in_training():
    assert psi(np.ones(500), np.concatenate([np.ones(300), np.zeros(200)])) > 0.25


def test_psi_still_handles_continuous_features_with_quantile_bins():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(size=5000), rng.normal(loc=1.5, size=2000)) > 0.25
