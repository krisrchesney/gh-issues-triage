import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from ghtriage.calibration import TimeCalibrated
from ghtriage.train import expected_calibration_error


class Overconfident(BaseEstimator, ClassifierMixin):
    """Ranks perfectly by the true rate but pushes every probability towards 0 or 1."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = X["true_p"].to_numpy()
        z = np.log(p / (1 - p)) * 4  # four times too steep
        q = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - q, q])


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    return pd.DataFrame({"true_p": p}), (rng.uniform(size=n) < p).astype(int)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_fixes_overconfidence(method):
    X, y = _data()
    raw = Overconfident().fit(X, y).predict_proba(X)[:, 1]
    cal = TimeCalibrated(Overconfident(), method=method).fit(X, y).predict_proba(X)[:, 1]
    assert expected_calibration_error(y, cal) < expected_calibration_error(y, raw) / 2


def test_sigmoid_calibration_never_reorders_predictions():
    X, y = _data()
    raw = Overconfident().fit(X, y).predict_proba(X)[:, 1]
    cal = TimeCalibrated(Overconfident(), method="sigmoid").fit(X, y).predict_proba(X)[:, 1]
    assert (np.argsort(raw, kind="stable") == np.argsort(cal, kind="stable")).all()


def test_calibrator_uses_only_the_latest_rows():
    """The calibration slice is the most recent `holdout` share, never earlier rows."""
    X, y = _data(1000)
    y = y.copy()
    y[-250:] = 1  # the latest quarter all closed
    model = TimeCalibrated(Overconfident(), method="sigmoid", holdout=0.25).fit(X, y)
    assert (
        model.calibrator_ is None
    )  # one class in the latest slice: left uncalibrated, not fitted on old rows
    np.testing.assert_allclose(model.predict_proba(X), Overconfident().fit(X, y).predict_proba(X))
