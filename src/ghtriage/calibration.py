"""Time-ordered probability calibration (kept free of training-only imports such as MLflow, because the
serving image unpickles models that use it)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class TimeCalibrated(BaseEstimator, ClassifierMixin):
    """Fit the model on the earlier part of the (time-ordered) training rows and a monotone calibrator on
    the latest `holdout` share, so probabilities match observed rates in the most recent period.

    "sigmoid" (Platt scaling on the logit) fixes over- or under-confidence and keeps the ranking strictly
    ordered; "isotonic" is more flexible but creates ties. A monotone map never reorders predictions; the
    cost is that the model itself trains on `holdout` less data, unless `refit` (the default): then the map
    learned on the holdout is applied to a model refitted on every row, which keeps the full model's
    ranking at the price of an approximate calibration. If the calibration slice holds only one class the
    model is left uncalibrated.
    """

    def __init__(self, estimator: Any, method: str = "sigmoid", holdout: float = 0.25, refit: bool = True):
        self.estimator = estimator
        self.method = method
        self.holdout = holdout
        self.refit = refit

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> TimeCalibrated:
        y = np.asarray(y)
        n_cal = max(1, int(len(y) * self.holdout))
        self.model_ = clone(self.estimator).fit(X.iloc[:-n_cal], y[:-n_cal])
        self.classes_ = np.array([0, 1])
        y_cal = y[-n_cal:]
        self.calibrator_ = None
        if len(np.unique(y_cal)) == 2:
            raw = self.model_.predict_proba(X.iloc[-n_cal:])[:, 1]
            if self.method == "isotonic":
                self.calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
                    raw, y_cal
                )
            else:
                self.calibrator_ = LogisticRegression().fit(_logit(raw).reshape(-1, 1), y_cal)
        if self.refit:
            # Rank with a model that has seen every row, including the newest; reuse the learned map.
            self.model_ = clone(self.estimator).fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model_.predict_proba(X)[:, 1]
        if self.calibrator_ is None:
            p = raw
        elif self.method == "isotonic":
            p = self.calibrator_.predict(raw)
        else:
            p = self.calibrator_.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p, p])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
