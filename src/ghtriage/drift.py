"""Drift monitoring and the retraining trigger.

Data drift uses the Population Stability Index (PSI) per feature against the training reference.
Common rule of thumb: PSI < 0.1 stable, 0.1 to 0.25 moderate shift, > 0.25 major shift.
Performance drift compares live ROC AUC (when labels have arrived) to the training test AUC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index of `actual` relative to `expected` using expected's quantile bins."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    categories = np.unique(expected)
    if len(categories) <= bins:
        # Discrete feature (flags, counts, small codes): quantile bins would collapse, so compare the share of
        # each value instead. A value never seen in training counts as a (large) shift.
        values = np.union1d(categories, np.unique(actual))
        exp_pct = np.array([(expected == v).mean() for v in values])
        act_pct = np.array([(actual == v).mean() for v in values])
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        exp_pct = np.histogram(expected, bins=edges)[0] / len(expected)
        act_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _status(value: float, warn: float, alert: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value >= alert:
        return "alert"
    if value >= warn:
        return "warn"
    return "ok"


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    drift_cfg: dict[str, Any],
    live_auc: float | None = None,
    baseline_auc: float | None = None,
) -> dict[str, Any]:
    """Compare current data to the training reference and decide whether to retrain."""
    if len(current) < 30:
        raise ValueError("Need at least 30 current rows for a meaningful drift check.")
    warn, alert = drift_cfg["psi_warn"], drift_cfg["psi_alert"]
    per_feature: dict[str, dict[str, Any]] = {}
    for col in features:
        value = psi(reference[col].to_numpy(), current[col].to_numpy(), bins=drift_cfg["bins"])
        per_feature[col] = {
            "psi": None if np.isnan(value) else round(value, 4),
            "status": _status(value, warn, alert),
            "missing_rate_current": round(float(current[col].isna().mean()), 4),
        }

    reasons: list[str] = []
    alerts = [c for c, v in per_feature.items() if v["status"] == "alert"]
    if alerts:
        reasons.append(f"Major data drift (PSI >= {alert}) in: {', '.join(alerts)}")

    performance: dict[str, Any] | None = None
    if live_auc is not None and baseline_auc is not None:
        drop = baseline_auc - live_auc
        performance = {
            "live_auc": round(live_auc, 4),
            "training_test_auc": round(baseline_auc, 4),
            "auc_drop": round(drop, 4),
        }
        if drop >= drift_cfg["auc_drop_alert"]:
            reasons.append(
                f"Live AUC fell {drop:.3f} below the training test AUC (limit {drift_cfg['auc_drop_alert']})"
            )

    statuses = {v["status"] for v in per_feature.values()}
    overall = "alert" if "alert" in statuses else "warn" if "warn" in statuses else "ok"
    return {
        "overall_status": overall,
        "retrain_recommended": bool(reasons),
        "reasons": reasons,
        "rows_reference": int(len(reference)),
        "rows_current": int(len(current)),
        "features": per_feature,
        "performance": performance,
        "thresholds": {"psi_warn": warn, "psi_alert": alert, "auc_drop_alert": drift_cfg["auc_drop_alert"]},
    }


def live_auc(model: Any, df: pd.DataFrame, features: list[str], target: str) -> float | None:
    """ROC AUC on labelled current rows, or None when labels are missing or single-class."""
    labelled = df.dropna(subset=[target])
    if len(labelled) < 30 or labelled[target].nunique() < 2:
        return None
    return float(roc_auc_score(labelled[target], model.predict_proba(labelled[features])[:, 1]))
