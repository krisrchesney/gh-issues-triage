"""Training: baseline vs gradient boosting, time-aware validation, MLflow tracking.

Selection rule (documented in the model card): the simpler baseline wins unless the challenger
beats its ROC AUC by at least `model.min_auc_improvement`. Simpler models are cheaper to
explain, monitor and debug, so the challenger has to earn promotion.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ghtriage.calibration import TimeCalibrated
from ghtriage.registry import promote
from ghtriage.storage import connect, read_table


def select_features(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    model_cfg = cfg["model"]
    explicit = model_cfg.get("features") or []
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(f"Configured features missing from data: {missing}")
        return list(explicit)
    skip = {model_cfg["target"], cfg["data"]["timestamp_col"], *model_cfg.get("exclude", [])}
    numeric = df.select_dtypes(include="number").columns
    features = [c for c in numeric if c not in skip]
    if not features:
        raise ValueError("No numeric feature columns found.")
    return features


def time_split(df: pd.DataFrame, ts_col: str, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the past, test on the most recent rows. Random splits leak future information."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    ordered = df.sort_values(ts_col).reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_fraction))
    if cut < 50 or len(ordered) - cut < 20:
        raise ValueError("Not enough rows for a meaningful train/test split.")
    return ordered.iloc[:cut], ordered.iloc[cut:]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """Weighted average gap between predicted probability and observed rate across bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def recall_at_top_fraction(y_true: np.ndarray, y_prob: np.ndarray, fraction: float = 0.05) -> float:
    """Share of all positives found when only the top `fraction` of scores are reviewed.

    The metric a triage team feels: "if we read the top 5%, how many real cases did we catch?"
    """
    y_true = np.asarray(y_true)
    positives = y_true.sum()
    if positives == 0:
        return float("nan")
    k = max(1, int(np.ceil(fraction * len(y_true))))
    top = np.argsort(-np.asarray(y_prob), kind="stable")[:k]
    return float(y_true[top].sum() / positives)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "recall_at_top_5pct": recall_at_top_fraction(y_true, y_prob, 0.05),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }


BASELINE = "logistic_baseline"


def build_candidates(
    seed: int, names: list[str] | None = None, calibration: str | None = None
) -> dict[str, Any]:
    """Build the requested candidate pipelines. The baseline is always included. With `calibration`
    ("sigmoid" or "isotonic"), every candidate is wrapped in TimeCalibrated."""
    built = _build_candidates(seed, names)
    if calibration and calibration != "none":
        return {name: TimeCalibrated(pipe, method=calibration) for name, pipe in built.items()}
    return built


def _build_candidates(seed: int, names: list[str] | None = None) -> dict[str, Pipeline]:
    """Build the requested candidate pipelines. The baseline is always included."""
    wanted = list(dict.fromkeys([BASELINE, *(names or [BASELINE, "hist_gbm"])]))
    factories = {
        BASELINE: lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "hist_gbm": lambda: Pipeline(
            [("clf", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, random_state=seed))]
        ),
        "lightgbm": lambda: _lightgbm_pipeline(seed),
    }
    unknown = [n for n in wanted if n not in factories]
    if unknown:
        raise ValueError(f"Unknown model candidates: {unknown}. Available: {sorted(factories)}")
    return {n: factories[n]() for n in wanted}


def _lightgbm_pipeline(seed: int) -> Pipeline:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError("LightGBM is not installed. Run: pip install -e '.[train]'") from exc
    return Pipeline(
        [
            (
                "clf",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=20,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    random_state=seed,
                    verbose=-1,
                ),
            )
        ]
    )


def train(cfg: dict[str, Any]) -> dict[str, Any]:
    """Train all candidates, track them in MLflow, promote the champion. Returns the model card."""
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    con = connect(data_cfg["db_path"])
    try:
        df = read_table(con, data_cfg["table"], order_by=data_cfg["timestamp_col"])
    finally:
        con.close()

    features = select_features(df, cfg)
    target, ts_col = model_cfg["target"], data_cfg["timestamp_col"]
    df = df.dropna(subset=[target])
    train_df, test_df = time_split(df, ts_col, model_cfg["test_fraction"])
    if train_df[target].nunique() < 2 or test_df[target].nunique() < 2:
        raise ValueError("Train and test sets must each contain both classes.")

    x_train, y_train = train_df[features], train_df[target].to_numpy()
    empty = [c for c in features if x_train[c].isna().all()]
    if empty:
        raise ValueError(
            f"Features with no values at all in the training period: {empty}. "
            "Remove them from model.features or collect data that populates them."
        )
    x_test, y_test = test_df[features], test_df[target].to_numpy()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    version = datetime.now(UTC).strftime("v%Y%m%dT%H%M%SZ")
    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}

    with mlflow.start_run(run_name=f"train-{version}"):
        mlflow.log_params(
            {
                "rows_train": len(train_df),
                "rows_test": len(test_df),
                "n_features": len(features),
                "test_fraction": model_cfg["test_fraction"],
                "positive_rate_train": round(float(y_train.mean()), 4),
            }
        )
        for name, pipeline in build_candidates(
            model_cfg["random_state"], model_cfg.get("candidates"), model_cfg.get("calibration")
        ).items():
            with mlflow.start_run(run_name=name, nested=True):
                pipeline.fit(x_train, y_train)
                metrics = evaluate(y_test, pipeline.predict_proba(x_test)[:, 1])
                mlflow.log_param("model", name)
                mlflow.log_metrics(metrics)
                results[name] = metrics
                fitted[name] = pipeline

        # Models are compared on `selection_metric` (default ROC AUC; use avg_precision for rare positives).
        metric = model_cfg.get("selection_metric", "roc_auc")
        if metric not in results[BASELINE]:
            raise ValueError(f"Unknown selection_metric '{metric}'. Available: {sorted(results[BASELINE])}")
        baseline_score = results[BASELINE][metric]
        challengers = [n for n in results if n != BASELINE]
        threshold = model_cfg["min_auc_improvement"]  # minimum gain in the selection metric
        best = max(challengers, key=lambda n: results[n][metric]) if challengers else None
        gain = results[best][metric] - baseline_score if best else 0.0
        if best and gain >= threshold:
            champion = best
            reason = (
                f"Best challenger ({best}) beat the baseline by {gain:.3f} {metric}, above the "
                f"{threshold} promotion threshold."
            )
        else:
            champion = BASELINE
            reason = (
                f"Best challenger gain of {gain:.3f} {metric} is below the {threshold} threshold, "
                "so the simpler baseline is kept."
            )

        card: dict[str, Any] = {
            "version": version,
            "trained_at": datetime.now(UTC).isoformat(),
            "champion": champion,
            "selection_reason": reason,
            "selection_metric": metric,
            "description": model_cfg.get("description"),
            "label_definition": model_cfg.get("label_definition"),
            "features": features,
            "target": target,
            "threshold": 0.5,
            "metrics": {"champion_auc": results[champion]["roc_auc"], "all_candidates": results},
            "data": {
                "rows_total": int(len(df)),
                "train_end": str(train_df[ts_col].max()),
                "test_start": str(test_df[ts_col].min()),
                "test_end": str(test_df[ts_col].max()),
            },
            "limitations": [
                "Evaluated on one time-ordered holdout; performance may differ in other periods.",
                "Default decision threshold is 0.5 and has not been tuned to a business cost.",
                *model_cfg.get("limitations", []),
            ],
        }

        out_dir = Path(model_cfg["registry_dir"]) / version
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted[champion], out_dir / "model.joblib")
        (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))
        x_train.to_parquet(out_dir / "reference.parquet", index=False)
        promote(model_cfg["registry_dir"], version)

        mlflow.set_tag("champion", champion)
        mlflow.log_metric("champion_roc_auc", results[champion]["roc_auc"])
        mlflow.log_artifact(str(out_dir / "model_card.json"))
    return card
