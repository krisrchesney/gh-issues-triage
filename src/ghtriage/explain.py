"""SHAP explanations (optional extra: pip install -e ".[explain]").

Uses SHAP's model-agnostic permutation explainer so the same code explains the logistic baseline,
HistGradientBoosting and LightGBM pipelines. Contributions are in probability units: they sum
(with the base value) to the model's predicted probability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MISSING_MSG = "SHAP is not installed. Run: pip install -e '.[explain]'"


def _import_shap():
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(MISSING_MSG) from exc
    return shap


def make_explainer(model: Any, background: pd.DataFrame, features: list[str], max_background: int = 100):
    shap = _import_shap()
    bg = background[features]
    if len(bg) > max_background:
        bg = bg.sample(max_background, random_state=0)

    def predict_positive(x: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(np.asarray(x), columns=features))[:, 1]

    masker = shap.maskers.Independent(bg.to_numpy(dtype=float), max_samples=max_background)
    return shap.Explainer(predict_positive, masker, algorithm="permutation", feature_names=features, seed=0)


def _max_evals(n_features: int) -> int:
    return max(500, 4 * n_features + 1)


def global_importance(
    explainer: Any, sample: pd.DataFrame, features: list[str], max_rows: int = 200
) -> dict[str, float]:
    """Mean absolute SHAP value per feature over a sample, sorted descending."""
    rows = sample[features].head(max_rows).to_numpy(dtype=float)
    values = explainer(rows, max_evals=_max_evals(len(features))).values
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1]
    return {features[i]: float(importance[i]) for i in order}


def explain_record(explainer: Any, record: dict[str, float], features: list[str]) -> dict[str, Any]:
    row = np.array([[float(record[f]) for f in features]])
    result = explainer(row, max_evals=_max_evals(len(features)))
    contributions = {f: float(v) for f, v in zip(features, result.values[0], strict=True)}
    ordered = dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True))
    return {"base_value": float(result.base_values[0]), "contributions": ordered}


def save_importance_plot(importance: dict[str, float], path: str | Path, top: int = 15) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(importance.items())[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(items) + 1)))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#3b6ea5")
    ax.set_xlabel("Mean |SHAP value| (probability points)")
    ax.set_title("Global feature importance")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
