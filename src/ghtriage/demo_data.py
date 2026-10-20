"""Synthetic placeholder data so the template runs end to end before a real source is wired in.

Replace this with a collector for your real data source (see collect.py). Clearly synthetic:
never present results from this data as real findings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def make_demo_data(
    n_rows: int = 5000,
    seed: int = 42,
    start: str = "2026-01-01",
    shift: float = 0.0,
) -> pd.DataFrame:
    """Generate a time-ordered binary classification set.

    `shift` moves the feature means to simulate drift, which lets you test the monitoring path.
    """
    x, y = make_classification(
        n_samples=n_rows,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.85, 0.15],
        flip_y=0.02,
        random_state=seed,
    )
    x = x + shift
    cols = [f"f{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=cols)
    df["label"] = y.astype(int)
    df["event_ts"] = pd.date_range(start=start, periods=n_rows, freq="15min")
    rng = np.random.default_rng(seed)
    df["row_id"] = rng.permutation(n_rows) + int(seed) * 1_000_000
    return df
