import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from ghtriage.train import rolling_fold_scores

SEEN: list[tuple[int, int, int, int]] = []


class Recorder(BaseEstimator, ClassifierMixin):
    """Remembers the time range it was fitted on and checks every prediction is strictly later."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.fit_max_ = int(X["t"].max())
        return self

    def predict_proba(self, X):
        assert int(X["t"].min()) > self.fit_max_, "validation rows must come after every fitted row"
        SEEN.append((self.fit_max_, int(X["t"].min()), int(X["t"].max()), len(X)))
        p = X["signal"].to_numpy()
        return np.column_stack([1 - p, p])


def test_rolling_folds_only_score_later_rows_and_cover_the_newest_half():
    SEEN.clear()
    n = 400
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"t": np.arange(n), "signal": rng.uniform(size=n)})
    df["label"] = (rng.uniform(size=n) < df["signal"]).astype(int)
    scores = rolling_fold_scores(df, ["t", "signal"], "label", {"m": Recorder()}, "roc_auc", folds=4)
    assert len(scores["m"]) == 4
    assert SEEN[0][1] == n // 2 and SEEN[-1][2] == n - 1  # blocks tile the newest half exactly
    assert all(a[2] < b[1] for a, b in zip(SEEN[:-1], SEEN[1:], strict=True))  # consecutive blocks
    assert all(0.5 < s <= 1 for s in scores["m"])  # the informative signal is rewarded in every block


def test_blocks_with_one_class_are_skipped():
    df = pd.DataFrame({"t": np.arange(100), "signal": np.linspace(0.1, 0.9, 100)})
    df["label"] = (np.arange(100) % 2).astype(int)
    df.loc[50:74, "label"] = 0  # the first validation block holds only negatives
    scores = rolling_fold_scores(df, ["t", "signal"], "label", {"m": Recorder()}, "roc_auc", folds=2)
    assert len(scores["m"]) == 1
