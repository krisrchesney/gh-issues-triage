"""Collector interface. Each project implements one Collector and registers it below.

A collector fetches the latest rows from a public source. `run_collector` upserts them into the Parquet
lake, so it is safe to run on a schedule and to re-run. Always respect each data source's terms, rate
limits and attribution rules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ghtriage.github_issues import GitHubIssuesCollector
from ghtriage.lake import upsert_partitioned


class Collector(Protocol):
    name: str
    key_cols: list[str]
    partition_col: str  # stable per key (e.g. creation time) so a row always lands in one file
    version_col: str | None  # a newer value replaces the stored row; None means insert-only
    keep_first_cols: list[str]  # "as first seen" columns that are never overwritten

    def fetch(self) -> pd.DataFrame: ...


class DemoCollector:
    """Emits a few synthetic rows stamped with the current time. Replace in real projects."""

    name = "demo"
    key_cols = ["row_id"]
    partition_col = "event_ts"
    version_col = None
    keep_first_cols: list[str] = []

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}

    def fetch(self) -> pd.DataFrame:
        rng = np.random.default_rng()
        n = 5
        df = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
        df["label"] = rng.integers(0, 2, size=n)
        df["event_ts"] = pd.Timestamp(datetime.now(UTC).replace(tzinfo=None))
        df["row_id"] = rng.integers(10**9, 10**10, size=n)
        return df


COLLECTORS: dict[str, type] = {"demo": DemoCollector, "gh_issues": GitHubIssuesCollector}


def run_collector(name: str, cfg: dict[str, Any]) -> tuple[int, int]:
    """Run one collector and return (inserted, updated) row counts."""
    if name not in COLLECTORS:
        raise KeyError(f"Unknown collector '{name}'. Registered: {sorted(COLLECTORS)}")
    collector = COLLECTORS[name](cfg)
    frame = collector.fetch()
    result = upsert_partitioned(
        cfg["lake"]["root"],
        collector.name,
        frame,
        key_cols=collector.key_cols,
        partition_col=collector.partition_col,
        version_col=collector.version_col,
        keep_first_cols=collector.keep_first_cols,
    )
    # Optional hook: save cursors/state only after the rows are safely stored.
    commit = getattr(collector, "commit", None)
    if callable(commit):
        commit()
    return result
