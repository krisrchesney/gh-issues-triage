"""Leak-free dataset construction for the issue triage model.

Prediction moment: the instant an issue is opened.
Question: will it be closed within `label_horizon_days` (default 14)?

Rules that keep the evaluation honest:

1. Only information available at creation time is used. Per-issue features come from the "as first
   seen" snapshot columns; context features (repo backlog, recent resolution speed, author history)
   use only other issues' facts that were already true strictly before the issue was created.
   `comments`, `labels`, `state` and `closed_at` of the issue itself are never features.
2. Right-censoring: an issue younger than the horizon cannot be labelled yet. Keeping the ones that
   happened to close fast while dropping the ones still open would manufacture positives, so ALL
   issues younger than the horizon are excluded (judged against each repo's own data freshness).
3. Coverage: the lake only holds issues updated since `coverage_start`. Issues created earlier but
   still active are a survivor-biased sample, so rows created before `coverage_start` are dropped
   before any feature is computed, and a `warmup_days` period is skipped so context features have
   history behind them.

`compute_features` is separately testable: tests/test_features.py proves that every feature of an
issue is unchanged when the world is truncated to the moment the issue was created.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ghtriage.lake import load_lake
from ghtriage.storage import connect, write_frame

DAY_NS = 86_400 * 10**9
HOUR_NS = 3_600 * 10**9
INF = np.iinfo(np.int64).max
MIN_SUPPORT = 5  # minimum matured issues needed before a recent resolution rate is reported

ISSUE_FEATURES = [
    "title_len",
    "title_words",
    "body_len",
    "body_words",
    "code_blocks",
    "has_stack_trace",
    "n_links",
    "n_images",
    "has_version",
    "n_questions",
    "n_checkboxes",
    "n_headings",
    "is_bot",
    "assoc_maintainer",
    "assoc_first_time",
    "assoc_contributor",
    "created_hour",
    "created_dow",
    "is_weekend",
]
CONTEXT_FEATURES = [
    "author_prior_issues",
    "repo_arrivals_7d",
    "repo_open_30d",
    "repo_median_ttc_30d_hours",
    "repo_closed_rate_recent",
]
# Title signals, computed from the title at creation (see attach_original_titles). The word lists are
# fixed a priori from general knowledge of issue trackers, never tuned on the labels.
TITLE_PATTERNS = {
    "title_kw_crash": r"\b(crash\w*|segfault|panic\w*|hang\w*|freez\w*|exception|error|fail\w*)\b",
    "title_kw_feature": r"\b(feature|support|add|allow|proposal|request|enhancement|option)\b",
    "title_kw_docs": r"\b(doc|docs|documentation|typo|readme|example)\b",
    "title_kw_perf": r"\b(slow|performance|memory|leak|latency)\b",
    "title_kw_regression": r"\b(regression|regressed|broke|broken|no longer)\b",
    "title_kw_question": r"(\?|^\s*(how|why|what|is it possible)\b)",
    "title_platform_tag": r"^\s*\[[^\]]{1,40}\]",
    "title_pkg_prefix": r"^\s*[\w./-]{2,40}:\s",
    "title_conventional": r"^\s*(feat|fix|docs|chore|refactor|perf|test|build|ci)(\([^)]*\))?!?:",
    "title_has_code": r"`",
}
TITLE_FEATURES = list(TITLE_PATTERNS)
# Which project an issue belongs to: close rates differ a lot between them.
REPO_FEATURES = ["repo_angular", "repo_flutter", "repo_go", "repo_tensorflow"]
REPO_FEATURE_OF = {
    "angular/angular": "repo_angular",
    "flutter/flutter": "repo_flutter",
    "golang/go": "repo_go",
    "tensorflow/tensorflow": "repo_tensorflow",
}
FEATURES = ISSUE_FEATURES + CONTEXT_FEATURES + TITLE_FEATURES + REPO_FEATURES
TITLE_SOURCE = "gh_titles"  # lake source with each backfilled issue's title at creation

REQUIRED_RAW = [
    "repo",
    "number",
    "created_at",
    "updated_at",
    "closed_at",
    "author_key",
    "is_bot",
    "author_association",
    *[
        c
        for c in ISSUE_FEATURES
        if c
        not in {
            "is_bot",
            "assoc_maintainer",
            "assoc_first_time",
            "assoc_contributor",
            "created_hour",
            "created_dow",
            "is_weekend",
        }
    ],
]


def _ns(series: pd.Series, nat_value: int) -> np.ndarray:
    """Datetime series -> int64 nanoseconds, whatever unit pandas chose, with NaT mapped to nat_value."""
    values = series.astype("datetime64[ns]").to_numpy().astype("int64")
    return np.where(series.isna().to_numpy(), nat_value, values)


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_RAW if c not in raw.columns]
    if missing:
        raise ValueError(f"Lake data is missing columns: {missing}")
    df = raw.copy()
    for col in ("created_at", "updated_at", "closed_at"):
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    return df


def attach_original_titles(issues: pd.DataFrame, titles: pd.DataFrame | None) -> pd.DataFrame:
    """Use each issue's title at creation where it is known.

    The lake keeps titles "as first seen". For issues first collected long after they were opened (the
    initial backfill), that can be a title a maintainer rewrote during triage, which leaks the outcome:
    renamed issues close far less often. `titles` (repo, number, title_at_creation) comes from GitHub's
    "renamed" events. Title length and word counts are recomputed from the original too.
    """
    df = issues.copy()
    if titles is None or titles.empty or "title" not in df.columns:
        return df
    orig = titles.drop_duplicates(["repo", "number"]).set_index(["repo", "number"])["title_at_creation"]
    keys = pd.MultiIndex.from_arrays([df["repo"], df["number"].astype(int)])
    found = orig.reindex(keys).to_numpy()
    has = pd.notna(found)
    if has.any():
        df.loc[has, "title"] = found[has]
        df.loc[has, "title_len"] = df.loc[has, "title"].str.len()
        df.loc[has, "title_words"] = df.loc[has, "title"].str.split().str.len()
    return df


def title_features(titles: pd.Series) -> pd.DataFrame:
    text = titles.fillna("").astype(str).str.lower()
    return pd.DataFrame(
        {
            name: text.str.contains(pattern, regex=True).astype(int)
            for name, pattern in TITLE_PATTERNS.items()
        },
        index=titles.index,
    )


def compute_features(issues: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Add creation-time features. `issues` must already be normalised and restricted to coverage."""
    df = issues.sort_values(["repo", "created_at", "number"]).reset_index(drop=True)
    n = len(df)
    horizon_ns = int(horizon_days) * DAY_NS
    titles = df["title"] if "title" in df.columns else pd.Series("", index=df.index)
    df = pd.concat([df, title_features(titles)], axis=1)
    for repo, col in REPO_FEATURE_OF.items():
        df[col] = (df["repo"] == repo).astype(int)

    created = df["created_at"]
    assoc = df["author_association"].fillna("NONE")
    df["is_bot"] = df["is_bot"].astype(int)
    df["assoc_maintainer"] = assoc.isin(["OWNER", "MEMBER", "COLLABORATOR"]).astype(int)
    df["assoc_first_time"] = assoc.isin(["FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER"]).astype(int)
    df["assoc_contributor"] = (assoc == "CONTRIBUTOR").astype(int)
    df["created_hour"] = created.dt.hour
    df["created_dow"] = created.dt.dayofweek
    df["is_weekend"] = (created.dt.dayofweek >= 5).astype(int)

    # An unknown author (deleted account) has no history we can trust: count zero prior issues.
    key = df["author_key"].fillna("__unknown__")
    prior = df.groupby(["repo", key]).cumcount()
    df["author_prior_issues"] = prior.where(df["author_key"].notna(), 0).astype(float)

    arrivals = np.full(n, np.nan)
    open_30 = np.full(n, np.nan)
    median_ttc = np.full(n, np.nan)
    closed_rate = np.full(n, np.nan)

    created_ns_all = _ns(created, 0)
    closed_ns_all = _ns(df["closed_at"], INF)

    for _, idx in df.groupby("repo").indices.items():
        c = created_ns_all[idx]  # ascending: the frame is sorted by repo, created_at, number
        cl = closed_ns_all[idx]  # INF while the issue is still open
        is_closed = cl != INF
        ttc_hours = np.where(is_closed, (cl - c) / HOUR_NS, np.nan)
        closed_fast = is_closed & ((cl - c) <= horizon_ns)
        prefix = np.concatenate([[0], np.cumsum(closed_fast)])

        before = np.searchsorted(c, c, side="left")  # issues created strictly before this one
        arrivals[idx] = before - np.searchsorted(c, c - 7 * DAY_NS, side="left")

        lo30 = np.searchsorted(c, c - 30 * DAY_NS, side="left")
        open_30[idx] = [(cl[lo30[k] : before[k]] >= c[k]).sum() for k in range(len(c))]

        order = np.argsort(cl[is_closed], kind="stable")
        closed_sorted = cl[is_closed][order]
        ttc_sorted = ttc_hours[is_closed][order]
        a = np.searchsorted(closed_sorted, c - 30 * DAY_NS, side="left")
        b = np.searchsorted(closed_sorted, c, side="left")  # closed strictly before creation
        median_ttc[idx] = [
            float(np.median(ttc_sorted[a[k] : b[k]])) if b[k] > a[k] else np.nan for k in range(len(c))
        ]

        # Issues old enough that their outcome was already settled when this one was opened.
        matured_hi = np.searchsorted(c, c - horizon_ns, side="right")
        matured_lo = np.searchsorted(c, c - 60 * DAY_NS, side="left")
        support = matured_hi - matured_lo
        rate = (prefix[matured_hi] - prefix[matured_lo]) / np.maximum(support, 1)
        closed_rate[idx] = np.where(support >= MIN_SUPPORT, rate, np.nan)

    df["repo_arrivals_7d"] = arrivals
    df["repo_open_30d"] = open_30
    df["repo_median_ttc_30d_hours"] = median_ttc
    df["repo_closed_rate_recent"] = closed_rate
    return df


def build_training_frame(
    raw: pd.DataFrame,
    *,
    coverage_start: str | pd.Timestamp,
    warmup_days: int,
    horizon_days: int,
    as_of: Mapping[str, pd.Timestamp] | None = None,
    min_rows: int = 200,
    titles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Labelled, leak-free training rows. `as_of` is each repo's data freshness (default: newest update).
    `titles` holds titles at creation for issues first seen late (see attach_original_titles)."""
    coverage_start = pd.Timestamp(coverage_start)
    df = attach_original_titles(normalise(raw), titles)
    df = df[df["created_at"] >= coverage_start]
    if df.empty:
        raise ValueError(f"No issues created on or after coverage_start={coverage_start.date()}.")
    feats = compute_features(df, horizon_days)

    freshness = pd.Series(dict(as_of)) if as_of is not None else df.groupby("repo")["updated_at"].max()
    cutoff = feats["repo"].map(freshness) - pd.Timedelta(days=horizon_days)
    mature = feats["created_at"] <= cutoff
    warm = feats["created_at"] >= coverage_start + pd.Timedelta(days=warmup_days)
    kept = feats[mature & warm].copy()

    horizon = pd.Timedelta(days=horizon_days)
    kept["label"] = (kept["closed_at"].notna() & (kept["closed_at"] - kept["created_at"] <= horizon)).astype(
        int
    )
    kept["event_ts"] = kept["created_at"]
    out = (
        kept[["event_ts", "label", "repo", "number", *FEATURES]]
        .sort_values("event_ts")
        .reset_index(drop=True)
    )

    if len(out) < min_rows:
        raise ValueError(
            f"Only {len(out)} labelled rows (need {min_rows}). "
            f"Collected {len(df)} issues since coverage_start; "
            f"{int((~mature).sum())} are younger than the {horizon_days}-day horizon and "
            f"{int((~warm).sum())} fall in the {warmup_days}-day warm-up. Collect more history first."
        )
    if out["label"].nunique() < 2:
        raise ValueError("Labels contain a single class; the dataset is not usable yet.")
    return out


def build_training_table(cfg: dict[str, Any]) -> int:
    """Entry point for `ghtriage build` (configured as data.builder). Returns the training row count."""
    data_cfg = cfg["data"]
    raw = load_lake(cfg["lake"]["root"], data_cfg["source"])
    if raw.empty:
        raise ValueError("The lake is empty. Run `collect gh_issues` (and sync the data branch) first.")
    frame = build_training_frame(
        raw,
        coverage_start=cfg["github"]["coverage_start"],
        warmup_days=int(data_cfg.get("warmup_days", 30)),
        horizon_days=int(data_cfg.get("label_horizon_days", 14)),
        min_rows=int(data_cfg.get("min_rows", 200)),
        titles=load_lake(cfg["lake"]["root"], TITLE_SOURCE),
    )
    con = connect(data_cfg["db_path"])
    try:
        return write_frame(con, data_cfg["table"], frame)
    finally:
        con.close()
