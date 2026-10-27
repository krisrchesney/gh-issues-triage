import numpy as np
import pandas as pd
import pytest
import yaml

from ghtriage.features import FEATURES, build_training_frame, compute_features, normalise
from tests.synth import make_raw_issues

H = 14


def _mini(rows):
    """rows: (number, created_day, closed_day or None, author). One repo, day 0 = 2026-01-01."""
    base = pd.Timestamp("2026-01-01")
    return pd.DataFrame(
        [
            {
                "repo": "org/x",
                "number": n,
                "created_at": base + pd.Timedelta(days=c),
                "updated_at": base + pd.Timedelta(days=(cl if cl is not None else c)),
                "closed_at": base + pd.Timedelta(days=cl) if cl is not None else pd.NaT,
                "author_key": a,
                "is_bot": False,
                "author_association": "NONE",
                **{
                    k: 1
                    for k in [
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
                    ]
                },
            }
            for n, c, cl, a in rows
        ]
    )


def test_context_features_match_hand_calculation():
    raw = _mini([(1, 0, 2, "X"), (2, 1, None, "Y"), (3, 3, 10, "X"), (4, 5, None, "X")])
    feats = compute_features(normalise(raw), H).set_index("number")
    d = feats.loc[4]
    assert d["repo_arrivals_7d"] == 3  # issues 1, 2, 3 were opened in the previous 7 days
    assert d["repo_open_30d"] == 2  # 2 still open; 3 closes on day 10, after day 5
    assert d["repo_median_ttc_30d_hours"] == pytest.approx(48.0)  # only issue 1 closed before day 5
    assert np.isnan(d["repo_closed_rate_recent"])  # nothing old enough to be settled yet
    assert d["author_prior_issues"] == 2  # author X opened issues 1 and 3 before
    assert feats.loc[1]["author_prior_issues"] == 0
    assert feats.loc[1]["repo_arrivals_7d"] == 0


def test_closed_rate_uses_only_settled_issues():
    # six issues opened on days 0..5; five close within 14 days, one stays open. Query opened at day 40.
    rows = [(i + 1, i, i + 2 if i < 5 else None, f"a{i}") for i in range(6)] + [(99, 40, None, "q")]
    feats = compute_features(normalise(_mini(rows)), H).set_index("number")
    assert feats.loc[99]["repo_closed_rate_recent"] == pytest.approx(5 / 6)


def test_no_future_information_leaks_into_features():
    """Truncating the world to the moment an issue opened must not change any of its features."""
    raw = normalise(make_raw_issues(n_per_repo=250, days=120, seed=3))
    full = compute_features(raw, H).set_index(["repo", "number"])
    rng = np.random.default_rng(0)
    sample = raw.iloc[rng.choice(len(raw), size=25, replace=False)]
    for _, row in sample.iterrows():
        t = row["created_at"]
        snap = raw[raw["created_at"] <= t].copy()
        snap.loc[snap["closed_at"] > t, "closed_at"] = pd.NaT  # outcomes after t were not known yet
        snap_feats = compute_features(snap, H).set_index(["repo", "number"])
        key = (row["repo"], row["number"])
        np.testing.assert_allclose(
            snap_feats.loc[key, FEATURES].to_numpy(dtype=float),
            full.loc[key, FEATURES].to_numpy(dtype=float),
            equal_nan=True,
            err_msg=f"future information leaked into the features of {key}",
        )


def test_build_excludes_issues_younger_than_horizon_even_if_closed():
    raw = make_raw_issues(n_per_repo=400, days=150, seed=1)
    out = build_training_frame(raw, coverage_start="2026-06-01", warmup_days=30, horizon_days=H)
    freshness = raw.groupby("repo")["updated_at"].max()
    limit = out["repo"].map(freshness) - pd.Timedelta(days=H)
    assert (out["event_ts"] <= limit).all()


def test_build_respects_coverage_and_warmup():
    raw = make_raw_issues(n_per_repo=400, days=150, seed=1)
    out = build_training_frame(raw, coverage_start="2026-06-01", warmup_days=30, horizon_days=H)
    assert out["event_ts"].min() >= pd.Timestamp("2026-07-01")
    later = build_training_frame(raw, coverage_start="2026-07-15", warmup_days=30, horizon_days=H)
    assert later["event_ts"].min() >= pd.Timestamp("2026-08-14")
    assert len(later) < len(out)


def test_label_boundary_is_inclusive_at_the_horizon():
    base = pd.Timestamp("2026-01-01")
    rows = []
    for i in range(300):
        c = base + pd.Timedelta(days=40, hours=i)
        rows.append((i + 1, c, c + pd.Timedelta(days=14)))  # closes exactly at the horizon -> 1
    for i in range(300):
        c = base + pd.Timedelta(days=40, hours=i)
        rows.append((1000 + i, c, c + pd.Timedelta(days=14, seconds=1)))  # one second late -> 0
    raw = _mini([(1, 0, 1, "a")])
    raw = pd.concat(
        [raw]
        + [
            _mini([(1, 0, 1, "a")]).assign(number=n, created_at=c, closed_at=cl, updated_at=cl)
            for n, c, cl in rows
        ],
        ignore_index=True,
    )
    raw = pd.concat([raw, _mini([(5000, 120, None, "z")])], ignore_index=True)  # pushes freshness forward
    out = build_training_frame(raw, coverage_start="2026-01-01", warmup_days=30, horizon_days=H, min_rows=100)
    on_time = out[out["number"] <= 300]
    late = out[(out["number"] >= 1000) & (out["number"] < 5000)]
    assert len(on_time) and (on_time["label"] == 1).all()
    assert len(late) and (late["label"] == 0).all()


def test_too_little_data_gives_a_clear_error():
    raw = make_raw_issues(n_per_repo=20, days=60, seed=2)
    with pytest.raises(ValueError, match="labelled rows"):
        build_training_frame(raw, coverage_start="2026-06-01", warmup_days=30, horizon_days=H)


def test_missing_raw_columns_are_reported():
    raw = make_raw_issues(n_per_repo=30, days=30).drop(columns=["body_len"])
    with pytest.raises(ValueError, match="body_len"):
        build_training_frame(raw, coverage_start="2026-06-01", warmup_days=5, horizon_days=H)


def test_config_feature_list_matches_code():
    cfg = yaml.safe_load(open("config/config.yaml"))
    assert cfg["model"]["features"] == FEATURES


def test_no_post_creation_column_is_a_feature():
    forbidden = {
        "comments",
        "labels",
        "state",
        "state_reason",
        "closed_at",
        "updated_at",
        "issue_id",
        "number",
    }
    assert forbidden.isdisjoint(FEATURES)


def test_original_titles_replace_renamed_ones_and_their_counts():
    from ghtriage.features import attach_original_titles

    issues = pd.DataFrame(
        {
            "repo": ["golang/go", "golang/go"],
            "number": [1, 2],
            "title": ["net/http: Client leaks on redirect", "untouched title"],
            "title_len": [34, 15],
            "title_words": [6, 2],
        }
    )
    titles = pd.DataFrame({"repo": ["golang/go"], "number": [1], "title_at_creation": ["http client leak?"]})
    out = attach_original_titles(issues, titles)
    assert list(out["title"]) == ["http client leak?", "untouched title"]
    assert list(out["title_len"]) == [17, 15] and list(out["title_words"]) == [3, 2]


def test_a_later_rename_cannot_change_title_features():
    """Regression guard: maintainers' retitling during triage must not reach the features."""
    from ghtriage.features import attach_original_titles, title_features

    before = pd.DataFrame(
        {"repo": ["golang/go"], "number": [7], "title": ["it crashes"], "title_len": [10], "title_words": [2]}
    )
    after = before.assign(title="runtime: crash in GC on arm64", title_len=29, title_words=6)
    orig = pd.DataFrame({"repo": ["golang/go"], "number": [7], "title_at_creation": ["it crashes"]})
    a = title_features(attach_original_titles(before, orig)["title"])
    b = title_features(attach_original_titles(after, orig)["title"])
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize(
    "title,flag",
    [
        ("App crashes on startup", "title_kw_crash"),
        ("Add support for dark mode", "title_kw_feature"),
        ("Typo in README", "title_kw_docs"),
        ("Memory leak in image cache", "title_kw_perf"),
        ("Scrolling broken since 3.22", "title_kw_regression"),
        ("How do I configure the router?", "title_kw_question"),
        ("[iOS] keyboard covers text field", "title_platform_tag"),
        ("net/http: client hangs", "title_pkg_prefix"),
        ("feat(router): lazy guards", "title_conventional"),
        ("`ngModel` ignores null", "title_has_code"),
    ],
)
def test_title_patterns(title, flag):
    from ghtriage.features import title_features

    row = title_features(pd.Series([title])).iloc[0]
    assert row[flag] == 1
    assert title_features(pd.Series(["plain words only"])).iloc[0].sum() == 0
