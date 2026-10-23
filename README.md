# gh-issues-triage

Predicts, **at the moment a GitHub issue is opened**, whether maintainers will close it within 14 days. Trained on issues from large Google-backed open-source repositories, served through a validated FastAPI endpoint, and monitored for data and performance drift.

> Status: pipeline and tests are complete. **No real-data results are published here yet**: the first collection run happens on your GitHub account. Numbers will be added to the Results section after that run; none are claimed before it.

## Why this project

Issue queues are ticket queues. The question "which new tickets are likely to stall?" is the same one a support or engineering team asks at intake, and public issue trackers give a free, continuously updating, auditable dataset (full history, so no waiting weeks to start).

## How it works

```
GitHub REST API --collect--> Parquet lake (data/raw/gh_issues/dt=YYYY-MM-DD)
                                   |  synced to an orphan `data` branch by scripts/data_sync.sh
                                   v
                 build (leak-free labels + creation-time features) --> DuckDB
                                   v
        train: logistic baseline vs HistGradientBoosting vs LightGBM  (MLflow)
                                   v
              models/<version>/ + model card  -->  FastAPI /predict /explain
                                   v
        drift: PSI per feature + live AUC after the training cutoff --> retrain flag
```

## Leakage and bias controls (the part worth reading)

Most "predict ticket outcome" projects quietly cheat. This one is built so it cannot:

| Risk | Control | Proof |
|---|---|---|
| Features that know the future (comments, labels, state, closed_at) | Explicit feature allowlist in `config.yaml`; post-creation columns are stored for analysis but blocked | `test_no_post_creation_column_is_a_feature`, `test_config_feature_list_matches_code` |
| Context features using later events | Every context feature uses only facts true strictly before the issue was opened | `test_no_future_information_leaks_into_features` truncates the world to the issue's creation moment and requires identical features |
| Right-censoring | Issues younger than the 14-day horizon are excluded **even if already closed**; otherwise fast closers would be over-represented | `test_build_excludes_issues_younger_than_horizon_even_if_closed` |
| Survivor bias from the backfill window | Issues created before `coverage_start` are dropped before any feature is computed; a 30-day warm-up is skipped | `test_build_respects_coverage_and_warmup` |
| Edited titles/bodies | Text statistics are frozen at first sight and never overwritten | `test_updates_replace_rows_but_keep_the_first_seen_snapshot` |
| Random splits leaking time | Time-ordered holdout (newest 20%) | `test_time_split_never_leaks_future` |
| Monitoring on training data | Drift and live AUC use only rows newer than the training cutoff | `test_monitoring_window_only_contains_rows_after_the_training_cutoff` |

## Privacy and data handling

- Issue **bodies are never stored**: text is reduced to counts (length, code blocks, links, stack-trace flag...) at ingest.
- Author ids are **salted and hashed**; no usernames are kept. Use a private `GH_AUTHOR_SALT` secret (see Setup).
- Only public data from public repositories is used. Respect GitHub's terms and rate limits.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                            # includes a no-network end-to-end run
export GITHUB_TOKEN=...                           # fine-grained token, public repo read only
export GH_AUTHOR_SALT="$(openssl rand -hex 16)"   # keep this value forever
python -m ghtriage.github_issues check-repos      # confirm each repo exists and is busy
python -m ghtriage.cli collect gh_issues          # first run backfills from coverage_start (resumable)
python -m ghtriage.cli build && python -m ghtriage.cli train
python -m ghtriage.cli drift && python -m ghtriage.cli explain
python -m ghtriage.cli serve                      # http://localhost:8000/docs
```

The first backfill is capped at `max_pages_per_repo` pages per repo per run; re-run until it reports `0 inserted` (the cursor lives in the lake, so it resumes automatically).

## Decisions and tradeoffs

- **Label = closed within 14 days, any reason.** Cheap to compute from the list endpoint (time-to-first-response would need one extra request per issue, which the 1,000 requests/hour Actions budget cannot afford). The cost: "closed" mixes completed, duplicate and not-planned.
- **Issues created before `coverage_start` are not stored.** The builder would drop them anyway, so keeping them only grows the data branch. A small `gh_state` progress marker keeps resumption correct when whole pages are skipped.
- **Pull requests are fetched and discarded.** The issues endpoint returns both and cannot filter them server-side, so PR-heavy repos spend pages on rows that are thrown away.
- **Counts instead of text.** Privacy and storage win; a language model on the text is the obvious next step and is not included.
- **The baseline must be beaten.** Logistic regression stays champion unless a challenger gains at least 0.01 ROC AUC.
- **DuckDB + Parquet, not a warehouse.** Free and local; no managed-warehouse experience is claimed.
- **Known collection gap.** Position-based pagination can skip an issue at a page boundary if an older issue is updated mid-run; expect a few missing rows per large backfill. Skips are unrelated to the outcome, so bias is negligible.
- **Collection bug found and repaired (Oct 2026).** GitHub's `sort=updated` list is not strictly ordered: during the flutter/flutter backfill, one item reported an `updated_at` two weeks ahead of its neighbours. The collector resumed from the newest value it had seen, so a capped run skipped every issue last updated between Aug 26 and Sep 9, 2026: 237 of 2,135 flutter issues created since `coverage_start`, 155 of them closed. That gap was **not** random with respect to the label. A check against GitHub's search counts found the other three repositories complete. Capped runs now resume from the median of their last page. The window was re-fetched with a one-off `github.rewind` entry (run on Oct 23, 2026: 237 issues restored). A re-check against GitHub's search API then found 0 of 4,648 issues created Jul 1 - Oct 21, 2026 missing in any of the four repositories, and 0 stored states differing from GitHub.

## Results

_Not yet generated._ After the first real run, add: ROC AUC / average precision / calibration (ECE) for the baseline and the champion, the SHAP importance chart from `reports/`, and the Evidently drift report.

## Limitations

- Closed-for-any-reason label; no distinction between resolved and dismissed.
- Context features start empty at `coverage_start` (hence the warm-up).
- Trained on a few large repositories; transfer to small projects is untested.
- The `/explain` endpoint is slow on cold start (permutation SHAP) and needs the `explain` extra.

## Data source

Data from the GitHub REST API (public repositories).
