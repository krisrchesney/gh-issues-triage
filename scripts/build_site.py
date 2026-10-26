# ruff: noqa: E501  (the HTML template below is easier to read unwrapped)
"""Build the static GitHub Pages site: new issues most likely to stall, plus an honest weekly replay.

Run after `build` and `train` (the champion in models/ is used for the board):

    python scripts/build_site.py --out site

Writes <out>/index.html and <out>/data.json.

Why a replay: an issue's 14-day outcome is known exactly 14 days after it opens, and every retrain uses
all known outcomes, so there is never a set of labelled issues the current model has not seen. Instead,
for each recent week the champion's model type (and the logistic baseline) is retrained on the outcomes
that were knowable when that week began (issues opened at least 14 days earlier) and scored on that
week's issues. That is what the model would have done live.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from site_charts import CHART_CSS, CHART_JS, scatter_chart, series_chart
from sklearn.metrics import roc_auc_score

from ghtriage.config import load_config
from ghtriage.features import build_training_frame, compute_features, normalise
from ghtriage.lake import load_lake
from ghtriage.registry import load_current
from ghtriage.train import BASELINE, build_candidates, select_features

REPO_VARS = {
    "angular/angular": "--s1",
    "flutter/flutter": "--s2",
    "golang/go": "--s3",
    "tensorflow/tensorflow": "--s4",
}
MIN_TRAIN_ROWS = 500


def load_issues(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_lake(cfg["lake"]["root"], "gh_issues")
    df = normalise(raw)
    return raw, df[df["created_at"] >= pd.Timestamp(cfg["github"]["coverage_start"])]


def board(cfg: dict, issues: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict, str]:
    """Open issues opened in the last `horizon` days, scored by the current champion."""
    feats = compute_features(issues, horizon)
    model, card, version = load_current(cfg["model"]["registry_dir"])
    fresh = issues["updated_at"].max()
    recent = feats[
        (feats["created_at"] > fresh - pd.Timedelta(days=horizon)) & feats["closed_at"].isna()
    ].copy()
    recent["p_close"] = model.predict_proba(recent[card["features"]])[:, 1]
    recent["age_days"] = (fresh - recent["created_at"]).dt.total_seconds() / 86400
    return recent.sort_values("p_close").reset_index(drop=True), card, version


def replay(cfg: dict, table: pd.DataFrame, champion: str, weeks: int = 8) -> pd.DataFrame:
    """Per week: train on outcomes knowable at the week's start, score that week's issues."""
    horizon = pd.Timedelta(days=int(cfg["data"].get("label_horizon_days", 14)))
    feats = select_features(table, cfg)
    target = cfg["model"]["target"]
    last_full = table["event_ts"].max().normalize() - pd.Timedelta(days=table["event_ts"].max().dayofweek + 7)
    starts = pd.date_range(end=last_full, periods=weeks, freq="7D")
    rows, preds = [], []
    for ws in starts:
        week = table[(table["event_ts"] >= ws) & (table["event_ts"] < ws + pd.Timedelta(days=7))]
        train = table[table["event_ts"] <= ws - horizon]
        if len(train) < MIN_TRAIN_ROWS or train[target].nunique() < 2 or week[target].nunique() < 2:
            continue
        pipes = build_candidates(cfg["model"]["random_state"], [BASELINE, champion])
        out = {
            "week": ws,
            "n": len(week),
            "closed_rate": float(week[target].mean()),
            "train_rows": len(train),
        }
        for name, pipe in pipes.items():
            pipe.fit(train[feats], train[target])
            p = pipe.predict_proba(week[feats])[:, 1]
            out[f"auc_{name}"] = float(roc_auc_score(week[target], p))
            if name == champion:
                preds.append(pd.DataFrame({"week": ws, "p": p, "y": week[target].to_numpy()}))
        rows.append(out)
    return pd.DataFrame(rows), (pd.concat(preds) if preds else pd.DataFrame(columns=["week", "p", "y"]))


def calibration_points(pred: pd.DataFrame, bins: int = 8) -> list[dict]:
    pred = pred.assign(bin=pd.qcut(pred["p"].rank(method="first"), bins, labels=False))
    pts = []
    for _, g in pred.groupby("bin"):
        mp, ar = float(g["p"].mean()), float(g["y"].mean())
        pts.append(
            {
                "x": mp,
                "y": ar,
                "tip": {
                    "title": f"{len(g)} issues",
                    "rows": [
                        ["var(--s1)", f"{mp * 100:.0f}%", "predicted"],
                        [None, f"{ar * 100:.0f}%", "actually closed within 14 days"],
                    ],
                },
            }
        )
    return pts


def weekly_close_rates(table: pd.DataFrame, min_n: int = 15) -> tuple[list[str], list[dict]]:
    t = table.assign(week=table["event_ts"].dt.to_period("W-SUN").dt.start_time)
    weeks = sorted(t["week"].unique())
    weeks = [w for w in weeks if (t["week"] == w).sum() >= 50]  # drop the thin first/last partial weeks
    series = []
    for repo, var in REPO_VARS.items():
        ys = []
        for w in weeks:
            g = t[(t["repo"] == repo) & (t["week"] == w)]
            ys.append(float(g["label"].mean()) if len(g) >= min_n else None)
        series.append({"name": repo, "short": repo.split("/")[1], "var": var, "y": ys})
    return [f"{pd.Timestamp(w):%b %d}" for w in weeks], series


def render(
    b: pd.DataFrame,
    card: dict,
    version: str,
    rp: pd.DataFrame,
    pred: pd.DataFrame,
    rates,
    built: str,
    top: int,
    horizon: int,
    champion: str,
) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(r.repo.split('/')[1])}</td>"
        f'<td><a href="https://github.com/{html.escape(r.repo)}/issues/{int(r.number)}">#{int(r.number)}</a></td>'
        f'<td class="title">{html.escape(str(r.title)[:90])}</td>'
        f"<td>{r.created_at:%Y-%m-%d}</td>"
        f'<td class="num">{r.age_days:.1f}</td>'
        f'<td class="num">{r.p_close * 100:.0f}%</td>'
        "</tr>"
        for r in b.head(top).itertuples()
    )
    weeks = [f"{w:%b %d}" for w in rp["week"]] if len(rp) else []
    if len(rp):
        auc_chart = series_chart(
            "replay-auc",
            "Replayed weekly ROC AUC: model vs logistic baseline",
            weeks,
            [
                {
                    "name": f"Model ({champion})",
                    "short": "model",
                    "var": "--s1",
                    "y": rp[f"auc_{champion}"].tolist(),
                },
                {
                    "name": "Logistic baseline",
                    "short": "baseline",
                    "var": "--s2",
                    "y": rp[f"auc_{BASELINE}"].tolist(),
                },
            ],
            0.4,
            0.9,
            fmt="num",
            ref=0.5,
            ref_label="coin flip",
            xlabel="week the issues were opened",
        )
        recent4 = rp.tail(4)
        m_auc, b_auc = recent4[f"auc_{champion}"].mean(), recent4[f"auc_{BASELINE}"].mean()
        cal = scatter_chart(
            "calibration",
            "Calibration over the last 4 replayed weeks",
            calibration_points(pred[pred["week"].isin(recent4["week"])]),
            "predicted chance of closing within 14 days (%)",
            "actually closed (%)",
        )
        replay_html = (
            f"<p>Each point is one week of issues, scored by a model trained only on outcomes known before that "
            f"week began. Averaged over the last 4 weeks: <b>{m_auc:.3f}</b> for the model vs <b>{b_auc:.3f}</b> "
            f"for the logistic baseline ({int(recent4['n'].sum()):,} issues). 0.5 is a coin flip.</p>"
            + auc_chart
            + "<h3>Are the probabilities honest?</h3><p>The same replayed predictions, grouped into eight "
            "equal-sized bins. On the diagonal, an issue given a 40% chance closes 40% of the time.</p>" + cal
        )
    else:
        replay_html = "<p>Not enough history yet for a replay.</p>"
        m_auc = None
    labels, series = rates
    rates_html = series_chart(
        "close-rates",
        "Share of issues closed within 14 days, by week opened",
        labels,
        series,
        0.0,
        1.0,
        fmt="pct",
        xlabel="week the issues were opened",
    )
    tile_auc = (
        f"<b>{m_auc:.3f}</b>replayed ROC AUC, last 4 weeks"
        if m_auc is not None
        else "<b>n/a</b>replayed ROC AUC"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Issue Triage Watch</title>
<meta name="description" content="New GitHub issues in tensorflow, flutter, angular and golang ranked by predicted chance of stalling, with a weekly out-of-time replay of the model's accuracy.">
<style>
:root {{ --bg:#fbfbfa; --fg:#1d1d1b; --muted:#66655f; --line:#e3e2dc; --accent:#1c5cab; --row:#f3f2ee; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#151513; --fg:#ecebe6; --muted:#a3a29b; --line:#33322e; --accent:#86b6ef; --row:#1e1e1b; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1040px; margin:0 auto; padding:32px 16px 48px; }}
h1 {{ font-size:1.6rem; margin:0 0 4px; letter-spacing:-.01em; }}
h2 {{ font-size:1.1rem; margin:32px 0 8px; }}
h3 {{ font-size:1rem; margin:24px 0 6px; }}
p, li {{ max-width:72ch; }}
.muted {{ color:var(--muted); }}
a {{ color:var(--accent); }}
.stats {{ display:flex; flex-wrap:wrap; gap:12px; margin:20px 0; }}
.stat {{ border:1px solid var(--line); border-radius:8px; padding:10px 14px; min-width:150px; }}
.stat b {{ display:block; font-size:1.3rem; }}
.wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; min-width:720px; font-size:.92rem; }}
th, td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
td.title {{ white-space:normal; min-width:280px; }}
th {{ font-weight:600; color:var(--muted); background:var(--row); position:sticky; top:0; }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
tr:last-child td {{ border-bottom:0; }}
footer {{ margin-top:40px; font-size:.85rem; }}
{CHART_CSS}
</style>
</head>
<body>
<main>
<h1>Issue Triage Watch</h1>
<p class="muted">Which newly opened issues in four large open-source projects are least likely to be closed within
{horizon} days? Predicted at the moment each issue is opened, from what was knowable then. Rebuilt every 6 hours.
Built {html.escape(built)}.</p>

<div class="stats">
  <div class="stat"><b>{len(b):,}</b>open issues opened in the last {horizon} days</div>
  <div class="stat">{tile_auc}</div>
  <div class="stat"><b>{html.escape(card["champion"])}</b>model <code>{html.escape(version)}</code></div>
  <div class="stat"><b>4</b>repos: tensorflow, flutter, angular, golang</div>
</div>

<h2>Most likely to stall</h2>
<p>Still-open issues from the last {horizon} days with the lowest predicted chance of being closed within {horizon}
days of opening. Closed for any reason counts as closed.</p>
<div class="wrap"><table>
<thead><tr><th>Repo</th><th>Issue</th><th>Title</th><th>Opened</th><th class="num">Age (days)</th><th class="num">Chance closed &le;{horizon}d</th></tr></thead>
<tbody>{rows}</tbody></table></div>

<h2>How accurate is it, week by week?</h2>
{replay_html}

<h2>Close rates are not stable</h2>
<p>The share of issues closed within {horizon} days moves from week to week and differs a lot by project, which is
part of why the model's probabilities drift. Weeks with fewer than 15 issues in a repo are left out.</p>
{rates_html}

<h2>How to read this</h2>
<ul>
<li>A triage aid, not a verdict: the model sees only creation-time facts (text statistics, the author's prior issues
in the repo, recent repo activity), never comments, labels or later edits.</li>
<li>"Closed" includes closed as duplicate or not planned, so a fast close is not always a fix.</li>
<li>Accuracy is modest. Details, leakage controls and caveats are in the
<a href="https://github.com/krisrchesney/gh-issues-triage#results">project README</a>.</li>
</ul>

<footer class="muted">
<p>Code: <a href="https://github.com/krisrchesney/gh-issues-triage">github.com/krisrchesney/gh-issues-triage</a>.
Data from the GitHub REST API (public repositories). No usernames or issue bodies are stored. Not affiliated with
GitHub or the projects shown.</p>
</footer>
</main>
<div id="tip" role="tooltip"></div>
<script>{CHART_JS}</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="site")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    horizon = int(cfg["data"].get("label_horizon_days", 14))
    raw, issues = load_issues(cfg)
    b, card, version = board(cfg, issues, horizon)
    table = build_training_frame(
        raw,
        coverage_start=cfg["github"]["coverage_start"],
        warmup_days=int(cfg["data"].get("warmup_days", 30)),
        horizon_days=horizon,
        min_rows=int(cfg["data"].get("min_rows", 200)),
    )
    rp, pred = replay(cfg, table, card["champion"])
    rates = weekly_close_rates(table)
    built = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(
        render(b, card, version, rp, pred, rates, built, args.top, horizon, card["champion"]),
        encoding="utf-8",
    )
    payload = {
        "built": built,
        "model_version": version,
        "champion": card["champion"],
        "board": json.loads(
            b.head(args.top)[["repo", "number", "title", "created_at", "p_close"]]
            .assign(created_at=lambda d: d["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%S"))
            .to_json(orient="records")
        ),
        "replay": json.loads(rp.assign(week=rp["week"].dt.strftime("%Y-%m-%d")).to_json(orient="records"))
        if len(rp)
        else [],
    }
    (out / "data.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {out / 'index.html'}: board {len(b)} open issues; replay weeks {len(rp)}")
    if len(rp):
        print(rp.to_string(index=False))


if __name__ == "__main__":
    main()
