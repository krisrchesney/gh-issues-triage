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

import numpy as np
import pandas as pd
from site_charts import CHART_CSS, CHART_JS, scatter_chart, series_chart
from sklearn.metrics import roc_auc_score

from ghtriage.config import load_config
from ghtriage.features import (
    TITLE_SOURCE,
    attach_original_titles,
    build_training_frame,
    compute_features,
    normalise,
)
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


def load_issues(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Raw lake rows, normalised in-coverage issues with titles at creation, and the title table."""
    raw = load_lake(cfg["lake"]["root"], "gh_issues")
    titles = load_lake(cfg["lake"]["root"], TITLE_SOURCE)
    df = attach_original_titles(normalise(raw), titles)
    return raw, df[df["created_at"] >= pd.Timestamp(cfg["github"]["coverage_start"])], titles


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
        pipes = build_candidates(
            cfg["model"]["random_state"], [BASELINE, champion], cfg["model"].get("calibration")
        )
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
    trust = 'Check <a href="#does-it-work">Does it work?</a> before relying on it.'
    if len(rp):
        weeks = [f"{w:%b %d}" for w in rp["week"]]
        recent4 = rp.tail(4)
        m_auc, b_auc = recent4[f"auc_{champion}"].mean(), recent4[f"auc_{BASELINE}"].mean()
        diff = m_auc - b_auc
        if diff >= 0.02:
            auc_head = "Week by week, the model beats a simple baseline, modestly"
            trust = 'It currently beats a simpler baseline, but only modestly: see <a href="#does-it-work">Does it work?</a>'
        elif diff <= -0.02:
            auc_head = "Week by week, a simple baseline does better than the model"
            trust = (
                'Right now a simpler baseline is more accurate: see <a href="#does-it-work">Does it work?</a>'
            )
        else:
            auc_head = "Week by week, the model is no better than a simple baseline"
            trust = 'Right now it is about as accurate as a much simpler baseline: see <a href="#does-it-work">Does it work?</a>'
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
            # 0.4-0.8 normally; widen in 0.1 steps if a week ever falls outside it.
            min(0.4, np.floor(rp[[f"auc_{champion}", f"auc_{BASELINE}"]].min().min() * 10) / 10),
            max(0.8, np.ceil(rp[[f"auc_{champion}", f"auc_{BASELINE}"]].max().max() * 10) / 10),
            fmt="num",
            tick_step=0.1,
            ref=0.5,
            ref_label="coin flip",
            xlabel="week the issues were opened",
        )
        pts = calibration_points(pred[pred["week"].isin(recent4["week"])])
        lo, hi = pts[0], pts[-1]
        lo_gap, hi_gap = lo["y"] - lo["x"], hi["x"] - hi["y"]
        if lo_gap > 0.08 and hi_gap > 0.08:
            cal_head = "Its probabilities are too extreme at both ends"
        elif max(abs(p["y"] - p["x"]) for p in pts) <= 0.08:
            cal_head = "Its probabilities roughly match what happens"
        else:
            cal_head = "Its probabilities are off in places"
        cal = scatter_chart(
            "calibration",
            "Calibration over the last 4 replayed weeks",
            pts,
            "predicted chance of closing within 14 days (%)",
            "actually closed (%)",
        )
        replay_html = (
            f"<h3>{auc_head}</h3>"
            "<p>Each point is one week of issues, scored by a model trained only on outcomes known before that week "
            f"began, exactly as it would have run live. Over the last 4 weeks: <b>{m_auc:.3f}</b> for the model vs "
            f"<b>{b_auc:.3f}</b> for a logistic-regression baseline ({int(recent4['n'].sum()):,} issues). ROC AUC is "
            "the chance that a randomly picked closed issue was ranked above a randomly picked stalled one: 0.5 is a "
            "coin flip, 1.0 is perfect.</p>" + auc_chart + f"<h3>{cal_head}</h3>"
            f"<p>The same replayed predictions in eight equal-sized groups. Issues given about {lo['x'] * 100:.0f}% "
            f"closed {lo['y'] * 100:.0f}% of the time; issues given about {hi['x'] * 100:.0f}% closed "
            f"{hi['y'] * 100:.0f}% of the time. On the diagonal, the predicted chance would match reality.</p>"
            + cal
        )
        tile_auc = f"<b>{m_auc:.3f}</b>replayed ROC AUC, last 4 weeks (baseline {b_auc:.3f})"
    else:
        replay_html = "<p>Not enough history yet for a replay.</p>"
        tile_auc = "<b>n/a</b>replayed ROC AUC"
    labels, series = rates
    recent_means = {
        s["name"]: float(np.mean(v)) for s in series if (v := [y for y in s["y"][-4:] if y is not None])
    }
    if recent_means:
        hi_repo = max(recent_means, key=recent_means.get)
        lo_repo = min(recent_means, key=recent_means.get)
        spread = recent_means[hi_repo] - recent_means[lo_repo]
        rates_head = (
            f"Close rates differ widely by project: {hi_repo.split('/')[1]} {recent_means[hi_repo] * 100:.0f}% vs "
            f"{lo_repo.split('/')[1]} {recent_means[lo_repo] * 100:.0f}%"
            if spread >= 0.10
            else "Close rates are similar across projects"
        )
    else:
        rates_head = "Close rates by project"
    rates_html = (
        f"<h3>{rates_head}</h3>"
        f"<p>Share of issues closed within {horizon} days, by the week they were opened (averages above are over the "
        "last 4 weeks). Rates move from week to week too, which is one reason a model's probabilities drift. Weeks "
        "with fewer than 15 issues in a project are left out.</p>"
        + series_chart(
            "close-rates",
            "Share of issues closed within 14 days, by week opened",
            labels,
            series,
            0.0,
            1.0,
            fmt="pct",
            xlabel="week the issues were opened",
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Issue Triage Watch</title>
<meta name="description" content="New GitHub issues in tensorflow, flutter, angular and golang ranked by predicted chance of stalling, with a weekly out-of-time replay of the model's accuracy.">
<style>
:root {{ --bg:#fbfbfa; --fg:#1d1d1b; --muted:#66655f; --line:#e3e2dc; --accent:#1c5cab; --row:#f3f2ee; --card:#f3f2ee; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#151513; --fg:#ecebe6; --muted:#a3a29b; --line:#33322e; --accent:#86b6ef; --row:#1e1e1b; --card:#1e1e1b; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1040px; margin:0 auto; padding:32px 16px 48px; }}
h1 {{ font-size:1.7rem; margin:0 0 6px; letter-spacing:-.01em; }}
h2 {{ font-size:1.25rem; margin:40px 0 8px; padding-top:16px; border-top:1px solid var(--line); }}
h3 {{ font-size:1rem; margin:28px 0 6px; }}
p, li {{ max-width:72ch; }}
.lede {{ font-size:1.05rem; max-width:68ch; }}
.muted {{ color:var(--muted); }}
a {{ color:var(--accent); }}
.guide {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:14px; margin:20px 0; }}
.guide section {{ background:var(--card); border-radius:8px; padding:14px 16px; }}
.guide h3 {{ margin:0 0 6px; }}
.guide ol, .guide ul {{ margin:0; padding-left:20px; }}
.guide li {{ margin:4px 0; }}
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
<p class="lede">Large open-source projects receive hundreds of new issues a week, and some sit unanswered. This page
looks at every issue opened in the last {horizon} days in four big projects (TensorFlow, Flutter, Angular and Go) and
estimates, as of the moment each one was opened, how likely it is to be closed within {horizon} days. The issues least
likely to be closed are listed first, as candidates for a maintainer's or contributor's attention.</p>
<p class="muted">Rebuilt every 6 hours. Built {html.escape(built)}.</p>

<div class="guide">
<section>
<h3>How it works</h3>
<ul>
<li>Every 6 hours, new and updated issues are collected from the GitHub API.</li>
<li>A model trained on earlier issues whose {horizon}-day outcome is known scores each new issue from facts available
when it was opened: title and body length, code blocks, links, stack traces, the author's earlier issues in the
project, and how busy the project was that week.</li>
<li>It never sees comments, labels or later edits, and "closed" means closed for any reason, including as a
duplicate or not planned.</li>
</ul>
</section>
<section>
<h3>How to use it</h3>
<ol>
<li>Open the <a href="#stalled">stalled-issue list</a>: issues are sorted by lowest chance of being closed. Click
one to open it on GitHub.</li>
<li>Use it to spot issues that may need a reply, a label or a reproduction, not as a forecast of what maintainers
will do.</li>
<li>{trust}</li>
</ol>
</section>
<section>
<h3>What it can't do</h3>
<ul>
<li>It does not read what an issue says, only measures of it (lengths and counts).</li>
<li>Its accuracy is modest, and its probabilities are not exact.</li>
<li>It was trained on four large projects and may not transfer to small ones.</li>
</ul>
</section>
</div>

<div class="stats">
  <div class="stat"><b>{len(b):,}</b>open issues opened in the last {horizon} days</div>
  <div class="stat">{tile_auc}</div>
  <div class="stat"><b>{html.escape(card["champion"])}</b>model <code>{html.escape(version)}</code></div>
  <div class="stat"><b>4</b>projects: tensorflow, flutter, angular, golang</div>
</div>

<h2 id="does-it-work">Does it work?</h2>
<p>Every outcome older than {horizon} days is already used for training, so the honest check is a replay: for each
recent week, train only on what was known before it started, then score that week's issues.</p>
{replay_html}
{rates_html}

<h2 id="data">The data</h2>
<h3 id="stalled">Most likely to stall: top {top} open issues from the last {horizon} days</h3>
<div class="wrap"><table>
<thead><tr><th>Project</th><th>Issue</th><th>Title</th><th>Opened</th><th class="num">Age (days)</th><th class="num">Chance closed &le;{horizon}d</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class="muted">Chance closed: the model's estimate, made when the issue was opened, that it would be closed within
{horizon} days. Titles are as first seen; no usernames or issue bodies are stored.</p>

<p><a href="data.json">Download this page's data (JSON)</a> · <a href="https://github.com/krisrchesney/gh-issues-triage#results">Full results and caveats in the README</a></p>

<footer class="muted">
<p>Code: <a href="https://github.com/krisrchesney/gh-issues-triage">github.com/krisrchesney/gh-issues-triage</a>.
Data from the GitHub REST API (public repositories). Not affiliated with GitHub or the projects shown.</p>
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
    raw, issues, titles = load_issues(cfg)
    b, card, version = board(cfg, issues, horizon)
    table = build_training_frame(
        raw,
        coverage_start=cfg["github"]["coverage_start"],
        warmup_days=int(cfg["data"].get("warmup_days", 30)),
        horizon_days=horizon,
        min_rows=int(cfg["data"].get("min_rows", 200)),
        titles=titles,
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
