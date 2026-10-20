"""Synthetic raw lake rows shaped like the real collector output (used only in tests)."""

import numpy as np
import pandas as pd


def make_raw_issues(
    n_per_repo: int = 500, days: int = 150, seed: int = 0, start: str = "2026-06-01"
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    end_ts = start_ts + pd.Timedelta(days=days)
    rows = []
    number = 0
    for r, repo in enumerate(["org/alpha", "org/beta"]):
        for off in np.sort(rng.uniform(0, days * 86400, n_per_repo)):
            number += 1
            created = start_ts + pd.Timedelta(seconds=float(off))
            maint = rng.random() < 0.2
            bot = rng.random() < 0.05
            body_len = int(rng.lognormal(6.0, 1.0))
            trace = rng.random() < 0.3
            p = 0.22 + 0.35 * maint + 0.2 * bot + 0.1 * trace - 0.1 * min(body_len / 3000, 1) + 0.05 * r
            fast = rng.random() < float(np.clip(p, 0.02, 0.95))
            if fast:
                ttc_days = rng.uniform(0.05, 13.5)
            else:
                ttc_days = rng.uniform(15, 90) if rng.random() < 0.4 else np.inf
            closed = created + pd.Timedelta(days=float(ttc_days)) if np.isfinite(ttc_days) else pd.NaT
            if pd.notna(closed) and closed > end_ts:
                closed = pd.NaT
            if pd.notna(closed):
                updated = closed
            else:
                updated = min(end_ts, created + pd.Timedelta(days=float(rng.uniform(0, 30))))
            assoc = "MEMBER" if maint else ("FIRST_TIME_CONTRIBUTOR" if rng.random() < 0.3 else "NONE")
            rows.append(
                {
                    "repo": repo,
                    "number": number,
                    "issue_id": number + 10**6,
                    "created_at": created,
                    "updated_at": updated,
                    "closed_at": closed,
                    "state": "closed" if pd.notna(closed) else "open",
                    "state_reason": "completed" if pd.notna(closed) else None,
                    "comments": int(rng.integers(0, 8)),
                    "labels": "",
                    "title": f"issue {number}",
                    "author_key": f"author{int(rng.integers(0, 60))}",
                    "is_bot": bot,
                    "author_association": assoc,
                    "title_len": int(rng.integers(10, 80)),
                    "title_words": int(rng.integers(2, 14)),
                    "body_len": body_len,
                    "body_words": body_len // 6,
                    "code_blocks": int(rng.integers(0, 3)),
                    "has_stack_trace": int(trace),
                    "n_links": int(rng.integers(0, 4)),
                    "n_images": int(rng.integers(0, 2)),
                    "has_version": int(rng.random() < 0.6),
                    "n_questions": int(rng.integers(0, 3)),
                    "n_checkboxes": int(rng.integers(0, 4)),
                    "n_headings": int(rng.integers(0, 5)),
                }
            )
    return pd.DataFrame(rows)
