"""GitHub issues collector (REST API) for the Parquet lake.

What it stores per issue (pull requests are skipped):
  * identifiers and timestamps: repo, number, created_at, updated_at, closed_at, state, state_reason
  * "as first seen" snapshot columns that never change after the first insert, so features cannot
    absorb later edits: title, author_key, is_bot, author_association and text statistics
  * the latest comment count and label names, kept for analysis only (they change after creation, so
    they are deliberately NOT model features)

What it does not store: issue bodies. Text is reduced to counts (length, code blocks, links...) at
ingest. That keeps the lake small and avoids copying other people's text around. Author ids are salted
and hashed, so the lake holds a stable pseudonym rather than a GitHub account.

Issues created before `coverage_start` are not stored: the dataset builder would drop them anyway
(they are a survivor-biased sample), so keeping them only grows the repository. Because skipped issues
leave no rows behind, progress is also recorded in a small lake source, `gh_state` (newest `updated_at`
processed per repo), written after the rows are stored. Without it a page holding only old issues would
leave the cursor where it was and a capped run would fetch the same pages forever.

Resumable: the cursor per repository is the newer of the newest stored `updated_at` and the `gh_state`
progress marker. Issues are fetched oldest-update-first, so a run cut short by the rate limit or
`max_pages_per_repo` resumes where it stopped.

Known limitation: list endpoints are paginated by position. If an already-passed issue is updated
while a run is mid-flight it moves to the end of the list and one issue at a page boundary can be
skipped. Missed issues are essentially random with respect to the outcome, so the bias is negligible,
but expect a few unfetched issues per large backfill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from ghtriage.config import load_config
from ghtriage.lake import lake_cursors, upsert_partitioned

API_ROOT = "https://api.github.com"
MAX_ATTEMPTS = 3

_URL = re.compile(r"https?://\S+")
_IMG = re.compile(r"!\[[^\]]*\]\(|<img\b", re.I)
_VERSION = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b")
_CHECKBOX = re.compile(r"^\s*[-*] \[[ xX]\]", re.M)
_HEADING = re.compile(r"^#{1,6} ", re.M)
_STACK = re.compile(
    r"Traceback \(most recent call last\)|Exception in thread|\bpanic:|Segmentation fault"
    r"|\bat [\w.$<>]+\([\w.]+:\d+\)|Unhandled exception",
    re.I,
)

TEXT_COLS = [
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


class GitHubError(RuntimeError):
    """A non-recoverable API problem (bad token, unknown repository, blocked access)."""


class RateLimited(Exception):
    """The API told us to stop. Not an error: whatever was fetched so far is still stored."""


def text_stats(title: str | None, body: str | None) -> dict[str, int]:
    """Reduce issue text to counts. The text itself is never stored."""
    title = title or ""
    body = body or ""
    return {
        "title_len": len(title),
        "title_words": len(title.split()),
        "body_len": len(body),
        "body_words": len(body.split()),
        "code_blocks": body.count("```") // 2,
        "has_stack_trace": int(bool(_STACK.search(body))),
        "n_links": len(_URL.findall(body)),
        "n_images": len(_IMG.findall(body)),
        "has_version": int(bool(_VERSION.search(f"{title}\n{body}"))),
        "n_questions": title.count("?") + body.count("?"),
        "n_checkboxes": len(_CHECKBOX.findall(body)),
        "n_headings": len(_HEADING.findall(body)),
    }


def author_key(user_id: int | None, salt: str) -> str | None:
    if user_id is None:
        return None
    return hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:16]


def normalise_issue(issue: dict[str, Any], repo: str, salt: str) -> dict[str, Any]:
    user = issue.get("user") or {}
    return {
        "repo": repo,
        "number": int(issue["number"]),
        "issue_id": int(issue["id"]),
        "created_at": issue["created_at"],
        "updated_at": issue["updated_at"],
        "closed_at": issue.get("closed_at"),
        "state": issue.get("state"),
        "state_reason": issue.get("state_reason"),
        "comments": int(issue.get("comments") or 0),
        "labels": "|".join(sorted(str(lb.get("name", "")) for lb in issue.get("labels") or [])),
        "title": (issue.get("title") or "")[:200],
        "author_key": author_key(user.get("id"), salt),
        "is_bot": bool(user.get("type") == "Bot" or str(user.get("login", "")).endswith("[bot]")),
        "author_association": issue.get("author_association"),
        **text_stats(issue.get("title"), issue.get("body")),
    }


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_localize(None).astype("datetime64[ns]")


class GitHubIssuesCollector:
    name = "gh_issues"
    key_cols = ["repo", "number"]
    partition_col = "created_at"
    version_col = "updated_at"
    keep_first_cols = [
        "title",
        "author_key",
        "is_bot",
        "author_association",
        *TEXT_COLS,
    ]

    def __init__(
        self,
        cfg: dict[str, Any],
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cfg = cfg
        gh = cfg["github"]
        self.repos: list[str] = list(gh["repos"])
        self.coverage_start = pd.Timestamp(gh["coverage_start"])
        self.per_page = int(gh.get("per_page", 100))
        self.max_pages = int(gh.get("max_pages_per_repo", 30))
        self.overlap = timedelta(minutes=int(gh.get("overlap_minutes", 60)))
        self.min_remaining = int(gh.get("min_rate_limit_remaining", 50))
        self.api_version = gh.get("api_version")
        self.salt = os.environ.get(gh.get("author_salt_env", "GH_AUTHOR_SALT"), "ghtriage-default-salt")
        self.token = os.environ.get(gh.get("token_env", "GITHUB_TOKEN"))
        self._transport = transport
        self._sleep = sleep
        self.pages_fetched = 0
        self.stopped_early = False
        self.skipped_before_coverage = 0
        self._progress: dict[str, pd.Timestamp] = {}

    # ---- HTTP -------------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "ghtriage-portfolio-collector"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.api_version:
            headers["X-GitHub-Api-Version"] = str(self.api_version)
        return headers

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=API_ROOT, headers=self._headers(), timeout=30.0, transport=self._transport
        )

    def _get(self, client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        resp: httpx.Response | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = client.get(url, params=params)
            except httpx.TransportError as exc:
                if attempt == MAX_ATTEMPTS:
                    raise GitHubError(f"Network error talking to GitHub: {exc}") from exc
                self._sleep(2**attempt)
                continue
            if resp.status_code >= 500 and attempt < MAX_ATTEMPTS:
                self._sleep(2**attempt)
                continue
            break
        assert resp is not None
        status = resp.status_code
        if status == 200:
            return resp
        text = resp.text[:300]
        remaining = resp.headers.get("x-ratelimit-remaining")
        if status in (403, 429) and (
            remaining == "0" or "retry-after" in resp.headers or "rate limit" in text.lower()
        ):
            raise RateLimited(text)
        if status == 401:
            raise GitHubError("GitHub rejected the token (401). Check GITHUB_TOKEN.")
        if status == 404:
            raise GitHubError(
                f"Not found (404): {url}. Check the repository name; private repos need a token."
            )
        raise GitHubError(f"GitHub returned {status} for {url}: {text}")

    @staticmethod
    def _next_url(resp: httpx.Response) -> str | None:
        link = resp.links.get("next")
        return link["url"] if link else None

    # ---- collection -------------------------------------------------------------------------

    def _since_for(self, cursor: pd.Timestamp | None) -> datetime:
        if cursor is None:
            return self.coverage_start.to_pydatetime().replace(tzinfo=UTC)
        since = cursor.to_pydatetime().replace(tzinfo=UTC) - self.overlap
        floor = self.coverage_start.to_pydatetime().replace(tzinfo=UTC)
        return max(since, floor)

    def _fetch_repo(
        self, client: httpx.Client, repo: str, since: datetime, out: list[dict[str, Any]]
    ) -> None:
        params: dict[str, Any] | None = {
            "state": "all",
            "sort": "updated",
            "direction": "asc",
            "per_page": self.per_page,
            "since": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        url: str | None = f"/repos/{repo}/issues"
        pages = 0
        while url and pages < self.max_pages:
            resp = self._get(client, url, params)
            params = None  # the "next" link already carries every parameter
            pages += 1
            self.pages_fetched += 1
            for item in resp.json():
                updated = pd.Timestamp(item["updated_at"]).tz_convert("UTC").tz_localize(None)
                if repo not in self._progress or updated > self._progress[repo]:
                    self._progress[repo] = updated
                if "pull_request" in item:
                    continue
                created = pd.Timestamp(item["created_at"]).tz_convert("UTC").tz_localize(None)
                if created < self.coverage_start:
                    self.skipped_before_coverage += 1
                    continue
                out.append(normalise_issue(item, repo, self.salt))
            remaining = resp.headers.get("x-ratelimit-remaining")
            if remaining is not None and int(remaining) < self.min_remaining:
                self.stopped_early = True
                raise RateLimited(f"only {remaining} requests left in this window")
            url = self._next_url(resp)

    def commit(self) -> None:
        """Runs after the rows are stored: record how far this run got, per repository."""
        if not self._progress:
            return
        state = pd.DataFrame(
            [{"repo": r, "updated_at": ts.as_unit("ns")} for r, ts in sorted(self._progress.items())]
        )
        upsert_partitioned(
            self.cfg["lake"]["root"],
            "gh_state",
            state,
            key_cols=["repo", "updated_at"],
            partition_col="updated_at",
        )

    def fetch(self) -> pd.DataFrame:
        root = self.cfg["lake"]["root"]
        cursors = lake_cursors(root, self.name, "updated_at", "repo")
        for repo_name, marker in lake_cursors(root, "gh_state", "updated_at", "repo").items():
            if repo_name not in cursors or marker > cursors[repo_name]:
                cursors[repo_name] = marker
        rows: list[dict[str, Any]] = []
        with self._client() as client:
            for repo in self.repos:
                since = self._since_for(cursors.get(repo))
                try:
                    self._fetch_repo(client, repo, since, rows)
                except RateLimited as exc:
                    print(f"[gh_issues] stopping early on {repo}: {exc}", file=sys.stderr)
                    self.stopped_early = True
                    break
        if not self.token:
            print(
                "[gh_issues] no GITHUB_TOKEN set: unauthenticated limit is 60 requests/hour", file=sys.stderr
            )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ("created_at", "updated_at", "closed_at"):
            df[col] = _to_utc(df[col])
        return df


def check_repos(cfg: dict[str, Any], transport: httpx.BaseTransport | None = None) -> list[dict[str, Any]]:
    """Confirm each configured repo exists and is active, and size the backfill (2 requests per repo)."""
    collector = GitHubIssuesCollector(cfg, transport=transport)
    since = (datetime.now(UTC) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    report: list[dict[str, Any]] = []
    with collector._client() as client:
        for repo in collector.repos:
            info = collector._get(client, f"/repos/{repo}").json()
            recent = collector._get(
                client,
                f"/repos/{repo}/issues",
                {"state": "all", "per_page": 100, "since": since, "sort": "updated", "direction": "desc"},
            ).json()
            prs = sum(1 for x in recent if "pull_request" in x)
            report.append(
                {
                    "repo": repo,
                    "archived": info.get("archived"),
                    "pushed_at": info.get("pushed_at"),
                    "updated_last_24h_issues": len(recent) - prs,
                    "updated_last_24h_prs": prs,
                    "page_was_full": len(recent) == 100,
                }
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ghtriage.github_issues")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-repos", help="Verify the configured repos exist and show recent activity")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "check-repos":
        print(json.dumps(check_repos(cfg), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
