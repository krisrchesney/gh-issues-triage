import re
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd
import pytest

from ghtriage import collect
from ghtriage.github_issues import GitHubError, GitHubIssuesCollector, check_repos, text_stats
from ghtriage.lake import load_lake


def gh_issue(
    number,
    updated,
    created="2026-07-02T10:00:00Z",
    closed=None,
    pr=False,
    body="",
    login="alice",
    uid=111,
    utype="User",
    assoc="NONE",
    title="Something broke",
    labels=(),
):
    issue = {
        "id": 10_000 + number,
        "number": number,
        "title": title,
        "body": body,
        "state": "closed" if closed else "open",
        "state_reason": "completed" if closed else None,
        "created_at": created,
        "updated_at": updated,
        "closed_at": closed,
        "comments": 2,
        "labels": [{"name": n} for n in labels],
        "user": {"login": login, "id": uid, "type": utype},
        "author_association": assoc,
    }
    if pr:
        issue["pull_request"] = {"url": "x"}
    return issue


class FakeGitHub:
    def __init__(self, issues, remaining=4000, status_for=None):
        self.issues = issues  # {"org/repo": [issue, ...]}
        self.remaining = remaining
        self.status_for = status_for or {}  # {"org/repo": [status, ...]} consumed in order
        self.requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.remaining -= 1
        headers = {"x-ratelimit-remaining": str(self.remaining)}
        info = re.fullmatch(r"/repos/([^/]+/[^/]+)", request.url.path)
        if info:
            return httpx.Response(
                200, json={"archived": False, "pushed_at": "2026-10-18T00:00:00Z"}, headers=headers
            )
        m = re.fullmatch(r"/repos/([^/]+/[^/]+)/issues", request.url.path)
        assert m, request.url
        repo = m.group(1)
        queue = self.status_for.get(repo)
        if queue:
            status = queue.pop(0)
            if status != 200:
                extra = {"x-ratelimit-remaining": "0"} if status == 403 else {}
                return httpx.Response(
                    status, json={"message": f"status {status}"}, headers={**headers, **extra}
                )
        if repo not in self.issues:
            return httpx.Response(404, json={"message": "Not Found"}, headers=headers)
        params = request.url.params
        since = params.get("since", "0000")
        per_page = int(params.get("per_page", 30))
        page = int(params.get("page", 1))
        items = sorted(
            (i for i in self.issues[repo] if i["updated_at"] >= since), key=lambda i: i["updated_at"]
        )
        chunk = items[(page - 1) * per_page : page * per_page]
        if page * per_page < len(items):
            nxt = request.url.copy_merge_params({"page": page + 1})
            headers["link"] = f'<{nxt}>; rel="next"'
        return httpx.Response(200, json=chunk, headers=headers)


@pytest.fixture
def gh_cfg(cfg):
    cfg["github"] = {
        "repos": ["org/alpha"],
        "coverage_start": "2026-07-01",
        "per_page": 2,
        "max_pages_per_repo": 10,
        "overlap_minutes": 60,
        "min_rate_limit_remaining": 5,
        "api_version": None,
        "token_env": "TEST_GH_TOKEN",
        "author_salt_env": "TEST_GH_SALT",
    }
    return cfg


def make(gh_cfg, server):
    return GitHubIssuesCollector(gh_cfg, transport=httpx.MockTransport(server.handler), sleep=lambda s: None)


def test_normalises_issues_skips_prs_and_drops_text(gh_cfg, monkeypatch):
    monkeypatch.setenv("TEST_GH_SALT", "pepper")
    body = "Steps:\n```\ncode\n```\nsee https://example.com and https://x.dev\nVersion 3.2.1 ?\n- [ ] done"
    server = FakeGitHub(
        {
            "org/alpha": [
                gh_issue(1, "2026-07-03T00:00:00Z", body=body, assoc="MEMBER", labels=("bug", "area/ui")),
                gh_issue(2, "2026-07-04T00:00:00Z", pr=True),
                gh_issue(3, "2026-07-05T00:00:00Z", utype="Bot", login="dependabot[bot]", uid=222),
            ]
        }
    )
    df = make(gh_cfg, server).fetch()
    assert sorted(df["number"]) == [1, 3]
    row = df[df["number"] == 1].iloc[0]
    assert row["code_blocks"] == 1 and row["n_links"] == 2 and row["has_version"] == 1
    assert row["n_checkboxes"] == 1 and row["n_questions"] == 1
    assert row["labels"] == "area/ui|bug"
    assert "body" not in df.columns
    assert "alice" not in df.to_csv() and "111" not in set(df["author_key"].astype(str))
    assert len(row["author_key"]) == 16
    assert bool(df[df["number"] == 3].iloc[0]["is_bot"]) is True
    assert str(df["created_at"].dtype).startswith("datetime64")


def test_author_hash_depends_on_the_salt(gh_cfg, monkeypatch):
    server = FakeGitHub({"org/alpha": [gh_issue(1, "2026-07-03T00:00:00Z")]})
    monkeypatch.setenv("TEST_GH_SALT", "one")
    a = make(gh_cfg, server).fetch()["author_key"].iloc[0]
    monkeypatch.setenv("TEST_GH_SALT", "two")
    b = make(gh_cfg, server).fetch()["author_key"].iloc[0]
    assert a != b


def test_follows_pagination_links(gh_cfg):
    issues = [gh_issue(n, f"2026-07-{n + 2:02d}T00:00:00Z") for n in range(1, 6)]
    server = FakeGitHub({"org/alpha": issues})
    df = make(gh_cfg, server).fetch()
    assert len(df) == 5
    assert len(server.requests) == 3  # 2 + 2 + 1 items at per_page=2


def test_max_pages_caps_a_run_and_the_next_run_resumes(gh_cfg):
    gh_cfg["github"]["max_pages_per_repo"] = 2
    issues = [
        gh_issue(n, f"2026-07-{n + 2:02d}T00:00:00Z", created=f"2026-07-{n + 1:02d}T00:00:00Z")
        for n in range(1, 8)
    ]
    server = FakeGitHub({"org/alpha": issues})
    collect.COLLECTORS["gh_issues"] = lambda c: GitHubIssuesCollector(
        c, transport=httpx.MockTransport(server.handler), sleep=lambda s: None
    )
    try:
        first = collect.run_collector("gh_issues", gh_cfg)
        second = collect.run_collector("gh_issues", gh_cfg)
        third = collect.run_collector("gh_issues", gh_cfg)
    finally:
        collect.COLLECTORS["gh_issues"] = GitHubIssuesCollector
    assert first == (4, 0)
    assert sum(r[0] for r in (first, second, third)) == 7
    assert sorted(load_lake(gh_cfg["lake"]["root"], "gh_issues")["number"]) == list(range(1, 8))


def test_first_run_uses_coverage_start_and_later_runs_use_the_cursor_with_overlap(gh_cfg):
    server = FakeGitHub({"org/alpha": [gh_issue(1, "2026-07-10T12:00:00Z")]})
    collect.COLLECTORS["gh_issues"] = lambda c: GitHubIssuesCollector(
        c, transport=httpx.MockTransport(server.handler), sleep=lambda s: None
    )
    try:
        collect.run_collector("gh_issues", gh_cfg)
        collect.run_collector("gh_issues", gh_cfg)
    finally:
        collect.COLLECTORS["gh_issues"] = GitHubIssuesCollector
    assert server.requests[0].url.params["since"] == "2026-07-01T00:00:00Z"
    assert server.requests[1].url.params["since"] == "2026-07-10T11:00:00Z"


def test_since_never_goes_before_coverage_start(gh_cfg):
    c = GitHubIssuesCollector(gh_cfg)
    since = c._since_for(pd.Timestamp("2026-07-01 00:20:00"))
    assert since.strftime("%Y-%m-%dT%H:%M:%SZ") == "2026-07-01T00:00:00Z"


def test_updates_replace_rows_but_keep_the_first_seen_snapshot(gh_cfg):
    server = FakeGitHub(
        {"org/alpha": [gh_issue(1, "2026-07-03T00:00:00Z", title="Original title", body="x" * 50)]}
    )
    collect.COLLECTORS["gh_issues"] = lambda c: GitHubIssuesCollector(
        c, transport=httpx.MockTransport(server.handler), sleep=lambda s: None
    )
    try:
        assert collect.run_collector("gh_issues", gh_cfg) == (1, 0)
        server.issues["org/alpha"] = [
            gh_issue(
                1, "2026-07-09T00:00:00Z", closed="2026-07-09T00:00:00Z", title="Edited title", body="y" * 400
            )
        ]
        assert collect.run_collector("gh_issues", gh_cfg) == (0, 1)
        assert collect.run_collector("gh_issues", gh_cfg) == (0, 0)  # re-run is a no-op
    finally:
        collect.COLLECTORS["gh_issues"] = GitHubIssuesCollector
    row = load_lake(gh_cfg["lake"]["root"], "gh_issues").iloc[0]
    assert row["state"] == "closed" and pd.notna(row["closed_at"])
    assert row["title"] == "Original title" and row["body_len"] == 50


def test_low_rate_limit_stops_cleanly_and_keeps_what_was_fetched(gh_cfg):
    gh_cfg["github"]["min_rate_limit_remaining"] = 100
    server = FakeGitHub(
        {"org/alpha": [gh_issue(n, f"2026-07-{n + 2:02d}T00:00:00Z") for n in range(1, 6)]}, remaining=100
    )
    c = make(gh_cfg, server)
    df = c.fetch()
    assert len(df) == 2 and c.stopped_early  # the first page was kept, then it stopped
    assert len(server.requests) == 1


def test_rate_limit_response_is_not_an_error(gh_cfg):
    server = FakeGitHub({"org/alpha": []}, status_for={"org/alpha": [403]})
    c = make(gh_cfg, server)
    assert c.fetch().empty and c.stopped_early


@pytest.mark.parametrize("status,message", [(404, "Not found"), (401, "token"), (418, "418")])
def test_hard_errors_raise_with_a_useful_message(gh_cfg, status, message):
    issues = {} if status == 404 else {"org/alpha": []}
    server = FakeGitHub(issues, status_for={"org/alpha": [status]})
    with pytest.raises(GitHubError, match=message):
        make(gh_cfg, server).fetch()


def test_blocked_access_403_is_a_hard_error_not_a_silent_stop(gh_cfg):
    def handler(request):
        return httpx.Response(403, json={"message": "Resource not accessible by integration"})

    c = GitHubIssuesCollector(gh_cfg, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    with pytest.raises(GitHubError, match="403"):
        c.fetch()


def test_server_errors_are_retried_then_succeed(gh_cfg):
    server = FakeGitHub(
        {"org/alpha": [gh_issue(1, "2026-07-03T00:00:00Z")]}, status_for={"org/alpha": [502, 502]}
    )
    slept = []
    c = GitHubIssuesCollector(gh_cfg, transport=httpx.MockTransport(server.handler), sleep=slept.append)
    assert len(c.fetch()) == 1
    assert slept == [2, 4]


def test_persistent_server_errors_give_up(gh_cfg):
    server = FakeGitHub({"org/alpha": []}, status_for={"org/alpha": [500, 500, 500]})
    with pytest.raises(GitHubError, match="500"):
        make(gh_cfg, server).fetch()


def test_token_is_sent_only_when_configured(gh_cfg, monkeypatch):
    server = FakeGitHub({"org/alpha": []})
    monkeypatch.delenv("TEST_GH_TOKEN", raising=False)
    make(gh_cfg, server).fetch()
    assert "authorization" not in server.requests[0].headers
    monkeypatch.setenv("TEST_GH_TOKEN", "abc123")
    make(gh_cfg, server).fetch()
    assert server.requests[-1].headers["authorization"] == "Bearer abc123"
    assert server.requests[-1].headers["user-agent"]


def test_text_stats_cases():
    assert text_stats(None, None)["body_len"] == 0
    s = text_stats(
        "Crash on start?", "Traceback (most recent call last):\n  boom\n![img](a.png)\n## Env\n1.2"
    )
    assert s["has_stack_trace"] == 1 and s["n_images"] == 1 and s["n_headings"] == 1
    assert s["has_version"] == 1 and s["n_questions"] == 1 and s["title_words"] == 3


def test_check_repos_reports_activity(gh_cfg):
    recent = (datetime.now(UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    server = FakeGitHub(
        {"org/alpha": [gh_issue(1, recent), gh_issue(2, recent, pr=True), gh_issue(3, recent)]}
    )
    report = check_repos(gh_cfg, transport=httpx.MockTransport(server.handler))
    assert report == [
        {
            "repo": "org/alpha",
            "archived": False,
            "pushed_at": "2026-10-18T00:00:00Z",
            "updated_last_24h_issues": 2,
            "updated_last_24h_prs": 1,
            "page_was_full": False,
        }
    ]
    assert len(server.requests) == 2
