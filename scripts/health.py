"""Is the live system healthy? One command for people, agents and the daily Command Center job.

    python scripts/health.py           # human summary, exit 1 if any check fails
    python scripts/health.py --json    # machine-readable result

Checks (GitHub REST, read-only): the latest collect run, CI on main, the site's Pages run and the
page's own "Built" time, how fresh the `data` branch is, and project-specific invariants. Uses a
token from GH_TOKEN or the GitHub CLI when available, else the public 60-requests/hour limit.
Standard library only, so it runs anywhere.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime

REPO = "krisrchesney/gh-issues-triage"
# Hours a signal may age before it counts as stale (cadence plus GitHub's usual schedule delay).
EXPECT = {
    "cve-kev-predictor": {
        "collect": 30,
        "data": 30,
        "pages": 30,
        "site": "https://krisrchesney.github.io/cve-kev-predictor/",
    },
    "gh-issues-triage": {
        "collect": 8,
        "data": 8,
        "pages": 8,
        "site": "https://krisrchesney.github.io/gh-issues-triage/",
        "data_paths": ["raw/gh_titles"],
    },
    "park-wait-forecast": {"collect": 8, "data": 8},
}[REPO.split("/")[1]]
GH_CLI = r"C:\Users\Kao\tools\gh\bin\gh.exe"


def token() -> str | None:
    if os.environ.get("GH_TOKEN"):
        return os.environ["GH_TOKEN"]
    exe = GH_CLI if os.path.exists(GH_CLI) else shutil.which("gh")
    if exe:
        out = subprocess.run([exe, "auth", "token"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return None


TOKEN = token()


def get(url: str):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "health-check"}
    req = urllib.request.Request(url, headers=headers)
    if TOKEN and url.startswith("https://api.github.com"):
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8")
    return json.loads(body) if url.startswith("https://api.github.com") else body


def hours_since(iso: str) -> float:
    return (datetime.now(UTC) - datetime.fromisoformat(iso.replace("Z", "+00:00"))).total_seconds() / 3600


def latest_run(workflow: str, branch: str | None = None) -> dict | None:
    q = f"https://api.github.com/repos/{REPO}/actions/workflows/{workflow}.yml/runs?per_page=5"
    if branch:
        q += f"&branch={branch}"
    runs = get(q)["workflow_runs"]
    return runs[0] if runs else None


def last_completed(workflow: str) -> dict | None:
    runs = get(
        f"https://api.github.com/repos/{REPO}/actions/workflows/{workflow}.yml/runs?per_page=10&status=completed"
    )
    return runs["workflow_runs"][0] if runs["workflow_runs"] else None


def main() -> int:
    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    def guarded(name: str, fn) -> None:
        try:
            fn()
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as exc:
            add(name, False, f"could not check: {exc}")

    def collect():
        run = last_completed("collect")
        if not run:
            return add("collect", False, "no completed collect run")
        age = hours_since(run["updated_at"])
        add(
            "collect",
            run["conclusion"] == "success" and age <= EXPECT["collect"],
            f"last completed run {run['conclusion']}, {age:.1f} h ago (limit {EXPECT['collect']} h)",
        )

    def ci():
        run = latest_run("ci", "main")
        if not run:
            return add("ci", False, "no CI run on main")
        add(
            "ci",
            run["status"] != "completed" or run["conclusion"] == "success",
            f"latest CI on main: {run['status']} {run['conclusion'] or ''} ({run['head_sha'][:7]})".strip(),
        )

    def data_branch():
        commit = get(f"https://api.github.com/repos/{REPO}/commits?sha=data&per_page=1")[0]
        age = hours_since(commit["commit"]["committer"]["date"])
        limit = EXPECT["data"]
        add("data branch", age <= limit, f"last data commit {age:.1f} h ago (limit {limit} h)")
        for path in EXPECT.get("data_paths", []):
            try:
                get(f"https://api.github.com/repos/{REPO}/contents/{path}?ref=data")
                add(f"data: {path}", True, "present on the data branch")
            except urllib.error.HTTPError:
                add(f"data: {path}", False, "missing from the data branch (a collect may have dropped it)")

    def pages():
        run = last_completed("pages")
        if run:
            add("pages run", run["conclusion"] == "success", f"last completed pages run {run['conclusion']}")
        html = get(EXPECT["site"] + f"?health={int(datetime.now(UTC).timestamp())}")
        m = re.search(r"Built (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC", html)
        if not m:
            return add("site", False, "page has no 'Built' time")
        age = hours_since(m.group(1).replace(" ", "T") + ":00+00:00")
        add("site", age <= EXPECT["pages"], f"page built {age:.1f} h ago (limit {EXPECT['pages']} h)")

    guarded("collect", collect)
    guarded("ci", ci)
    guarded("data branch", data_branch)
    if "site" in EXPECT:
        guarded("site", pages)

    result = {
        "repo": REPO,
        "checkedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
    }
    if "--json" in sys.argv:
        print(json.dumps(result, indent=1))
    else:
        for c in checks:
            print(f"{'OK  ' if c['ok'] else 'FAIL'} {c['name']}: {c['detail']}")
        print("health: OK" if result["ok"] else "health: FAILED")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
