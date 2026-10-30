"""One command that answers "is the repo healthy?" - used by people, agents and the Stop hook.

    python scripts/check.py

Runs ruff and the pytest suite with the repo's .venv. Exits non-zero on any failure, using the
tools' own exit codes. One local exception: tests/test_data_sync.py needs `git worktree add
--quiet` (git 2.17+); on an older local git those tests are skipped here with a loud warning.
CI runs them on every push.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = "Scripts" if os.name == "nt" else "bin"
VENV_PY = os.path.join(ROOT, ".venv", BIN, "python.exe" if os.name == "nt" else "python")


def main() -> int:
    py = VENV_PY if os.path.exists(VENV_PY) else sys.executable
    failed = False

    lint = subprocess.run([py, "-m", "ruff", "check", "."], cwd=ROOT)
    if lint.returncode != 0:
        print("check: ruff found problems")
        failed = True

    args = [py, "-m", "pytest", "-q", "-p", "no:warnings"]
    worktree_help = subprocess.run(["git", "worktree", "add", "-h"], cwd=ROOT, capture_output=True, text=True)
    if "--quiet" not in (worktree_help.stdout + worktree_help.stderr):
        print("WARNING: local git is too old for tests/test_data_sync.py (needs `git worktree add --quiet`);")
        print("         skipping them here. CI runs them on every push.")
        print("         Upgrading Git for Windows removes this skip.")
        args += ["--deselect", "tests/test_data_sync.py"]
    tests = subprocess.run(args, cwd=ROOT)
    if tests.returncode != 0:
        print("check: tests failed")
        failed = True

    print("check: FAILED" if failed else "check: OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
