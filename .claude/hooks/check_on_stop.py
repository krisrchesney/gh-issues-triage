"""Stop hook: before Claude ends a turn, run scripts/check.py.

Claude Code sends a JSON event on stdin. Exit code 2 + a message on stderr blocks the stop and
hands that message back to Claude, so it keeps working until the checks pass.
`stop_hook_active` is true when Claude is already continuing because of this hook; we let it
stop then, so a check it can't fix never traps it in a loop (it still sees the failure and has
to report it).
"""

import json
import os
import subprocess
import sys

event = json.load(sys.stdin)
root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()

# Nothing changed -> nothing to check (keeps question-only turns fast).
status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
if not status.stdout.strip():
    sys.exit(0)

result = subprocess.run(
    [sys.executable, os.path.join(root, "scripts", "check.py")],
    cwd=root,
    capture_output=True,
    text=True,
)
if result.returncode == 0:
    sys.exit(0)

if event.get("stop_hook_active"):
    sys.exit(0)

tail = (result.stdout + result.stderr)[-3000:]
print(f"scripts/check.py failed. Fix it before stopping:\n{tail}", file=sys.stderr)
sys.exit(2)
