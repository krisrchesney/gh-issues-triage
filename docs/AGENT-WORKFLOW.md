# How the agent workflow works

The same setup as the Boxwatch repo, with one addition for an ML project: a results auditor,
because this project's main promise is that every published number is true. Every piece is a
plain file you can read and change.

## The loop

```
 you: "I want X"
   |
   v
 planner agent --writes--> TASKS.md (tickets with "Owns" + "Done when")
   |
   v
 builder sessions, one per ticket, each in its own worktree/branch
   |   (optional: test-writer agent writes the tests first)
   |   Stop hook runs scripts/check.py; a builder can't say "done" while it fails
   v
 reviewer agent (read-only) --> APPROVE / CHANGES NEEDED (+ "needs results-auditor")
   |
   v
 results-auditor agent (read-only, only if a number changed) --> CLEAN / FIX NEEDED
   |
   v
 you merge the branch to main
```

You stay the prioritizer and the person who merges. The agents do the work in between.

## The pieces

| File | What it is | Why |
|---|---|---|
| `CLAUDE.md` | Loaded into every session and agent. | Rules in one place: honesty rules for numbers, leakage rules, data-branch safety, commit identity. |
| `TASKS.md` | The shared backlog. | Sessions coordinate through this file and git. "Owns" stops two builders editing the same code. |
| `.claude/agents/planner.md` | Idea -> tickets. Sonnet. Can't touch project code. | Splitting work so it can run in parallel; flags tickets that change published numbers. |
| `.claude/agents/test-writer.md` | Writes tests from "Done when". Edits only `tests/`. | Tests from the spec catch the builder's misreadings; favours no-leakage and time-order tests. |
| `.claude/agents/reviewer.md` | Read-only review of one branch. | The author is the worst reviewer of their own code; checks for leakage and weakened tests. |
| `.claude/agents/results-auditor.md` | Read-only re-run of every published claim. | A number that no run produces, or a "win" inside the noise, is caught before it ships. |
| `scripts/check.py` | ruff + pytest with the repo's `.venv`, judged by their exit codes. | One definition of "the code is healthy". |
| `scripts/health.py` | Read-only GitHub checks: collect, CI, data branch, site freshness. | One definition of "the live system is healthy"; the daily Command Center job runs it. |
| `.claude/settings.json` + `.claude/hooks/check_on_stop.py` | Stop hook: runs check.py when Claude tries to end a turn with changes; failure sends Claude back to work. | The harness enforces "the tests passed", so it can't be skipped. |

## Running it (desktop app)

1. **Plan.** In a session started in this repo folder:
   > Use the planner agent to turn this into tickets: <your idea>
2. **Build in parallel.** For each Ready ticket, start a new session with a worktree and say:
   > Take T-2 from TASKS.md.
3. **Review.** When a builder says it's in Review:
   > Use the reviewer agent on T-2, branch ticket/T-2-...
   If it says "needs results-auditor":
   > Use the results-auditor agent on branch ticket/T-2-...
4. **Merge.** You merge the approved branch (or ask a session to, and watch it).

Start sessions inside the repo folder: sessions started elsewhere don't load these agents or the
Stop hook.

## Status in Command Center

This project has a record on the Command Center Projects tab. Sessions working here update it
(see CLAUDE.md, "Command Center asks"), and the daily `ml-portfolio-health` job runs
`scripts/health.py` and raises an ask on the record when something is red.
