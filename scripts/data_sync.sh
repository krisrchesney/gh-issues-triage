#!/usr/bin/env bash
# Sync the local Parquet lake (data/raw) with an orphan "data" branch of this same repository.
# Scheduled collectors run on throwaway CI machines, so history has to be stored somewhere durable.
# Keeping it on its own branch keeps the main branch history small and clean.
#
#   scripts/data_sync.sh pull   restore data/raw from the data branch (no-op if the branch is absent)
#   scripts/data_sync.sh push   commit data/raw to the data branch (creates the branch on first use)
#
# Uses a git worktree of the current repo, so it inherits the credentials actions/checkout configured.
set -euo pipefail

cmd="${1:-}"
branch="${DATA_BRANCH:-data}"
remote="${DATA_REMOTE:-origin}"
data_dir="${DATA_DIR:-data/raw}"
subdir="raw"

[[ "$cmd" == "pull" || "$cmd" == "push" ]] || { echo "usage: $0 pull|push" >&2; exit 64; }

root="$(git rev-parse --show-toplevel)"
cd "$root"
tmp="$(mktemp -d)"
cleanup() { git worktree remove --force "$tmp/wt" >/dev/null 2>&1 || true; rm -rf "$tmp"; }
trap cleanup EXIT

remote_has_branch() { git ls-remote --exit-code --heads "$remote" "$branch" >/dev/null 2>&1; }

if [[ "$cmd" == "pull" ]]; then
  mkdir -p "$data_dir"
  if remote_has_branch; then
    git fetch --quiet --depth 1 "$remote" "$branch"
    git worktree add --quiet --detach "$tmp/wt" FETCH_HEAD
    if [[ -d "$tmp/wt/$subdir" ]]; then cp -R "$tmp/wt/$subdir/." "$data_dir/"; fi
    echo "data_sync: restored $(find "$data_dir" -name '*.parquet' | wc -l | tr -d ' ') parquet file(s) from $remote/$branch"
  else
    echo "data_sync: no '$branch' branch on $remote yet; starting empty"
  fi
  exit 0
fi

# push
[[ -d "$data_dir" ]] || { echo "data_sync: nothing to push ($data_dir missing)"; exit 0; }
for attempt in 1 2 3; do
  rm -rf "$tmp/wt"; git worktree prune
  if remote_has_branch; then
    git fetch --quiet --depth 1 "$remote" "$branch"
    git worktree add --quiet --detach "$tmp/wt" FETCH_HEAD
  else
    git worktree add --quiet --detach "$tmp/wt" HEAD
    git -C "$tmp/wt" checkout --quiet --orphan "$branch"
    git -C "$tmp/wt" rm -rf --quiet . >/dev/null 2>&1 || true
  fi
  rm -rf "${tmp:?}/wt/$subdir"; mkdir -p "$tmp/wt/$subdir"
  cp -R "$data_dir/." "$tmp/wt/$subdir/"
  git -C "$tmp/wt" add -A
  if git -C "$tmp/wt" diff --cached --quiet; then echo "data_sync: no changes to push"; exit 0; fi
  git -C "$tmp/wt" -c user.name="${GIT_AUTHOR_NAME:-data-bot}" -c user.email="${GIT_AUTHOR_EMAIL:-data-bot@users.noreply.github.com}" \
    commit --quiet -m "data: update $(date -u +%Y-%m-%dT%H:%M:%SZ) [skip ci]"
  if git -C "$tmp/wt" push --quiet "$remote" "HEAD:refs/heads/$branch"; then
    echo "data_sync: pushed to $remote/$branch"; exit 0
  fi
  echo "data_sync: push attempt $attempt failed, retrying" >&2; sleep $((attempt * 3))
done
echo "data_sync: giving up after 3 attempts" >&2; exit 1
