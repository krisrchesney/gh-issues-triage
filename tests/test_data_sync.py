"""Exercises scripts/data_sync.sh against a local bare remote (no network, no GitHub)."""

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data_sync.sh"
pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None, reason="needs git and bash"
)


def run(cmd, cwd, check=True):
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def make_clone(tmp_path: Path, name: str, remote: Path) -> Path:
    clone = tmp_path / name
    run(["git", "clone", "-q", str(remote), str(clone)], tmp_path)
    run(["git", "config", "user.email", "t@example.com"], clone)
    run(["git", "config", "user.name", "t"], clone)
    return clone


@pytest.fixture
def remote_and_clone(tmp_path):
    remote = tmp_path / "remote.git"
    run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], tmp_path)
    seed = make_clone(tmp_path, "seed", remote)
    (seed / "README.md").write_text("code")
    run(["git", "add", "."], seed)
    run(["git", "commit", "-q", "-m", "init"], seed)
    run(["git", "push", "-q", "origin", "HEAD:main"], seed)
    return remote, make_clone(tmp_path, "worker1", remote)


def write_part(clone: Path, day: str, content: bytes) -> Path:
    path = clone / "data" / "raw" / "src" / f"dt={day}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_pull_with_no_data_branch_is_a_noop(remote_and_clone):
    _, clone = remote_and_clone
    out = run(["bash", str(SCRIPT), "pull"], clone)
    assert "starting empty" in out.stdout


def test_push_creates_branch_then_pull_restores_it_elsewhere(tmp_path, remote_and_clone):
    remote, worker1 = remote_and_clone
    write_part(worker1, "2026-01-01", b"one")
    assert "pushed" in run(["bash", str(SCRIPT), "push"], worker1).stdout

    heads = run(["git", "ls-remote", "--heads", str(remote)], tmp_path).stdout
    assert "refs/heads/data" in heads and "refs/heads/main" in heads
    # the data branch is an orphan: it must not carry the code history
    files = run(["git", "ls-tree", "-r", "--name-only", "data"], remote).stdout.split()
    assert files == ["raw/src/dt=2026-01-01/part.parquet"]

    worker2 = make_clone(tmp_path, "worker2", remote)
    run(["bash", str(SCRIPT), "pull"], worker2)
    assert (worker2 / "data/raw/src/dt=2026-01-01/part.parquet").read_bytes() == b"one"


def test_second_run_accumulates_and_unchanged_data_is_not_recommitted(tmp_path, remote_and_clone):
    remote, worker1 = remote_and_clone
    write_part(worker1, "2026-01-01", b"one")
    run(["bash", str(SCRIPT), "push"], worker1)

    worker2 = make_clone(tmp_path, "worker2", remote)
    run(["bash", str(SCRIPT), "pull"], worker2)
    write_part(worker2, "2026-01-02", b"two")
    assert "pushed" in run(["bash", str(SCRIPT), "push"], worker2).stdout
    assert "no changes" in run(["bash", str(SCRIPT), "push"], worker2).stdout

    files = sorted(run(["git", "ls-tree", "-r", "--name-only", "data"], remote).stdout.split())
    assert files == ["raw/src/dt=2026-01-01/part.parquet", "raw/src/dt=2026-01-02/part.parquet"]
    assert int(run(["git", "rev-list", "--count", "data"], remote).stdout) == 2
