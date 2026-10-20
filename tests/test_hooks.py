import sys
import types

import pandas as pd

from ghtriage import collect
from ghtriage.cli import main


class _FakeCollector:
    name = "fake"
    key_cols = ["id"]
    partition_col = "ts"
    version_col = None
    keep_first_cols: list[str] = []
    committed = False

    def __init__(self, cfg=None):
        pass

    def fetch(self):
        return pd.DataFrame({"id": [1], "ts": [pd.Timestamp("2026-01-01")]})

    def commit(self):
        type(self).committed = True


def test_commit_hook_runs_after_rows_are_stored(cfg, monkeypatch):
    monkeypatch.setitem(collect.COLLECTORS, "fake", _FakeCollector)
    assert collect.run_collector("fake", cfg) == (1, 0)
    assert _FakeCollector.committed is True


def test_commit_hook_not_called_when_storing_fails(cfg, monkeypatch):
    class Broken(_FakeCollector):
        committed = False

        def fetch(self):
            return pd.DataFrame({"id": [1], "ts": ["not a date"]})

    monkeypatch.setitem(collect.COLLECTORS, "broken", Broken)
    try:
        collect.run_collector("broken", cfg)
    except Exception:
        pass
    assert Broken.committed is False


def test_build_uses_custom_builder_when_configured(cfg, tmp_path, monkeypatch):
    module = types.ModuleType("fake_builder_mod")
    module.build = lambda config: 42
    monkeypatch.setitem(sys.modules, "fake_builder_mod", module)
    cfg["data"]["builder"] = "fake_builder_mod:build"
    import yaml

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main(["--config", str(path), "build"]) == 0
