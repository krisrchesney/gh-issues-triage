import pandas as pd
import pytest

from ghtriage.collect import run_collector
from ghtriage.lake import build_table, lake_cursors, load_lake, upsert_partitioned
from ghtriage.storage import connect, read_table


def _rows(ids, updated, created="2026-01-01T10:00:00Z", val=1.0, first="a"):
    return pd.DataFrame(
        {
            "id": ids,
            "repo": "r/x",
            "created_at": pd.to_datetime(created, utc=True),
            "updated_at": pd.to_datetime(updated, utc=True),
            "val": val,
            "first_title": first,
        }
    )


def _upsert(root, df, keep_first=("first_title",)):
    return upsert_partitioned(
        root,
        "src",
        df,
        key_cols=["id"],
        partition_col="created_at",
        version_col="updated_at",
        keep_first_cols=keep_first,
    )


def test_insert_creates_date_partitions(tmp_path):
    df = pd.concat(
        [
            _rows([1], "2026-01-02T00:00:00Z"),
            _rows([2], "2026-01-03T00:00:00Z", created="2026-01-02T09:00:00Z"),
        ]
    )
    assert _upsert(tmp_path, df) == (2, 0)
    assert (tmp_path / "src" / "dt=2026-01-01" / "part.parquet").exists()
    assert (tmp_path / "src" / "dt=2026-01-02" / "part.parquet").exists()
    assert sorted(load_lake(tmp_path, "src")["id"]) == [1, 2]


def test_newer_version_replaces_older_is_ignored(tmp_path):
    _upsert(tmp_path, _rows([1], "2026-01-02T00:00:00Z", val=1.0))
    assert _upsert(tmp_path, _rows([1], "2026-01-05T00:00:00Z", val=9.0)) == (0, 1)
    assert load_lake(tmp_path, "src")["val"].iloc[0] == 9.0
    assert _upsert(tmp_path, _rows([1], "2026-01-03T00:00:00Z", val=5.0)) == (0, 0)  # stale
    assert load_lake(tmp_path, "src")["val"].iloc[0] == 9.0


def test_keep_first_columns_survive_updates(tmp_path):
    _upsert(tmp_path, _rows([1], "2026-01-02T00:00:00Z", first="original title"))
    _upsert(tmp_path, _rows([1], "2026-01-09T00:00:00Z", val=3.0, first="edited title"))
    row = load_lake(tmp_path, "src").iloc[0]
    assert row["first_title"] == "original title"
    assert row["val"] == 3.0


def test_duplicates_within_one_batch_keep_the_newest(tmp_path):
    batch = pd.concat(
        [_rows([1], "2026-01-02T00:00:00Z", val=1.0), _rows([1], "2026-01-04T00:00:00Z", val=2.0)]
    )
    assert _upsert(tmp_path, batch) == (1, 0)
    assert load_lake(tmp_path, "src")["val"].iloc[0] == 2.0


def test_rerunning_the_same_batch_changes_nothing(tmp_path):
    df = _rows([1, 2, 3], "2026-01-02T00:00:00Z")
    _upsert(tmp_path, df)
    assert _upsert(tmp_path, df) == (0, 0)


def test_cursors_return_latest_per_group(tmp_path):
    other = _rows([5], "2026-02-01T00:00:00Z").assign(repo="r/y")
    _upsert(
        tmp_path, pd.concat([_rows([1], "2026-01-02T00:00:00Z"), _rows([2], "2026-01-08T00:00:00Z"), other])
    )
    cursors = lake_cursors(tmp_path, "src", "updated_at", "repo")
    assert cursors["r/x"] == pd.Timestamp("2026-01-08")
    assert cursors["r/y"] == pd.Timestamp("2026-02-01")


def test_empty_lake_behaviour(tmp_path):
    assert load_lake(tmp_path, "nothing").empty
    assert lake_cursors(tmp_path, "nothing", "updated_at", "repo") == {}
    con = connect(tmp_path / "t.duckdb")
    with pytest.raises(ValueError):
        build_table(con, tmp_path, "nothing", "t")


def test_unsafe_source_name_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_lake(tmp_path, "../escape")


def test_demo_collector_to_table(cfg):
    inserted, _ = run_collector("demo", cfg)
    assert inserted == 5
    con = connect(cfg["data"]["db_path"])
    assert build_table(con, cfg["lake"]["root"], "demo", cfg["data"]["table"]) == 5
    assert len(read_table(con, cfg["data"]["table"])) == 5
