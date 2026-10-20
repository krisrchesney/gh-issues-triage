import pandas as pd
import pytest

from ghtriage.storage import append_dedup, connect, read_table, table_exists, write_frame


def test_roundtrip(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    df = pd.DataFrame({"id": [1, 2], "v": [0.1, 0.2]})
    assert write_frame(con, "things", df) == 2
    assert table_exists(con, "things")
    assert read_table(con, "things", order_by="id")["v"].tolist() == [0.1, 0.2]


def test_append_dedup_skips_existing_keys(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    first = pd.DataFrame({"id": [1, 2], "v": [1.0, 2.0]})
    assert append_dedup(con, "things", first, ["id"]) == 2
    second = pd.DataFrame({"id": [2, 3], "v": [9.0, 3.0]})
    assert append_dedup(con, "things", second, ["id"]) == 1
    assert len(read_table(con, "things")) == 3


def test_append_dedup_empty_frame(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    assert append_dedup(con, "things", pd.DataFrame({"id": []}), ["id"]) == 0


def test_unsafe_identifier_rejected(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    with pytest.raises(ValueError):
        read_table(con, "things; DROP TABLE things")
