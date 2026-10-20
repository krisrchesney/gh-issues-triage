"""DuckDB storage with Parquet snapshots. Free, local, and fast enough for portfolio-scale data."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_identifier(name: str) -> str:
    """Table and column names are interpolated into SQL, so only allow plain identifiers."""
    if not _IDENTIFIER.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


check_identifier = _check_identifier


def connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    _check_identifier(table)
    row = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()
    return bool(row and row[0])


def write_frame(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """Replace a table with the given frame. Returns the row count written."""
    _check_identifier(table)
    con.register("_incoming", df)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _incoming")
    con.unregister("_incoming")
    return len(df)


def append_dedup(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, key_cols: list[str]) -> int:
    """Append rows, skipping any whose key already exists. Safe to re-run a collector.

    Returns the number of rows actually inserted.
    """
    _check_identifier(table)
    for col in key_cols:
        _check_identifier(col)
    if df.empty:
        return 0
    if not table_exists(con, table):
        return write_frame(con, table, df)
    before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    con.register("_incoming", df)
    join = " AND ".join(f"t.{c} = i.{c}" for c in key_cols)
    con.execute(
        f"INSERT INTO {table} SELECT i.* FROM _incoming i "
        f"WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE {join})"
    )
    con.unregister("_incoming")
    after = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return after - before


def read_table(con: duckdb.DuckDBPyConnection, table: str, order_by: str | None = None) -> pd.DataFrame:
    _check_identifier(table)
    query = f"SELECT * FROM {table}"
    if order_by:
        query += f" ORDER BY {_check_identifier(order_by)}"
    return con.execute(query).df()


def export_parquet(con: duckdb.DuckDBPyConnection, table: str, out_path: str | Path) -> Path:
    """Snapshot a table to compressed Parquet (small enough to keep in Git LFS or release assets)."""
    _check_identifier(table)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY {table} TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    return out
