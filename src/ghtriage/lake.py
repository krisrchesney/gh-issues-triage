"""Parquet data lake: raw/<source>/dt=YYYY-MM-DD/part.parquet.

Why a lake on top of DuckDB: scheduled collectors run on ephemeral CI machines, so history has to live
somewhere durable. Small date-partitioned Parquet files diff cleanly, compress well, and can be synced
to a dedicated git branch (see scripts/data_sync.sh) instead of bloating the main branch.

All timestamps in the lake are stored as UTC, timezone-naive.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd

from ghtriage.storage import check_identifier, write_frame


def to_utc_naive(series: pd.Series) -> pd.Series:
    """Convert any datetime-like column to UTC, timezone-naive."""
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def partition_file(root: str | Path, source: str, day: str) -> Path:
    return Path(root) / source / f"dt={day}" / "part.parquet"


def _dedupe_batch(df: pd.DataFrame, key_cols: list[str], version_col: str | None) -> pd.DataFrame:
    """Within one incoming batch keep a single row per key (highest version, else last seen)."""
    if version_col:
        df = df.sort_values(version_col, kind="stable")
    return df.drop_duplicates(subset=key_cols, keep="last")


def _merge(
    old: pd.DataFrame,
    new: pd.DataFrame,
    key_cols: list[str],
    version_col: str | None,
    keep_first_cols: Sequence[str],
) -> tuple[pd.DataFrame, int, int]:
    cols = list(dict.fromkeys([*old.columns, *new.columns]))
    old = old.reindex(columns=cols)
    new = _dedupe_batch(new.reindex(columns=cols), key_cols, version_col)

    old_keys = pd.MultiIndex.from_frame(old[key_cols])
    new_keys = pd.MultiIndex.from_frame(new[key_cols])
    inserted = new[~new_keys.isin(old_keys)]
    candidates = new[new_keys.isin(old_keys)]

    updated = candidates.iloc[0:0]
    if version_col and len(candidates):
        joined = candidates.merge(
            old[[*key_cols, version_col]], on=key_cols, how="left", suffixes=("", "__old")
        )
        newer = (joined[version_col] > joined[f"{version_col}__old"]).to_numpy()
        updated = candidates.iloc[newer]

    kept = old
    if len(updated):
        updated_keys = pd.MultiIndex.from_frame(updated[key_cols])
        kept = old[~old_keys.isin(updated_keys)]
        if keep_first_cols:
            first = old[[*key_cols, *keep_first_cols]]
            updated = updated.drop(columns=list(keep_first_cols)).merge(first, on=key_cols, how="left")
            updated = updated[cols]

    frames = [f for f in (kept, updated, inserted) if len(f)]
    result = pd.concat(frames, ignore_index=True) if frames else old
    return result.sort_values(key_cols).reset_index(drop=True), len(inserted), len(updated)


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def upsert_partitioned(
    root: str | Path,
    source: str,
    df: pd.DataFrame,
    key_cols: list[str],
    partition_col: str,
    version_col: str | None = None,
    keep_first_cols: Sequence[str] = (),
) -> tuple[int, int]:
    """Merge rows into the lake. Returns (inserted, updated).

    - New keys are inserted.
    - Existing keys are replaced only when `version_col` is greater than the stored one.
    - `keep_first_cols` are never overwritten after first insert. Use them for "as first seen"
      snapshots so features cannot leak information that arrived after the prediction moment.
    - `partition_col` must be stable per key (e.g. creation time), so a row always lands in the same file.
    """
    check_identifier(source)
    if df.empty:
        return 0, 0
    df = df.copy()
    df[partition_col] = to_utc_naive(df[partition_col])
    if version_col and pd.api.types.is_datetime64_any_dtype(df[version_col]):
        df[version_col] = to_utc_naive(df[version_col])
    days = df[partition_col].dt.strftime("%Y-%m-%d")

    total_inserted = total_updated = 0
    for day, chunk in df.groupby(days):
        path = partition_file(root, source, str(day))
        if path.exists():
            merged, ins, upd = _merge(pd.read_parquet(path), chunk, key_cols, version_col, keep_first_cols)
        else:
            merged = _dedupe_batch(chunk, key_cols, version_col).sort_values(key_cols)
            ins, upd = len(merged), 0
        if ins or upd:
            _write_atomic(merged, path)
        total_inserted += ins
        total_updated += upd
    return total_inserted, total_updated


def _glob(root: str | Path, source: str) -> str:
    return (Path(root) / source / "dt=*" / "part.parquet").as_posix()


def has_data(root: str | Path, source: str) -> bool:
    return any((Path(root) / source).glob("dt=*/part.parquet"))


def load_lake(root: str | Path, source: str) -> pd.DataFrame:
    """Read every partition of a source into one frame (empty frame if nothing is stored yet)."""
    check_identifier(source)
    if not has_data(root, source):
        return pd.DataFrame()
    con = duckdb.connect()
    try:
        return con.execute(
            "SELECT * FROM read_parquet(?, union_by_name=true, hive_partitioning=false)",
            [_glob(root, source)],
        ).df()
    finally:
        con.close()


def lake_cursors(root: str | Path, source: str, value_col: str, group_col: str) -> dict[str, pd.Timestamp]:
    """Latest value of `value_col` per `group_col`, e.g. the newest updated_at seen per repository."""
    check_identifier(source)
    check_identifier(value_col)
    check_identifier(group_col)
    if not has_data(root, source):
        return {}
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"SELECT {group_col}, max({value_col}) FROM read_parquet(?, union_by_name=true, "
            f"hive_partitioning=false) GROUP BY {group_col}",
            [_glob(root, source)],
        ).fetchall()
    finally:
        con.close()
    return {str(k): pd.Timestamp(v) for k, v in rows if v is not None}


def build_table(con: duckdb.DuckDBPyConnection, root: str | Path, source: str, table: str) -> int:
    """Materialise a lake source into a DuckDB table. Returns the row count."""
    df = load_lake(root, source)
    if df.empty:
        raise ValueError(f"No data in lake source '{source}' under {Path(root).resolve()}.")
    return write_frame(con, table, df)
