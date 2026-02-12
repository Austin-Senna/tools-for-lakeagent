"""
Memory-aware join engine.

Port of ``aurum/DoD/data_processing_utils.py``.

Key algorithms
--------------
- ``estimate_output_row_size`` — estimate bytes-per-row of a join
- ``does_join_fit_in_memory`` — check before allocating
- ``join_ab_on_key_optimizer`` — chunked equi-join with memory guard
- ``materialize_join_graph`` — execute a full join graph sequentially
"""

from __future__ import annotations

import functools
import math
import os
from pathlib import Path
from typing import Sequence

import polars as pl
import psutil

from aurum.config import aurumConfig
from aurum.graph.field_network import Hit


# ---------------------------------------------------------------------------
# Memory estimation  (ported from data_processing_utils helpers)
# ---------------------------------------------------------------------------

def estimate_output_row_size(df_a: pl.DataFrame, df_b: pl.DataFrame) -> int:
    """Estimate bytes per row in the joined output.

    Uses the average row size of each DataFrame.
    """
    size_a = df_a.estimated_size("b") / max(df_a.height, 1)
    size_b = df_b.estimated_size("b") / max(df_b.height, 1)
    return int(size_a + size_b)


def does_join_fit_in_memory(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    fraction: float = 0.6,
) -> bool:
    """Conservatively check whether the join result fits in memory.

    Uses ``psutil`` to read available RAM.  The default *fraction*
    matches Aurum's ``memory_limit_fraction`` (0.6).
    """
    available = psutil.virtual_memory().available
    # Worst case: Cartesian product (we divide by a heuristic factor).
    row_size = estimate_output_row_size(df_a, df_b)
    estimated_rows = min(df_a.height * df_b.height, max(df_a.height, df_b.height) * 10)
    estimated_bytes = row_size * estimated_rows
    return estimated_bytes < available * fraction


# ---------------------------------------------------------------------------
# Chunked equi-join  (ported from join_ab_on_key_optimizer)
# ---------------------------------------------------------------------------

def join_on_key(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    key_a: str,
    key_b: str,
    how: str = "inner",
    cfg: aurumConfig | None = None,
) -> pl.DataFrame:
    """Memory-aware equi-join.

    Ported from ``data_processing_utils.join_ab_on_key_optimizer``.

    * If the join fits in memory → single Polars join.
    * Otherwise → chunk *df_a* and join each chunk, then ``vstack``.
    """
    if cfg is None:
        cfg = aurumConfig()

    # Cast join keys to common string type for safety
    df_a = df_a.with_columns(pl.col(key_a).cast(pl.Utf8).alias(key_a))
    df_b = df_b.with_columns(pl.col(key_b).cast(pl.Utf8).alias(key_b))

    if does_join_fit_in_memory(df_a, df_b, cfg.memory_limit_fraction):
        if key_a == key_b:
            return df_a.join(df_b, on=key_a, how=how, suffix="_right")
        return df_a.join(
            df_b,
            left_on=key_a,
            right_on=key_b,
            how=how,
            suffix="_right",
        )

    # Chunked join (memory-constrained path)
    available = psutil.virtual_memory().available * cfg.memory_limit_fraction
    row_size = estimate_output_row_size(df_a, df_b)
    chunk_rows = max(int(available / (row_size * 10 + 1)), 1_000)

    parts: list[pl.DataFrame] = []
    for offset in range(0, df_a.height, chunk_rows):
        chunk = df_a.slice(offset, chunk_rows)
        if key_a == key_b:
            joined = chunk.join(df_b, on=key_a, how=how, suffix="_right")
        else:
            joined = chunk.join(
                df_b,
                left_on=key_a,
                right_on=key_b,
                how=how,
                suffix="_right",
            )
        if joined.height > 0:
            parts.append(joined)

    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="vertical_relaxed")


# ---------------------------------------------------------------------------
# Apply filter (ported from data_processing_utils.apply_filter)
# ---------------------------------------------------------------------------

def apply_filter(
    df: pl.DataFrame,
    column: str,
    value: str,
) -> pl.DataFrame:
    """Filter rows where *column* contains *value* (case-insensitive).

    Ported from ``data_processing_utils.apply_filter``.
    """
    return df.filter(
        pl.col(column).cast(pl.Utf8).str.to_lowercase().str.contains(value.lower())
    )


# ---------------------------------------------------------------------------
# Materialise a full join graph
# ---------------------------------------------------------------------------

def _load_table(source_name: str, data_dir: Path) -> pl.DataFrame:
    """Load a CSV or Parquet by source name."""
    path = data_dir / source_name
    if not path.exists():
        # Try adding common extensions
        for ext in (".csv", ".parquet", ".pq"):
            candidate = path.with_suffix(ext)
            if candidate.exists():
                path = candidate
                break
    if path.suffix in (".parquet", ".pq"):
        return pl.read_parquet(path)
    return pl.read_csv(path, infer_schema_length=10_000)


def materialize_join_graph(
    join_graph: list[tuple[Hit, Hit]],
    data_dir: Path,
    cfg: aurumConfig | None = None,
    project_columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Execute a join graph left-to-right and return the result.

    Ported from ``data_processing_utils.materialize_join_graph``.

    Parameters
    ----------
    join_graph:
        List of ``(left_hit, right_hit)`` pairs, each encoding one
        equi-join hop.
    data_dir:
        Root directory where source tables live.
    project_columns:
        If given, only keep these columns in the final output.
    """
    if cfg is None:
        cfg = aurumConfig()
    if not join_graph:
        return pl.DataFrame()

    # Load the first table
    left_hit, right_hit = join_graph[0]
    result = _load_table(left_hit.source_name, data_dir)
    right = _load_table(right_hit.source_name, data_dir)
    result = join_on_key(result, right, left_hit.field_name, right_hit.field_name, cfg=cfg)

    for left_hit, right_hit in join_graph[1:]:
        right = _load_table(right_hit.source_name, data_dir)
        result = join_on_key(result, right, left_hit.field_name, right_hit.field_name, cfg=cfg)

    if project_columns:
        available = [c for c in project_columns if c in result.columns]
        if available:
            result = result.select(available)

    return result
