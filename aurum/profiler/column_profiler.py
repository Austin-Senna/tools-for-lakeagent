"""
Column profiler — extract per-column statistics and signatures.

Replaces the legacy ``DDProfiler`` (Java) + ``ElasticStore`` pipeline.
Instead of profiling into Elasticsearch, we profile directly into
in-memory dataclasses and persist with Pickle / Parquet.

Key statistics per column (ported from the ES ``profile`` index schema):
- data_type: "N" (numeric) or "T" (text)
- total_values / unique_values / cardinality_ratio
- For numeric: median, IQR, min, max
- For text: MinHash signature (datasketch)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
from datasketch import MinHash

from aurum.config import aurumConfig
from aurum.profiler.text_utils import normalise_value


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ColumnId:
    """Globally unique identifier for a column.

    Mirrors Aurum's ``compute_field_id`` which used
    ``str(CRC32(db_name + source_name + field_name))``.
    We use a truncated MD5 for lower collision probability.
    """

    db_name: str
    source_name: str
    field_name: str

    @property
    def nid(self) -> str:
        raw = f"{self.db_name}:{self.source_name}:{self.field_name}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.nid)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ColumnId):
            return self.nid == other.nid
        return NotImplemented

    def __str__(self) -> str:
        return f"{self.source_name}.{self.field_name}"


@dataclass
class NumericProfile:
    """Summary statistics for a numeric column.

    Ported from the Elasticsearch ``profile`` document fields
    ``median``, ``iqr``, ``minValue``, ``maxValue``.
    """

    median: float
    iqr: float
    min_value: float
    max_value: float

    @property
    def core_left(self) -> float:
        """Left boundary of the IQR core range (median − IQR)."""
        return self.median - self.iqr

    @property
    def core_right(self) -> float:
        """Right boundary of the IQR core range (median + IQR)."""
        return self.median + self.iqr

    @property
    def domain(self) -> float:
        """Width of the IQR core range."""
        return self.core_right - self.core_left


@dataclass
class ColumnProfile:
    """Complete profile for a single column in the data lake.

    Combines metadata that was previously split across the ``profile``
    and ``text`` Elasticsearch indices.
    """

    col_id: ColumnId
    data_type: str  # "N" or "T"
    total_values: int
    unique_values: int
    # Numeric stats (populated only when data_type == "N")
    numeric: NumericProfile | None = None
    # MinHash signature (populated only when data_type == "T")
    minhash: MinHash | None = None

    @property
    def cardinality_ratio(self) -> float:
        """``unique / total`` — the core PK-candidate heuristic.

        Ported from ``fieldnetwork.init_meta_schema``:
        ``cardinality_ratio = float(unique_values) / float(total_values)``
        """
        if self.total_values == 0:
            return 0.0
        return self.unique_values / self.total_values


# ---------------------------------------------------------------------------
# Profiling logic
# ---------------------------------------------------------------------------

def _build_minhash(values: pl.Series, num_perm: int) -> MinHash:
    """Build a MinHash from the unique string values in a column.

    Replaces the Java-interop ``ss_utils.minhash`` with standard datasketch.
    Preprocessing is the same: ``str(v).lower().encode('utf-8')``.
    """
    mh = MinHash(num_perm=num_perm)
    for v in values.drop_nulls().unique().to_list():
        mh.update(normalise_value(str(v)).encode("utf-8"))
    return mh


def _is_numeric_column(series: pl.Series) -> bool:
    """Decide if a Polars Series should be treated as numeric."""
    return series.dtype.is_numeric()


def profile_column(
    series: pl.Series,
    *,
    db_name: str,
    source_name: str,
    field_name: str,
    cfg: aurumConfig = aurumConfig(),
) -> ColumnProfile:
    """Profile a single column and return a ``ColumnProfile``.

    This is the modern replacement for DDProfiler (Java) +
    ElasticStore ingestion.
    """
    col_id = ColumnId(db_name=db_name, source_name=source_name, field_name=field_name)
    total = len(series)
    unique = series.n_unique()

    if _is_numeric_column(series):
        clean = series.drop_nulls().cast(pl.Float64)
        if len(clean) == 0:
            num_profile = NumericProfile(0.0, 0.0, 0.0, 0.0)
        else:
            np_arr = clean.to_numpy()
            q25, q50, q75 = float(np.percentile(np_arr, 25)), float(np.median(np_arr)), float(np.percentile(np_arr, 75))
            num_profile = NumericProfile(
                median=q50,
                iqr=q75 - q25,
                min_value=float(np.min(np_arr)),
                max_value=float(np.max(np_arr)),
            )
        return ColumnProfile(
            col_id=col_id,
            data_type="N",
            total_values=total,
            unique_values=unique,
            numeric=num_profile,
        )
    else:
        mh = _build_minhash(series.cast(pl.Utf8), num_perm=cfg.minhash_perms)
        return ColumnProfile(
            col_id=col_id,
            data_type="T",
            total_values=total,
            unique_values=unique,
            minhash=mh,
        )


def profile_dataframe(
    df: pl.DataFrame,
    *,
    db_name: str,
    source_name: str,
    cfg: aurumConfig = aurumConfig(),
) -> list[ColumnProfile]:
    """Profile every column in a Polars DataFrame."""
    profiles: list[ColumnProfile] = []
    # Sample if too large
    if len(df) > cfg.profiler_sample_rows:
        df = df.sample(n=cfg.profiler_sample_rows, seed=42)

    for col_name in df.columns:
        prof = profile_column(
            df[col_name],
            db_name=db_name,
            source_name=source_name,
            field_name=col_name,
            cfg=cfg,
        )
        profiles.append(prof)
    return profiles


def profile_directory(
    directory: Path,
    *,
    db_name: str = "default",
    glob: str = "*.csv",
    cfg: aurumConfig = aurumConfig(),
) -> Iterator[ColumnProfile]:
    """Walk a directory and yield ``ColumnProfile`` for every column found.

    Supports CSV and Parquet via Polars auto-detection.
    """
    for path in sorted(directory.rglob(glob)):
        source_name = path.name
        try:
            if path.suffix == ".parquet":
                df = pl.read_parquet(path)
            else:
                df = pl.read_csv(path, infer_schema_length=5000, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Skipping {path}: {exc}")
            continue

        for prof in profile_dataframe(df, db_name=db_name, source_name=source_name, cfg=cfg):
            yield prof
