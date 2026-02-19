"""
Column profiler — Python replacement for the Java ``ddprofiler``.

Reads columns from :mod:`source_readers`, computes per-column statistics
(type, cardinality, MinHash signatures, numeric range), and stores
profile + text documents into DuckDB and/or Elasticsearch.

This is the critical **Stage 0** of the Aurum pipeline.

Storage targets (A/B testable):

* **DuckDB** — embedded ``.db`` file with FTS. Single-writer constraint:
  multiprocessing workers profile columns, main thread bulk-inserts.
* **Elasticsearch** — ``profile`` + ``text`` indices via ``helpers.bulk``.
"""

from __future__ import annotations

import logging
import math
import re
from concurrent import futures
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from datasketch import HyperLogLog, MinHash

if TYPE_CHECKING:
    from aurum_v2.config import AurumConfig
    from aurum_v2.profiler.source_readers import SourceReader
    from aurum_v2.store.duck_store import DuckStore
    from aurum_v2.store.elastic_store import ElasticStore

__all__ = [
    "ColumnProfile",
    "Profiler",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ColumnProfile — per-column statistics container
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    """All computed statistics for a single column.

    Python equivalent of the Java ``WorkerTaskResult``.
    """

    nid: str
    """CRC32 node ID."""

    db_name: str
    source_name: str
    column_name: str
    data_type: str
    """``"T"`` for text, ``"N"`` for numeric."""

    total_values: int = 0
    unique_values: int = 0

    # ── Text column stats ──────────────────────────────────────────────
    entities: str = ""
    """Comma-separated NER entity labels."""

    minhash: list[int] = field(default_factory=list)
    """MinHash signature array (k=512). Empty for numeric columns."""

    # ── Numeric column stats ───────────────────────────────────────────
    min_value: float = 0.0
    max_value: float = 0.0
    avg_value: float = 0.0
    median: float = 0.0
    iqr: float = 0.0

    # ── Raw values (for text index) ────────────────────────────────────
    raw_values: list[str] = field(default_factory=list, repr=False)
    """Top-k unique values stored in the keyword search index."""

    # ── Path (filesystem / S3 URI) ─────────────────────────────────────
    path: str = ""
    """Path to the data source that contains this column."""


# ---------------------------------------------------------------------------
# KMinHash  (uses datasketch — clean reprofile, no legacy compat needed)
# ---------------------------------------------------------------------------

_TOKENIZER = re.compile(r"[\s_\-]+")


def compute_kmin_hash(values: list[str], k: int) -> list[int]:
    """Compute a MinHash signature over tokenised text values.

    Tokenisation: lowercase, split on whitespace/underscores/hyphens.
    Each token is UTF-8 encoded and fed to ``datasketch.MinHash(num_perm=k)``.

    Returns a Python ``list[int]`` of signed 64-bit hash values (safe for
    JSON serialisation and DuckDB ``BIGINT[]`` storage).
    """
    m = MinHash(num_perm=k)
    for val in values:
        tokens = _TOKENIZER.split(str(val).lower())
        for token in tokens:
            if token:
                m.update(token.encode("utf-8"))
    # Convert uint64 ndarray -> signed int64 list for storage compat
    return m.hashvalues.astype(np.int64).tolist()


# ---------------------------------------------------------------------------
# Cardinality  (HyperLogLog, p=18 matching legacy precision)
# ---------------------------------------------------------------------------

def compute_cardinality(values: list[str]) -> int:
    """Approximate unique-value count via HyperLogLog (p=16, ~0.4% error)."""
    hll = HyperLogLog(p=16)
    for val in values:
        hll.update(val.encode("utf-8"))
    return int(hll.count())


# ---------------------------------------------------------------------------
# Numeric range stats
# ---------------------------------------------------------------------------

def compute_numeric_stats(
    values: list[str],
) -> tuple[float, float, float, float, float]:
    """Compute ``(min, max, avg, median, iqr)`` safely for numeric strings.

    Filters out non-finite values (NaN, Inf) to prevent stat corruption.
    """
    valid_floats: list[float] = []
    for v in values:
        try:
            f = float(v)
            if math.isfinite(f):
                valid_floats.append(f)
        except (ValueError, OverflowError):
            pass

    if not valid_floats:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    arr = np.array(valid_floats)
    q75, q25 = np.percentile(arr, [75, 25])
    return (
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.mean(arr)),
        float(np.median(arr)),
        float(q75 - q25),
    )


# ---------------------------------------------------------------------------
# Profile a single column  (runs in worker process)
# ---------------------------------------------------------------------------

def profile_column(
    db_name: str,
    table_name: str,
    column_name: str,
    values: list[str],
    aurum_type: str,
    minhash_num_perm: int = 512,
    path: str = "",
) -> ColumnProfile:
    """Profile a single column and return a :class:`ColumnProfile`.

    Parameters
    ----------
    minhash_num_perm : int
        Passed explicitly so worker processes don't depend on a config
        instance (which may not pickle across ``ProcessPoolExecutor``.
    values
        Already truncated by the source reader when
        ``config.limit_text_values`` is True.
    """
    from aurum_v2.models.hit import compute_field_id

    unique_values = compute_cardinality(values)
    kmin_hash: list[int] = []
    numeric_stats = (0.0, 0.0, 0.0, 0.0, 0.0)

    if aurum_type == "T":
        kmin_hash = compute_kmin_hash(values, k=minhash_num_perm)
    else:
        numeric_stats = compute_numeric_stats(values)

    nid = compute_field_id(
        db_name=db_name, source_name=table_name, field_name=column_name,
    )

    # Store deduplicated raw values for the text index
    raw_vals = list(dict.fromkeys(values)) if aurum_type == "T" else []

    return ColumnProfile(
        nid=nid,
        db_name=db_name,
        source_name=table_name,
        column_name=column_name,
        data_type=aurum_type,
        total_values=len(values),
        unique_values=unique_values,
        minhash=kmin_hash,
        min_value=numeric_stats[0],
        max_value=numeric_stats[1],
        avg_value=numeric_stats[2],
        median=numeric_stats[3],
        iqr=numeric_stats[4],
        raw_values=raw_vals,
        path=path,
    )


# ---------------------------------------------------------------------------
# Main Profiler class
# ---------------------------------------------------------------------------

class Profiler:
    """Orchestrates profiling of data sources and stores results.

    Usage::

        profiler = Profiler(config)
        profiler.run(readers)
        profiler.store_profiles(duck=my_duck, es=my_es)   # A/B test
    """

    def __init__(self, config: AurumConfig) -> None:
        self._config = config
        self._profiles: list[ColumnProfile] = []

    # ------------------------------------------------------------------
    # Profiling pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        readers: list[SourceReader],
        *,
        max_workers: int = 4,
    ) -> None:
        """Profile all columns from all *readers* using multiprocessing.

        Column profiling is CPU-bound (hashing, numeric stats) and runs in
        a ``ProcessPoolExecutor``.  The main thread collects results into
        ``self._profiles`` for subsequent storage.
        """
        cfg = self._config
        MAX_QUEUE_SIZE = max_workers * 2

        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            active_futures: set[futures.Future[ColumnProfile]] = set()
            
            for reader in readers:
                for db_name, table_name, col_name, aurum_type, values, path in reader.read_columns():
                    
                    fut = executor.submit(
                        profile_column,
                        db_name=db_name,
                        table_name=table_name,
                        column_name=col_name,
                        values=values,
                        aurum_type=aurum_type,
                        minhash_num_perm=cfg.minhash_num_perm,
                        path=path,
                    )
                    active_futures.add(fut)
                    
                    # THROTTLE: If the queue is full, wait for at least one worker to finish 
                    # before pulling the next column from the generator.
                    if len(active_futures) >= MAX_QUEUE_SIZE:
                        done, active_futures = futures.wait(
                            active_futures, 
                            return_when=futures.FIRST_COMPLETED
                        )
                        for d in done:
                            try:
                                self._profiles.append(d.result())
                            except Exception as e:
                                logger.error("Worker failed: %s", e)

            # CLEANUP: Drain whatever is left in the queue after the reader runs dry
            for fut in futures.as_completed(active_futures):
                try:
                    self._profiles.append(fut.result())
                except Exception as e:
                    logger.error("Worker failed during cleanup: %s", e)

    # ------------------------------------------------------------------
    # Dual storage  (main-thread only)
    # ------------------------------------------------------------------

    def store_profiles(
        self,
        *,
        duck: DuckStore | None = None,
        es: ElasticStore | None = None,
    ) -> dict[str, int]:
        """Flush collected profiles to one or both storage backends.

        **DuckDB constraint**: all writes happen here on the main thread
        (never from worker processes).

        **ES constraint**: uses ``elasticsearch.helpers.bulk`` internally.

        Parameters
        ----------
        duck : DuckStore, optional
            If provided, inserts into DuckDB.
        es : ElasticStore, optional
            If provided, inserts into Elasticsearch.

        Returns
        -------
        dict[str, int]
            ``{"duckdb": n, "elasticsearch": m}`` counts per backend.
        """
        result: dict[str, int] = {}
        max_tv = self._config.max_text_values

        if duck is not None:
            n = duck.bulk_insert_profiles(
                self._profiles, max_text_values=max_tv,
            )
            result["duckdb"] = n
            logger.info("Stored %d profiles to DuckDB", n)

        if es is not None:
            n = es.bulk_insert_profiles(
                self._profiles, max_text_values=max_tv,
            )
            result["elasticsearch"] = n
            logger.info("Stored %d actions to Elasticsearch", n)

        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def profiles(self) -> list[ColumnProfile]:
        """Return all collected profiles."""
        return list(self._profiles)
