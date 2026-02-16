"""Temporary script to write duck_store.py and column_profiler.py."""
import pathlib

BASE = pathlib.Path("/Users/austinsenna/Documents/projects/tools-for-lakeagent/aurum_v2")

# ── duck_store.py ──────────────────────────────────────────────────────────

(BASE / "store" / "duck_store.py").write_text('''\
"""
DuckDB store — embedded ingestion + retrieval for column profiles.

Replaces Elasticsearch with a single ``.db`` file using DuckDB native
Full-Text Search (FTS) extension. No JVM, no server, no infrastructure.

Schema mirrors the legacy ES indices:

* **``profile``** table — one row per column (metadata + minhash + numeric stats).
* **``text_index``** table — one row per column (top-k unique values for FTS).

**Single-writer constraint**: DuckDB allows only one concurrent writer.
All inserts must go through the main thread (never from multiprocessing
workers). The :meth:`bulk_insert_profiles` method handles this.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from aurum_v2.models.hit import Hit

if TYPE_CHECKING:
    from aurum_v2.config import AurumConfig
    from aurum_v2.profiler.column_profiler import ColumnProfile

__all__ = ["KWType", "DuckStore"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword-search scope (shared enum with ElasticStore for API compat)
# ---------------------------------------------------------------------------

class KWType(Enum):
    """Which field to query against."""

    KW_CONTENT = 0   # FTS on text_index.text
    KW_SCHEMA = 1    # column name
    KW_ENTITIES = 2  # entity annotations
    KW_TABLE = 3     # table / source name


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_PROFILE = """
CREATE TABLE IF NOT EXISTS profile (
    nid           VARCHAR PRIMARY KEY,
    db_name       VARCHAR,
    path          VARCHAR DEFAULT '',
    source_name   VARCHAR NOT NULL,
    column_name   VARCHAR NOT NULL,
    data_type     VARCHAR NOT NULL,   -- 'T' or 'N'
    total_values  BIGINT  DEFAULT 0,
    unique_values BIGINT  DEFAULT 0,
    entities      VARCHAR DEFAULT '',
    minhash       BIGINT[],           -- k=512 hash values
    min_value     DOUBLE  DEFAULT 0,
    max_value     DOUBLE  DEFAULT 0,
    avg_value     DOUBLE  DEFAULT 0,
    median        DOUBLE  DEFAULT 0,
    iqr           DOUBLE  DEFAULT 0
);
"""

_CREATE_TEXT_INDEX = """
CREATE TABLE IF NOT EXISTS text_index (
    nid         VARCHAR PRIMARY KEY,
    db_name     VARCHAR,
    source_name VARCHAR,
    column_name VARCHAR,
    text        VARCHAR          -- top-k unique values joined by space
);
"""


# ---------------------------------------------------------------------------
# DuckStore
# ---------------------------------------------------------------------------

class DuckStore:
    """Embedded DuckDB store for column profiles + FTS keyword search.

    Parameters
    ----------
    config : AurumConfig
        System configuration.
    db_path : str | Path
        Path to the ``.db`` file. Use ``":memory:"`` for testing.
    """

    def __init__(self, config: AurumConfig, db_path: str | Path = "aurum.db") -> None:
        self._config = config
        self._db_path = str(db_path)
        self._con: duckdb.DuckDBPyConnection = duckdb.connect(self._db_path)

    # ==================================================================
    # Schema management
    # ==================================================================

    def init_tables(self, *, recreate: bool = False) -> None:
        """Create ``profile`` and ``text_index`` tables + FTS index.

        If *recreate* is True, existing tables are dropped first.
        """
        if recreate:
            self._con.execute("DROP TABLE IF EXISTS text_index;")
            self._con.execute("DROP TABLE IF EXISTS profile;")
            logger.info("Dropped existing DuckDB tables")

        self._con.execute(_CREATE_PROFILE)
        self._con.execute(_CREATE_TEXT_INDEX)
        logger.info("DuckDB tables ready at %s", self._db_path)

    def _rebuild_fts(self) -> None:
        """(Re)create the FTS index on ``text_index.text``.

        Must be called after bulk inserts — DuckDB FTS indexes are not
        automatically updated on INSERT.
        """
        self._con.execute("INSTALL fts; LOAD fts;")
        # Drop existing FTS index if present
        try:
            self._con.execute("PRAGMA drop_fts_index('text_index');")
        except duckdb.CatalogException:
            pass  # no existing index
        self._con.execute(
            "PRAGMA create_fts_index("
            "  'text_index', 'nid', 'text', 'column_name', 'source_name',"
            "  stemmer='english', stopwords='english'"
            ");"
        )
        logger.info("FTS index rebuilt on text_index")

    # ==================================================================
    # Ingestion  (single-writer — call from main thread only)
    # ==================================================================

    def bulk_insert_profiles(
        self,
        profiles: list[ColumnProfile],
        *,
        max_text_values: int = 1_000,
    ) -> int:
        """Insert profiles into ``profile`` and ``text_index`` tables.

        **Must be called from the main thread** — DuckDB only supports a
        single concurrent writer.

        Uses DuckDB ``INSERT OR REPLACE`` via ``executemany`` for efficient
        batched writes.

        Parameters
        ----------
        profiles : list[ColumnProfile]
            Column profiles gathered by the Profiler.
        max_text_values : int
            Max unique values stored per column in the text index.

        Returns
        -------
        int
            Number of profiles inserted.
        """
        if not profiles:
            return 0

        profile_rows = []
        text_rows = []

        for p in profiles:
            profile_rows.append((
                p.nid,
                p.db_name,
                getattr(p, "path", ""),
                p.source_name,
                p.column_name,
                p.data_type,
                p.total_values,
                p.unique_values,
                p.entities,
                p.minhash,          # DuckDB BIGINT[] accepts list[int]
                p.min_value,
                p.max_value,
                p.avg_value,
                p.median,
                p.iqr,
            ))

            if p.data_type == "T" and p.raw_values:
                unique_vals = list(dict.fromkeys(p.raw_values))[:max_text_values]
                text_rows.append((
                    p.nid,
                    p.db_name,
                    p.source_name,
                    p.column_name,
                    " ".join(unique_vals),
                ))

        self._con.execute("BEGIN TRANSACTION;")
        try:
            self._con.executemany(
                """INSERT OR REPLACE INTO profile
                   (nid, db_name, path, source_name, column_name, data_type,
                    total_values, unique_values, entities, minhash,
                    min_value, max_value, avg_value, median, iqr)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                profile_rows,
            )
            if text_rows:
                self._con.executemany(
                    """INSERT OR REPLACE INTO text_index
                       (nid, db_name, source_name, column_name, text)
                       VALUES (?, ?, ?, ?, ?)""",
                    text_rows,
                )
            self._con.execute("COMMIT;")
        except Exception:
            self._con.execute("ROLLBACK;")
            raise

        # Rebuild FTS after bulk load
        self._rebuild_fts()

        logger.info(
            "Inserted %d profile rows, %d text rows into DuckDB",
            len(profile_rows), len(text_rows),
        )
        return len(profile_rows)

    # ==================================================================
    # Retrieval — used by network-building pipeline
    # ==================================================================

    def get_all_fields(self) -> Iterator[tuple[str, str, str, str, int, int, str]]:
        """Scan ``profile`` table.

        Yields ``(nid, db_name, source_name, column_name,
        total_values, unique_values, data_type)``.
        """
        rows = self._con.execute(
            "SELECT nid, db_name, source_name, column_name,"
            "       total_values, unique_values, data_type"
            "  FROM profile"
        ).fetchall()
        for row in rows:
            yield row  # type: ignore[misc]

    def get_all_mh_text_signatures(self) -> list[tuple[str, list[int]]]:
        """Return ``[(nid, minhash_array), ...]`` for all text columns."""
        rows = self._con.execute(
            "SELECT nid, minhash FROM profile"
            " WHERE data_type = 'T' AND minhash IS NOT NULL"
        ).fetchall()
        results: list[tuple[str, list[int]]] = []
        for nid, mh in rows:
            if mh:
                results.append((nid, list(mh)))
        return results

    def get_all_fields_num_signatures(
        self,
    ) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Return ``[(nid, (median, iqr, min_val, max_val)), ...]`` for numeric cols."""
        rows = self._con.execute(
            "SELECT nid, median, iqr, min_value, max_value"
            "  FROM profile WHERE data_type = 'N'"
        ).fetchall()
        return [(nid, (med, iq, mn, mx)) for nid, med, iq, mn, mx in rows]

    # ==================================================================
    # Keyword search — DuckDB FTS (used by Algebra / API)
    # ==================================================================

    def search_keywords(
        self, keywords: str, kw_type: KWType, max_hits: int = 15,
    ) -> Iterator[Hit]:
        """Relevance-ranked keyword search via DuckDB FTS or ILIKE."""
        if kw_type == KWType.KW_CONTENT:
            yield from self._fts_search(keywords, max_hits)
        elif kw_type == KWType.KW_SCHEMA:
            yield from self._profile_like_search("column_name", keywords, max_hits)
        elif kw_type == KWType.KW_ENTITIES:
            yield from self._profile_like_search("entities", keywords, max_hits)
        elif kw_type == KWType.KW_TABLE:
            yield from self._profile_like_search("source_name", keywords, max_hits)

    def exact_search_keywords(
        self, keywords: str, kw_type: KWType, max_hits: int = 15,
    ) -> Iterator[Hit]:
        """Exact-match keyword search."""
        if kw_type == KWType.KW_CONTENT:
            rows = self._con.execute(
                "SELECT t.nid, p.db_name, p.source_name, p.column_name"
                "  FROM text_index t"
                "  JOIN profile p ON t.nid = p.nid"
                " WHERE t.text LIKE ?"
                " LIMIT ?",
                [f"% {keywords} %", max_hits],
            ).fetchall()
        elif kw_type == KWType.KW_SCHEMA:
            rows = self._con.execute(
                "SELECT nid, db_name, source_name, column_name"
                "  FROM profile WHERE column_name = ? LIMIT ?",
                [keywords, max_hits],
            ).fetchall()
        elif kw_type == KWType.KW_ENTITIES:
            rows = self._con.execute(
                "SELECT nid, db_name, source_name, column_name"
                "  FROM profile WHERE entities LIKE ? LIMIT ?",
                [f"%{keywords}%", max_hits],
            ).fetchall()
        elif kw_type == KWType.KW_TABLE:
            rows = self._con.execute(
                "SELECT nid, db_name, source_name, column_name"
                "  FROM profile WHERE source_name = ? LIMIT ?",
                [keywords, max_hits],
            ).fetchall()
        else:
            return

        for nid, db, src, col in rows:
            yield Hit(nid=nid, db_name=db, source_name=src,
                      field_name=col, score=1.0)

    def fuzzy_keyword_match(
        self, keywords: str, max_hits: int = 15,
    ) -> Iterator[Hit]:
        """Fuzzy keyword search on text_index (FTS stemming handles most cases)."""
        yield from self._fts_search(keywords, max_hits)

    # ==================================================================
    # Internals
    # ==================================================================

    def _fts_search(self, keywords: str, max_hits: int) -> Iterator[Hit]:
        """Run a DuckDB FTS query against the ``text_index`` table."""
        try:
            rows = self._con.execute(
                "SELECT t.nid, p.db_name, p.source_name, p.column_name,"
                "       fts_main_text_index.match_bm25(t.nid, ?) AS score"
                "  FROM text_index t"
                "  JOIN profile p ON t.nid = p.nid"
                " WHERE score IS NOT NULL"
                " ORDER BY score DESC"
                " LIMIT ?",
                [keywords, max_hits],
            ).fetchall()
        except duckdb.Error as e:
            logger.warning("FTS query failed: %s", e)
            return
        for nid, db, src, col, score in rows:
            yield Hit(nid=nid, db_name=db, source_name=src,
                      field_name=col, score=float(score))

    def _profile_like_search(
        self, field: str, keywords: str, max_hits: int,
    ) -> Iterator[Hit]:
        """ILIKE search on a profile column."""
        rows = self._con.execute(
            f"SELECT nid, db_name, source_name, column_name"
            f"  FROM profile"
            f" WHERE {field} ILIKE ?"
            f" LIMIT ?",
            [f"%{keywords}%", max_hits],
        ).fetchall()
        for nid, db, src, col in rows:
            yield Hit(nid=nid, db_name=db, source_name=src,
                      field_name=col, score=1.0)

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._con.close()
''')
print("duck_store.py written")


# ── column_profiler.py ─────────────────────────────────────────────────────

(BASE / "profiler" / "column_profiler.py").write_text('''\
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

_TOKENIZER = re.compile(r"[\\s_\\-]+")


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
    """Approximate unique-value count via HyperLogLog (p=18, ~0.4% error)."""
    hll = HyperLogLog(p=18)
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
    max_text_values: int = 1_000,
    path: str = "",
) -> ColumnProfile:
    """Profile a single column and return a :class:`ColumnProfile`.

    Parameters
    ----------
    minhash_num_perm : int
        Passed explicitly so worker processes don't depend on a config
        instance (which may not pickle across ``ProcessPoolExecutor``).
    max_text_values : int
        Cap on unique raw values retained for keyword search.
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

    # Deduplicate + cap raw values for the text index
    raw_vals = (
        list(dict.fromkeys(values))[:max_text_values]
        if aurum_type == "T"
        else []
    )

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
        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            pending: list[futures.Future[ColumnProfile]] = []
            for reader in readers:
                for db_name, table_name, col_name, aurum_type, values in (
                    reader.read_columns()
                ):
                    fut = executor.submit(
                        profile_column,
                        db_name=db_name,
                        table_name=table_name,
                        column_name=col_name,
                        values=values,
                        aurum_type=aurum_type,
                        minhash_num_perm=cfg.minhash_num_perm,
                        max_text_values=cfg.max_text_values,
                    )
                    pending.append(fut)

            for fut in futures.as_completed(pending):
                try:
                    self._profiles.append(fut.result())
                except Exception as e:
                    logger.error("Worker failed: %s", e)

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
''')
print("column_profiler.py written")


# ── elastic_store.py ───────────────────────────────────────────────────────

(BASE / "store" / "elastic_store.py").write_text('''\
"""
Elasticsearch store — ingestion + retrieval for column profiles.

Provides two ES indices matching the legacy ``NativeElasticStore.java``:

* **``profile``** — one doc per column (metadata + minhash + numeric stats).
* **``text``** — one doc per column (top-k unique values for keyword search).

Retrieval methods satisfy the interface expected by
:mod:`aurum_v2.builder.coordinator` and :mod:`aurum_v2.discovery.api`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from enum import Enum
from typing import TYPE_CHECKING, Any

from elasticsearch import Elasticsearch  # type: ignore[import-untyped]
from elasticsearch.helpers import bulk  # type: ignore[import-untyped]

from aurum_v2.models.hit import Hit

if TYPE_CHECKING:
    from aurum_v2.config import AurumConfig
    from aurum_v2.profiler.column_profiler import ColumnProfile

__all__ = ["KWType", "ElasticStore"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword-search scope
# ---------------------------------------------------------------------------

class KWType(Enum):
    """Which Elasticsearch field to query against."""

    KW_CONTENT = 0   # full-text index ("text")
    KW_SCHEMA = 1    # column name  ("profile.columnName")
    KW_ENTITIES = 2  # entity annotations
    KW_TABLE = 3     # table / source name


# ---------------------------------------------------------------------------
# Index mappings (mirrors NativeElasticStore.initStore)
# ---------------------------------------------------------------------------

_PROFILE_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "id":           {"type": "keyword"},
            "dbName":       {"type": "keyword", "index": False},
            "path":         {"type": "keyword", "index": False},
            "sourceName":   {"type": "text", "analyzer": "standard"},
            "sourceNameNA": {"type": "keyword"},
            "columnName":   {"type": "text", "analyzer": "standard"},
            "columnNameNA": {"type": "keyword"},
            "dataType":     {"type": "keyword"},
            "totalValues":  {"type": "long"},
            "uniqueValues": {"type": "long"},
            "entities":     {"type": "keyword"},
            "minhash":      {"type": "long"},
            "minValue":     {"type": "double"},
            "maxValue":     {"type": "double"},
            "avgValue":     {"type": "double"},
            "median":       {"type": "double"},
            "iqr":          {"type": "double"},
        }
    }
}

_TEXT_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "id":         {"type": "keyword"},
            "dbName":     {"type": "keyword", "index": False},
            "sourceName": {"type": "keyword", "index": False},
            "columnName": {"type": "keyword", "index": False},
            "text":       {"type": "text", "analyzer": "english"},
        }
    }
}


# ---------------------------------------------------------------------------
# ElasticStore
# ---------------------------------------------------------------------------

class ElasticStore:
    """Elasticsearch ingestion + retrieval for column profiles.

    Parameters
    ----------
    config : AurumConfig
        Must have ``es_host`` and ``es_port``.
    """

    def __init__(self, config: AurumConfig) -> None:
        self._config = config
        self._client: Elasticsearch = Elasticsearch(
            [{"host": config.es_host, "port": int(config.es_port)}]
        )

    # ==================================================================
    # Index management
    # ==================================================================

    def init_indices(self, *, recreate: bool = False) -> None:
        """Create ``profile`` and ``text`` indices if they don't exist.

        If *recreate* is True, existing indices are deleted first.
        """
        for name, mapping in [("profile", _PROFILE_MAPPING),
                               ("text", _TEXT_MAPPING)]:
            if recreate and self._client.indices.exists(index=name):
                self._client.indices.delete(index=name)
                logger.info("Deleted existing index '%s'", name)
            if not self._client.indices.exists(index=name):
                self._client.indices.create(index=name, body=mapping)
                logger.info("Created index '%s'", name)

    # ==================================================================
    # Ingestion  (bulk API — called from Profiler.store_profiles)
    # ==================================================================

    def bulk_insert_profiles(
        self,
        profiles: list[ColumnProfile],
        *,
        max_text_values: int = 1_000,
    ) -> int:
        """Bulk-index a list of :class:`ColumnProfile` into ES.

        For each profile, two documents are produced:

        * A **profile document** in the ``profile`` index.
        * A **text document** in the ``text`` index (top-k unique values).

        Uses ``elasticsearch.helpers.bulk`` for efficient batching.

        Parameters
        ----------
        profiles : list[ColumnProfile]
            Completed column profiles from the profiler.
        max_text_values : int
            Maximum unique values stored in the ``text`` index per column.

        Returns
        -------
        int
            Number of actions successfully indexed.
        """
        actions: list[dict[str, Any]] = []

        for p in profiles:
            # -- profile doc --
            actions.append({
                "_index": "profile",
                "_id": p.nid,
                "_source": {
                    "id":           p.nid,
                    "dbName":       p.db_name,
                    "path":         getattr(p, "path", ""),
                    "sourceName":   p.source_name,
                    "sourceNameNA": p.source_name,
                    "columnName":   p.column_name,
                    "columnNameNA": p.column_name,
                    "dataType":     p.data_type,
                    "totalValues":  p.total_values,
                    "uniqueValues": p.unique_values,
                    "entities":     p.entities,
                    "minhash":      p.minhash,
                    "minValue":     p.min_value,
                    "maxValue":     p.max_value,
                    "avgValue":     p.avg_value,
                    "median":       p.median,
                    "iqr":          p.iqr,
                },
            })

            # -- text doc (top-k unique values for keyword search) --
            if p.data_type == "T" and p.raw_values:
                unique_vals = list(dict.fromkeys(p.raw_values))[:max_text_values]
                actions.append({
                    "_index": "text",
                    "_id": p.nid,
                    "_source": {
                        "id":         p.nid,
                        "dbName":     p.db_name,
                        "sourceName": p.source_name,
                        "columnName": p.column_name,
                        "text":       " ".join(unique_vals),
                    },
                })

        if not actions:
            return 0

        success, errors = bulk(self._client, actions, raise_on_error=False)
        if errors:
            logger.warning("ES bulk insert had %d errors", len(errors))
        return success

    # ==================================================================
    # Retrieval — used by network-building pipeline
    # ==================================================================

    def get_all_fields(
        self,
    ) -> Iterator[tuple[str, str, str, str, int, int, str]]:
        """Scroll ``profile`` index.

        Yields ``(nid, db_name, source_name, column_name,
        total_values, unique_values, data_type)``.
        """
        body: dict[str, Any] = {"query": {"match_all": {}}}
        source_fields = [
            "dbName", "sourceName", "columnName",
            "totalValues", "uniqueValues", "dataType",
        ]
        res = self._client.search(
            index="profile", body=body, scroll="5m",
            size=500, _source=source_fields,
        )
        scroll_id = res["_scroll_id"]
        while True:
            hits = res["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                s = h["_source"]
                yield (
                    h["_id"],
                    s.get("dbName", ""),
                    s["sourceName"],
                    s["columnName"],
                    s.get("totalValues", 0),
                    s.get("uniqueValues", 0),
                    s.get("dataType", "T"),
                )
            res = self._client.scroll(scroll_id=scroll_id, scroll="5m")
            scroll_id = res["_scroll_id"]
        self._client.clear_scroll(scroll_id=scroll_id)

    def get_all_mh_text_signatures(self) -> list[tuple[str, list[int]]]:
        """Return ``[(nid, minhash_array), ...]`` for all text columns."""
        body: dict[str, Any] = {
            "query": {"bool": {"filter": [{"term": {"dataType": "T"}}]}}
        }
        results: list[tuple[str, list[int]]] = []
        res = self._client.search(
            index="profile", body=body, scroll="5m",
            size=500, _source=["minhash"],
        )
        scroll_id = res["_scroll_id"]
        while True:
            hits = res["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                mh = h["_source"].get("minhash", [])
                if mh:
                    results.append((h["_id"], mh))
            res = self._client.scroll(scroll_id=scroll_id, scroll="5m")
            scroll_id = res["_scroll_id"]
        self._client.clear_scroll(scroll_id=scroll_id)
        return results

    def get_all_fields_num_signatures(
        self,
    ) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Return ``[(nid, (median, iqr, min_val, max_val)), ...]``."""
        body: dict[str, Any] = {
            "query": {"bool": {"filter": [{"term": {"dataType": "N"}}]}}
        }
        results: list[tuple[str, tuple[float, float, float, float]]] = []
        res = self._client.search(
            index="profile", body=body, scroll="5m",
            size=500, _source=["median", "iqr", "minValue", "maxValue"],
        )
        scroll_id = res["_scroll_id"]
        while True:
            hits = res["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                s = h["_source"]
                results.append((
                    h["_id"],
                    (s["median"], s["iqr"], s["minValue"], s["maxValue"]),
                ))
            res = self._client.scroll(scroll_id=scroll_id, scroll="5m")
            scroll_id = res["_scroll_id"]
        self._client.clear_scroll(scroll_id=scroll_id)
        return results

    # ==================================================================
    # Keyword search — used by Algebra / API
    # ==================================================================

    def search_keywords(
        self, keywords: str, kw_type: KWType, max_hits: int = 15,
    ) -> Iterator[Hit]:
        """Relevance-ranked keyword search (ES ``match`` query)."""
        index, field = self._resolve_kw(kw_type, exact=False)
        body: dict[str, Any] = {
            "from": 0, "size": max_hits,
            "query": {"match": {field: keywords}},
        }
        yield from self._run_kw_query(index, body)

    def exact_search_keywords(
        self, keywords: str, kw_type: KWType, max_hits: int = 15,
    ) -> Iterator[Hit]:
        """Exact-term keyword search (ES ``term`` query)."""
        index, field = self._resolve_kw(kw_type, exact=True)
        body: dict[str, Any] = {
            "from": 0, "size": max_hits,
            "query": {"term": {field: keywords}},
        }
        yield from self._run_kw_query(index, body)

    def fuzzy_keyword_match(
        self, keywords: str, max_hits: int = 15,
    ) -> Iterator[Hit]:
        """Fuzzy keyword search on the text index."""
        body: dict[str, Any] = {
            "from": 0, "size": max_hits,
            "query": {"match": {
                "text": {"query": keywords, "fuzziness": "AUTO"},
            }},
        }
        yield from self._run_kw_query("text", body)

    # ==================================================================
    # Internals
    # ==================================================================

    @staticmethod
    def _resolve_kw(kw_type: KWType, *, exact: bool) -> tuple[str, str]:
        """Return ``(index_name, field_name)`` for a keyword scope."""
        if kw_type == KWType.KW_CONTENT:
            return ("text", "text")
        if kw_type == KWType.KW_SCHEMA:
            return ("profile", "columnNameNA" if exact else "columnName")
        if kw_type == KWType.KW_ENTITIES:
            return ("profile", "entities")
        if kw_type == KWType.KW_TABLE:
            return ("profile", "sourceNameNA" if exact else "sourceName")
        raise ValueError(f"Unknown KWType: {kw_type}")

    def _run_kw_query(self, index: str, body: dict) -> Iterator[Hit]:
        """Execute an ES search and yield Hits."""
        filter_path = [
            "hits.total", "hits.hits._source.id", "hits.hits._score",
            "hits.hits._source.dbName", "hits.hits._source.sourceName",
            "hits.hits._source.columnName",
        ]
        res = self._client.search(
            index=index, body=body, filter_path=filter_path,
        )
        total = res.get("hits", {}).get("total", 0)
        if isinstance(total, dict):
            total = total.get("value", 0)
        if total == 0:
            return
        for el in res["hits"]["hits"]:
            s = el["_source"]
            yield Hit(
                nid=str(s.get("id", el.get("_id", ""))),
                db_name=s.get("dbName", ""),
                source_name=s["sourceName"],
                field_name=s["columnName"],
                score=el["_score"],
            )

    def close(self) -> None:
        """Close the underlying transport."""
        self._client.close()
''')
print("elastic_store.py written")
