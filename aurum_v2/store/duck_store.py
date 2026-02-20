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

import contextlib
import logging
import re
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

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
        self._fts_ready = False

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
        """(Re)create the FTS indexes on text_index and profile."""
        self._con.execute("INSTALL fts; LOAD fts;")
        
        # 1. Content Index (For data values) — English stemming is appropriate here
        with contextlib.suppress(duckdb.CatalogException):
            self._con.execute("PRAGMA drop_fts_index('text_index');")
        self._con.execute(
            "PRAGMA create_fts_index("
            "  'text_index', 'nid', 'text',"
            "  stemmer='english', stopwords='english'"
            ");"
        )

        # 2. Metadata Index — English stemmer so "experiment" matches
        #    "experimental".  stopwords='none' so identifiers like
        #    "Id", "Name", "No" remain findable.
        with contextlib.suppress(duckdb.CatalogException):
            self._con.execute("PRAGMA drop_fts_index('profile');")
        self._con.execute(
            "PRAGMA create_fts_index("
            "  'profile', 'nid', 'source_name', 'column_name', 'entities',"
            "  stemmer='english', stopwords='none'"
            ");"
        )
        self._fts_ready = True
        logger.info("FTS indexes rebuilt on text_index and profile")

    def _ensure_fts(self) -> None:
        """Lazily rebuild FTS indexes on first search if not already done."""
        if self._fts_ready:
            return
        # Check if the FTS index already exists (from a prior bulk_insert)
        tables = self._con.execute(
            "SELECT table_name FROM information_schema.tables"
            " WHERE table_name LIKE 'fts_main_profile%'"
        ).fetchall()
        if tables:
            self._con.execute("INSTALL fts; LOAD fts;")
            self._fts_ready = True
        else:
            # Build from scratch (e.g. after --rebuild or fresh connection)
            self._rebuild_fts()

    # ==================================================================
    # Ingestion  (single-writer — call from main thread only)
    # ==================================================================

    def bulk_insert_profiles(
        self,
        profiles: list[ColumnProfile],
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
                unique_vals = list(dict.fromkeys(p.raw_values))
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

    _FETCH_CHUNK = 10_000

    def get_all_fields_name(self) -> Iterator[tuple[str, str]]:
        """Chunked streaming of ``(nid, column_name)`` from profile."""
        cursor = self._con.execute(
            "SELECT nid, column_name FROM profile"
        )
        while True:
            chunk = cursor.fetchmany(self._FETCH_CHUNK)
            if not chunk:
                break
            yield from chunk

    def get_all_fields(self) -> Iterator[tuple[str, str, str, str, int, int, str]]:
        """Chunked streaming of full field metadata from profile."""
        cursor = self._con.execute(
            "SELECT nid, db_name, source_name, column_name,"
            "       total_values, unique_values, data_type"
            "  FROM profile"
        )
        while True:
            chunk = cursor.fetchmany(self._FETCH_CHUNK)
            if not chunk:
                break
            yield from chunk

    def get_all_mh_text_signatures(self) -> Iterator[tuple[str, list[int]]]:
        """Chunked streaming of ``(nid, minhash_array)`` for text columns."""
        cursor = self._con.execute(
            "SELECT nid, minhash FROM profile"
            " WHERE data_type = 'T' AND minhash IS NOT NULL"
        )
        while True:
            chunk = cursor.fetchmany(self._FETCH_CHUNK)
            if not chunk:
                break
            for nid, mh in chunk:
                if mh:
                    yield (nid, list(mh))

    def get_all_fields_num_signatures(
        self,
    ) -> Iterator[tuple[str, tuple[float, float, float, float]]]:
        """Chunked streaming of ``(nid, (median, iqr, min, max))`` for numeric cols."""
        cursor = self._con.execute(
            "SELECT nid, median, iqr, min_value, max_value"
            "  FROM profile WHERE data_type = 'N'"
        )
        while True:
            chunk = cursor.fetchmany(self._FETCH_CHUNK)
            if not chunk:
                break
            for nid, med, iq, mn, mx in chunk:
                yield (nid, (med, iq, mn, mx))

    # ==================================================================
    # Keyword search — DuckDB FTS (used by Algebra / API)
    # ==================================================================

    def search_keywords(
        self, keywords: str, kw_type: KWType, max_hits: int = 15,
    ) -> Iterator[Hit]:
        """Relevance-ranked keyword search via DuckDB FTS."""
        self._ensure_fts()
        if kw_type == KWType.KW_CONTENT:
            yield from self._fts_search(keywords, max_hits)
        elif kw_type == KWType.KW_SCHEMA:
            yield from self._profile_fts_search("column_name", keywords, max_hits)
        elif kw_type == KWType.KW_ENTITIES:
            yield from self._profile_fts_search("entities", keywords, max_hits)
        elif kw_type == KWType.KW_TABLE:
            yield from self._profile_fts_search("source_name", keywords, max_hits)

    def exact_search_keywords(
        self, keywords: str, kw_type: KWType, max_hits: int = 15,
    ) -> Iterator[Hit]:
        """Exact-match keyword search."""
        if kw_type == KWType.KW_CONTENT:
            rows = self._con.execute(
                "SELECT t.nid, p.db_name, p.source_name, p.column_name"
                "  FROM text_index t"
                "  JOIN profile p ON t.nid = p.nid"
                " WHERE regexp_matches(t.text, ?)"
                " LIMIT ?",
                [rf"\b{re.escape(keywords)}\b", max_hits],
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
        """
        True fuzzy matching using Levenshtein edit distance.
        Allows up to 2 character edits (typos).
        """
        try:
            rows = self._con.execute(
                """
                WITH tokens AS (
                    SELECT nid, unnest(string_split(text, ' ')) AS token
                    FROM text_index
                ),
                scored AS (
                    -- Calculate the minimum edit distance for any token in the text
                    SELECT nid, min(levenshtein(token, ?)) AS edits
                    FROM tokens
                    GROUP BY nid
                    HAVING edits <= 2  -- Allow a maximum of 2 typos
                )
                SELECT s.nid, p.db_name, p.source_name, p.column_name, s.edits
                FROM scored s
                JOIN profile p ON s.nid = p.nid
                ORDER BY s.edits ASC  -- Ascending because lower edit distance is better
                LIMIT ?
                """,
                [keywords, max_hits],
            ).fetchall()
        except duckdb.Error as e:
            logger.warning("Fuzzy query failed: %s", e)
            return
            
        for nid, db, src, col, score in rows:
            # We convert the edit distance back to a pseudo-score (e.g., 1.0 for exact, lower for more edits)
            # so it matches the expected Hit format.
            pseudo_score = 1.0 / (1.0 + float(score)) 
            yield Hit(nid=nid, db_name=db, source_name=src, field_name=col, score=pseudo_score)

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

    def _profile_fts_search(self, field: str, keywords: str, max_hits: int) -> Iterator[Hit]:
        """FTS search on profile metadata, with ILIKE fallback.

        DuckDB FTS matches whole stemmed tokens — it won't find
        "experimental" when the user types "experiment".  For identifier
        fields (column_name, source_name) an ILIKE substring match is a
        better UX, so we try FTS first then fall back.
        """
        hits: list[Hit] = []

        # 1. Try FTS (BM25 ranked)
        try:
            rows = self._con.execute(
                f"SELECT nid, db_name, source_name, column_name,"
                f"       fts_main_profile.match_bm25(nid, ?, fields := '{field}') AS score"
                f"  FROM profile"
                f" WHERE score IS NOT NULL"
                f" ORDER BY score DESC"
                f" LIMIT ?",
                [keywords, max_hits],
            ).fetchall()
            for nid, db, src, col, score in rows:
                hits.append(Hit(nid=nid, db_name=db, source_name=src,
                                field_name=col, score=float(score)))
        except duckdb.Error as e:
            logger.warning("Profile FTS query failed: %s", e)

        # 2. Fill remaining slots with ILIKE substring matches
        seen = {h.nid for h in hits}
        remaining = max_hits - len(hits)
        if remaining > 0:
            try:
                rows = self._con.execute(
                    f"SELECT nid, db_name, source_name, column_name"
                    f"  FROM profile"
                    f" WHERE {field} ILIKE ?"
                    f" LIMIT ?",
                    [f"%{keywords}%", remaining + len(seen)],
                ).fetchall()
                for nid, db, src, col in rows:
                    if nid not in seen:
                        hits.append(Hit(nid=nid, db_name=db, source_name=src,
                                        field_name=col, score=0.5))
                        seen.add(nid)
                        if len(hits) >= max_hits:
                            break
            except duckdb.Error as e:
                logger.warning("ILIKE fallback failed: %s", e)

        yield from hits

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

    # ==================================================================
    # Path resolution
    # ==================================================================

    def get_path_of(self, nid: str) -> str:
        """Return the data-source path (S3 URI or local) for column *nid*."""
        row = self._con.execute(
            "SELECT path FROM profile WHERE nid = ? LIMIT 1", [nid],
        ).fetchone()
        return row[0] if row and row[0] else ""

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._con.close()
