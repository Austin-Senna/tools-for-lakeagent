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
from elasticsearch import helpers

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

    def get_all_fields(self) -> Iterator[tuple[str, str, str, str, int, int, str]]:
        """Scroll ``profile`` index using memory-safe generators."""
        query = {"query": {"match_all": {}}}
        source_fields = [
            "dbName", "sourceName", "columnName",
            "totalValues", "uniqueValues", "dataType",
        ]
        
        for doc in helpers.scan(self._client, index="profile", query=query, _source=source_fields):
            s = doc.get("_source", {})
            yield (
                doc["_id"],
                s.get("dbName", ""),
                s.get("sourceName", ""),
                s.get("columnName", ""),
                s.get("totalValues", 0),
                s.get("uniqueValues", 0),
                s.get("dataType", "T"),
            )

    def get_all_mh_text_signatures(self) -> Iterator[tuple[str, list[int]]]:
        """Yield ``(nid, minhash_array)`` for all text columns."""
        query = {"query": {"term": {"dataType": "T"}}}
        
        for doc in helpers.scan(self._client, index="profile", query=query, _source=["minhash"]):
            s = doc.get("_source", {})
            mh = s.get("minhash", [])
            if mh:
                yield (doc["_id"], mh)

    def get_all_fields_name(self) -> Iterator[tuple[str, str]]:
        """Yield ``(nid, column_name)`` for all columns."""
        query = {"query": {"match_all": {}}}
        for doc in helpers.scan(self._client, index="profile", query=query, _source=["columnName"]):
            s = doc.get("_source", {})
            yield (doc["_id"], s.get("columnName", ""))

    def get_all_fields_num_signatures(self) -> Iterator[tuple[str, tuple[float, float, float, float]]]:
        """Yield ``(nid, (median, iqr, min_val, max_val))`` for numeric cols safely."""
        query = {"query": {"term": {"dataType": "N"}}}
        source_fields = ["median", "iqr", "minValue", "maxValue"]
        
        for doc in helpers.scan(self._client, index="profile", query=query, _source=source_fields):
            s = doc.get("_source", {})
            yield (
                doc["_id"],
                (
                    float(s.get("median", 0.0)),
                    float(s.get("iqr", 0.0)),
                    float(s.get("minValue", 0.0)),
                    float(s.get("maxValue", 0.0)),
                )
            )

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


# Backward-compat alias used by discovery.api
StoreHandler = ElasticStore
