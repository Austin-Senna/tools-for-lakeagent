"""
Elasticsearch store handler — reads column profiles and performs keyword searches.

Direct port of ``modelstore/elasticstore.py`` from the legacy codebase.
The only change is that configuration is injected via :class:`AurumConfig`
instead of module‑level globals.
"""

from __future__ import annotations

from enum import Enum
from collections.abc import Iterator
from typing import TYPE_CHECKING

from elasticsearch import Elasticsearch

from aurum_v2.models.hit import Hit

if TYPE_CHECKING:
    from aurum_v2.config import AurumConfig

__all__ = ["KWType", "StoreHandler"]


# ---------------------------------------------------------------------------
# Keyword‑search scope
# ---------------------------------------------------------------------------

class KWType(Enum):
    """Which Elasticsearch field to query against."""

    KW_CONTENT = 0      # full‑text index ("text")
    KW_SCHEMA = 1       # column name ("profile.columnName")
    KW_ENTITIES = 2     # entity annotations
    KW_TABLE = 3        # table / source name
    KW_METADATA = 4     # metadata annotations


# ---------------------------------------------------------------------------
# StoreHandler
# ---------------------------------------------------------------------------

class StoreHandler:
    """Thin wrapper around an Elasticsearch client.

    Parameters
    ----------
    config : AurumConfig
        System configuration carrying ``es_host`` and ``es_port``.
    """

    def __init__(self, config: AurumConfig) -> None:
        self._config = config
        self._client: Elasticsearch = Elasticsearch(
            [{"host": config.es_host, "port": config.es_port}]
        )

    # ------------------------------------------------------------------
    # Field / profile retrieval (used by network‑building pipeline)
    # ------------------------------------------------------------------

    def get_all_fields(self) -> Iterator[tuple[str, str, str, str, int, int, str]]:
        """Scroll over all documents in the ``profile`` index.

        Yields
        ------
        (nid, db_name, source_name, column_name, total_values, unique_values, data_type)
            One tuple per profiled column.
        """
        raise NotImplementedError

    def get_all_mh_text_signatures(self) -> Iterator[tuple[str, list]]:
        """Yield ``(nid, minhash_signature_array)`` for every text column."""
        raise NotImplementedError

    def get_all_fields_num_signatures(
        self,
    ) -> Iterator[tuple[str, tuple[float, float, float, float]]]:
        """Yield ``(nid, (median, iqr, min_val, max_val))`` for every numeric column."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Keyword search (used by Algebra)
    # ------------------------------------------------------------------

    def search_keywords(
        self, keywords: str, elasticfieldname: KWType, max_hits: int = 15
    ) -> Iterator[Hit]:
        """Fuzzy / relevance‑ranked keyword search.

        Maps *elasticfieldname* to the correct ES index and query body,
        then yields :class:`Hit` objects with ES ``_score``.
        """
        raise NotImplementedError

    def exact_search_keywords(
        self, keywords: str, elasticfieldname: KWType, max_hits: int = 15
    ) -> Iterator[Hit]:
        """Exact‑term keyword search (uses ES ``term`` query)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def get_path_of(self, nid: str) -> str:
        """Return the filesystem path to the data source that contains *nid*.

        Performs a point query on the ES ``profile`` index for the ``path``
        field.
        """
        raise NotImplementedError

    def suggest_schema(self, suggestion_string: str, max_hits: int = 5):
        """Auto‑complete suggestions for column names (ES suggest API)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the ES client (placeholder)."""
        raise NotImplementedError
