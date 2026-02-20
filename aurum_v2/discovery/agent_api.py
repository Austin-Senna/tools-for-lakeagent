"""
Agent-friendly API — 3 functions for AI agent consumption.

Wraps the full Algebra into a minimal surface:

* ``search_value(query)``  — find columns by data content
* ``search_field(query)``  — find columns/tables by attribute or table name
* ``neighbor(table, relation)`` — graph traversal, cross-table only

All functions return plain ``list[dict]`` — no Aurum internals leak out.
"""

from __future__ import annotations

from typing import Literal

from aurum_v2.discovery.api import API, init_system_duck
from aurum_v2.models.relation import Relation

__all__ = ["AgentAPI", "init_agent"]

# Map user-facing relation strings → internal Relation enum
_RELATION_MAP: dict[str, Relation] = {
    "content": Relation.CONTENT_SIM,
    "schema": Relation.SCHEMA_SIM,
    "pkfk": Relation.PKFK,
}


def _hit_to_dict(hit) -> dict:
    """Convert a Hit to a plain dict."""
    return {
        "nid": hit.nid,
        "table": hit.source_name,
        "column": hit.field_name,
        "score": round(hit.score, 4),
    }


def _dedup_tables(dicts: list[dict]) -> list[dict]:
    """Collapse column-level results to unique tables, keeping best score."""
    best: dict[str, dict] = {}
    for d in dicts:
        t = d["table"]
        if t not in best or d["score"] > best[t]["score"]:
            best[t] = d
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


class AgentAPI:
    """Minimal API surface for AI agent tool-calling.

    Parameters
    ----------
    api : API
        Fully initialised Aurum API instance.
    """

    def __init__(self, api: API) -> None:
        self._api = api

    # ------------------------------------------------------------------
    # 1. search_value
    # ------------------------------------------------------------------

    def search_value(
        self,
        query: str,
        top_k: int = 10,
        dedup_tables: bool = False,
    ) -> list[dict]:
        """Find columns whose *data values* match ``query``.

        Parameters
        ----------
        query : str
            Value to search for in profiled column data.
        top_k : int
            Maximum results to return.
        dedup_tables : bool
            If True, collapse to unique tables keeping the best score.

        Returns list of ``{nid, table, column, score}`` dicts.
        """
        drs = self._api.search_content(query, top_k)
        results = [_hit_to_dict(h) for h in drs]
        if dedup_tables:
            results = _dedup_tables(results)
        return results[:top_k]

    # ------------------------------------------------------------------
    # 2. search_field
    # ------------------------------------------------------------------

    def search_field(
        self,
        query: str,
        top_k: int = 10,
        dedup_tables: bool = False,
    ) -> list[dict]:
        """Find columns/tables whose *name* matches ``query``.

        Combines attribute search (column names) and table search
        (source names), deduplicates, and returns the top results
        sorted by score.

        Parameters
        ----------
        query : str
            Name to search for in column/table names.
        top_k : int
            Maximum results to return.
        dedup_tables : bool
            If True, collapse to unique tables keeping the best score.

        Returns list of ``{nid, table, column, score}`` dicts.
        """
        attr_drs = self._api.search_attribute(query, top_k)
        tbl_drs = self._api.search_table(query, top_k)

        seen: set[str] = set()
        results: list[dict] = []
        for drs in (attr_drs, tbl_drs):
            for hit in drs:
                d = _hit_to_dict(hit)
                if d["nid"] not in seen:
                    seen.add(d["nid"])
                    results.append(d)

        results.sort(key=lambda x: x["score"], reverse=True)
        if dedup_tables:
            results = _dedup_tables(results)
        return results[:top_k]

    # ------------------------------------------------------------------
    # 3. neighbor
    # ------------------------------------------------------------------

    def neighbor(
        self,
        input: str,
        relation: Literal["content", "schema", "pkfk"] = "pkfk",
        top_k: int | None = None,
    ) -> list[dict]:
        """Return cross-table neighbors via ``relation``.

        Parameters
        ----------
        input : str
            Table name (e.g. ``"dataset/file.txt"``) **or** column nid
            (e.g. ``"dataset/file.txt.col_name"``).  Both are accepted.
        relation : str
            Edge type: ``"content"``, ``"schema"``, or ``"pkfk"``.
        top_k : int | None
            If set, cap the number of results returned.

        Returns list of ``{nid, table, column, score}`` dicts,
        sorted by score descending.  Same-table results are excluded.
        """
        rel = _RELATION_MAP.get(relation)
        if rel is None:
            raise ValueError(
                f"Unknown relation {relation!r}. "
                f"Choose from: {', '.join(_RELATION_MAP)}"
            )

        drs = self._api._neighbor_search(input, rel)

        # Determine source table for same-table filtering.
        # nid format: "source_name.field_name" — table is everything before last dot.
        input_table = input.rsplit(".", 1)[0] if "." in input else input

        results: list[dict] = []
        seen: set[str] = set()
        for hit in drs:
            if hit.source_name == input_table:
                continue  # skip same-table
            d = _hit_to_dict(hit)
            if d["nid"] not in seen:
                seen.add(d["nid"])
                results.append(d)

        results.sort(key=lambda x: x["score"], reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results


def init_agent(
    model_path: str,
    db_path: str = "aurum.db",
) -> AgentAPI:
    """One-liner to spin up an agent-ready API.

    >>> agent = init_agent("/path/to/model", "aurum.db")
    >>> agent.search_value("California")
    """
    api = init_system_duck(model_path, db_path)
    return AgentAPI(api)
