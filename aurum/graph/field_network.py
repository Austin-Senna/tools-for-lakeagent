"""
Field network — a multi-relation graph over columns in a data lake.

This is the modern rewrite of ``aurum/knowledgerepr/fieldnetwork.py``.

Architecture
------------
- Nodes  = column IDs (``ColumnId.nid`` strings — 16-hex-char MD5 prefix)
- Node attrs = ``cardinality`` (``unique / total``)
- Edges  = typed by ``Relation``, with a ``score`` attribute
- Backend = ``networkx.MultiGraph`` (same as Aurum)

Serialisation uses ``pickle`` (Aurum used ``nx.write_gpickle``).
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import networkx as nx

from aurum.graph.relations import Relation, OP
from aurum.profiler.column_profiler import ColumnId, ColumnProfile


# ---------------------------------------------------------------------------
# Hit — lightweight result object (ported from apiutils.Hit namedtuple)
# ---------------------------------------------------------------------------

class Hit:
    """A single search / traversal result.

    Mirrors Aurum's ``Hit(nid, db_name, source_name, field_name, score)``
    but as a proper class with hashing on ``nid``.
    """

    __slots__ = ("nid", "db_name", "source_name", "field_name", "score")

    def __init__(
        self,
        nid: str,
        db_name: str,
        source_name: str,
        field_name: str,
        score: float = 0.0,
    ) -> None:
        self.nid = nid
        self.db_name = db_name
        self.source_name = source_name
        self.field_name = field_name
        self.score = score

    # Hashing / equality on nid only (same as Aurum)
    def __hash__(self) -> int:
        return hash(self.nid)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Hit):
            return self.nid == other.nid
        return NotImplemented

    def __repr__(self) -> str:
        return f"Hit({self.source_name}.{self.field_name}, score={self.score:.3f})"

    def __str__(self) -> str:
        return f"{self.db_name}:{self.source_name}.{self.field_name}"


# ---------------------------------------------------------------------------
# FieldNetwork
# ---------------------------------------------------------------------------

class FieldNetwork:
    """Multi-relation graph over columns in a data lake.

    Direct port of ``aurum/knowledgerepr/fieldnetwork.py :: FieldNetwork``.
    """

    def __init__(
        self,
        graph: nx.MultiGraph | None = None,
        id_names: dict[str, tuple[str, str, str, str]] | None = None,
        source_ids: dict[str, list[str]] | None = None,
    ) -> None:
        self._graph: nx.MultiGraph = graph or nx.MultiGraph()
        # nid → (db_name, source_name, field_name, data_type)
        self._id_names: dict[str, tuple[str, str, str, str]] = id_names or {}
        # source_name → [nid, ...]
        self._source_ids: dict[str, list[str]] = source_ids or defaultdict(list)

    # ── Bulk initialisation ──────────────────────────────────────────

    def init_from_profiles(self, profiles: Iterator[ColumnProfile] | list[ColumnProfile]) -> None:
        """Populate the graph skeleton from column profiles.

        Ported from ``FieldNetwork.init_meta_schema``.
        """
        for prof in profiles:
            nid = prof.col_id.nid
            self._id_names[nid] = (
                prof.col_id.db_name,
                prof.col_id.source_name,
                prof.col_id.field_name,
                prof.data_type,
            )
            self._source_ids[prof.col_id.source_name].append(nid)
            self._graph.add_node(nid, cardinality=prof.cardinality_ratio)

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def order(self) -> int:
        return len(self._id_names)

    @property
    def num_tables(self) -> int:
        return len(self._source_ids)

    def iterate_ids(self) -> Iterator[str]:
        yield from self._id_names

    def iterate_ids_by_type(self, data_type: str) -> Iterator[str]:
        for nid, (_, _, _, dt) in self._id_names.items():
            if dt == data_type:
                yield nid

    def iterate_values(self) -> Iterator[tuple[str, str, str, str]]:
        yield from self._id_names.values()

    def get_fields_of_source(self, source: str) -> list[str]:
        return self._source_ids.get(source, [])

    def get_data_type_of(self, nid: str) -> str:
        return self._id_names[nid][3]

    def get_info_for(self, nids: list[str]) -> list[tuple[str, str, str, str]]:
        """Return ``(nid, db_name, source_name, field_name)`` for each nid."""
        out: list[tuple[str, str, str, str]] = []
        for nid in nids:
            db, sn, fn, _ = self._id_names[nid]
            out.append((nid, db, sn, fn))
        return out

    def get_cardinality_of(self, nid: str) -> float:
        data = self._graph.nodes.get(nid, {})
        return data.get("cardinality", 0.0) or 0.0

    def get_hits_from_table(self, table: str) -> list[Hit]:
        nids = self.get_fields_of_source(table)
        hits: list[Hit] = []
        for nid in nids:
            db, sn, fn, _ = self._id_names[nid]
            hits.append(Hit(nid, db, sn, fn, 0.0))
        return hits

    # ── Mutation ─────────────────────────────────────────────────────

    def add_relation(
        self,
        src: Hit | str,
        tgt: Hit | str,
        relation: Relation,
        score: float,
    ) -> None:
        """Add or update an edge.  Accepts nid strings or Hit objects."""
        src_nid = src.nid if isinstance(src, Hit) else src
        tgt_nid = tgt.nid if isinstance(tgt, Hit) else tgt
        self._graph.add_edge(src_nid, tgt_nid, key=relation, score=score)

    # ── Traversal ────────────────────────────────────────────────────

    def neighbors_id(self, hit_or_nid: Hit | str, relation: Relation) -> list[Hit]:
        """Return all neighbours connected by *relation*.

        Ported from ``FieldNetwork.neighbors_id``.
        """
        nid = hit_or_nid.nid if isinstance(hit_or_nid, Hit) else str(hit_or_nid)
        results: list[Hit] = []
        if nid not in self._graph:
            return results
        for neighbour, edge_dict in self._graph[nid].items():
            if relation in edge_dict:
                score = edge_dict[relation].get("score", 0.0)
                db, sn, fn, _ = self._id_names[neighbour]
                results.append(Hit(neighbour, db, sn, fn, score))
        return results

    def find_path(
        self,
        source: Hit | str,
        target: Hit | str,
        relation: Relation,
        max_hops: int = 5,
    ) -> list[Hit]:
        """DFS path-finding between two columns.

        Simplified port of ``FieldNetwork.find_path_hit``.
        """
        src_nid = source.nid if isinstance(source, Hit) else str(source)
        tgt_nid = target.nid if isinstance(target, Hit) else str(target)

        visited: set[str] = set()

        def _dfs(current: str, depth: int) -> list[Hit] | None:
            if depth <= 0:
                return None
            visited.add(current)
            for n in self.neighbors_id(current, relation):
                if n.nid == tgt_nid:
                    return [n]
                if n.nid not in visited:
                    sub = _dfs(n.nid, depth - 1)
                    if sub is not None:
                        return [n, *sub]
            return None

        path = _dfs(src_nid, max_hops)
        if path is None:
            return []
        # Prepend source Hit
        db, sn, fn, _ = self._id_names[src_nid]
        return [Hit(src_nid, db, sn, fn, 0.0), *path]

    # ── Serialisation ────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Persist the network to disk (3 pickle files, same as Aurum)."""
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "graph.pkl", "wb") as f:
            pickle.dump(self._graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(directory / "id_info.pkl", "wb") as f:
            pickle.dump(self._id_names, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(directory / "table_ids.pkl", "wb") as f:
            pickle.dump(dict(self._source_ids), f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, directory: Path) -> FieldNetwork:
        """Deserialise from disk.  Mirrors ``fieldnetwork.deserialize_network``."""
        with open(directory / "graph.pkl", "rb") as f:
            graph = pickle.load(f)  # noqa: S301
        with open(directory / "id_info.pkl", "rb") as f:
            id_names = pickle.load(f)  # noqa: S301
        with open(directory / "table_ids.pkl", "rb") as f:
            source_ids = pickle.load(f)  # noqa: S301
        return cls(graph=graph, id_names=id_names, source_ids=source_ids)
