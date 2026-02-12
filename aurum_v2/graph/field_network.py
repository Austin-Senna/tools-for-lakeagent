"""
FieldNetwork — the core multi‑relation column graph.

Wraps a ``networkx.MultiGraph`` where:

* **Nodes** = column *nid* strings (CRC32 of db+source+field).
  Each node carries a ``cardinality`` attribute (unique / total).
* **Edges** are keyed by :class:`Relation` and carry a ``score`` dict.

This is a faithful port of ``knowledgerepr/fieldnetwork.py``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import networkx as nx

from aurum_v2.models.drs import DRS
from aurum_v2.models.hit import Hit
from aurum_v2.models.relation import OP, Relation

__all__ = ["FieldNetwork", "serialize_network", "deserialize_network"]


class FieldNetwork:
    """Multi‑relation graph over profiled columns.

    Parameters
    ----------
    graph : nx.MultiGraph | None
        Pre‑existing graph (used by :func:`deserialize_network`).
    id_names : dict | None
        ``nid → (db_name, source_name, field_name, data_type)``
    source_ids : dict | None
        ``source_name → [nid, …]``
    """

    def __init__(
        self,
        graph: nx.MultiGraph | None = None,
        id_names: dict | None = None,
        source_ids: dict | None = None,
    ) -> None:
        self._graph: nx.MultiGraph = graph if graph is not None else nx.MultiGraph()
        self._id_names: dict[str, tuple[str, str, str, str]] = id_names or {}
        self._source_ids: dict[str, list[str]] = source_ids or defaultdict(list)

    # ------------------------------------------------------------------
    # Metadata queries
    # ------------------------------------------------------------------

    def graph_order(self) -> int:
        """Number of registered columns (nodes)."""
        return len(self._id_names)

    def get_number_tables(self) -> int:
        return len(self._source_ids)

    def iterate_ids(self) -> Iterator[str]:
        """Yield every registered *nid*."""
        yield from self._id_names.keys()

    def iterate_ids_text(self) -> Iterator[str]:
        """Yield nids of text‑typed columns only."""
        for nid, (_, _, _, dtype) in self._id_names.items():
            if dtype == "T":
                yield nid

    def iterate_values(self) -> Iterator[tuple[str, str, str, str]]:
        """Yield ``(db_name, source_name, field_name, data_type)`` for every column."""
        yield from self._id_names.values()

    def get_fields_of_source(self, source: str) -> list[str]:
        """Return list of *nid*s belonging to *source*."""
        return self._source_ids[source]

    def get_data_type_of(self, nid: str) -> str:
        """``'T'`` for text, ``'N'`` for numeric."""
        return self._id_names[nid][3]

    def get_info_for(self, nids: list[str]) -> list[tuple[str, str, str, str]]:
        """Return ``[(nid, db_name, source_name, field_name), …]``."""
        info = []
        for nid in nids:
            db_name, source_name, field_name, _ = self._id_names[nid]
            info.append((nid, db_name, source_name, field_name))
        return info

    def get_hits_from_table(self, table: str) -> list[Hit]:
        """Return all :class:`Hit` objects for columns in *table*."""
        nids = self.get_fields_of_source(table)
        info = self.get_info_for(nids)
        return [Hit(nid, db, sn, fn, 0) for nid, db, sn, fn in info]

    def get_cardinality_of(self, nid: str) -> float:
        """Return the node's cardinality ratio (``unique / total``), or 0."""
        card = self._graph.nodes[nid].get("cardinality")
        return card if card is not None else 0

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def init_meta_schema(
        self,
        fields: Iterator[tuple[str, str, str, str, int, int, str]],
    ) -> None:
        """Populate the graph skeleton from profiled column tuples.

        Each element of *fields* is
        ``(nid, db_name, source_name, field_name, total_values, unique_values, data_type)``.

        Creates one node per column with a ``cardinality`` attribute and
        populates ``_id_names`` and ``_source_ids``.
        """
        raise NotImplementedError

    def add_field(self, nid: str, cardinality: float | None = None) -> str:
        """Add a single graph node for column *nid* with optional cardinality."""
        raise NotImplementedError

    def add_relation(
        self,
        node_src: str,
        node_target: str,
        relation: Relation,
        score: float,
    ) -> None:
        """Add (or update) a typed edge between two columns.

        The ``relation`` serves as the MultiGraph *key*, and ``score`` is
        stored in the edge‑data dict.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Relation → OP mapping
    # ------------------------------------------------------------------

    @staticmethod
    def get_op_from_relation(relation: Relation) -> OP:
        """Map a :class:`Relation` edge type to the corresponding provenance :class:`OP`."""
        _map = {
            Relation.CONTENT_SIM: OP.CONTENT_SIM,
            Relation.ENTITY_SIM: OP.ENTITY_SIM,
            Relation.PKFK: OP.PKFK,
            Relation.SCHEMA: OP.TABLE,
            Relation.SCHEMA_SIM: OP.SCHEMA_SIM,
            Relation.MEANS_SAME: OP.MEANS_SAME,
            Relation.MEANS_DIFF: OP.MEANS_DIFF,
            Relation.SUBCLASS: OP.SUBCLASS,
            Relation.SUPERCLASS: OP.SUPERCLASS,
            Relation.MEMBER: OP.MEMBER,
            Relation.CONTAINER: OP.CONTAINER,
        }
        return _map[relation]

    # ------------------------------------------------------------------
    # Neighbor traversal
    # ------------------------------------------------------------------

    def neighbors_id(self, hit: Hit | str, relation: Relation) -> DRS:
        """Return a :class:`DRS` of all neighbours connected by *relation*.

        Each neighbour is turned into a :class:`Hit` with the edge ``score``.
        The returned DRS carries provenance wired as
        ``Operation(op_from_relation, params=[hit])``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Path finding — field level
    # ------------------------------------------------------------------

    def find_path_hit(
        self,
        source: Hit,
        target: Hit,
        relation: Relation,
        max_hops: int = 5,
    ) -> DRS:
        """DFS path search between two *columns* via *relation*.

        On success, returns a DRS whose provenance DAG contains the hop‑by‑hop
        chain::

            ORIGIN(source) → PKFK → intermediate₁ → PKFK → … → target

        On failure returns an empty carrier DRS.

        .. note:: ``max_hops`` is hard‑coded to 5 in the legacy code regardless
           of the parameter value passed in.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Path finding — table level
    # ------------------------------------------------------------------

    def find_path_table(
        self,
        source: Hit,
        target: Hit,
        relation: Relation,
        api: object,
        max_hops: int = 3,
        lean_search: bool = False,
    ) -> DRS:
        """DFS path search between two *tables* via *relation*.

        Unlike :meth:`find_path_hit`, this performs a **table‑level** DFS:
        for each neighbour column, all sibling columns in the same table are
        expanded (via ``api.drs_from_table_hit``), and the provenance chain
        records both the cross‑table PKFK hop *and* the same‑table OP.TABLE
        link.

        Parameters
        ----------
        api : Algebra | API
            Required to expand table neighbours via ``drs_from_table_hit``.
        lean_search : bool
            If ``True``, use ``_drs_from_table_hit_lean_no_provenance`` to
            skip provenance construction in table expansion (faster).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Degree / diagnostics
    # ------------------------------------------------------------------

    def fields_degree(self, topk: int) -> list[tuple[str, int]]:
        """Return the *topk* highest-degree nodes in the graph.

        Port of ``FieldNetwork.fields_degree`` from
        ``knowledgerepr/fieldnetwork.py``.

        Algorithm:

        1. Compute the degree of every node via ``G.degree()``.
        2. Sort by degree descending.
        3. Return the first *topk* entries as ``(nid, degree)`` tuples.

        Useful for diagnostics — high-degree columns are typically join
        hubs (e.g. ``county``, ``state``, ``year``).

        Parameters
        ----------
        topk : int
            Number of top-degree nodes to return.

        Returns
        -------
        list[tuple[str, int]]
            ``[(nid, degree), ...]`` sorted descending by degree.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Enumeration / debug helpers
    # ------------------------------------------------------------------

    def enumerate_relation(
        self, relation: Relation, as_str: bool = True
    ) -> Iterator:
        """Iterate over all distinct edge pairs for *relation*."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal graph access (used by serialisation & CSV export)
    # ------------------------------------------------------------------

    def _get_underlying_repr_graph(self) -> nx.MultiGraph:
        return self._graph

    def _get_underlying_repr_id_to_field_info(self) -> dict:
        return self._id_names

    def _get_underlying_repr_table_to_ids(self) -> dict:
        return self._source_ids


# ======================================================================
# Serialisation (module‑level functions, matching legacy layout)
# ======================================================================

def serialize_network(network: FieldNetwork, path: str) -> None:
    """Pickle the three core artefacts to *path*.

    Creates ``graph.pickle``, ``id_info.pickle``, ``table_ids.pickle``.
    """
    raise NotImplementedError


def deserialize_network(path: str) -> FieldNetwork:
    """Reconstruct a :class:`FieldNetwork` from pickled artefacts at *path*."""
    raise NotImplementedError
