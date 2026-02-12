"""
Algebra — the user‑facing discovery query API.

Provides keyword search, neighbour traversal, set operations, path finding,
and BFS‑based graph traversal.  All methods accept flexible "general input"
(nid int, table‑name string, Hit, or DRS) and return :class:`DRS`.

Line‑for‑line logic port of ``algebra.py`` (the ``Algebra`` class) from the
legacy codebase.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, List, Optional

from aurum_v2.models.drs import DRS
from aurum_v2.models.hit import Hit, compute_field_id
from aurum_v2.models.relation import DRSMode, OP, Operation, Relation
from aurum_v2.store.elastic_store import KWType

if TYPE_CHECKING:
    from aurum_v2.graph.field_network import FieldNetwork
    from aurum_v2.store.elastic_store import StoreHandler

__all__ = ["Algebra"]


class Algebra:
    """Core discovery algebra.

    Parameters
    ----------
    network : FieldNetwork
        The loaded column‑relation graph.
    store_client : StoreHandler
        Elasticsearch connection for keyword searches and path lookups.
    """

    def __init__(self, network: "FieldNetwork", store_client: "StoreHandler") -> None:
        self._network = network
        self._store_client = store_client

    # ==================================================================
    # Basic search API
    # ==================================================================

    def search(self, kw: str, kw_type: KWType, max_results: int = 10) -> DRS:
        """Keyword search over the ES store.

        Parameters
        ----------
        kw : str
            Search term.
        kw_type : KWType
            Where to search (content, schema, table, entities).
        max_results : int
            Maximum hits.

        Returns a DRS with ``OP.KW_LOOKUP`` provenance.
        """
        raise NotImplementedError

    def exact_search(self, kw: str, kw_type: KWType, max_results: int = 10) -> DRS:
        """Like :meth:`search` but exact‑match only."""
        raise NotImplementedError

    # Convenience wrappers matching legacy names:

    def search_content(self, kw: str, max_results: int = 10) -> DRS:
        return self.search(kw, KWType.KW_CONTENT, max_results)

    def search_attribute(self, kw: str, max_results: int = 10) -> DRS:
        return self.search(kw, KWType.KW_SCHEMA, max_results)

    def search_exact_attribute(self, kw: str, max_results: int = 10) -> DRS:
        return self.exact_search(kw, KWType.KW_SCHEMA, max_results)

    def search_table(self, kw: str, max_results: int = 10) -> DRS:
        return self.search(kw, KWType.KW_TABLE, max_results)

    # ==================================================================
    # Neighbour traversal API
    # ==================================================================

    def _neighbor_search(self, input_data, relation: Relation) -> DRS:
        """Generic neighbour expansion through *relation*.

        1. Convert *input_data* to DRS.
        2. Create a carrier DRS and absorb input's provenance.
        3. If in TABLE mode, expand to field‑level DRS.
        4. For each Hit, call ``network.neighbors_id`` and absorb the result.
        """
        raise NotImplementedError

    def content_similar_to(self, general_input) -> DRS:
        return self._neighbor_search(general_input, Relation.CONTENT_SIM)

    def schema_similar_to(self, general_input) -> DRS:
        return self._neighbor_search(general_input, Relation.SCHEMA_SIM)

    def pkfk_of(self, general_input) -> DRS:
        return self._neighbor_search(general_input, Relation.PKFK)

    # ==================================================================
    # Transitive‑closure / BFS traversal
    # ==================================================================

    def traverse(self, a: DRS, primitive: Relation, max_hops: int = 2) -> DRS:
        """Breadth‑first multi‑hop traversal following *primitive*.

        Each hop expands the fringe via ``_neighbor_search`` and unions the
        results into the accumulator.
        """
        raise NotImplementedError

    # ==================================================================
    # Path finding
    # ==================================================================

    def paths(
        self,
        drs_a: DRS,
        drs_b: DRS,
        relation: Relation = Relation.PKFK,
        max_hops: int = 2,
        lean_search: bool = False,
    ) -> DRS:
        """Find transitive paths between elements of *drs_a* and *drs_b*.

        For every ``(h1, h2)`` in the Cartesian product of *drs_a* × *drs_b*:

        * **FIELDS mode** → ``network.find_path_hit(h1, h2, relation)``
        * **TABLE mode**  → ``network.find_path_table(h1, h2, relation, self)``

        Results are absorbed into a single output DRS.
        """
        raise NotImplementedError

    # ==================================================================
    # Combiner (set operations)
    # ==================================================================

    def intersection(self, a: DRS, b: DRS) -> DRS:
        raise NotImplementedError

    def union(self, a: DRS, b: DRS) -> DRS:
        raise NotImplementedError

    def difference(self, a: DRS, b: DRS) -> DRS:
        raise NotImplementedError

    # ==================================================================
    # General‑input helpers
    # ==================================================================

    def make_drs(self, general_input) -> DRS:
        """Coerce any supported input type into a DRS.

        Accepts: ``int`` / ``str`` (nid), ``str`` (table name), ``tuple``
        ``(db, source, field)``, ``Hit``, ``DRS``, or ``list`` thereof.
        """
        raise NotImplementedError

    def drs_from_table_hit(self, hit: Hit) -> DRS:
        """Expand a single Hit into a DRS of all columns in the same table.

        Provenance: ``Operation(OP.TABLE, params=[hit])``.
        """
        raise NotImplementedError

    def _drs_from_table_hit_lean_no_provenance(self, hit: Hit) -> DRS:
        """Like :meth:`drs_from_table_hit` but with ``lean_drs=True``."""
        raise NotImplementedError

    def _general_to_drs(self, general_input) -> DRS:
        """Convert nid / table‑name / tuple / Hit / DRS → DRS.

        Conversion rules (checked in order):
        1. Already a DRS → pass through.
        2. ``None`` → empty carrier DRS.
        3. Integer (or string that parses as int) → look up nid via
           ``network.get_info_for``, wrap as single‑Hit DRS.
        4. String → interpret as table name → ``get_hits_from_table``.
        5. Non‑Hit tuple ``(db, source, field)`` → compute nid → Hit.
        6. Hit with empty ``field_name`` → table mode expansion.
        7. Hit with field → single‑element ORIGIN DRS.
        """
        raise NotImplementedError

    def _general_to_field_drs(self, general_input) -> DRS:
        """Expand a table‑mode DRS to field‑level by absorbing all table columns."""
        raise NotImplementedError

    def _nid_to_hit(self, nid: int | str) -> Hit:
        """Look up metadata for *nid* and return a Hit (score = 0)."""
        raise NotImplementedError

    def _node_to_hit(self, node: tuple) -> Hit:
        """Given ``(db, source, field)``, compute nid and return a Hit."""
        raise NotImplementedError

    def _hit_to_drs(self, hit: Hit, table_mode: bool = False) -> DRS:
        """Wrap a Hit in a DRS.  If *table_mode*, expand to whole table."""
        raise NotImplementedError

    @staticmethod
    def _assert_same_mode(a: DRS, b: DRS) -> None:
        assert a.mode == b.mode, "Input DRS objects are not in the same mode"

    @staticmethod
    def _represents_int(s) -> bool:
        try:
            int(s)
            return True
        except (ValueError, TypeError):
            return False
