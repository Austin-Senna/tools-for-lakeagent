"""
Algebra — the user‑facing discovery query API.

Provides keyword search, neighbour traversal, set operations, path finding,
and BFS‑based graph traversal.  All methods accept flexible "general input"
(nid int, table‑name string, Hit, or DRS) and return :class:`DRS`.

Line‑for‑line logic port of ``algebra.py`` (the ``Algebra`` class) from the
legacy codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aurum_v2.models.drs import DRS
from aurum_v2.models.hit import Hit
from aurum_v2.models.relation import Relation, OP, Operation, DRSMode
from aurum_v2.store.elastic_store import KWType


if TYPE_CHECKING:
    from aurum_v2.graph.field_network import FieldNetwork
    from aurum_v2.store.elastic_store import ElasticStore
    from aurum_v2.store.duck_store import DuckStore

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

    def __init__(self, 
                 network: FieldNetwork,
                 duck: DuckStore | None = None,
                 es: ElasticStore | None = None) -> None:
        self._network = network
        self._store_client = duck if es is None else es

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

        Returns a DRS with ``OP.KW_LOOKUP`` operation.
        """
        hits = self._store_client.search_keywords(keywords=kw, kw_type=kw_type, max_hits = max_results)
        return DRS(list(hits), Operation(OP.KW_LOOKUP, params = [kw, kw_type]))


    def exact_search(self, kw: str, kw_type: KWType, max_results: int = 10) -> DRS:
        """Like :meth:`search` but exact‑match only."""
        hits = self._store_client.exact_search_keywords(keywords=kw, kw_type=kw_type, max_hits = max_results)
        return DRS(list(hits), Operation(OP.KW_LOOKUP, params = [kw, kw_type]))

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
        i_drs = self._general_to_drs(input_data)

        o_drs =  DRS(data=[], operation=Operation(OP.NONE))
        o_drs.absorb_provenance(i_drs)
        o_drs.set_fields_mode()

        for hit in i_drs:
            o_drs.absorb(self._network.neighbors_id(hit, relation))
        return o_drs

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

        Returns the UNION of all nodes visited during the walk.
        """
        # 1. Standardize Input
        i_drs = self._general_to_drs(a)
        
        # 2. Prepare the Accumulator (Snowball)
        # We start with an empty DRS and absorb the input immediately
        o_drs = DRS(data=[], operation=Operation(OP.NONE))
        o_drs.absorb(i_drs) 

        if i_drs.mode == DRSMode.TABLE:
            raise ValueError('Input mode DRSMode.TABLE not supported for traversing.')

        # 3. The BFS Loop
        current_frontier = i_drs
        
        while max_hops > 0:
            max_hops -= 1
            
            # Create a holding pen for this hop's discoveries
            next_frontier = DRS(data=[], operation=Operation(OP.NONE))
            
            # Find neighbors for every node in the current frontier
            for hit in current_frontier:
                neighbors_drs = self._network.neighbors_id(hit, primitive)
                next_frontier.absorb(neighbors_drs)
            
            # "Snowball" step: Add everything we just found to the final output
            o_drs.absorb(next_frontier)
            
            # Move the frontier forward for the next hop
            current_frontier = next_frontier
            
            # Optimization: If we hit a dead end, stop early
            if current_frontier.size() == 0:
                break
                
        return o_drs

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
        src_drs = self._general_to_drs(drs_a)
        tgt_drs = self._general_to_drs(drs_b)
        
        # 2. Output Accumulator
        o_drs = DRS([], Operation(OP.NONE))

        # 3. Cartesian Product (Compare every Start to every End)
        # This looks O(N^2), but usually N is small (user selects specific tables)
        for src in src_drs:
            for tgt in tgt_drs:
                # TABLE MODE: Use the teleporting table-pathfinder
                if src_drs.mode == DRSMode.TABLE:
                    path_drs = self._network.find_path_table(
                        src, tgt, relation, self, max_hops, lean_search
                    )
                
                # FIELDS MODE: Use the direct point-to-point pathfinder
                else:
                    path_drs = self._network.find_path_hit(
                        src, tgt, relation, max_hops
                    )
                
                # Add any found paths to the result
                o_drs.absorb(path_drs)

        return o_drs

    # ==================================================================
    # Combiner (set operations)
    # ==================================================================

    def intersection(self, a: DRS, b: DRS) -> DRS:
        a = self._general_to_drs(a)
        b = self._general_to_drs(b)
        self._assert_same_mode(a, b)

        return a.intersection(b)

    def union(self, a: DRS, b: DRS) -> DRS:
        a = self._general_to_drs(a)
        b = self._general_to_drs(b)
        self._assert_same_mode(a, b)

        return a.union(b)

    def difference(self, a: DRS, b: DRS) -> DRS:
        a = self._general_to_drs(a)
        b = self._general_to_drs(b)
        self._assert_same_mode(a, b)

        return a.set_difference(b)

    # ==================================================================
    # General‑input helpers
    # ==================================================================

    def make_drs(self, general_input) -> DRS:
        """Coerce any supported input (or list of inputs) into a single DRS.

        Handles:
        * Lists: Merges multiple inputs into one DRS (Union).
        * Singles: Delegates to _general_to_drs.
        """
        # 1. Handle Lists (Batch Processing)
        if isinstance(general_input, list):
            if not general_input:
                return DRS([], Operation(OP.NONE))
            
            # Convert the first item
            final_drs = self._general_to_drs(general_input[0])
            
            # Union the rest
            for item in general_input[1:]:
                next_drs = self._general_to_drs(item)
                final_drs = final_drs.absorb(next_drs)
            return final_drs
        
        return self._general_to_drs(general_input)

    def drs_from_table_hit(self, hit: Hit) -> DRS:
        """Expand a single Hit into a DRS of all columns in the same table.

        Provenance: ``Operation(OP.TABLE, params=[hit])``.
        """
        # Get all sibling columns from the network
        siblings = self._network.get_hits_from_table(hit.source_name)
        # Return new DRS with TABLE operation provenance
        return DRS(siblings, Operation(OP.TABLE, params=[hit]))
    
    def _drs_from_table_hit_lean_no_provenance(self, hit: Hit) -> DRS:
        """Like :meth:`drs_from_table_hit` but with ``lean_drs=True``."""
        hits = self._network.get_hits_from_table(hit.source_name)
        drs = DRS([x for x in hits], Operation(OP.TABLE, params=[hit]), lean_drs=True)
        return drs

    def _general_to_drs(self, general_input) -> DRS:
        """Convert a SINGLE input (nid/table/tuple/Hit) -> DRS.
        
        This is the low-level type switch logic.
        """
        # 1. Already a DRS? Pass it through.
        if isinstance(general_input, DRS):
            return general_input

        # 2. None? Return empty carrier.
        if general_input is None:
            return DRS([], Operation(OP.NONE))

        # 3. Hit object? Wrap it.
        if isinstance(general_input, Hit):
            return self._hit_to_drs(general_input)

        # 4. Integer (or string ID)? Look up the Hit by ID.
        if isinstance(general_input, int) or (isinstance(general_input, str) and general_input.isdigit()):
            hit = self._nid_to_hit(general_input)
            return self._hit_to_drs(hit)

        # 5. Tuple (db, source, field)? Compute ID -> Hit.
        if isinstance(general_input, tuple):
            hit = self._node_to_hit(general_input)
            return self._hit_to_drs(hit)

        # 6. String (Table Name)? Return ALL columns in that table.
        if isinstance(general_input, str):
            hits = self._network.get_hits_from_table(general_input)
            return DRS(hits, Operation(OP.ORIGIN))

        raise ValueError(f"Input type {type(general_input)} not supported by make_drs")
        

    def _nid_to_hit(self, nid: int | str) -> Hit:
        """Look up metadata for *nid* and return a Hit (score = 0)."""
        nid = str(nid)
        # Network lookup returns: (db, source, field, type)
        info = self._network.get_info_for([nid])
        if not info:
             raise ValueError(f"NID {nid} not found in network")
        _, db, src, field = info[0]
        return Hit(nid, db, src, field, 0)

    def _node_to_hit(self, node: tuple) -> Hit:
        """Given (db, source, field), compute nid and return a Hit."""
        from aurum_v2.models.hit import compute_field_id
        db, src, field = node
        nid = compute_field_id(db, src, field) # DB is usually ignored in ID hash
        return self._nid_to_hit(nid)

    def _hit_to_drs(self, hit: Hit, table_mode: bool = False) -> DRS:
        """Wrap a Hit in a DRS.  If *table_mode*, expand to whole table."""
        if table_mode:
            return self.drs_from_table_hit(hit)
        return DRS([hit], Operation(OP.ORIGIN))

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
