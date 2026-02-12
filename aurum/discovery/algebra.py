"""
Query algebra — the user-facing API for data discovery.

Port of ``aurum/algebra.py`` (the newer API) and ``aurum/ddapi.py``
(the original API).  Both are unified here into a single ``Algebra``
class.

Usage::

    from aurum.discovery.algebra import Algebra
    api = Algebra(network)
    results = api.search_attribute("employee_name")
    similar = api.content_similar_to(results)
    paths   = api.paths(table_a_drs, table_b_drs)
"""

from __future__ import annotations

import itertools

from aurum.config import aurumConfig
from aurum.discovery.result_set import DRS, DRSMode, Operation
from aurum.graph.field_network import FieldNetwork, Hit
from aurum.graph.relations import OP, Relation
from aurum.profiler.text_utils import tokenize_name


class Algebra:
    """High-level discovery API over a ``FieldNetwork``.

    Combines ``aurum/algebra.py :: Algebra`` and ``aurum/ddapi.py :: DDAPI``.
    """

    def __init__(
        self,
        network: FieldNetwork,
        cfg: aurumConfig | None = None,
    ) -> None:
        self._net = network
        self._cfg = cfg or aurumConfig()

    # ── Seed / Lookup ────────────────────────────────────────────────

    def make_drs(self, general_input: str | int | Hit | DRS) -> DRS:
        """Convert arbitrary input (table name, nid, Hit, DRS) into a DRS.

        Ported from ``algebra.Algebra._general_to_drs``.
        """
        if isinstance(general_input, DRS):
            return general_input
        if isinstance(general_input, Hit):
            return DRS([general_input], Operation(OP.ORIGIN))
        if isinstance(general_input, str):
            # Try as table name first
            hits = self._net.get_hits_from_table(general_input)
            if hits:
                return DRS(hits, Operation(OP.ORIGIN))
            # Otherwise try as nid
            try:
                info = self._net.get_info_for([general_input])
                if info:
                    nid, db, sn, fn = info[0]
                    return DRS([Hit(nid, db, sn, fn, 0.0)], Operation(OP.ORIGIN))
            except KeyError:
                pass
            return DRS([], Operation(OP.NONE))
        if isinstance(general_input, int):
            return self.make_drs(str(general_input))
        raise ValueError(f"Cannot convert {type(general_input)} to DRS")

    def drs_from_table(self, table_name: str) -> DRS:
        """Return a DRS containing all columns of *table_name*."""
        hits = self._net.get_hits_from_table(table_name)
        return DRS(hits, Operation(OP.ORIGIN))

    # ── Keyword search (in-memory, over column names) ────────────────

    def search_attribute(self, keyword: str, max_results: int | None = None) -> DRS:
        """Search column *names* for a keyword (substring match).

        Replaces the ES-backed ``store_client.search_keywords(..., KW_SCHEMA)``.
        """
        if max_results is None:
            max_results = self._cfg.max_search_results
        kw_lower = keyword.lower()
        kw_tokens = set(tokenize_name(keyword))
        hits: list[Hit] = []
        for nid, (db, sn, fn, _dt) in self._net._id_names.items():
            fn_tokens = set(tokenize_name(fn))
            # Match if keyword is substring OR any token overlaps
            if kw_lower in fn.lower() or kw_tokens & fn_tokens:
                hits.append(Hit(nid, db, sn, fn, 1.0))
                if len(hits) >= max_results:
                    break
        return DRS(hits, Operation(OP.KW_LOOKUP, params=[keyword]))

    def search_exact_attribute(self, keyword: str, max_results: int | None = None) -> DRS:
        """Exact match on column names."""
        if max_results is None:
            max_results = self._cfg.max_search_results
        kw_lower = keyword.lower()
        hits: list[Hit] = []
        for nid, (db, sn, fn, _dt) in self._net._id_names.items():
            if fn.lower() == kw_lower:
                hits.append(Hit(nid, db, sn, fn, 1.0))
                if len(hits) >= max_results:
                    break
        return DRS(hits, Operation(OP.KW_LOOKUP, params=[keyword]))

    def search_content(self, keyword: str, max_results: int | None = None) -> DRS:
        """Placeholder for content search (would require a text index)."""
        # In the full implementation this would query a content index.
        # For now, delegate to attribute search as a fallback.
        return self.search_attribute(keyword, max_results)

    # ── Neighbour traversal ──────────────────────────────────────────

    def _neighbor_search(self, input_data: str | Hit | DRS, relation: Relation) -> DRS:
        """Traverse graph edges of a given relation type.

        Ported from ``algebra.Algebra.__neighbor_search``.
        """
        i_drs = self.make_drs(input_data)
        o_drs = DRS([], Operation(OP.NONE))
        o_drs.absorb_provenance(i_drs)
        for h in i_drs:
            neighbours = self._net.neighbors_id(h, relation)
            neighbour_drs = DRS(neighbours, Operation(relation_to_op(relation), params=[h]))
            o_drs = o_drs.absorb(neighbour_drs)
        return o_drs

    def content_similar_to(self, general_input: str | Hit | DRS) -> DRS:
        return self._neighbor_search(general_input, Relation.CONTENT_SIM)

    def schema_similar_to(self, general_input: str | Hit | DRS) -> DRS:
        return self._neighbor_search(general_input, Relation.SCHEMA_SIM)

    def pkfk_of(self, general_input: str | Hit | DRS) -> DRS:
        return self._neighbor_search(general_input, Relation.PKFK)

    def inclusion_dependency_of(self, general_input: str | Hit | DRS) -> DRS:
        return self._neighbor_search(general_input, Relation.INCLUSION_DEPENDENCY)

    # ── Path finding (transitive closure) ────────────────────────────

    def paths(
        self,
        drs_a: str | Hit | DRS,
        drs_b: str | Hit | DRS,
        relation: Relation = Relation.PKFK,
        max_hops: int | None = None,
    ) -> DRS:
        """Find join paths between elements in *drs_a* and *drs_b*.

        Ported from ``algebra.Algebra.paths``.
        """
        a = self.make_drs(drs_a)
        b = self.make_drs(drs_b)
        if max_hops is None:
            max_hops = self._cfg.max_hops

        o_drs = DRS([], Operation(OP.NONE))
        o_drs.absorb_provenance(a)
        if b is not a:
            o_drs.absorb_provenance(b)

        if a.mode == DRSMode.TABLE:
            # Table-level: find paths between any field of each table
            for h1, h2 in itertools.product(a, b):
                if h1.nid == h2.nid:
                    continue
                path = self._net.find_path(h1, h2, relation, max_hops=max_hops)
                if path:
                    path_drs = DRS(path, Operation(OP.PKFK, params=[h1]))
                    o_drs = o_drs.absorb(path_drs)
        else:
            for h1, h2 in itertools.product(a, b):
                if h1.nid == h2.nid:
                    continue
                path = self._net.find_path(h1, h2, relation, max_hops=max_hops)
                if path:
                    path_drs = DRS(path, Operation(OP.PKFK, params=[h1]))
                    o_drs = o_drs.absorb(path_drs)

        return o_drs

    # ── Set algebra (convenience wrappers) ───────────────────────────

    def union(self, a: DRS, b: DRS) -> DRS:
        return a | b

    def intersection(self, a: DRS, b: DRS) -> DRS:
        return a & b

    def difference(self, a: DRS, b: DRS) -> DRS:
        return a - b

    # ── Helpers ──────────────────────────────────────────────────────

    def get_path_nid(self, nid: str) -> str | None:
        """Return filesystem path for a source that contains *nid*."""
        info = self._net.get_info_for([nid])
        if info:
            _, _db, sn, _fn = info[0]
            return sn
        return None


def relation_to_op(relation: Relation) -> OP:
    """Map a Relation to its corresponding provenance OP."""
    mapping = {
        Relation.CONTENT_SIM: OP.CONTENT_SIM,
        Relation.SCHEMA_SIM: OP.SCHEMA_SIM,
        Relation.PKFK: OP.PKFK,
        Relation.ENTITY_SIM: OP.ENTITY_SIM,
        Relation.SCHEMA: OP.TABLE,
        Relation.INCLUSION_DEPENDENCY: OP.PKFK,  # treated as join-like
        Relation.MEANS_SAME: OP.MEANS_SAME,
        Relation.MEANS_DIFF: OP.MEANS_DIFF,
        Relation.SUBCLASS: OP.SUBCLASS,
        Relation.SUPERCLASS: OP.SUPERCLASS,
        Relation.MEMBER: OP.MEMBER,
        Relation.CONTAINER: OP.CONTAINER,
    }
    return mapping.get(relation, OP.NONE)
