"""
Provenance — a directed acyclic graph tracking how DRS results were derived.

Backed by ``networkx.MultiDiGraph``.

* **Nodes** are :class:`~aurum_v2.models.hit.Hit` objects (including synthetic
  origin Hits created for keyword‑lookup operations).
* **Edges** carry an :class:`~aurum_v2.models.relation.OP` label describing the
  operation that produced the relationship.

This is a faithful port of the ``Provenance`` class in the legacy
``api/apiutils.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from aurum_v2.models.relation import OP

if TYPE_CHECKING:
    from aurum_v2.models.hit import Hit

__all__ = ["Provenance"]

# Module‑level counter that guarantees globally unique synthetic origin IDs.
_global_origin_id: int = 0


class Provenance:
    """Provenance DAG embedded inside every :class:`~aurum_v2.models.drs.DRS`.

    Parameters
    ----------
    data : list[Hit]
        Initial result set to record.
    operation : Operation
        The operation that created *data*.
    """

    def __init__(self, data: list, operation) -> None:
        self._p_graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._cached_leafs_and_heads: tuple[list | None, list | None] = (None, None)
        self._populate_provenance(data, operation.op, operation.params)

    # ------------------------------------------------------------------
    # Graph access
    # ------------------------------------------------------------------

    def prov_graph(self) -> nx.MultiDiGraph:
        """Return the underlying ``MultiDiGraph`` (invalidates leaf/head cache for safety)."""
        self._invalidate_cache()
        return self._p_graph

    def swap_p_graph(self, new_graph: nx.MultiDiGraph) -> None:
        """Replace the internal graph wholesale (used during provenance merges)."""
        self._invalidate_cache()
        self._p_graph = new_graph

    # ------------------------------------------------------------------
    # Populate
    # ------------------------------------------------------------------

    def _populate_provenance(self, data: list, op: OP, params: list | None) -> None:
        """Wire the initial nodes/edges according to *op*.

        Logic mirrors legacy ``Provenance.populate_provenance`` exactly:

        * ``OP.NONE``  → carrier DRS, skip.
        * ``OP.ORIGIN`` → add each element as a standalone node.
        * ``OP.KW_LOOKUP / SCHNAME_LOOKUP / ENTITY_LOOKUP`` → create a synthetic
          origin Hit (keyword as all name fields), connect to each element.
        * Everything else → *params[0]* is the source Hit; connect it to each
          element with the given *op* label.
        """
        if op == OP.NONE:
            return
            
        elif op == OP.ORIGIN:
            for element in data:
                self._p_graph.add_node(element)
                
        elif op in {OP.KW_LOOKUP, OP.SCHNAME_LOOKUP, OP.ENTITY_LOOKUP}:
            global _global_origin_id
            # Create a synthetic origin Hit to represent the user's text search
            kw = str(params[0]) if params else "unknown"
            synthetic_hit = Hit(
                nid=f"synthetic_origin_{_global_origin_id}",
                db_name=kw, source_name=kw, field_name=kw, score=-1.0
            )
            _global_origin_id += 1
            
            self._p_graph.add_node(synthetic_hit)
            for element in data:
                self._p_graph.add_node(element)
                # NetworkX MultiDiGraph uses 'key' to store multiple edges
                self._p_graph.add_edge(synthetic_hit, element, key=op)
                
        else:
            # All other operations (like PKFK) must have a source Hit in params[0]
            if not params:
                return
            src_hit = params[0]
            self._p_graph.add_node(src_hit)
            for element in data:
                self._p_graph.add_node(element)
                self._p_graph.add_edge(src_hit, element, key=op)
                
        self._invalidate_cache()

    # ------------------------------------------------------------------
    # Leaf / head helpers
    # ------------------------------------------------------------------

    def get_leafs_and_heads(self) -> tuple[list[Hit], list[Hit]]:
        """Return ``(leafs, heads)`` of the provenance DAG."""
        if self._cached_leafs_and_heads[0] is not None:
            return self._cached_leafs_and_heads  # type: ignore

        # Modern NetworkX: leafs have no incoming edges, heads have no outgoing edges
        leafs = [n for n, d in self._p_graph.in_degree() if d == 0]
        heads = [n for n, d in self._p_graph.out_degree() if d == 0]

        self._cached_leafs_and_heads = (leafs, heads)
        return leafs, heads

    def _invalidate_cache(self) -> None:
        self._cached_leafs_and_heads = (None, None)

    # ------------------------------------------------------------------
    # Path computation
    # ------------------------------------------------------------------

    def compute_paths_from_origin_to(
        self,
        a: Hit,
        leafs: list | None = None,
        heads: list | None = None,
    ) -> list[list[Hit]]:
        """All simple paths from any *leaf* to *a*."""
        if leafs is None:
            leafs, _ = self.get_leafs_and_heads()
            
        all_paths = []
        for leaf in leafs:
            if nx.has_path(self._p_graph, leaf, a):
                all_paths.extend(list(nx.all_simple_paths(self._p_graph, leaf, a)))
        return all_paths

    def compute_all_paths(
        self,
        leafs: list | None = None,
        heads: list | None = None,
    ) -> list[list[Hit]]:
        """Every leaf → head simple path in the DAG."""
        if leafs is None or heads is None:
            leafs, heads = self.get_leafs_and_heads()
            
        all_paths = []
        for leaf in leafs:
            for head in heads:
                if nx.has_path(self._p_graph, leaf, head):
                    all_paths.extend(list(nx.all_simple_paths(self._p_graph, leaf, head)))
        return all_paths

    def compute_paths_with(
        self,
        a: Hit,
        leafs: list | None = None,
        heads: list | None = None,
    ) -> list[list[Hit]]:
        """All paths that *contain* node *a* (upstream + downstream stitched)."""
        if leafs is None or heads is None:
            leafs, heads = self.get_leafs_and_heads()
            
        all_paths = []
        if a in leafs:
            for head in heads:
                if nx.has_path(self._p_graph, a, head):
                    all_paths.extend(list(nx.all_simple_paths(self._p_graph, a, head)))
        elif a in heads:
            for leaf in leafs:
                if nx.has_path(self._p_graph, leaf, a):
                    all_paths.extend(list(nx.all_simple_paths(self._p_graph, leaf, a)))
        else:
            upstreams = self.compute_paths_from_origin_to(a, leafs=leafs)
            downstreams = []
            for head in heads:
                if nx.has_path(self._p_graph, a, head):
                    downstreams.extend(list(nx.all_simple_paths(self._p_graph, a, head)))
                    
            # Stitch upstream paths with downstream paths
            for u in upstreams:
                for d in downstreams:
                    # d[1:] skips duplicating node 'a' where they join
                    all_paths.append(u + d[1:])
                    
        return all_paths

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain_path(self, path: list[Hit]) -> str:
        """Human‑readable explanation of *path* by traversing edge labels."""
        explanation = []
        
        # Cleanly iterate over pairs of adjacent nodes
        for src, trg in zip(path, path[1:]):
            edge_data = self._p_graph[src][trg]
            
            # The edge key is the OP Enum, so we extract its name
            ops = [k.name for k in edge_data.keys() if isinstance(k, OP)]
            op_string = ", ".join(ops)
            
            explanation.append(
                f"{src.source_name}:{src.field_name} --[{op_string}]--> {trg.source_name}:{trg.field_name}"
            )
            
        return "\n".join(explanation)