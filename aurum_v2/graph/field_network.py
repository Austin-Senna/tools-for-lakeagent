"""
FieldNetwork — the core multi‑relation column graph.

Wraps a ``networkx.MultiGraph`` where:

* **Nodes** = column *nid* strings (CRC32 of db+source+field).
  Each node carries a ``cardinality`` attribute (unique / total).
* **Edges** are keyed by :class:`Relation` and carry a ``score`` dict.

This is a faithful port of ``knowledgerepr/fieldnetwork.py``.
"""

from __future__ import annotations
import pickle
from pathlib import Path
from collections import defaultdict
from collections.abc import Iterator

import networkx as nx

from aurum_v2.models.drs import DRS
from aurum_v2.models.hit import Hit
from aurum_v2.models.relation import OP, Relation, Operation
import logging 

logger = logging.getLogger(__name__)

__all__ = ["FieldNetwork", "serialize_network", "deserialize_network", "serialize_network_to_csv"]


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
        for (nid, db_name, sn_name, fn_name, total_values, unique_values, data_type) in fields:
            self._id_names[nid] = (db_name, sn_name, fn_name, data_type)
            self._source_ids[sn_name].append(nid)
            cardinality_ratio = float(unique_values) / float(total_values) if total_values > 0 else 0.0
            self.add_field(nid, cardinality_ratio)
        
        logger.info(
            "Graph initialized with %d nodes across %d tables.", 
            self.graph_order(), self.get_number_tables()
        )

    def add_field(self, nid: str, cardinality: float | None = None) -> str:
        """Add a single graph node for column *nid* with optional cardinality."""
        self._graph.add_node(nid, cardinality = cardinality)
        return nid

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
        self._graph.add_edge(node_src, node_target, key=relation, score=score)

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
            Relation.INCLUSION_DEPENDENCY: OP.CONTENT_SIM,
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
        nid = str(hit.nid) if isinstance(hit, Hit) else str(hit)
        data = []
        neighbors = self._graph[nid]

        for k,v in neighbors.items():
            if relation in v:
                score = v[relation]['score']
                (db_name, source_name, field_name, data_type) = self._id_names[k]
                data.append(Hit(k, db_name, source_name, field_name, score))
        op = self.get_op_from_relation(relation)
        return DRS(data, Operation(op, params=[hit]))

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
        """BFS path search between two *columns* via *relation*.
        Much better than original DFS search in Aurum Paper"""
        
        # 1. Edge case: The source IS the target
        if source.nid == target.nid:
            return DRS([target], Operation(OP.ORIGIN, params=[source]))

        # 2. Iterative BFS to find the shortest path matching the relation
        # Queue stores: (current_node_id, path_of_hits_so_far)
        queue = [(str(source.nid), [source])]
        visited = {str(source.nid)}
        found_path = None

        while queue:
            current_nid, path = queue.pop(0)

            # Stop exploring this branch if we hit the user's hop limit
            if len(path) - 1 >= max_hops:
                continue

            # Fast NetworkX adjacency lookup: self._graph[nid] returns neighbors & edges
            for neighbor_nid, edges in self._graph[current_nid].items():
                if relation in edges:
                    if neighbor_nid not in visited:
                        # Reconstruct the Hit object for the neighbor
                        db, src, field, dtype = self._id_names[neighbor_nid]
                        neighbor_hit = Hit(
                            nid=neighbor_nid, 
                            db_name=db, 
                            source_name=src, 
                            field_name=field, 
                            score=edges[relation]['score']
                        )
                        
                        new_path = path + [neighbor_hit]
                        if neighbor_nid == str(target.nid):
                            found_path = new_path
                            break
                        
                        visited.add(neighbor_nid)
                        queue.append((neighbor_nid, new_path))
            
            if found_path:
                break
        
        # If we exhausted the queue and found nothing
        if not found_path:
            return DRS([], Operation(OP.NONE))

        # 3. Assemble the Provenance Chain cleanly (No more hardcoded PKFK!)
        op_type = self.get_op_from_relation(relation)
        
        # Start the chain with the ORIGIN operation
        o_drs = DRS([found_path[0]], Operation(OP.ORIGIN))
        
        # Chain the rest of the hops together dynamically
        prev_hit = found_path[0]
        for current_hit in found_path[1:]:
            step_drs = DRS([current_hit], Operation(op_type, params=[prev_hit]))
            o_drs = o_drs.absorb(step_drs)
            prev_hit = current_hit
            
        return o_drs

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
        """BFS path search between two *tables* via *relation*."""
        
        # 1. Edge Case: Already in the same table
        if source.source_name == target.source_name:
            return DRS([target], Operation(OP.ORIGIN, params=[source]))

        # 2. Initialize Queue
        # Queue stores: (current_outbound_hit, path_of_tuples)
        # Path Tuple: (outbound_hit, inbound_hit_that_brought_us_to_this_table)
        queue = [(source, [(source, None)])]
        
        # Track visited TABLES (not columns) to prevent infinite loops
        visited_tables = {source.source_name}
        found_path = None

        while queue:
            current_hit, path = queue.pop(0)

            if len(path) - 1 >= max_hops:
                continue

            # Find cross-table neighbors directly from NetworkX
            neighbor_nids = []
            for n_nid, edges in self._graph[str(current_hit.nid)].items():
                if relation in edges:
                    neighbor_nids.append((n_nid, edges[relation]['score']))

            for n_nid, score in neighbor_nids:
                db, src, field, dtype = self._id_names[n_nid]

                # Skip edges that point back to our own table
                if src == current_hit.source_name:
                    continue

                neighbor_hit = Hit(n_nid, db, src, field, score)

                # Did we reach the target's table?
                if src == target.source_name:
                    found_path = path + [(target, neighbor_hit)]
                    break

                # If it's a completely new table, expand it
                if src not in visited_tables:
                    visited_tables.add(src)

                    # Get all sibling columns in this new table via the API
                    if lean_search and hasattr(api, "_drs_from_table_hit_lean_no_provenance"):
                        siblings = api._drs_from_table_hit_lean_no_provenance(neighbor_hit)
                    else:
                        siblings = api.drs_from_table_hit(neighbor_hit)

                    # Queue every sibling as a potential outbound jump for the next hop
                    for sibling in siblings:
                        queue.append((sibling, path + [(sibling, neighbor_hit)]))

            if found_path:
                break

        # 3. Handle Failure
        if not found_path:
            return DRS([], Operation(OP.NONE))

        # 4. Assemble the Provenance Chain cleanly
        op_type = self.get_op_from_relation(relation)
        
        # Start with the origin
        src_outbound, _ = found_path[0]
        o_drs = DRS([src_outbound], Operation(OP.ORIGIN))
        
        prev_outbound = src_outbound

        # Stitch the hops together
        for current_outbound, current_inbound in found_path[1:]:
            
            # Step A: The cross-table jump (e.g., PKFK)
            jump_drs = DRS([current_inbound], Operation(op_type, params=[prev_outbound]))
            o_drs = o_drs.absorb_provenance(jump_drs)
            
            # Step B: The intra-table jump (e.g., TABLE linking column A to column B)
            if current_inbound.nid != current_outbound.nid:
                table_drs = DRS([current_outbound], Operation(OP.TABLE, params=[current_inbound]))
                o_drs = o_drs.absorb(table_drs)
            else:
                # Same column — just merge the data in
                o_drs = o_drs.absorb(jump_drs)
                
            prev_outbound = current_outbound
            
        return o_drs

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
        # 1. NetworkX's .degree() returns a DegreeView (an iterator of (node, degree) tuples)
        degree_list = list(self._graph.degree())
        
        # 2. Sort descending based on the degree (which is the second item in the tuple: x[1])
        degree_list.sort(key=lambda x: x[1], reverse=True)
        
        # 3. Return the top K slice
        return degree_list[:topk]

    # ------------------------------------------------------------------
    # Enumeration / debug helpers
    # ------------------------------------------------------------------

    def enumerate_relation(
        self, relation: Relation, as_str: bool = True
    ) -> Iterator:
        """Iterate over all distinct edge pairs for *relation*."""
        # By passing keys=True, NetworkX yields (node_src, node_target, edge_key)
        # Because the graph is undirected, NetworkX guarantees it only yields each pair once.
        for u, v, key in self._graph.edges(keys=True):
            if key == relation:
                if as_str:
                    # Translate the raw nids into human-readable "table.column" strings
                    _, src_u, col_u, _ = self._id_names[u]
                    _, src_v, col_v, _ = self._id_names[v]
                    yield (f"{src_u}.{col_u}", f"{src_v}.{col_v}")
                else:
                    # Just yield the raw nids
                    yield (u, v)
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
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use HIGHEST_PROTOCOL for massive speed and compression benefits
    with open(out_dir / "graph.pickle", "wb") as f:
        pickle.dump(network._get_underlying_repr_graph(), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out_dir / "id_info.pickle", "wb") as f:
        pickle.dump(network._get_underlying_repr_id_to_field_info(), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out_dir / "table_ids.pickle", "wb") as f:
        pickle.dump(network._get_underlying_repr_table_to_ids(), f, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_network(path: str) -> FieldNetwork:
    """Reconstruct a :class:`FieldNetwork` from pickled artefacts at *path*."""
    in_dir = Path(path)

    with open(in_dir / "graph.pickle", "rb") as f:
        graph = pickle.load(f)

    with open(in_dir / "id_info.pickle", "rb") as f:
        id_info = pickle.load(f)

    with open(in_dir / "table_ids.pickle", "rb") as f:
        table_ids = pickle.load(f)

    # Re-instantiate the graph using the constructor we defined earlier
    return FieldNetwork(graph=graph, id_names=id_info, source_ids=table_ids)

def serialize_network_to_csv(network: FieldNetwork, path: str) -> None:
    """Export the graph as nodes and edges CSV files for visualization tools like Gephi."""
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    nodes = set()
    G = network._get_underlying_repr_graph()
    
    with open(out_dir / "edges.csv", "w") as f:
        # Modern NetworkX uses .edges() instead of the deprecated .edges_iter()
        for src, tgt in G.edges():
            f.write(f"{src},{tgt},1\n")
            nodes.add(src)
            nodes.add(tgt)
            
    with open(out_dir / "nodes.csv", "w") as f:
        for n in nodes:
            f.write(f"{n},node\n")
