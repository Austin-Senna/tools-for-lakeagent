"""
DRS — Domain Result Set.

The universal data carrier in Aurum: every query and traversal operation
takes and returns a DRS.  It bundles:

* ``data``        – the set of :class:`Hit` results
* ``provenance``  – the :class:`Provenance` DAG
* ``mode``        – iteration mode (FIELDS or TABLE)
* ranking scores  – certainty / coverage, computed lazily

This is a line‑for‑line logic port of the legacy ``DRS`` class in
``api/apiutils.py``.
"""

from __future__ import annotations
from dataclasses import asdict

from collections import defaultdict
from enum import Enum
from typing import Any
import networkx as nx

from aurum_v2.models.hit import Hit
from aurum_v2.models.provenance import Provenance
from aurum_v2.models.relation import DRSMode, Operation, OP

__all__ = ["DRS"]


class DRS:
    """Domain Result Set — the universal result container.

    Parameters
    ----------
    data : list[Hit]
        Initial result hits.
    operation : Operation
        The operation that produced *data* (drives provenance wiring).
    lean_drs : bool
        If ``True``, skip provenance construction entirely (speed optimisation
        used by ``find_path_table`` during lean search).
    """

    # ------------------------------------------------------------------
    # Ranking criteria
    # ------------------------------------------------------------------

    class RankingCriteria(Enum):
        CERTAINTY = 0
        COVERAGE = 1

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        data: list[Hit],
        operation: Operation,
        lean_drs: bool = False,
    ) -> None:
        self._data: list[Hit] = data
        self._provenance: Provenance | None = None
        if not lean_drs:
            self._provenance = Provenance(data, operation)

        # Table‑mode lazy view
        self._table_view: list[str] = []

        # Iteration state
        self._idx: int = 0
        self._idx_table: int = 0
        self._mode: DRSMode = DRSMode.FIELDS

        # Ranking state
        self._ranked: bool = False
        self._rank_data: dict[Hit, dict[str, Any]] = defaultdict(dict)
        self._ranking_criteria: DRS.RankingCriteria | None = None
        self._chosen_rank: list[tuple[Hit, Any]] = []
        self._origin_values_coverage: dict[Hit, int] = {}

    # ------------------------------------------------------------------
    # Iteration (fields or table mode)
    # ------------------------------------------------------------------

    def __iter__(self) -> DRS:
        return self

    def __next__(self) -> Hit | str:
        """Yield the next element according to the current mode."""
        if self._mode == DRSMode.FIELDS:
            if self._idx < len(self._data):
                item = self._data[self._idx]
                self._idx += 1
                return item
            self._idx = 0
            raise StopIteration
        elif self._mode == DRSMode.TABLE:
            if not self._table_view:
                # Lazy load unique tables preserving insertion order
                seen = set()
                self._table_view = [
                    h.source_name for h in self._data 
                    if not (h.source_name in seen or seen.add(h.source_name))
                ]
            if self._idx_table < len(self._table_view):
                item_table = self._table_view[self._idx_table]
                self._idx_table += 1
                return item_table
            self._idx_table = 0
            raise StopIteration
    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def data(self) -> list[Hit]:
        return self._data

    def set_data(self, data: list[Hit]) -> DRS:
        """Replace internal data; resets table view, indices, ranking."""
        self._data = list(data)
        self._table_view = []
        self._idx = 0
        self._idx_table = 0
        self._mode = DRSMode.FIELDS
        self._ranked = False
        return self
    
    @property
    def mode(self) -> DRSMode:
        return self._mode

    def size(self) -> int:
        return len(self._data)

    def get_provenance(self) -> Provenance:
        assert self._provenance is not None, "Provenance not available (lean DRS)"
        return self._provenance

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def set_fields_mode(self) -> None:
        self._mode = DRSMode.FIELDS

    def set_table_mode(self) -> None:
        self._mode = DRSMode.TABLE

    # ------------------------------------------------------------------
    # Provenance merge operations
    # ------------------------------------------------------------------

    def absorb_provenance(
        self,
        drs: DRS,
        annotate_and_edges: bool = False,
        annotate_or_edges: bool = False,
    ) -> DRS:
        """Merge the *provenance graph* of *drs* into ``self`` (data untouched).

        Optionally annotate shared edges with ``AND`` or ``OR`` labels for
        intersection / union bookkeeping.
        """
        self._ranked = False
        if not self._provenance or not drs._provenance:
            return self

        merged_graph = nx.compose(
            self._provenance.prov_graph(), 
            drs.get_provenance().prov_graph()
        )

        if annotate_and_edges or annotate_or_edges:
            label = 'AND' if annotate_and_edges else 'OR'
            # Find nodes present in both results
            disjoint = set(self._data).intersection(set(drs.data))
            
            for node in disjoint:
                # Annotate incoming edges directly in the new merged graph
                for src, tgt, key, edge_data in merged_graph.in_edges(node, data=True, keys=True):
                    merged_graph[src][tgt][key][label] = 1

        self._provenance.swap_p_graph(merged_graph)
        return self


    def absorb(self, drs: DRS) -> DRS:
        """Merge *both* data (set‑union) and provenance of *drs* into ``self``."""
        self._ranked = False
        new_data = list(set(self._data).union(set(drs.data)))
        self.set_data(new_data)
        self.absorb_provenance(drs)
        return self

    # ------------------------------------------------------------------
    # Set operations  (each returns a *new* DRS)
    # ------------------------------------------------------------------

    def intersection(self, drs: DRS) -> DRS:
        """Set intersection. In TABLE mode, matches on ``source_name``."""
        result = DRS([], Operation(OP.NONE))
        
        if drs.mode == DRSMode.TABLE:
            my_tables = {h.source_name: h for h in self._data}
            their_tables = {h.source_name: h for h in drs.data}
            shared_tables = set(my_tables.keys()).intersection(their_tables.keys())
            
            new_data = []
            for t in shared_tables:
                new_data.extend([my_tables[t], their_tables[t]])
        else:
            new_data = list(set(self._data).intersection(set(drs.data)))

        result.set_data(new_data)
        if self._provenance and drs._provenance:
            result.absorb_provenance(self, annotate_and_edges=True)
            result.absorb_provenance(drs, annotate_and_edges=True)
            
        return result

    def union(self, drs: DRS) -> DRS:
        """Set union of data, provenance composed."""
        result = DRS([], Operation(OP.NONE))
        result.set_data(list(set(self._data).union(set(drs.data))))
        
        if self._provenance and drs._provenance:
            result.absorb_provenance(self)
            result.absorb_provenance(drs)
            
        return result

    def set_difference(self, drs: DRS) -> DRS:
        """``self − drs`` with provenance composed."""
        result = DRS([], Operation(OP.NONE))
        result.set_data(list(set(self._data) - set(drs.data)))
        
        if self._provenance and drs._provenance:
            result.absorb_provenance(self)
            result.absorb_provenance(drs)
            
        return result

    # ------------------------------------------------------------------
    # Provenance query helpers
    # ------------------------------------------------------------------
    def paths(self) -> list[list[Hit]]:
        """All leaf→head paths in the provenance DAG."""
        return self.get_provenance().compute_all_paths()

    def path(self, a: Hit) -> list[list[Hit]]:
        """All paths that contain *a*."""
        return self.get_provenance().compute_paths_with(a)

    def why(self, a: Hit) -> list[Hit]:
        """Origin results that led to *a* appearing in this DRS."""
        if a not in self._data:
            return []
        paths = self.get_provenance().compute_paths_from_origin_to(a)
        return list({p[0] for p in paths if p})

    def why_id(self, nid: int | str) -> list[Hit]:
        """Convenience: :meth:`why` by nid integer."""
        hit = next((x for x in self._data if str(x.nid) == str(nid)), None)
        return self.why(hit) if hit else []

    def how(self, a: Hit) -> list[str]:
        """Human‑readable derivation explanations for *a*."""
        if a not in self._data:
            return []
        paths = self.get_provenance().compute_paths_from_origin_to(a)
        return [self.get_provenance().explain_path(p) for p in paths]

    def how_id(self, nid: int | str) -> list[str]:
        """Convenience: :meth:`how` by nid integer."""
        hit = next((x for x in self._data if str(x.nid) == str(nid)), None)
        return self.how(hit) if hit else []

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def compute_ranking_scores(self) -> None:
        """Compute both certainty and coverage scores (called lazily)."""
        self._compute_certainty_scores()
        self._compute_coverage_scores()
        self._ranked = True

    def _compute_certainty_scores(self) -> None:
        """Traverse reversed provenance graph to aggregate edge scores.

        Strategy: recursive continuous‑path score aggregation from each
        result node upward.  When a fork is encountered the branch with
        the maximum score is chosen.
        """
        """Traverse reversed provenance graph to aggregate edge scores."""
        if not self._provenance:
            return
            
        def get_score(pg: nx.MultiDiGraph, src: Hit, visited: set) -> float:
            current_score = float(src.score)
            neighbors = [n for n in pg.neighbors(src) if n not in visited]
            
            if not neighbors:
                return current_score
                
            # Aggregate max branch score
            max_branch = 0.0
            for n in neighbors:
                visited.add(n)
                branch_score = get_score(pg, n, visited)
                if branch_score > max_branch:
                    max_branch = branch_score
                    
            return current_score + max_branch

        pg_reversed = self._provenance.prov_graph().reverse(copy=False)
        visited: set[Hit] = set()
        
        for el in self._data:
            if el not in visited:
                score = get_score(pg_reversed, el, visited)
                self._rank_data[el]['certainty_score'] = score

    def _compute_coverage_scores(self) -> None:
        """Rank by the fraction of origin leaves that led to this result."""
        if not self._provenance:
            return
            
        leafs, _ = self.get_provenance().get_leafs_and_heads()
        total_origins = len(leafs) if leafs else 1

        for el in self._data:
            # Replaced the weird C-extension 'bitarray' with native Python sets
            covered_origins = set(self.why(el))
            coverage = len(covered_origins) / total_origins
            self._rank_data[el]['coverage_score'] = (coverage, covered_origins)

    def rank_certainty(self) -> DRS:
        """Sort ``self.data`` by descending certainty score.  Returns ``self``."""
        if not self._ranked:
            self.compute_ranking_scores()

        elements = [
            (el, self._rank_data[el].get('certainty_score', 0.0)) 
            for el in self._data
        ]
        elements.sort(key=lambda x: x[1], reverse=True)
        
        self._data = [el for el, _ in elements]
        self._ranking_criteria = self.RankingCriteria.CERTAINTY
        self._chosen_rank = elements
        return self

    def rank_coverage(self) -> DRS:
        """Sort ``self.data`` by descending coverage score.  Returns ``self``."""
        if not self._ranked:
            self.compute_ranking_scores()

        elements = [
            (el, self._rank_data[el].get('coverage_score', (0.0, set()))[0]) 
            for el in self._data
        ]
        elements.sort(key=lambda x: x[1], reverse=True)
        
        self._data = [el for el, _ in elements]
        self._ranking_criteria = self.RankingCriteria.COVERAGE
        self._chosen_rank = elements
        return self

    # ------------------------------------------------------------------
    # Serialisation & Output
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON‑friendly output for web APIs."""
        original_mode = self._mode
        self.set_fields_mode()
        
        sources = {}
        for x in self:
            table = x.source_name
            if table not in sources:
                sources[table] = {
                    'source_res': asdict(x),  # <--- Modern safely converts dataclass to dict
                    'field_res': []
                }
            sources[table]['field_res'].append(asdict(x))

        edges = []
        if self._provenance:
            # Modern edge extraction
            for u, v in self._provenance.prov_graph().edges():
                edges.append((asdict(u), asdict(v)))

        self._mode = original_mode
        return {'sources': sources, 'edges': edges}

    # ------------------------------------------------------------------
    # Debug / display
    # ------------------------------------------------------------------

    def debug_print(self) -> None:
        print(f"Total data: {len(self._data)}")
        if self._provenance:
            print(f"Total nodes prov graph: {len(self._provenance.prov_graph())}")

    def visualize_provenance(self, labels: bool = False) -> None:
        """Quick ``matplotlib`` visualisation of the provenance DAG."""
        raise NotImplementedError

    def print_tables(self) -> None:
        original = self._mode
        self.set_table_mode()
        for x in self:
            print(x)
        self._mode = original

    def print_columns(self) -> None:
        original = self._mode
        self.set_fields_mode()
        for x in set(self):
            print(x)
        self._mode = original

    def print_tables_with_scores(self) -> None:
        raise NotImplementedError

    def print_columns_with_scores(self) -> None:
        raise NotImplementedError

    def pretty_print_columns(self) -> None:
        original = self._mode
        self.set_table_mode()
        for x in self:
            print(f"DB: {x.db_name:20} TABLE: {x.source_name:30} FIELD: {x.field_name:30}")
        self._mode = original

    def pretty_print_columns_with_scores(self) -> None:
        for x, score in self._chosen_rank:
            print(f"DB: {x.db_name:20} TABLE: {x.source_name:30} FIELD: {x.field_name:30} SCORE: {score}")
