"""
Domain Result Set (DRS) — the query-result container with provenance.

Port of ``aurum/api/apiutils.py``: ``Hit``, ``DRS``, ``Provenance``.

A DRS wraps a list of ``Hit`` objects and records a DAG of operations
that produced them, so every result is *explainable*.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Iterator

import networkx as nx

from aurum.graph.field_network import Hit
from aurum.graph.relations import OP


# ---------------------------------------------------------------------------
# DRS mode
# ---------------------------------------------------------------------------

class DRSMode(Enum):
    FIELDS = 0
    TABLE = 1


# ---------------------------------------------------------------------------
# Operation (provenance atom)
# ---------------------------------------------------------------------------

class Operation:
    """A single provenance step: what operation + what parameters."""

    __slots__ = ("op", "params")

    def __init__(self, op: OP, params: list | None = None) -> None:
        self.op = op
        self.params = params or []


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class Provenance:
    """A DAG that tracks how each ``Hit`` was derived.

    Ported from ``apiutils.Provenance``.  Every node in the graph is a
    ``Hit``; edges are labelled with the ``OP`` that connected them.

    Keyword / schema-name / entity lookups create a **synthetic origin
    Hit** so that the search term itself appears as a root in the DAG,
    matching the legacy behaviour.
    """

    # Global counter for synthetic origin node IDs (mirrors legacy _gnid)
    _gnid: int = 0

    def __init__(self, data: list[Hit], operation: Operation) -> None:
        self._p_graph = nx.MultiDiGraph()
        self.populate_provenance(data, operation.op, operation.params)
        self._cached_leafs_and_heads: tuple[list[Hit] | None, list[Hit] | None] = (None, None)

    # ── Graph accessors ──────────────────────────────────────────────

    def prov_graph(self) -> nx.MultiDiGraph:
        """Return the internal provenance graph (invalidates cache for safety)."""
        self.invalidate_leafs_heads_cache()
        return self._p_graph

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Property-style accessor (read-only, no cache invalidation)."""
        return self._p_graph

    def swap_p_graph(self, new: nx.MultiDiGraph) -> None:
        """Replace the internal provenance graph (used by AND/OR annotation merges)."""
        self.invalidate_leafs_heads_cache()
        self._p_graph = new

    # ── Population ───────────────────────────────────────────────────

    def populate_provenance(self, data: list[Hit], op: OP, params: list | None) -> None:
        """Add nodes/edges to the provenance graph for a single operation.

        Handles four cases (matching the legacy implementation):
        1. ``OP.NONE``  – carrier DRS, skip.
        2. ``OP.ORIGIN`` – seed data, add bare nodes.
        3. ``KW_LOOKUP / SCHNAME_LOOKUP / ENTITY_LOOKUP`` – create a
           **synthetic origin Hit** from the search keyword so that the
           provenance DAG has a trackable root for every search.
        4. Everything else – the first param is a parent ``Hit``;
           connect it to each result with an edge labelled by *op*.
        """
        if op == OP.NONE:
            return

        if op == OP.ORIGIN:
            for element in data:
                self._p_graph.add_node(element)
            return

        # Keyword / schema-name / entity lookup → synthetic origin
        if op in (OP.KW_LOOKUP, OP.SCHNAME_LOOKUP, OP.ENTITY_LOOKUP):
            keyword = params[0] if params else ""
            hit = Hit(
                nid=str(Provenance._gnid),
                db_name=str(keyword),
                source_name=str(keyword),
                field_name=str(keyword),
                score=-1.0,
            )
            Provenance._gnid += 1
            self._p_graph.add_node(hit)
            for element in data:
                self._p_graph.add_node(element)
                self._p_graph.add_edge(hit, element, op=op)
            self.invalidate_leafs_heads_cache()
            return

        # Generic operation with a Hit parameter
        if params:
            parent = params[0]
            self._p_graph.add_node(parent)
            for element in data:
                self._p_graph.add_node(element)
                self._p_graph.add_edge(parent, element, op=op)
            self.invalidate_leafs_heads_cache()

    # ── Leafs / heads (DAG roots & terminals) ────────────────────────

    def invalidate_leafs_heads_cache(self) -> None:
        self._cached_leafs_and_heads = (None, None)

    def get_leafs_and_heads(self) -> tuple[list[Hit], list[Hit]]:
        """Compute DAG roots (leafs) and terminals (heads) with cycle handling.

        Results are cached until invalidated.
        """
        if self._cached_leafs_and_heads[0] is not None and self._cached_leafs_and_heads[1] is not None:
            return self._cached_leafs_and_heads[0], self._cached_leafs_and_heads[1]

        leafs: list[Hit] = []
        heads: list[Hit] = []
        for node in self._p_graph.nodes():
            pre = set(self._p_graph.predecessors(node))
            suc = set(self._p_graph.successors(node))
            # Remove cycles: predecessors that are also successors
            pre = pre - suc
            if len(pre) == 0 and len(suc) == 0:
                continue  # isolated node
            if len(pre) == 0:
                leafs.append(node)
            if len(suc) == 0:
                heads.append(node)

        self._cached_leafs_and_heads = (leafs, heads)
        return leafs, heads

    # ── Merge ────────────────────────────────────────────────────────

    def merge(self, other: Provenance) -> None:
        """Absorb another provenance graph (union of edges/nodes)."""
        self._p_graph = nx.compose(self._p_graph, other._p_graph)
        self.invalidate_leafs_heads_cache()

    # ── Path computation ─────────────────────────────────────────────

    def compute_paths_from_origin_to(self, a: Hit) -> list[list[Hit]]:
        """Return all paths from any leaf (origin) node to *a*.

        Ported from ``apiutils.Provenance.compute_paths_from_origin_to``.
        """
        leafs, _heads = self.get_leafs_and_heads()
        all_paths: list[list[Hit]] = []
        for leaf in leafs:
            try:
                paths = nx.all_simple_paths(self._p_graph, leaf, a)
                all_paths.extend(paths)
            except nx.NodeNotFound:
                continue
        return all_paths

    def compute_all_paths(self) -> list[list[Hit]]:
        """Return all leaf → head paths in the provenance graph.

        Ported from ``apiutils.Provenance.compute_all_paths``.
        """
        _leafs, heads = self.get_leafs_and_heads()
        all_paths: list[list[Hit]] = []
        for h in heads:
            paths = self.compute_paths_with(h)
            all_paths.extend(paths)
        return all_paths

    def compute_paths_with(self, a: Hit) -> list[list[Hit]]:
        """Return all paths that pass through *a*.

        If *a* is a leaf, returns all paths from *a* to any head.
        If *a* is a head, returns all paths from any leaf to *a*.
        Otherwise, stitches upstream (leaf→a) and downstream (a→head) paths.

        Ported from ``apiutils.Provenance.compute_paths_with``.
        """
        leafs, heads = self.get_leafs_and_heads()
        all_paths: list[list[Hit]] = []

        if a in leafs:
            for h in heads:
                try:
                    paths = nx.all_simple_paths(self._p_graph, a, h)
                    all_paths.extend(paths)
                except nx.NodeNotFound:
                    continue
        elif a in heads:
            for leaf in leafs:
                try:
                    paths = nx.all_simple_paths(self._p_graph, leaf, a)
                    all_paths.extend(paths)
                except nx.NodeNotFound:
                    continue
        else:
            # Stitch upstream + downstream
            upstreams: list[list[Hit]] = []
            for leaf in leafs:
                try:
                    paths = nx.all_simple_paths(self._p_graph, leaf, a)
                    upstreams.extend(paths)
                except nx.NodeNotFound:
                    continue
            downstreams: list[list[Hit]] = []
            for h in heads:
                try:
                    paths = nx.all_simple_paths(self._p_graph, a, h)
                    downstreams.extend(paths)
                except nx.NodeNotFound:
                    continue

            if len(downstreams) > len(upstreams):
                for d in downstreams:
                    for u in upstreams:
                        all_paths.append(u + d)
            else:
                for u in upstreams:
                    for d in downstreams:
                        all_paths.append(u + d)

        return all_paths

    # ── Explanation ──────────────────────────────────────────────────

    def explain_path(self, p: list[Hit]) -> str:
        """Given a specific path, walk its edges and return a human-readable story.

        Ported from ``apiutils.Provenance.explain_path``.
        """
        def _name(h: Hit) -> str:
            return f"{h.source_name}:{h.field_name}"

        def _edge_str(edge_info: dict) -> str:
            return ", ".join(str(k) for k in edge_info.keys())

        explanation = ""
        for idx in range(len(p) - 1):
            src, trg = p[idx], p[idx + 1]
            edge_info = self._p_graph[src][trg]
            explanation += f"{_name(src)} -> {_edge_str(edge_info)} -> {_name(trg)}\n"
        return explanation

    def explain(self, target: Hit) -> str:
        """Return a human-readable explanation of how *target* was reached.

        Finds all simple paths from roots to *target* and formats them.
        """
        roots = [n for n in self._p_graph if self._p_graph.in_degree(n) == 0]
        lines: list[str] = []
        for root in roots:
            try:
                for path in nx.all_simple_paths(self._p_graph, root, target):
                    steps: list[str] = []
                    for i in range(len(path) - 1):
                        src, tgt = path[i], path[i + 1]
                        edge_data = self._p_graph[src][tgt]
                        ops = [str(v.get("op", "?")) for v in edge_data.values()]
                        steps.append(f"{src} -[{','.join(ops)}]-> {tgt}")
                    lines.append(" | ".join(steps))
            except nx.NodeNotFound:
                continue
        return "\n".join(lines) if lines else "(no provenance)"


# ---------------------------------------------------------------------------
# DRS — Domain Result Set
# ---------------------------------------------------------------------------

class DRS:
    """An iterable result set with full provenance tracking.

    Port of ``apiutils.DRS``.  Supports set algebra (``&``, ``|``, ``-``)
    with provenance propagation, AND/OR edge annotations, path queries,
    why/how provenance queries, and certainty/coverage ranking.
    """

    class RankingCriteria(Enum):
        CERTAINTY = 0
        COVERAGE = 1

    def __init__(
        self,
        data: list[Hit],
        operation: Operation,
        mode: DRSMode = DRSMode.FIELDS,
        lean_drs: bool = False,
    ) -> None:
        self.data = list(data)
        self.mode = mode
        if not lean_drs:
            self._provenance = Provenance(data, operation)
        else:
            # Lightweight mode: create an empty provenance (no graph work)
            self._provenance = Provenance([], Operation(OP.NONE))
        self._table_view: list[str] = []
        # Ranking state
        self._ranked: bool = False
        self._rank_data: dict[Hit, dict] = defaultdict(dict)
        self._ranking_criteria: DRS.RankingCriteria | None = None
        self._chosen_rank: list[tuple[Hit, object]] = []
        self._origin_values_coverage: dict[Hit, int] = {}

    # ── Data mutation (ported from legacy set_data) ──────────────────

    def set_data(self, data: list[Hit]) -> DRS:
        """Replace the data list and reset iteration/ranking state."""
        self.data = list(data)
        self._table_view = []
        self._ranked = False
        return self

    # ── Iteration ────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Hit]:
        if self.mode == DRSMode.TABLE:
            # Lazy de-duplication by source_name (matches legacy table-mode)
            if not self._table_view:
                seen: set[str] = set()
                for h in self.data:
                    if h.source_name not in seen:
                        self._table_view.append(h.source_name)
                        seen.add(h.source_name)
            return iter(self._table_view)  # type: ignore[return-value]
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __bool__(self) -> bool:
        return len(self.data) > 0

    def size(self) -> int:
        return len(self.data)

    # ── Provenance accessors ─────────────────────────────────────────

    @property
    def provenance(self) -> Provenance:
        return self._provenance

    def get_provenance(self) -> Provenance:
        """Legacy-compatible accessor."""
        return self._provenance

    # ── Provenance: absorb with AND/OR annotations ───────────────────

    def absorb_provenance(
        self,
        drs: DRS,
        annotate_and_edges: bool = False,
        annotate_or_edges: bool = False,
    ) -> DRS:
        """Merge provenance of *drs* into self, **not** the data.

        Optionally annotates overlapping edges with ``AND`` or ``OR``
        labels so the provenance graph records intersection/union logic.

        Ported from ``apiutils.DRS.absorb_provenance``.
        """
        def _annotate_union_edges(merged: nx.MultiDiGraph, label: str) -> None:
            my_data = set(self.data)
            merging_data = set(drs.data)
            overlap = my_data & merging_data
            for el in overlap:
                if el not in self._provenance.graph or el not in drs._provenance.graph:
                    continue
                for edges in (
                    self._provenance.graph.in_edges(el, data=True),
                    drs._provenance.graph.in_edges(el, data=True),
                ):
                    for src, tar, _dic in edges:
                        if src in merged and tar in merged[src]:
                            edge_data = merged[src][tar]
                            for e in edge_data:
                                edge_data[e][label] = 1

        self._ranked = False
        prov_graph_of_merging = drs.get_provenance().prov_graph()
        merged = nx.compose(self._provenance.prov_graph(), prov_graph_of_merging)

        if annotate_and_edges:
            _annotate_union_edges(merged, "AND")
        if annotate_or_edges:
            _annotate_union_edges(merged, "OR")

        self._provenance.swap_p_graph(merged)
        return self

    def absorb(self, other: DRS) -> DRS:
        """Merge *other* into self (data union + provenance merge). Returns self."""
        self._ranked = False
        merging_data = set(other.data)
        my_data = set(self.data)
        new_data = merging_data | my_data
        self.set_data(list(new_data))
        self.absorb_provenance(other)
        return self

    # ── Set operations (with AND/OR provenance annotations) ──────────

    def union(self, other: DRS) -> DRS:
        """``self | other`` with provenance merge."""
        self._ranked = False
        result = DRS([], Operation(OP.NONE), mode=self.mode)
        merging_data = set(other.data)
        my_data = set(self.data)
        result.set_data(list(merging_data | my_data))
        result.absorb_provenance(self)
        result.absorb_provenance(other)
        return result

    def __or__(self, other: DRS) -> DRS:
        return self.union(other)

    def intersection(self, other: DRS) -> DRS:
        """``self & other`` — keep only Hits present in both, with AND annotation."""
        self._ranked = False
        result = DRS([], Operation(OP.NONE), mode=self.mode)

        if other.mode == DRSMode.TABLE:
            merging_tables = [(h.source_name, h) for h in other.data]
            my_tables = [(h.source_name, h) for h in self.data]
            new_data: list[Hit] = []
            for table, hit_ext in merging_tables:
                for t, hit_in in my_tables:
                    if table == t:
                        new_data.append(hit_ext)
                        new_data.append(hit_in)
            result.set_data(new_data)
        else:
            merging_data = set(other.data)
            my_data = set(self.data)
            result.set_data(list(merging_data & my_data))

        result.absorb_provenance(self, annotate_and_edges=True)
        result.absorb_provenance(other, annotate_and_edges=True)
        return result

    def __and__(self, other: DRS) -> DRS:
        return self.intersection(other)

    def difference(self, other: DRS) -> DRS:
        """``self - other``."""
        self._ranked = False
        result = DRS([], Operation(OP.NONE), mode=self.mode)
        merging_data = set(other.data)
        my_data = set(self.data)
        result.set_data(list(my_data - merging_data))
        result.absorb_provenance(self)
        result.absorb_provenance(other)
        return result

    def set_difference(self, other: DRS) -> DRS:
        """Legacy alias for ``difference``."""
        return self.difference(other)

    def __sub__(self, other: DRS) -> DRS:
        return self.difference(other)

    # ── Mode ─────────────────────────────────────────────────────────

    def set_table_mode(self) -> None:
        self.mode = DRSMode.TABLE

    def set_fields_mode(self) -> None:
        self.mode = DRSMode.FIELDS

    # ── Path functions ───────────────────────────────────────────────

    def paths(self) -> list[list[Hit]]:
        """Return all paths contained in the provenance graph.

        Ported from ``apiutils.DRS.paths``.
        """
        return self._provenance.compute_all_paths()

    def path(self, a: Hit) -> list[list[Hit]]:
        """Return all provenance paths that contain *a*.

        Ported from ``apiutils.DRS.path``.
        """
        return self._provenance.compute_paths_with(a)

    # ── Query provenance: why / how ──────────────────────────────────

    def why(self, a: Hit) -> list[Hit]:
        """Return the origin Hits that led to *a* appearing in this DRS.

        Walks provenance backward from *a* to find all leaf (origin) nodes.

        Ported from ``apiutils.DRS.why``.
        """
        if a not in self.data:
            return []
        paths = self._provenance.compute_paths_from_origin_to(a)
        origins: set[Hit] = set()
        for p in paths:
            origins.add(p[0])
        return list(origins)

    def why_id(self, nid: int | str) -> list[Hit]:
        """Like ``why`` but accepts a nid instead of a Hit."""
        nid_str = str(nid)
        hit = None
        for x in self.data:
            if str(x.nid) == nid_str:
                hit = x
                break
        if hit is None:
            return []
        return self.why(hit)

    def how(self, a: Hit) -> list[str]:
        """Return human-readable explanation strings for how *a* was derived.

        Ported from ``apiutils.DRS.how``.
        """
        if a not in self.data:
            return []
        paths = self._provenance.compute_paths_from_origin_to(a)
        return [self._provenance.explain_path(p) for p in paths]

    def how_id(self, nid: int | str) -> list[str]:
        """Like ``how`` but accepts a nid instead of a Hit."""
        nid_str = str(nid)
        hit = None
        for x in self.data:
            if str(x.nid) == nid_str:
                hit = x
                break
        if hit is None:
            return []
        return self.how(hit)

    # ── Ranking ──────────────────────────────────────────────────────

    def _compute_certainty_scores(self) -> None:
        """Compute certainty scores by recursive traversal of the reversed provenance graph.

        Ported from ``apiutils.DRS._compute_certainty_scores``.
        """
        def _get_score(pg: nx.MultiDiGraph, src: Hit, visited: set[Hit]) -> float:
            current_score = float(src.score) if src.score is not None else 0.0
            ns = [x for x in pg.neighbors(src) if x not in visited]
            if len(ns) == 1:
                visited.add(ns[0])
                current_score += _get_score(pg, ns[0], visited)
            elif len(ns) > 1:
                max_score = 0.0
                for n in ns:
                    visited.add(n)
                    s = _get_score(pg, n, visited)
                    if s > max_score:
                        max_score = s
                current_score += max_score
            return current_score

        pg = self._provenance.prov_graph().reverse()
        visited: set[Hit] = set()
        for el in self.data:
            if el not in visited:
                score = _get_score(pg, el, visited)
                self._rank_data[el]["certainty_score"] = score

    def _compute_coverage_scores(self) -> None:
        """Compute coverage scores using origin-set tracking.

        Uses a simple set-based approach (no bitarray dependency) that is
        functionally equivalent to the legacy bitarray implementation.

        Ported from ``apiutils.DRS._compute_coverage_scores``.
        """
        leafs, _heads = self._provenance.get_leafs_and_heads()
        total_number = len(leafs) if leafs else 1

        # Assign indices to origin values
        self._origin_values_coverage = {}
        for i, origin in enumerate(leafs):
            self._origin_values_coverage[origin] = i

        for el in self.data:
            elements = self.why(el)
            coverage = float(len(elements)) / float(total_number)
            coverage_set = set(self._origin_values_coverage.get(e, -1) for e in elements)
            self._rank_data[el]["coverage_score"] = (coverage, coverage_set)

    def compute_ranking_scores(self) -> None:
        """Compute both certainty and coverage scores."""
        self._compute_certainty_scores()
        self._compute_coverage_scores()
        self._ranked = True

    def rank_certainty(self) -> DRS:
        """Rank results by certainty (aggregate provenance scores).

        Ported from ``apiutils.DRS.rank_certainty``.
        """
        if not self._ranked:
            self.compute_ranking_scores()

        elements: list[tuple[Hit, float]] = []
        for el, score_dict in self._rank_data.items():
            value = (el, score_dict.get("certainty_score", 0.0))
            elements.append(value)
        elements.sort(key=lambda a: a[1], reverse=True)
        self.data = [el for el, _score in elements]
        self._ranking_criteria = self.RankingCriteria.CERTAINTY
        self._chosen_rank = elements
        return self

    def rank_coverage(self) -> DRS:
        """Rank results by coverage (how many origins they trace back to).

        Ported from ``apiutils.DRS.rank_coverage``.
        """
        if not self._ranked:
            self.compute_ranking_scores()

        elements: list[tuple[Hit, object]] = []
        for el, score_dict in self._rank_data.items():
            value = (el, score_dict.get("coverage_score", (0.0, set())))
            elements.append(value)
        elements.sort(key=lambda a: a[1][0], reverse=True)  # type: ignore[index]
        self.data = [el for el, _score in elements]
        self._ranking_criteria = self.RankingCriteria.COVERAGE
        self._chosen_rank = elements
        return self

    # ── Display / debug ──────────────────────────────────────────────

    def debug_print(self) -> None:
        """Print data count vs provenance node count (legacy ``prov_size``)."""
        len_data = len(self.data)
        total_nodes = len(self._provenance.prov_graph().nodes())
        print(f"Total data: {len_data}")
        print(f"Total nodes prov graph: {total_nodes}")

    def prov_size(self) -> None:
        """Alias for ``debug_print``."""
        self.debug_print()

    def visualize_provenance(self, labels: bool = False) -> None:
        """Render the provenance graph with matplotlib.

        Ported from ``apiutils.DRS.visualize_provenance``.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for visualize_provenance()")
            return
        if labels:
            nx.draw_networkx(self.get_provenance().prov_graph())
        else:
            nx.draw(self.get_provenance().prov_graph())
        plt.show()

    def to_dict(self) -> dict:
        """Serialize data + provenance edges to a JSON-friendly dict.

        Ported from ``apiutils.DRS.__dict__``.
        """
        saved_mode = self.mode
        sources: dict[str, dict] = {}
        edges: list[tuple[dict, dict]] = []
        self.set_fields_mode()

        for x in self.data:
            table = x.source_name
            hit_dict = {
                "nid": x.nid,
                "db_name": x.db_name,
                "source_name": x.source_name,
                "field_name": x.field_name,
                "score": x.score,
            }
            if table not in sources:
                sources[table] = {"source_res": hit_dict, "field_res": []}
            sources[table]["field_res"].append(hit_dict)

        for src, dst in self.get_provenance().prov_graph().edges():
            origin = {
                "nid": src.nid, "db_name": src.db_name,
                "source_name": src.source_name, "field_name": src.field_name,
                "score": src.score,
            }
            destination = {
                "nid": dst.nid, "db_name": dst.db_name,
                "source_name": dst.source_name, "field_name": dst.field_name,
                "score": dst.score,
            }
            edges.append((origin, destination))

        self.mode = saved_mode
        return {"sources": sources, "edges": edges}

    # ── Print helpers ────────────────────────────────────────────────

    def print_tables(self) -> None:
        saved = self.mode
        self.set_table_mode()
        for x in self:
            print(x)
        self.mode = saved

    def print_columns(self) -> None:
        saved = self.mode
        self.set_fields_mode()
        seen: set = set()
        for x in self.data:
            if x.nid not in seen:
                print(x)
                seen.add(x.nid)
        self.mode = saved

    def pretty_print_columns(self) -> None:
        saved = self.mode
        self.set_fields_mode()
        seen: set = set()
        for x in self.data:
            if x.nid not in seen:
                print(f"DB: {x.db_name:20s} TABLE: {x.source_name:30s} FIELD: {x.field_name:30s}")
                seen.add(x.nid)
        self.mode = saved

    def print_tables_with_scores(self) -> None:
        """Print table-level aggregated scores (certainty or coverage)."""
        if not self._chosen_rank:
            return

        saved = self.mode
        self.set_fields_mode()
        group: dict[str, object] = {}

        if self._ranking_criteria == self.RankingCriteria.CERTAINTY:
            for x, score in self._chosen_rank:
                old = group.get(x.source_name, 0.0)
                group[x.source_name] = old + score  # type: ignore[operator]
            ranked = sorted(group.items(), key=lambda a: a[1], reverse=True)  # type: ignore[arg-type]
        elif self._ranking_criteria == self.RankingCriteria.COVERAGE:
            for x, score in self._chosen_rank:
                _cov, new_set = score  # type: ignore[misc]
                old_set = group.get(x.source_name, set())
                group[x.source_name] = old_set | new_set  # type: ignore[operator]
            ranked = sorted(
                [(t, len(s) / max(len(s), 1)) for t, s in group.items()],  # type: ignore[arg-type]
                key=lambda a: a[1], reverse=True,
            )
        else:
            ranked = []

        for item in ranked:
            print(item)
        self.mode = saved

    def print_columns_with_scores(self) -> None:
        saved = self.mode
        self.set_fields_mode()
        seen: set = set()
        for el, score in self._chosen_rank:
            if el.nid not in seen:
                print(f"{el} -> {score}")
                seen.add(el.nid)
        self.mode = saved

    def pretty_print_columns_with_scores(self) -> None:
        saved = self.mode
        self.set_fields_mode()
        seen: set = set()
        for x, score in self._chosen_rank:
            if x.nid not in seen:
                print(
                    f"DB: {x.db_name:20s} TABLE: {x.source_name:30s} "
                    f"FIELD: {x.field_name:30s} SCORE: {str(score):10s}"
                )
                seen.add(x.nid)
        self.mode = saved

    def __repr__(self) -> str:
        return f"DRS({len(self.data)} hits, mode={self.mode.name})"

    def pretty(self) -> str:
        lines = [repr(self)]
        for h in self.data:
            lines.append(f"  {h}")
        return "\n".join(lines)
