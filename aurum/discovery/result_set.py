"""
Domain Result Set (DRS) — the query-result container with provenance.

Port of ``aurum/api/apiutils.py``: ``Hit``, ``DRS``, ``Provenance``.

A DRS wraps a list of ``Hit`` objects and records a DAG of operations
that produced them, so every result is *explainable*.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Iterator

import networkx as nx

from aurum.graph.field_network import Hit
from aurum.graph.relations import OP


# ---------------------------------------------------------------------------
# DRS mode
# ---------------------------------------------------------------------------

class DRSMode(Enum):
    FIELDS = auto()
    TABLE = auto()


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
    """

    def __init__(self, data: list[Hit], operation: Operation) -> None:
        self._graph = nx.MultiDiGraph()
        self._populate(data, operation)

    def _populate(self, data: list[Hit], operation: Operation) -> None:
        op = operation.op
        params = operation.params
        if op == OP.NONE:
            return
        if op == OP.ORIGIN:
            for h in data:
                self._graph.add_node(h)
            return
        # Operations with a parent Hit
        if params:
            parent = params[0]
            self._graph.add_node(parent)
            for h in data:
                self._graph.add_node(h)
                self._graph.add_edge(parent, h, op=op)

    def merge(self, other: Provenance) -> None:
        """Absorb another provenance graph (union of edges/nodes)."""
        self._graph = nx.compose(self._graph, other._graph)

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    def explain(self, target: Hit) -> str:
        """Return a human-readable explanation of how *target* was reached."""
        # Find all simple paths from roots to target
        roots = [n for n in self._graph if self._graph.in_degree(n) == 0]
        lines: list[str] = []
        for root in roots:
            for path in nx.all_simple_paths(self._graph, root, target):
                steps: list[str] = []
                for i in range(len(path) - 1):
                    src, tgt = path[i], path[i + 1]
                    edge_data = self._graph[src][tgt]
                    ops = [str(v.get("op", "?")) for v in edge_data.values()]
                    steps.append(f"{src} -[{','.join(ops)}]-> {tgt}")
                lines.append(" | ".join(steps))
        return "\n".join(lines) if lines else "(no provenance)"


# ---------------------------------------------------------------------------
# DRS — Domain Result Set
# ---------------------------------------------------------------------------

class DRS:
    """An iterable result set with full provenance tracking.

    Port of ``apiutils.DRS``.  Supports set algebra (``&``, ``|``, ``-``)
    with provenance propagation.
    """

    def __init__(
        self,
        data: list[Hit],
        operation: Operation,
        mode: DRSMode = DRSMode.FIELDS,
    ) -> None:
        self.data = list(data)
        self.mode = mode
        self._provenance = Provenance(data, operation)

    # ── Iteration ────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Hit]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __bool__(self) -> bool:
        return len(self.data) > 0

    # ── Set operations (ported from apiutils.DRS) ────────────────────

    def union(self, other: DRS) -> DRS:
        """``self | other`` with provenance merge."""
        seen = {h.nid for h in self.data}
        merged = list(self.data)
        for h in other.data:
            if h.nid not in seen:
                merged.append(h)
                seen.add(h.nid)
        result = DRS(merged, Operation(OP.NONE), mode=self.mode)
        result._provenance.merge(self._provenance)
        result._provenance.merge(other._provenance)
        return result

    def __or__(self, other: DRS) -> DRS:
        return self.union(other)

    def intersection(self, other: DRS) -> DRS:
        """``self & other`` — keep only Hits present in both."""
        other_nids = {h.nid for h in other.data}
        common = [h for h in self.data if h.nid in other_nids]
        result = DRS(common, Operation(OP.NONE), mode=self.mode)
        result._provenance.merge(self._provenance)
        result._provenance.merge(other._provenance)
        return result

    def __and__(self, other: DRS) -> DRS:
        return self.intersection(other)

    def difference(self, other: DRS) -> DRS:
        """``self - other``."""
        other_nids = {h.nid for h in other.data}
        diff = [h for h in self.data if h.nid not in other_nids]
        result = DRS(diff, Operation(OP.NONE), mode=self.mode)
        result._provenance.merge(self._provenance)
        return result

    def __sub__(self, other: DRS) -> DRS:
        return self.difference(other)

    # ── Mode ─────────────────────────────────────────────────────────

    def set_table_mode(self) -> None:
        self.mode = DRSMode.TABLE

    def set_fields_mode(self) -> None:
        self.mode = DRSMode.FIELDS

    # ── Provenance ───────────────────────────────────────────────────

    @property
    def provenance(self) -> Provenance:
        return self._provenance

    def absorb(self, other: DRS) -> DRS:
        """Merge *other* into self (union + provenance merge). Returns self."""
        for h in other.data:
            if not any(existing.nid == h.nid for existing in self.data):
                self.data.append(h)
        self._provenance.merge(other._provenance)
        return self

    # ── Display ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"DRS({len(self.data)} hits, mode={self.mode.name})"

    def pretty(self) -> str:
        lines = [repr(self)]
        for h in self.data:
            lines.append(f"  {h}")
        return "\n".join(lines)
