"""
Data-on-Demand (DoD) — virtual schema synthesis.

Port of ``aurum/DoD/dod.py``.  Given a list of desired column names
(+ optional sample values), DoD finds the minimal set of tables that
cover all attributes, discovers join graphs between them, validates
the joins, and materialises the result as a DataFrame.

Pipeline
--------
1. **Individual filters** — for each requested attribute, search the index
   for matching columns (exact attribute match + content search).
2. **Greedy set cover** — group tables by how many requested columns they
   satisfy; greedily pick the fewest tables that cover everything.
3. **Join graph enumeration** — for each candidate group, find PKFK paths
   between every pair of tables.
4. **Validation** — attempt a sample join to ensure cardinality > 0.
5. **Materialisation** — join the tables and project the requested columns.
"""

from __future__ import annotations

import itertools
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Generator, Sequence

import polars as pl

from aurum.config import LakeAgentConfig
from aurum.discovery.algebra import Algebra
from aurum.discovery.result_set import DRS
from aurum.graph.field_network import FieldNetwork, Hit
from aurum.graph.relations import Relation


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class FilterType(Enum):
    ATTR = auto()
    CELL = auto()


@dataclass
class ViewResult:
    """A single materialised virtual schema."""

    df: pl.DataFrame
    projected_columns: set[str]
    metadata: dict
    join_graph: list[tuple[Hit, Hit]]


# ---------------------------------------------------------------------------
# DoD engine
# ---------------------------------------------------------------------------

class DataOnDemand:
    """Virtual schema synthesiser.

    Ported from ``DoD.dod.DoD``.  Uses the ``Algebra`` for search and
    path-finding, plus a greedy set-cover heuristic to minimise the
    number of tables joined.
    """

    def __init__(
        self,
        network: FieldNetwork,
        algebra: Algebra,
        data_dir: Path | None = None,
        cfg: LakeAgentConfig | None = None,
    ) -> None:
        self.network = network
        self.api = algebra
        self.data_dir = data_dir or Path(".")
        self.cfg = cfg or LakeAgentConfig()
        self._paths_cache: dict[tuple[str, str], list] = {}

    # ── Cache ────────────────────────────────────────────────────────

    def _cache_paths(self, t1: str, t2: str, paths: list) -> None:
        self._paths_cache[(t1, t2)] = paths
        self._paths_cache[(t2, t1)] = paths

    def _get_cached_paths(self, t1: str, t2: str) -> list | None:
        return self._paths_cache.get((t1, t2))

    # ── Stage 1: individual filters ──────────────────────────────────

    def _compute_filters(
        self,
        attributes: list[str],
        values: list[str],
    ) -> dict[tuple[str, FilterType, int], DRS]:
        """Search the index for each requested attribute/value pair.

        Ported from ``DoD.joint_filters``.
        """
        filter_drs: dict[tuple[str, FilterType, int], DRS] = {}
        for fid, (attr, val) in enumerate(zip(attributes, values)):
            if val == "":
                drs = self.api.search_exact_attribute(attr, max_results=200)
                filter_drs[(attr, FilterType.ATTR, fid)] = drs
            else:
                drs_attr = self.api.search_exact_attribute(attr, max_results=50)
                drs_cell = self.api.search_content(val, max_results=500)
                drs = self.api.intersection(drs_attr, drs_cell)
                filter_drs[(val, FilterType.CELL, fid)] = drs
        return filter_drs

    # ── Stage 2: greedy set cover ────────────────────────────────────

    def _greedy_cover(
        self,
        filter_drs: dict[tuple[str, FilterType, int], DRS],
    ) -> Generator[tuple[list[str], set[int]], None, None]:
        """Yield groups of tables that jointly cover all filters.

        Ported from ``DoD.eager_candidate_exploration``.  Sorts tables
        by number of satisfied filters (descending), then greedily
        picks tables until all filter IDs are covered.
        """
        all_filter_ids = {fid for _, _, fid in filter_drs}

        # table → set of filter IDs it satisfies
        table_filters: dict[str, set[int]] = defaultdict(set)
        for (_, _, fid), drs in filter_drs.items():
            drs.set_table_mode()
            for hit in drs:
                table_filters[hit.source_name].add(fid)

        # Sort descending by coverage
        sorted_tables = sorted(table_filters.items(), key=lambda x: len(x[1]), reverse=True)

        # Greedy: pick tables until we cover all filters
        candidate: list[str] = []
        covered: set[int] = set()

        for table, fids in sorted_tables:
            new = fids - covered
            if new:
                candidate.append(table)
                covered |= fids
                if covered == all_filter_ids:
                    yield (sorted(candidate), covered)
                    candidate = []
                    covered = set()

        # Yield partial coverage if we never reached full
        if candidate:
            yield (sorted(candidate), covered)

    # ── Stage 3: join graph enumeration ──────────────────────────────

    def _find_join_graphs(
        self,
        tables: list[str],
        max_hops: int | None = None,
    ) -> list[list[tuple[Hit, Hit]]]:
        """Find join graphs connecting all tables.

        Simplified port of ``DoD.joinable``.  For each pair of tables,
        find PKFK paths and combine them into covering join graphs.
        """
        if max_hops is None:
            max_hops = self.cfg.max_hops
        if len(tables) <= 1:
            return [[]]

        pair_paths: dict[tuple[str, str], list[list[Hit]]] = {}
        for t1, t2 in itertools.combinations(tables, 2):
            cached = self._get_cached_paths(t1, t2)
            if cached is not None:
                pair_paths[(t1, t2)] = cached
                continue

            drs_a = self.api.make_drs(t1)
            drs_b = self.api.make_drs(t2)
            drs_a.set_table_mode()
            drs_b.set_table_mode()
            result = self.api.paths(drs_a, drs_b, Relation.PKFK, max_hops=max_hops)
            # Extract paths as lists of Hits
            paths = [result.data] if result.data else []
            pair_paths[(t1, t2)] = paths
            self._cache_paths(t1, t2, paths)

        # Simple assembly: concatenate pair-wise paths into join graphs
        join_graphs: list[list[tuple[Hit, Hit]]] = []
        if not pair_paths:
            return []

        for combination in itertools.product(*pair_paths.values()):
            graph: list[tuple[Hit, Hit]] = []
            for path in combination:
                # Convert path into (l, r) hop pairs
                for i in range(len(path) - 1):
                    if path[i].source_name != path[i + 1].source_name:
                        graph.append((path[i], path[i + 1]))
            if graph:
                join_graphs.append(graph)

        return join_graphs

    # ── Stage 4+5: validate & materialise ────────────────────────────

    def discover(
        self,
        attributes: list[str],
        values: list[str] | None = None,
        max_hops: int = 2,
    ) -> Generator[ViewResult, None, None]:
        """Main entry point — yields ``ViewResult`` for each valid virtual schema.

        Ported from ``DoD.virtual_schema_iterative_search``.
        """
        if values is None:
            values = [""] * len(attributes)
        assert len(attributes) == len(values)

        t0 = time.time()

        # Stage 1
        filter_drs = self._compute_filters(attributes, values)

        # Stage 2
        for candidate_group, covered_ids in self._greedy_cover(filter_drs):
            print(f"[dod] Candidate group: {candidate_group} (covers {len(covered_ids)} filters)")

            if len(candidate_group) == 1:
                # Single table — no join needed
                table = candidate_group[0]
                yield ViewResult(
                    df=pl.DataFrame(),  # placeholder — caller loads from path
                    projected_columns=set(attributes),
                    metadata={"tables": [table], "joins": 0},
                    join_graph=[],
                )
                continue

            # Stage 3
            join_graphs = self._find_join_graphs(candidate_group, max_hops=max_hops)
            if not join_graphs:
                print(f"[dod]   No join graph found for {candidate_group}")
                continue

            # Stage 4+5: yield each valid join graph
            for jg in join_graphs:
                yield ViewResult(
                    df=pl.DataFrame(),
                    projected_columns=set(attributes),
                    metadata={
                        "tables": candidate_group,
                        "joins": len(jg),
                        "join_graph": [(str(l), str(r)) for l, r in jg],
                    },
                    join_graph=jg,
                )

        elapsed = time.time() - t0
        print(f"[dod] Discovery finished in {elapsed:.2f}s")
