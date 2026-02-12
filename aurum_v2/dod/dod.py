"""
Data on Demand (DoD) — discover and materialise virtual schemas.

Given a list of desired attributes (and optional sample values), DoD:

1. Searches the ES store for tables matching each filter.
2. Groups tables by filter coverage (eager candidate exploration).
3. Finds join graphs connecting tables in each group (via ``api.paths``).
4. Validates that each join graph is materializable (trial joins).
5. Materializes the join graph and projects the requested columns.

Direct port of ``DoD/dod.py``.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple

import itertools
import pandas as pd

from aurum_v2.dod import join_utils as dpu
from aurum_v2.models.drs import DRS
from aurum_v2.models.hit import Hit
from aurum_v2.models.relation import Relation

if TYPE_CHECKING:
    from aurum_v2.config import AurumConfig
    from aurum_v2.discovery.api import API

__all__ = ["DoD", "FilterType"]


class FilterType(Enum):
    ATTR = 0
    CELL = 1


class DoD:
    """Data on Demand engine.

    Parameters
    ----------
    network : FieldNetwork
        The loaded column‑relation graph.
    store_client : StoreHandler
        Elasticsearch connection.
    csv_separator : str
        Column separator used when reading CSV data files.
    """

    def __init__(
        self,
        network,
        store_client,
        csv_separator: str = ",",
    ) -> None:
        from aurum_v2.discovery.api import API

        self.aurum_api: API = API(network=network, store_client=store_client)
        self.paths_cache: Dict[Tuple[str, str], list] = {}
        dpu.configure_csv_separator(csv_separator)

    # ------------------------------------------------------------------
    # Path cache
    # ------------------------------------------------------------------

    def place_paths_in_cache(self, t1: str, t2: str, paths: list) -> None:
        self.paths_cache[(t1, t2)] = paths
        self.paths_cache[(t2, t1)] = paths

    def are_paths_in_cache(self, t1: str, t2: str) -> Optional[list]:
        return self.paths_cache.get((t1, t2)) or self.paths_cache.get((t2, t1))

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    def individual_filters(self, sch_def: dict) -> Dict[tuple, DRS]:
        """Obtain DRS sets that fulfil individual attribute / cell filters."""
        raise NotImplementedError

    def joint_filters(self, sch_def: dict) -> Dict[tuple, DRS]:
        """Obtain DRS sets using joint (attribute ∩ cell) filters.

        If a value is empty, use attribute search only; otherwise intersect
        attribute DRS with content DRS.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def virtual_schema_iterative_search(
        self,
        list_attributes: List[str],
        list_samples: List[str],
        perf_stats: dict,
        max_hops: int = 2,
        debug_enumerate_all_jps: bool = False,
    ) -> Iterator[Tuple[pd.DataFrame, set, dict]]:
        """Main DoD entry point — a generator yielding materialised views.

        Yields
        ------
        (materialized_df, attrs_to_project, view_metadata)

        Algorithm (5 stages, mirrors legacy ``dod.py``):

        **Stage 1** — Build joint filters from attributes + sample values.

        **Stage 2** — Group tables by filter coverage.
            Sort tables by (num unique filters covered, lexicographic).
            Use ``eager_candidate_exploration()`` to enumerate groups that
            cover all filters.

        **Stage 3** — For each candidate group with >1 table, discover join
            graphs via ``self.joinable()``.

        **Stage 4** — Validate each join graph with
            ``self.is_join_graph_materializable()`` (trial joins per hop).

        **Stage 5** — Materialise valid join graphs via
            ``dpu.materialize_join_graph_sample()`` and yield results.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Candidate group enumeration
    # ------------------------------------------------------------------

    def _eager_candidate_exploration(
        self,
        table_fulfilled_filters: OrderedDict,
        filter_drs: dict,
    ) -> Iterator[Tuple[list, set]]:
        """Eagerly enumerate groups of tables covering as many filters as possible.

        For each pivot table, greedily add further tables that contribute new
        filters until full coverage is achieved.  Backup groups (partial
        coverage) are yielded at the end.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Join‑graph discovery
    # ------------------------------------------------------------------

    def joinable(
        self,
        group_tables: List[str],
        cache_unjoinable_pairs: dict,
        max_hops: int = 2,
    ) -> List[List[Tuple[Hit, Hit]]]:
        """Find all join graphs connecting *group_tables*.

        Algorithm:

        1. For each pair ``(t1, t2)`` in ``combinations(group_tables, 2)``:
           a. Check the unjoinable‑pair cache; skip if present.
           b. ``api.paths(t1, t2, PKFK, max_hops, lean_search=True)``
           c. Extract ``drs.paths()`` — list of leaf→head paths.
           d. Track which *group_tables* each path covers.

        2. Enumerate all combinations of pair‑wise paths via
           ``itertools.product``.

        3. For each pair of paths, combine if they jointly cover more tables
           and share at least one table.

        4. Transform paths into pair‑hop format
           ``[(left_hit, right_hit), …]``.

        5. Deduplicate by ``compute_join_graph_id``.

        6. Filter to only those covering *all* group tables.

        7. Sort by number of joins (ascending — prefer fewer hops).
        """
        raise NotImplementedError

    def transform_join_path_to_pair_hop(
        self, join_path: list
    ) -> List[Tuple[Hit, Hit]]:
        """Convert a linear provenance path into ``[(l, r), …]`` hop pairs.

        Pairs where both sides are in the same table are removed (no
        self‑join needed).
        """
        raise NotImplementedError

    def compute_join_graph_id(self, join_graph: List[Tuple[Hit, Hit]]) -> int:
        """Hash a join graph for deduplication (sum of nid hashes)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Materialisation validation
    # ------------------------------------------------------------------

    def is_join_graph_materializable(
        self,
        join_graph: List[Tuple[Hit, Hit]],
        table_fulfilled_filters: dict,
    ) -> bool:
        """Validate a join graph by performing trial joins per hop.

        For each ``(l, r)`` hop:

        1. Read ``l``'s CSV, apply any cell‑value filters.
        2. Read ``r``'s CSV, apply any cell‑value filters.
        3. Perform ``pd.merge`` on the join key.
        4. If the result has 0 rows → return ``False``.

        Returns ``True`` only if all hops succeed.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def materialize_join_graphs(
        self,
        materializable_join_graphs: List[Tuple[list, set]],
    ) -> List[Tuple[pd.DataFrame, set, dict]]:
        """Materialise a batch of validated join graphs.

        For each ``(join_graph, filters)``:
        1. Determine attributes to project.
        2. Call ``dpu.materialize_join_graph_sample(jg, self)``.
        3. Collect ``(df, attrs_to_project, view_metadata)``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # View metadata
    # ------------------------------------------------------------------

    def format_join_graph_into_nodes_edges(
        self, join_graph: List[Tuple[Hit, Hit]]
    ) -> dict:
        """Produce ``{"nodes": [...], "edges": [...]}`` for the web UI."""
        raise NotImplementedError
