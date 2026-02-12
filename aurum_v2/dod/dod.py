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

from collections import OrderedDict
from collections.abc import Iterator
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

from aurum_v2.dod import join_utils as dpu
from aurum_v2.models.drs import DRS
from aurum_v2.models.hit import Hit

if TYPE_CHECKING:
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
        self.paths_cache: dict[tuple[str, str], list] = {}
        dpu.configure_csv_separator(csv_separator)

    # ------------------------------------------------------------------
    # Path cache
    # ------------------------------------------------------------------

    def place_paths_in_cache(self, t1: str, t2: str, paths: list) -> None:
        self.paths_cache[(t1, t2)] = paths
        self.paths_cache[(t2, t1)] = paths

    def are_paths_in_cache(self, t1: str, t2: str) -> list | None:
        return self.paths_cache.get((t1, t2)) or self.paths_cache.get((t2, t1))

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    def individual_filters(self, sch_def: dict) -> dict[tuple, DRS]:
        """Obtain DRS sets that fulfil individual attribute / cell filters."""
        raise NotImplementedError

    def joint_filters(self, sch_def: dict) -> dict[tuple, DRS]:
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
        list_attributes: list[str],
        list_samples: list[str],
        perf_stats: dict,
        max_hops: int = 2,
        debug_enumerate_all_jps: bool = False,
    ) -> Iterator[tuple[pd.DataFrame, set, dict]]:
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
    ) -> Iterator[tuple[list, set]]:
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
        group_tables: list[str],
        cache_unjoinable_pairs: dict,
        max_hops: int = 2,
    ) -> list[list[tuple[Hit, Hit]]]:
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
    ) -> list[tuple[Hit, Hit]]:
        """Convert a linear provenance path into ``[(l, r), …]`` hop pairs.

        Pairs where both sides are in the same table are removed (no
        self‑join needed).
        """
        raise NotImplementedError

    def compute_join_graph_id(self, join_graph: list[tuple[Hit, Hit]]) -> int:
        """Hash a join graph for deduplication (sum of nid hashes)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Materialisation validation
    # ------------------------------------------------------------------

    def is_join_graph_materializable(
        self,
        join_graph: list[tuple[Hit, Hit]],
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
        materializable_join_graphs: list[tuple[list, set]],
    ) -> list[tuple[pd.DataFrame, set, dict]]:
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
        self, join_graph: list[tuple[Hit, Hit]]
    ) -> dict:
        """Produce ``{"nodes": [...], "edges": [...]}`` for the web UI."""
        raise NotImplementedError


# ======================================================================
# Module-level helpers  (missing from original v2 skeleton)
# ======================================================================


def obtain_table_paths(
    set_nids: dict[str, str],
    dod: DoD,
) -> dict[str, str]:
    """Batch-resolve filesystem/S3 paths for a set of tables.

    Port of ``DoD/data_processing_utils.obtain_table_paths``.

    Parameters
    ----------
    set_nids : dict[str, str]
        Mapping ``{table_name: nid}``.
    dod : DoD
        DoD instance whose ``api`` gives access to the store's
        :meth:`~StoreHandler.get_path_of`.

    Returns
    -------
    dict[str, str]
        ``{table_name: filesystem_or_s3_path}``.

    Algorithm:
        For each (table_name, nid) pair, call
        ``dod.api.helper.get_path_nid(nid)`` to resolve the filesystem
        path and store it in the result dict.
    """
    raise NotImplementedError


def rank_materializable_join_graphs(
    materializable_join_paths: list,
    table_path: dict[str, str],
    dod: DoD,
) -> list:
    """Score and sort join graphs by key-likelihood quality.

    Port of ``DoD/data_processing_utils.rank_materializable_join_graphs``.

    Parameters
    ----------
    materializable_join_paths : list
        List of materializable join paths, where each path is a list of
        ``(left_hit, right_hit, relation)`` triples.
    table_path : dict[str, str]
        Mapping ``{table_name: path}`` (from :func:`obtain_table_paths`).
    dod : DoD
        DoD instance (used to resolve field info).

    Returns
    -------
    list
        The input join paths sorted **descending** by average
        key-likelihood score.

    Algorithm:

    1. For each join path, iterate over hops.
    2. For the left-side table of each hop, load the CSV and call
       ``dpu.key_likelihood_ranking()`` to rank columns by how likely
       they are to be join keys (high cardinality, low nulls).
    3. Look up the join field's score in that ranking.
    4. Average hop-scores per join graph.
    5. Sort descending.

    Caches per-table key-likelihood so each CSV is loaded at most once.
    """
    raise NotImplementedError


def rank_materializable_join_paths_piece(
    materializable_join_paths: list,
    candidate_group: set[str],
    table_path: dict[str, str],
    dod: DoD,
) -> list:
    """Re-order join path hops by per-field key-likelihood rank.

    Port of
    ``DoD/data_processing_utils.rank_materializable_join_paths_piece``.

    Parameters
    ----------
    materializable_join_paths : list
        List of annotated join paths.
    candidate_group : set[str]
        Table names involved in this candidate group.
    table_path : dict[str, str]
        ``{table_name: path}``.
    dod : DoD
        DoD instance.

    Returns
    -------
    list
        Reordered join paths (same length), with hops sorted so that
        the most key-likely fields come first.

    Algorithm:

    1. For each table in *candidate_group*, load CSV and compute
       key-likelihood via ``dpu.key_likelihood_ranking(df)``.
    2. Build per-table rank maps ``{field_name: rank_index}``.
    3. Split each join path into per-hop buckets.
    4. Sort each hop's bucket by the right-side field's rank.
    5. Reassemble sorted hop buckets into complete join paths.
    """
    raise NotImplementedError
