"""
Relation‑building functions for the FieldNetwork graph.

Each ``build_*`` function reads data from the store (or from signatures
provided by the coordinator), computes similarity / overlap, and calls
``network.add_relation()`` for every discovered pair.

Algorithms and thresholds are identical to the legacy
``knowledgerepr/networkbuilder.py``.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)
from datasketch import MinHashLSH, MinHash  # type: ignore[import-untyped]
from nearpy import Engine  # type: ignore[import-untyped]
from nearpy.distances import CosineDistance  # type: ignore[import-untyped]
from nearpy.hashes import RandomBinaryProjections  # type: ignore[import-untyped]
from aurum_v2.models.relation import Relation
import math
from collections import defaultdict
from sklearn.cluster import DBSCAN
import heapq

if TYPE_CHECKING:
    from aurum_v2.config import AurumConfig
    from aurum_v2.graph.field_network import FieldNetwork

__all__ = [
    "build_schema_sim_relation",
    "build_content_sim_mh_text",
    "build_content_sim_relation_num_overlap_distr",
    "build_pkfk_relation",
]


# ======================================================================
# Internal NearPy LSH wrapper (mirrors legacy LSHRandomProjectionsIndex)
# ======================================================================

class _LSHIndex:
    """NearPy engine for approximate nearest‑neighbour search on TF‑IDF vectors."""

    def __init__(self, num_features: int, projection_count: int = 30) -> None:
        self._rbp = RandomBinaryProjections("default", projection_count)
        self._engine = Engine(
            num_features,
            lshashes=[self._rbp],
            distance=CosineDistance(),
        )

    def index(self, vector: np.ndarray, key: str) -> None:
        self._engine.store_vector(vector, key)

    def query(self, vector: np.ndarray) -> list:
        """Return list of ``(data, key, distance)`` triples."""
        return self._engine.neighbours(vector)


# ======================================================================
# 1. Schema similarity  (TF‑IDF on column names → NearPy LSH)
# ======================================================================
def build_schema_sim_relation(
    network: FieldNetwork,
    fields: Iterable[tuple[str, str]],  # List/Iterator of (nid, field_name)
    config: AurumConfig | None = None,
) -> _LSHIndex:
    """Build ``Relation.SCHEMA_SIM`` edges.

    Algorithm (mirrors legacy exactly):

    1. Collect all ``field_name`` strings from the network.
    2. Compute a TF‑IDF matrix over these strings (each field name is one
       document).
    3. Index every TF‑IDF vector into a :class:`_LSHIndex` (NearPy engine
       with :class:`RandomBinaryProjections`, cosine distance).
    4. Query each vector against the index.  For every neighbour pair
       ``(nid₁, nid₂)`` with ``nid₁ ≠ nid₂``, call
       ``network.add_relation(nid₁, nid₂, SCHEMA_SIM, cosine_distance)``.

    Returns the :class:`_LSHIndex` so it can be serialised alongside the
    network artefacts.
    """
    max_degree = config.max_degrees if config and hasattr(config, 'max_degrees') else 100

    # 1. Safely unpack tuples to guarantee nids and docs are perfectly aligned
    fields_list = list(fields)
    if not fields_list:
        log.warning("build_schema_sim_relation: no fields provided, skipping")
        return _LSHIndex(num_features=0)

    nids, docs = zip(*fields_list)
    log.debug("  schema_sim: %d fields to process (max_degree=%d)", len(nids), max_degree)

    # 2. Compute TF-IDF Matrix
    max_features = config.tfidf_max_features if config and hasattr(config, 'tfidf_max_features') else 10_000
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(docs)
    num_features = tfidf_matrix.shape[1]
    log.debug("  schema_sim: TF-IDF matrix shape=%s (vocab size=%d)", tfidf_matrix.shape, num_features)

    lsh = _LSHIndex(num_features=num_features)

    # 3. Index vectors into NearPy
    for i, nid in enumerate(nids):
        vec = tfidf_matrix[i].todense().A[0]
        lsh.index(vec, nid)
    log.debug("  schema_sim: indexed %d vectors into LSH", len(nids))

    # 4. Query and connect nodes
    edges_added = 0
    for i, nid in enumerate(nids):
        query_vec = tfidf_matrix[i].todense().A[0]
        neighbors = lsh.query(query_vec)

        top_k_edges = []
        for _, r_nid, distance in neighbors:
            if nid != r_nid:
                score = 1.0
                if len(top_k_edges) < max_degree:
                    heapq.heappush(top_k_edges, (score, r_nid))
                else:
                    break

        for score, r_nid in top_k_edges:
            network.add_relation(nid, r_nid, Relation.SCHEMA_SIM, score)
            edges_added += 1

    # tfidf_matrix and vectorizer no longer needed — free before returning
    del tfidf_matrix, vectorizer
    gc.collect()

    log.info("  schema_sim: added %d SCHEMA_SIM edges across %d fields", edges_added, len(nids))
    return lsh


# ======================================================================
# 2. Content similarity — text columns  (MinHash LSH)
# ======================================================================
# LEGACY
# def build_content_sim_mh_text(
#     network: FieldNetwork,
#     mh_signatures: Iterator[tuple[str, list]],
#     config: AurumConfig | None = None,
# ) -> MinHashLSH:
#     """Build ``Relation.CONTENT_SIM`` edges for text columns.

#     Algorithm:

#     1. For each ``(nid, minhash_array)`` from the store, reconstruct a
#        ``datasketch.MinHash`` object (``num_perm=512``) and insert into a
#        ``MinHashLSH(threshold=0.7, num_perm=512)``.
#     2. Query each object.  For every match ``r_nid ≠ nid``, add a
#        ``CONTENT_SIM`` edge with ``score = 1``.

#     Returns the ``MinHashLSH`` index for optional serialisation.
#     """
#     threshold = config.minhash_threshold if config else 0.7
#     num_perm = config.minhash_num_perm if config else 512
    
#     lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
#     minhashes = {}

#     # 1. Reconstruct MinHash objects from the raw arrays and index them
#     for nid, sig_array in mh_signatures:
#         # Datasketch requires the numpy array of hash values
#         m = MinHash(num_perm=num_perm, hashvalues=np.array(sig_array, dtype=np.uint64))
#         lsh.insert(nid, m)
#         minhashes[nid] = m


#     # 2. Query the index for collisions and compute actual Jaccard similarity
#     for nid, m in minhashes.items():
#         result = lsh.query(m)
#         for r_nid in result:
#             if r_nid != nid and r_nid in minhashes:
#                 score = m.jaccard(minhashes[r_nid])
#                 network.add_relation(nid, r_nid, Relation.CONTENT_SIM, score)

#     return lsh

# ======================================================================
# 2. Content similarity — text columns  (MinHash LSH)
# ======================================================================

def build_content_sim_mh_text(
    network: FieldNetwork,
    mh_signatures: Iterator[tuple[str, list]],
    config: AurumConfig | None = None,
) -> MinHashLSH:
    """Build ``Relation.CONTENT_SIM`` edges for text columns.

    Algorithm:

    1. For each ``(nid, minhash_array)`` from the store, reconstruct a
       ``datasketch.MinHash`` object (``num_perm=512``) and insert into a
       ``MinHashLSH(threshold=0.7, num_perm=512)``.
    2. Query each object.  For every match ``r_nid ≠ nid``, add a
       ``CONTENT_SIM`` edge with ``score = 1``.

    Returns the ``MinHashLSH`` index for optional serialisation.
    """
    threshold = config.minhash_threshold if config else 0.7
    num_perm = config.minhash_num_perm if config else 512
    max_degree = config.max_degrees if config else 100

    log.debug("  content_sim_text: threshold=%.2f, num_perm=%d, max_degree=%d",
              threshold, num_perm, max_degree)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}

    # 1. Reconstruct MinHash objects from the raw arrays and index them
    for nid, sig_array in mh_signatures:
        m = MinHash(num_perm=num_perm, hashvalues=np.array(sig_array, dtype=np.uint64))
        lsh.insert(nid, m)
        minhashes[nid] = m

    log.debug("  content_sim_text: indexed %d MinHash signatures", len(minhashes))

    edges_added = 0
    pruned_hubs = 0

    # 2. Query the index for collisions and compute actual Jaccard similarity
    for nid, m in minhashes.items():
        neighbors = lsh.query(m)
        top_neighbors = []

        for r_nid in neighbors:
            if r_nid != nid:
                score = m.jaccard(minhashes[r_nid])
                if score >= threshold:
                    if len(top_neighbors) < max_degree:
                        heapq.heappush(top_neighbors, (score, r_nid))
                    elif score > top_neighbors[0][0]:
                        heapq.heappop(top_neighbors)
                        heapq.heappush(top_neighbors, (score, r_nid))

        if len(neighbors) > max_degree:
            pruned_hubs += 1

        for score, r_nid in top_neighbors:
            network.add_relation(nid, r_nid, Relation.CONTENT_SIM, round(score, 4))
            edges_added += 1

    # minhashes dict no longer needed after querying — free before returning
    del minhashes
    gc.collect()

    log.info("  content_sim_text: added %d CONTENT_SIM edges; pruned %d oversized hubs",
             edges_added, pruned_hubs)
    return lsh

# ======================================================================
# 3. Content similarity — numeric columns  (distribution overlap)
# ======================================================================
# LEGACY
# def build_content_sim_relation_num_overlap_distr(
#     network: FieldNetwork,
#     id_sig: Iterator[tuple[str, tuple[float, float, float, float]]],
#     config: AurumConfig | None = None,
# ) -> None:
#     """Build ``CONTENT_SIM`` and ``INCLUSION_DEPENDENCY`` edges for numeric columns.

#     For each pair of numeric columns, the function checks:

#     * **Content similarity**: whether the IQR‑based ranges overlap by ≥ 85 %.
#       Three overlap cases are handled (full containment, left partial, right
#       partial), matching legacy ``compute_overlap``.
#     * **Inclusion dependency**: whether one column's ``[min, max]`` is entirely
#       contained in the other's *and* the core (median ± IQR) overlap ≥ 30 %.
#       Only positive‑valued columns are considered.
#     * **Single‑point clustering**: columns whose IQR domain = 0 are clustered
#       with ``DBSCAN(eps=0.1, min_samples=2)`` on their median value.  Columns
#       in the same cluster get ``CONTENT_SIM`` edges.

#     Thresholds are taken from *config* (defaults: ``num_overlap_th=0.85``,
#     ``inclusion_dep_th=0.3``, ``dbscan_eps=0.1``).
#     """
#     # 1. Fetch configurations or use defaults
#     overlap_th = config.num_overlap_th if config and hasattr(config, 'num_overlap_th') else 0.85
#     inc_dep_th = config.inclusion_dep_th if config and hasattr(config, 'inclusion_dep_th') else 0.3
#     dbscan_eps = config.dbscan_eps if config and hasattr(config, 'dbscan_eps') else 0.1

#     def compute_overlap(ref_left: float, ref_right: float, left: float, right: float) -> float:
#         """Calculates overlap ratio relative to the reference domain width."""
#         ref_width = ref_right - ref_left
#         if ref_width == 0:
#             return 0.0
            
#         # Calculate geometric intersection bounds
#         overlap_left = max(ref_left, left)
#         overlap_right = min(ref_right, right)
        
#         if overlap_left < overlap_right:
#             return (overlap_right - overlap_left) / ref_width
#         return 0.0

#     # 2. Prepare and sort domain statistics
#     entries = []
#     for nid, (c_median, c_iqr, c_min, c_max) in id_sig:
#         x_left = c_median - c_iqr
#         x_right = c_median + c_iqr
#         domain = x_right - x_left
#         entries.append((domain, nid, c_min, x_left, x_right, c_max))
        
#     # Sort descending by domain size
#     entries.sort(reverse=True, key=lambda x: x[0])
    
#     single_points = []

#     # 3. Compare pairs for Content Similarity and Inclusion Dependency
#     for ref_domain, ref_nid, ref_min, ref_left, ref_right, ref_max in entries:
#         if ref_domain == 0:
#             # Save for DBSCAN clustering later
#             single_points.append((ref_nid, ref_left)) 
#             continue

#         for cand_domain, cand_nid, cand_min, cand_left, cand_right, cand_max in entries:
#             if cand_nid == ref_nid:
#                 continue
#             if cand_domain == 0:
#                 continue
            
#             # Early termination: entries are sorted descending by domain.
#             # If the candidate domain is so small that even full containment
#             # can't reach the overlap threshold, skip the rest.
#             if cand_domain / ref_domain < overlap_th:
#                 break
            
#             # Content Similarity Check
#             actual_overlap = compute_overlap(ref_left, ref_right, cand_left, cand_right)
#             if actual_overlap >= overlap_th:
#                 network.add_relation(cand_nid, ref_nid, Relation.CONTENT_SIM, actual_overlap)

#             # Inclusion Dependency Check
#             if not (math.isinf(ref_min) or math.isinf(ref_max) or math.isinf(cand_min) or math.isinf(cand_max)):
#                 if cand_min >= ref_min and cand_max <= ref_max:
#                     if cand_min >= 0: # Only positive numbers as IDs
#                         if actual_overlap >= inc_dep_th:
#                             network.add_relation(cand_nid, ref_nid, Relation.INCLUSION_DEPENDENCY, 1.0)

#     # 4. Final clustering for single points (Domain == 0)
#     if not single_points:
#         return

#     fields = [pt[0] for pt in single_points]
#     medians = np.array([[pt[1]] for pt in single_points]) # Reshaped for DBSCAN

#     db_median = DBSCAN(eps=dbscan_eps, min_samples=2).fit(medians)
    
#     # Group by cluster labels
#     clusters = defaultdict(list)
#     for idx, label in enumerate(db_median.labels_):
#         if label != -1:
#             clusters[label].append(fields[idx])

#     # Connect clustered nodes bidirectionally
#     # You would intuitively think the similarity score should be 1.0 (100% match), but its not?
#     for cluster_nodes in clusters.values():
#         for i in range(len(cluster_nodes)):
#             for j in range(i + 1, len(cluster_nodes)):
#                 network.add_relation(cluster_nodes[i], cluster_nodes[j], Relation.CONTENT_SIM, overlap_th)
#                 network.add_relation(cluster_nodes[j], cluster_nodes[i], Relation.CONTENT_SIM, overlap_th)

def build_content_sim_relation_num_overlap_distr(
    network, # FieldNetwork
    id_sig: Iterator[tuple[str, tuple[float, float, float, float]]],
    config: AurumConfig | None = None
) -> None:
    """Build CONTENT_SIM and INCLUSION_DEPENDENCY edges for numeric columns.
        O(N^2), but works more closely to O(N).
    """
    
    max_degree = config.max_degrees if config and hasattr(config, 'max_degrees') else 100
    overlap_th = config.num_overlap_th if config and hasattr(config, 'num_overlap_th') else 0.85
    inc_dep_th = config.inclusion_dep_th if config and hasattr(config, 'inclusion_dep_th') else 0.3
    dbscan_eps = config.dbscan_eps if config and hasattr(config, 'dbscan_eps') else 0.1

    log.debug("  content_sim_num: overlap_th=%.2f, inc_dep_th=%.2f, dbscan_eps=%.2f, max_degree=%d",
              overlap_th, inc_dep_th, dbscan_eps, max_degree)

    def compute_overlap(ref_left: float, ref_right: float, left: float, right: float) -> float:
        ref_width = ref_right - ref_left
        if ref_width == 0:
            return 0.0
        overlap_left = max(ref_left, left)
        overlap_right = min(ref_right, right)
        if overlap_left < overlap_right:
            return (overlap_right - overlap_left) / ref_width
        return 0.0

    entries = []
    for nid, (c_median, c_iqr, c_min, c_max) in id_sig:
        x_left = c_median - c_iqr
        x_right = c_median + c_iqr
        domain = x_right - x_left
        entries.append((domain, nid, c_min, x_left, x_right, c_max))

    entries.sort(reverse=True, key=lambda x: x[0])
    single_points = []

    degree_tracker = defaultdict(int)
    pruned_hubs = 0
    content_edges = 0
    inclusion_edges = 0

    log.debug("  content_sim_num: %d numeric column entries to compare", len(entries))

    for ref_domain, ref_nid, ref_min, ref_left, ref_right, ref_max in entries:
        if ref_domain == 0:
            single_points.append((ref_nid, ref_left))
            continue

        top_content_sim = []
        top_inclusion = []

        for cand_domain, cand_nid, cand_min, cand_left, cand_right, cand_max in entries:
            if cand_nid == ref_nid or cand_domain == 0:
                continue

            ratio = cand_domain / ref_domain
            if ratio < inc_dep_th:
                break

            actual_overlap = compute_overlap(ref_left, ref_right, cand_left, cand_right)

            # 1. Content Similarity Check (Requires 85%)
            if ratio >= overlap_th and actual_overlap >= overlap_th:
                if len(top_content_sim) < max_degree:
                    heapq.heappush(top_content_sim, (actual_overlap, cand_nid))
                elif actual_overlap > top_content_sim[0][0]:
                    heapq.heappop(top_content_sim)
                    heapq.heappush(top_content_sim, (actual_overlap, cand_nid))

            # 2. Inclusion Dependency Check (Requires 30% + Min/Max containment)
            if not (math.isinf(ref_min) or math.isinf(ref_max) or math.isinf(cand_min) or math.isinf(cand_max)):
                if cand_min >= ref_min and cand_max <= ref_max:
                    if cand_min >= 0:
                        if actual_overlap >= inc_dep_th:
                            if len(top_inclusion) < max_degree:
                                heapq.heappush(top_inclusion, (actual_overlap, cand_nid))
                            elif actual_overlap > top_inclusion[0][0]:
                                heapq.heappop(top_inclusion)
                                heapq.heappush(top_inclusion, (actual_overlap, cand_nid))

        for actual_overlap, cand_nid in top_content_sim:
            network.add_relation(cand_nid, ref_nid, Relation.CONTENT_SIM, round(actual_overlap, 4))
            degree_tracker[ref_nid] += 1
            degree_tracker[cand_nid] += 1
            content_edges += 1

        for actual_overlap, cand_nid in top_inclusion:
            network.add_relation(cand_nid, ref_nid, Relation.INCLUSION_DEPENDENCY, 1.0)
            degree_tracker[ref_nid] += 1
            degree_tracker[cand_nid] += 1
            inclusion_edges += 1

    # Final clustering for single points (Domain == 0)
    dbscan_edges = 0
    if single_points:
        log.debug("  content_sim_num: clustering %d single-point (zero-IQR) columns with DBSCAN",
                  len(single_points))
        fields = [pt[0] for pt in single_points]
        medians = np.array([[pt[1]] for pt in single_points])

        db_median = DBSCAN(eps=dbscan_eps, min_samples=2).fit(medians)

        clusters = defaultdict(list)
        for idx, label in enumerate(db_median.labels_):
            if label != -1:
                clusters[label].append((fields[idx], medians[idx][0]))

        log.debug("  content_sim_num: DBSCAN found %d clusters from %d single-point columns",
                  len(clusters), len(single_points))

        for cluster_nodes in clusters.values():
            if len(cluster_nodes) > max_degree:
                pruned_hubs += 1
                continue

            for i in range(len(cluster_nodes)):
                for j in range(i + 1, len(cluster_nodes)):
                    nid_i, med_i = cluster_nodes[i]
                    nid_j, med_j = cluster_nodes[j]

                    dist = abs(med_i - med_j)
                    if dbscan_eps == 0 or dist == 0:
                        actual_score = 1.0
                    else:
                        actual_score = 1.0 - ((dist / dbscan_eps) * (1.0 - overlap_th))

                    actual_score = round(max(overlap_th, min(1.0, actual_score)), 4)

                    network.add_relation(nid_i, nid_j, Relation.CONTENT_SIM, actual_score)
                    network.add_relation(nid_j, nid_i, Relation.CONTENT_SIM, actual_score)
                    dbscan_edges += 2

    log.info(
        "  content_sim_num: added %d CONTENT_SIM + %d INCLUSION_DEPENDENCY + %d DBSCAN edges; "
        "pruned %d oversized hubs",
        content_edges, inclusion_edges, dbscan_edges, pruned_hubs,
    )

# ======================================================================
# 4. Primary‑key / foreign‑key relation
# ======================================================================

def build_pkfk_relation(
    network: FieldNetwork,
    config: AurumConfig | None = None,
) -> None:
    """Build ``Relation.PKFK`` edges.

    For every column *n* with ``cardinality > 0.7``:

    1. Get its neighbourhood via ``INCLUSION_DEPENDENCY`` (numeric) or
       ``CONTENT_SIM`` (text).
    2. For each neighbour *ne*, add a ``PKFK`` edge scored as
       ``max(card_n, card_ne)``.
    """
    cardinality_th = (
        config.pkfk_cardinality_th if config and hasattr(config, 'pkfk_cardinality_th') else 0.7
    )
    log.debug("  pkfk: cardinality_th=%.2f", cardinality_th)

    def get_neighborhood(nid: str):
        data_type = network.get_data_type_of(nid)
        if data_type == "N":
            return network.neighbors_id(nid, Relation.INCLUSION_DEPENDENCY)
        elif data_type == "T":
            return network.neighbors_id(nid, Relation.CONTENT_SIM)
        return []

    pk_candidates = 0
    edges_added = 0

    for n in network.iterate_ids():
        n_card = network.get_cardinality_of(n)

        if n_card > cardinality_th:
            pk_candidates += 1
            neighbors = get_neighborhood(n)

            for ne in neighbors:
                ne_nid = ne.nid if hasattr(ne, 'nid') else ne

                if ne_nid != n:
                    ne_card = network.get_cardinality_of(ne_nid)
                    highest_card = max(n_card, ne_card)
                    network.add_relation(n, ne_nid, Relation.PKFK, highest_card)
                    edges_added += 1

    log.info("  pkfk: %d PK candidates found; added %d PKFK edges", pk_candidates, edges_added)
