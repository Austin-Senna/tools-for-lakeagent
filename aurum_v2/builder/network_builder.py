"""
Relation‑building functions for the FieldNetwork graph.

Each ``build_*`` function reads data from the store (or from signatures
provided by the coordinator), computes similarity / overlap, and calls
``network.add_relation()`` for every discovered pair.

Algorithms and thresholds are identical to the legacy
``knowledgerepr/networkbuilder.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Iterable

import numpy as np
from datasketch import MinHashLSH, MinHash  # type: ignore[import-untyped]
from nearpy import Engine  # type: ignore[import-untyped]
from nearpy.distances import CosineDistance  # type: ignore[import-untyped]
from nearpy.hashes import RandomBinaryProjections  # type: ignore[import-untyped]
from aurum_v2.models.relation import Relation
import math
from collections import defaultdict
from sklearn.cluster import DBSCAN

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
    
    # 1. Safely unpack tuples to guarantee nids and docs are perfectly aligned
    fields_list = list(fields)
    if not fields_list:
        return _LSHIndex(num_features=0)
        
    nids, docs = zip(*fields_list)

    # 2. Compute TF-IDF Matrix
    # (Replaces the legacy da.get_tfidf_docs call to keep dependencies standard)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    num_features = tfidf_matrix.shape[1]
    lsh = _LSHIndex(num_features=num_features)

    # 3. Index vectors into NearPy and cache dense arrays
    dense_vectors: list[np.ndarray] = []
    for i, nid in enumerate(nids):
        vec = tfidf_matrix[i].todense().A[0]
        dense_vectors.append(vec)
        lsh.index(vec, nid)

    # 3b. Rarity penalty — penalise ubiquitous column names (e.g. "Id" × 13)
    #     so they don't all saturate at 1.0.
    #     rarity(nid) = 1 / log2(1 + count_of_columns_with_same_name)
    from collections import Counter
    name_counts = Counter(docs)
    nid_rarity: dict[str, float] = {}
    for nid_val, doc in zip(nids, docs):
        nid_rarity[nid_val] = 1.0 / math.log2(1 + name_counts[doc])

    # 4. Query and connect nodes (reuse cached dense vectors)
    for i, nid in enumerate(nids):
        neighbors = lsh.query(dense_vectors[i])
        
        # NearPy returns a list of (data, key, distance)
        for _, r_nid, distance in neighbors:
            if nid != r_nid:
                # Cosine similarity × geometric mean of rarity penalties
                cosine_sim = 1.0 - distance
                rarity = math.sqrt(nid_rarity[nid] * nid_rarity[r_nid])
                score = round(cosine_sim * rarity, 4)
                network.add_relation(nid, r_nid, Relation.SCHEMA_SIM, score)

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
    max_degree = config.max_degrees if config else 500

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    mh_sig_obj = []

    # 1. Reconstruct MinHash objects from the raw arrays and index them
    for nid, sig_array in mh_signatures:
        # Datasketch requires the numpy array of hash values
        m = MinHash(num_perm=num_perm, hashvalues=np.array(sig_array, dtype=np.uint64))
        lsh.insert(nid, m)
        minhashes[nid] = m
        mh_sig_obj.append((nid, m))

    edges_added = 0
    pruned_hubs = 0

    # 2. Query the index for collisions and compute actual Jaccard similarity
    for nid, m in minhashes.items():
        neighbors = lsh.query(m)

        if len(neighbors) > max_degree:
            pruned_hubs += 1
            continue

        for r_nid in neighbors:
            if r_nid != nid:
                # Calculate the ACTUAL Jaccard similarity to give the edge a real weight
                r_mh_obj = minhashes[r_nid]
                score = m.jaccard(r_mh_obj)

                # Final safety check: only add if score actually meets the threshold
                if score >= threshold:
                    network.add_relation(nid, r_nid, Relation.CONTENT_SIM, round(score, 4))
                    edges_added += 1
    print(f"CONTENT_SIM (Text) complete. Added {edges_added} edges. Pruned {pruned_hubs} massive hubs.")
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
    """Build CONTENT_SIM and INCLUSION_DEPENDENCY edges for numeric columns."""
    
    max_degree = config.max_degrees if config and hasattr(config, 'max_degrees') else 500
    overlap_th = config.num_overlap_th if config and hasattr(config, 'num_overlap_th') else 0.85
    inc_dep_th = config.inclusion_dep_th if config and hasattr(config, 'inclusion_dep_th') else 0.3
    dbscan_eps = config.dbscan_eps if config and hasattr(config, 'dbscan_eps') else 0.1

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
    
    # Trackers for the Hairball Killer
    degree_tracker = defaultdict(int)
    pruned_hubs = 0

    for ref_domain, ref_nid, ref_min, ref_left, ref_right, ref_max in entries:
        if ref_domain == 0:
            single_points.append((ref_nid, ref_left)) 
            continue

        for cand_domain, cand_nid, cand_min, cand_left, cand_right, cand_max in entries:
            if cand_nid == ref_nid or cand_domain == 0:
                continue
            
            # The Hairball Killer Check
            if degree_tracker[ref_nid] >= max_degree:
                pruned_hubs += 1
                break # Stop adding edges for this specific super-hub
            
            ratio = cand_domain / ref_domain
            
            # FIX: Only break if it drops below the LOWEST threshold (inc_dep_th)
            if ratio < inc_dep_th:
                break
            
            actual_overlap = compute_overlap(ref_left, ref_right, cand_left, cand_right)

            # 1. Content Similarity Check (Requires 85%)
            if ratio >= overlap_th and actual_overlap >= overlap_th:
                network.add_relation(cand_nid, ref_nid, Relation.CONTENT_SIM, round(actual_overlap, 4))
                degree_tracker[ref_nid] += 1
                degree_tracker[cand_nid] += 1

            # 2. Inclusion Dependency Check (Requires 30% + Min/Max containment)
            # Notice we check this independently now so it doesn't get skipped!
            if not (math.isinf(ref_min) or math.isinf(ref_max) or math.isinf(cand_min) or math.isinf(cand_max)):
                if cand_min >= ref_min and cand_max <= ref_max:
                    if cand_min >= 0: # Only positive numbers
                        if actual_overlap >= inc_dep_th:
                            network.add_relation(cand_nid, ref_nid, Relation.INCLUSION_DEPENDENCY, 1.0)
                            degree_tracker[ref_nid] += 1
                            degree_tracker[cand_nid] += 1

    # Final clustering for single points (Domain == 0)
    if single_points:
        fields = [pt[0] for pt in single_points]
        medians = np.array([[pt[1]] for pt in single_points])

        db_median = DBSCAN(eps=dbscan_eps, min_samples=2).fit(medians)
        
        clusters = defaultdict(list)
        for idx, label in enumerate(db_median.labels_):
            if label != -1:
                clusters[label].append(fields[idx])

        for cluster_nodes in clusters.values():
            # Apply hairball killer to DBSCAN clusters too
            if len(cluster_nodes) > max_degree:
                pruned_hubs += 1
                continue 
                
            for i in range(len(cluster_nodes)):
                for j in range(i + 1, len(cluster_nodes)):
                    network.add_relation(cluster_nodes[i], cluster_nodes[j], Relation.CONTENT_SIM, overlap_th)
                    network.add_relation(cluster_nodes[j], cluster_nodes[i], Relation.CONTENT_SIM, overlap_th)

    print(f"Numeric builder complete. Pruned {pruned_hubs} generic numeric super-hubs.")

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
    # 1. Fetch configurations or use defaults
    cardinality_th = config.pkfk_cardinality_th if config and hasattr(config, 'pkfk_cardinality_th') else 0.7

    def get_neighborhood(nid: str):
        """Helper to fetch candidate neighbors based on column data type."""
        data_type = network.get_data_type_of(nid)
        if data_type == "N":
            return network.neighbors_id(nid, Relation.INCLUSION_DEPENDENCY)
        elif data_type == "T":
            return network.neighbors_id(nid, Relation.CONTENT_SIM)
        return []

    # 2. Scan the network for Primary Key candidates
    for n in network.iterate_ids():
        n_card = network.get_cardinality_of(n)
        
        # A valid PK candidate must have high uniqueness (cardinality)
        if n_card > cardinality_th:
            neighbors = get_neighborhood(n)
            
            for ne in neighbors:
                # Safely handle the neighbor object depending on network's return type
                ne_nid = ne.nid if hasattr(ne, 'nid') else ne
                
                if ne_nid != n:
                    # Calculate the edge score based on the highest cardinality
                    ne_card = network.get_cardinality_of(ne_nid)
                    highest_card = max(n_card, ne_card)
                    
                    # Add the directional PKFK relationship
                    network.add_relation(n, ne_nid, Relation.PKFK, highest_card)