"""
Network builder — construct all similarity / relationship edges.

This is the algorithmic heart of Aurum, ported from
``aurum/knowledgerepr/networkbuilder.py``.

Each ``build_*`` function creates one class of edge in the ``FieldNetwork``:

1. **Schema similarity**   — column-name embeddings, cosine sim
2. **Content similarity (text)** — MinHash LSH, Jaccard threshold
3. **Content similarity (numeric)** — median ± IQR interval overlap
4. **Inclusion dependency** — full [min,max] containment of numeric ranges
5. **PK / FK**             — high cardinality + content/inclusion overlap

The ``build_all`` orchestrator calls them in order (same as
``networkbuildercoordinator.main``).
"""

from __future__ import annotations

import time
from collections import defaultdict
from math import isinf
from typing import Sequence

import numpy as np
from datasketch import MinHashLSH
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from aurum.config import aurumConfig
from aurum.graph.field_network import FieldNetwork, Hit
from aurum.graph.relations import Relation
from aurum.profiler.column_profiler import ColumnProfile
from aurum.profiler.text_utils import tokenize_name


# ---------------------------------------------------------------------------
# 1. Schema similarity (replaces TF-IDF + LSH Random Binary Projections)
# ---------------------------------------------------------------------------

def build_schema_sim(
    network: FieldNetwork,
    profiles: Sequence[ColumnProfile],
    cfg: aurumConfig,
) -> None:
    """Create ``SCHEMA_SIM`` edges between columns with similar names.

    Legacy path: TfidfVectorizer(sublinear_tf=True) → RandomBinaryProjections(30)
    → CosineDistance → NearPy Engine.

    Modern path: Sentence-Transformer embeddings → all-pairs cosine similarity
    (or FAISS for >100k columns).

    Falls back to TF-IDF if sentence-transformers is unavailable.
    """
    t0 = time.time()
    nids: list[str] = []
    texts: list[str] = []
    for p in profiles:
        nids.append(p.col_id.nid)
        # Prepare the "document" — Aurum fed raw field_name into TfidfVectorizer.
        # We feed the curated token string into sentence-transformers.
        tokens = tokenize_name(p.col_id.field_name, min_length=cfg.text_token_min_length)
        texts.append(" ".join(tokens) if tokens else p.col_id.field_name.lower())

    if len(nids) == 0:
        return

 
    from sklearn.feature_extraction.text import TfidfVectorizer

    vect = TfidfVectorizer(min_df=1, sublinear_tf=cfg.tfidf_sublinear_tf, use_idf=True)
    tfidf = vect.fit_transform(texts)
    embeddings = tfidf.toarray()  # dense for cosine_similarity

    sim_matrix = cosine_similarity(embeddings)

    edges_added = 0
    threshold = cfg.schema_sim_threshold
    for i in range(len(nids)):
        for j in range(i + 1, len(nids)):
            score = float(sim_matrix[i, j])
            if score >= threshold:
                network.add_relation(nids[i], nids[j], Relation.SCHEMA_SIM, score)
                edges_added += 1

    print(f"[schema_sim] {edges_added} edges in {time.time() - t0:.2f}s")


# ---------------------------------------------------------------------------
# 2. Content similarity — text columns (MinHash LSH)
# ---------------------------------------------------------------------------

def build_content_sim_text(
    network: FieldNetwork,
    profiles: Sequence[ColumnProfile],
    cfg: aurumConfig,
) -> MinHashLSH:
    """Create ``CONTENT_SIM`` edges between text columns with similar values.

    Direct port of ``networkbuilder.build_content_sim_mh_text``:
    - ``MinHashLSH(threshold=0.7, num_perm=512)``
    - Score is binary (1) — same as Aurum.
    """
    t0 = time.time()
    text_profiles = [p for p in profiles if p.data_type == "T" and p.minhash is not None]

    lsh = MinHashLSH(threshold=cfg.jaccard_threshold, num_perm=cfg.minhash_perms)
    sig_objects: list[tuple[str, object]] = []

    for p in text_profiles:
        nid = p.col_id.nid
        try:
            lsh.insert(nid, p.minhash)
        except ValueError:
            # Duplicate key — skip
            continue
        sig_objects.append((nid, p.minhash))

    edges_added = 0
    for nid, mh_obj in sig_objects:
        results = lsh.query(mh_obj)
        for r_nid in results:
            if r_nid != nid:
                network.add_relation(nid, r_nid, Relation.CONTENT_SIM, 1.0)
                edges_added += 1

    print(f"[content_sim_text] {edges_added} edges in {time.time() - t0:.2f}s")
    return lsh


# ---------------------------------------------------------------------------
# 3. Content similarity — numeric columns (median ± IQR overlap)
# ---------------------------------------------------------------------------

def _compute_interval_overlap(
    ref_left: float,
    ref_right: float,
    cand_left: float,
    cand_right: float,
) -> float:
    """Compute what fraction of the *ref* interval is overlapped by *cand*.

    Ported from ``networkbuilder.build_content_sim_relation_num_overlap_distr``
    → inner ``compute_overlap``.

    Three cases:
    - Full containment: cand inside ref  →  ``(cand_right - cand_left) / ref_width``
    - Left partial:  cand starts inside ref  →  ``(ref_right - cand_left) / ref_width``
    - Right partial: cand ends inside ref  →  ``(cand_right - ref_left) / ref_width``
    """
    ref_width = ref_right - ref_left
    if ref_width <= 0:
        return 0.0

    if cand_left >= ref_left and cand_right <= ref_right:
        return (cand_right - cand_left) / ref_width
    if cand_left >= ref_left and cand_left <= ref_right:
        return (ref_right - cand_left) / ref_width
    if cand_right <= ref_right and cand_right >= ref_left:
        return (cand_right - ref_left) / ref_width
    return 0.0


def build_content_sim_numeric(
    network: FieldNetwork,
    profiles: Sequence[ColumnProfile],
    cfg: aurumConfig,
) -> None:
    """Create ``CONTENT_SIM`` and ``INCLUSION_DEPENDENCY`` edges for numeric columns.

    Port of ``networkbuilder.build_content_sim_relation_num_overlap_distr``.

    Algorithm:
    1. Sort columns by domain width (``2 * IQR``) descending.
    2. For each pair, compute ``_compute_interval_overlap`` using median ± IQR.
    3. If overlap ≥ 0.85 → ``CONTENT_SIM`` edge.
    4. If candidate [min, max] ⊆ reference [min, max] AND min ≥ 0
       AND core overlap ≥ 0.3 → ``INCLUSION_DEPENDENCY`` edge.
    5. Single-point columns (IQR = 0) are clustered with DBSCAN(ε=0.1).
    """
    t0 = time.time()
    num_profiles = [p for p in profiles if p.data_type == "N" and p.numeric is not None]

    # Build sortable entries: (nid, domain, min, left, right, max)
    entries: list[tuple[str, float, float, float, float, float]] = []
    for p in num_profiles:
        n = p.numeric
        assert n is not None
        entries.append((
            p.col_id.nid,
            n.domain,
            n.min_value,
            n.core_left,
            n.core_right,
            n.max_value,
        ))

    # Sort by domain descending (largest first) — same as Aurum
    entries.sort(key=lambda e: e[1], reverse=True)

    single_points: list[tuple[str, float]] = []
    content_edges = 0
    inclusion_edges = 0
    overlap_th = cfg.numeric_overlap_threshold
    incl_th = cfg.inclusion_dep_overlap_threshold

    for i, ref in enumerate(entries):
        ref_nid, ref_domain, ref_min, ref_left, ref_right, ref_max = ref

        if ref_domain == 0:
            single_points.append((ref_nid, ref_right - ref_right / 2))
            continue

        for j, cand in enumerate(entries):
            if i == j:
                continue
            cand_nid, cand_domain, cand_min, cand_left, cand_right, cand_max = cand

            # Skip infinite values
            if any(isinf(v) for v in (ref_min, ref_max, cand_min, cand_max)):
                continue

            # ── Inclusion dependency check ───────────────────────
            if cand_min >= ref_min and cand_max <= ref_max and cand_min >= 0:
                actual_ov = _compute_interval_overlap(ref_left, ref_right, cand_left, cand_right)
                if actual_ov >= incl_th:
                    network.add_relation(cand_nid, ref_nid, Relation.INCLUSION_DEPENDENCY, 1.0)
                    inclusion_edges += 1

            # ── Content similarity check ─────────────────────────
            actual_ov = _compute_interval_overlap(ref_left, ref_right, cand_left, cand_right)
            if actual_ov >= overlap_th:
                network.add_relation(cand_nid, ref_nid, Relation.CONTENT_SIM, actual_ov)
                content_edges += 1

    # ── DBSCAN clustering for single-point (constant) columns ────
    if single_points:
        nids_sp = [nid for nid, _ in single_points]
        medians_sp = np.array([m for _, m in single_points]).reshape(-1, 1)
        db = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples).fit(medians_sp)
        clusters: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(db.labels_):
            if label >= 0:
                clusters[label].append(idx)
        for members in clusters.values():
            for a in members:
                for b in members:
                    if a != b:
                        network.add_relation(
                            nids_sp[a], nids_sp[b], Relation.CONTENT_SIM, overlap_th
                        )
                        content_edges += 1

    print(
        f"[content_sim_num] {content_edges} content + "
        f"{inclusion_edges} inclusion edges in {time.time() - t0:.2f}s"
    )


# ---------------------------------------------------------------------------
# 4. PK / FK relation
# ---------------------------------------------------------------------------

def build_pkfk(
    network: FieldNetwork,
    cfg: aurumConfig,
) -> None:
    """Create ``PKFK`` edges.

    Port of ``networkbuilder.build_pkfk_relation``:
    - For every column with ``cardinality > 0.7``:
      - Numeric columns → check ``INCLUSION_DEPENDENCY`` neighbours
      - Text columns → check ``CONTENT_SIM`` neighbours
      - Edge score = ``max(card_src, card_neighbour)``
    """
    t0 = time.time()
    total = 0
    for nid in network.iterate_ids():
        card = network.get_cardinality_of(nid)
        if card < cfg.pk_cardinality_threshold:
            continue

        dt = network.get_data_type_of(nid)
        if dt == "N":
            neighbours = network.neighbors_id(nid, Relation.INCLUSION_DEPENDENCY)
        else:
            neighbours = network.neighbors_id(nid, Relation.CONTENT_SIM)

        for ne in neighbours:
            if ne.nid == nid:
                continue
            ne_card = network.get_cardinality_of(ne.nid)
            score = max(card, ne_card)
            network.add_relation(nid, ne.nid, Relation.PKFK, score)
            total += 1

    print(f"[pkfk] {total} edges in {time.time() - t0:.2f}s")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_all(
    network: FieldNetwork,
    profiles: Sequence[ColumnProfile],
    cfg: aurumConfig | None = None,
) -> dict[str, object]:
    """Build the complete field network — same order as Aurum's coordinator.

    Returns a dict of index artefacts (MinHashLSH, etc.) for optional
    persistence.

    Orchestration order (``networkbuildercoordinator.main``):
    1. Schema similarity (column names)
    2. Content similarity — text (MinHash)
    3. Content similarity — numeric (interval overlap)
    4. PK / FK detection
    """
    if cfg is None:
        cfg = aurumConfig()

    t0 = time.time()
    print(f"Building network over {len(profiles)} columns…")

    # 1. Schema sim
    build_schema_sim(network, profiles, cfg)

    # 2. Content sim — text
    content_lsh = build_content_sim_text(network, profiles, cfg)

    # 3. Content sim — numeric
    build_content_sim_numeric(network, profiles, cfg)

    # 4. PK / FK
    build_pkfk(network, cfg)

    print(f"[build_all] Done in {time.time() - t0:.2f}s — "
          f"{network.order} nodes, {network._graph.number_of_edges()} edges")

    return {"content_lsh": content_lsh}
