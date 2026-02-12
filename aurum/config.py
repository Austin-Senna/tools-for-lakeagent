"""
aurum configuration — every tunable knob in one place.

All thresholds are ported from the legacy Aurum codebase with their original
values documented.  Override via ``aurumConfig(minhash_perms=128, ...)``.

Origin mapping
--------------
- minhash_perms / jaccard_threshold → networkbuilder.build_content_sim_mh_text
- numeric_overlap_threshold         → networkbuilder.build_content_sim_relation_num_overlap_distr
- inclusion_dep_overlap_threshold   → same function, inclusion-dependency branch
- pk_cardinality_threshold          → networkbuilder.build_pkfk_relation
- join_overlap_threshold            → config.join_overlap_th
- schema_sim_model                  → replaces TF-IDF + LSH random projections
- tfidf_sublinear_tf                → dataanalysis.vect (TfidfVectorizer param)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AurumConfig:
    """Immutable configuration for all aurum subsystems."""

    # ── MinHash / Content Similarity (text columns) ──────────────────
    minhash_perms: int = 512
    """Number of MinHash permutations.  Aurum used 512; 256 is sufficient
    with modern datasketch (smaller memory, ~same accuracy)."""

    jaccard_threshold: float = 0.7
    """Minimum estimated Jaccard similarity to create a CONTENT_SIM edge
    between two text columns.  Ported from ``MinHashLSH(threshold=0.7)``."""

    # ── Numeric Column Overlap ───────────────────────────────────────
    numeric_overlap_threshold: float = 0.85
    """Minimum interval-overlap ratio (based on median ± IQR) to link two
    numeric columns with CONTENT_SIM."""

    inclusion_dep_overlap_threshold: float = 0.3
    """Minimum core-range overlap to declare an INCLUSION_DEPENDENCY when
    one column's [min, max] is fully contained in another's."""

    # ── PK / FK Detection ────────────────────────────────────────────
    pk_cardinality_threshold: float = 0.7
    """A column with ``nunique / count ≥ this`` is a primary-key candidate.
    Ported from ``build_pkfk_relation``."""

    # ── Join / Set Overlap ───────────────────────────────────────────
    join_overlap_threshold: float = 0.4
    """Minimum Jaccard-like overlap of *values* to consider two columns
    joinable (used in DoD materialisation checks)."""
    
    # # ── Schema Similarity ────────────────────────────────────────────
    max_distance_schema_similarity = 10
    # schema_sim_model: str = "all-MiniLM-L6-v2"
    # """Sentence-transformer model for column-name embeddings.  Replaces the
    # legacy TF-IDF + Random Binary Projections + Cosine Distance pipeline."""

    # schema_sim_threshold: float = 0.55
    # """Cosine similarity threshold for SCHEMA_SIM edges.  Tuned to
    # approximate the recall of the legacy LSH approach."""

    tfidf_sublinear_tf: bool = True
    """Legacy TF-IDF used ``sublinear_tf=True`` (log-dampened TF).  Kept as
    an option for the fallback TF-IDF path."""

    # ── Profiler ─────────────────────────────────────────────────────
    profiler_sample_rows: int = 50_000
    """Max rows to sample when profiling very large tables."""

    text_token_min_length: int = 2
    """Discard tokens shorter than this during column-name cleaning."""

    # ── Join Materialisation (DoD) ───────────────────────────────────
    join_chunksize: int = 1_000
    """Rows per chunk for memory-managed join estimation."""

    memory_limit_fraction: float = 0.6
    """Maximum fraction of system RAM allowed for join output."""

    join_timeout_seconds: float = 180.0
    """Abort a single join if it exceeds this wall-clock time."""

    sample_size_for_validation: int = 1_000
    """Number of rows to sample when validating a join graph."""

    # ── DBSCAN (numeric single-point clustering) ─────────────────────
    dbscan_eps: float = 0.1
    """DBSCAN ε for clustering constant-valued numeric columns."""

    dbscan_min_samples: int = 2
    """DBSCAN minimum cluster size."""

    # ── Paths & Serialisation ────────────────────────────────────────
    index_dir: Path = field(default_factory=lambda: Path(".lake_index"))
    """Default directory for persisted index artefacts."""

    max_search_results: int = 50
    """Default cap on keyword-search results."""

    max_hops: int = 3
    """Maximum hops when searching for join paths between tables."""


# Alias for backward compatibility with code using the lowercase name
aurumConfig = AurumConfig
