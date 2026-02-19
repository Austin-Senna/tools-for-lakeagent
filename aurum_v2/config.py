"""
Central configuration for the Aurum v2 system.

All thresholds, connection strings, and tunables live here.
Values match the legacy ``config.py`` defaults.
"""

from dataclasses import dataclass


@dataclass
class AurumConfig:
    """Immutable‑by‑convention configuration container."""

    # ── Elasticsearch ──────────────────────────────────────────────────
    es_host: str = "localhost"
    es_port: str = "9200"
    es_index: str = "profile"

    # ── Overlap / similarity thresholds ────────────────────────────────
    join_overlap_th: float = 0.4
    num_overlap_th: float = 0.85
    inclusion_dep_th: float = 0.3
    pkfk_cardinality_th: float = 0.7
    schema_sim_max_distance: int = 10

    # ── MinHash LSH ────────────────────────────────────────────────────
    minhash_num_perm: int = 512
    minhash_threshold: float = 0.7

    # ── DBSCAN (single‑point numerical clustering) ─────────────────────
    dbscan_eps: float = 0.1
    dbscan_min_samples: int = 2

    # ── NearPy LSH (schema similarity) ─────────────────────────────────
    nearpy_projection_count: int = 30

    # ── DoD / join materialisation ─────────────────────────────────────
    csv_separator: str = ","
    join_chunksize: int = 1000
    memory_limit_fraction: float = 0.6   # fraction of total RAM
    join_timeout_seconds: int = 180      # 3 minutes

    # --- Spacy Entity Recognition -----
    spacy_model: str = "en_core_web_sm"
    spacy_size: int = 1000

    # ── Text index limits ──────────────────────────────────────────────
    limit_text_values: bool = False
    max_text_values: int = 1_000
    """Max unique values stored per column in the keyword search index."""

    # ── DuckDB ─────────────────────────────────────────────────────────
    duckdb_path: str = "aurum.db"

    # ── Serialisation paths ────────────────────────────────────────────
    graph_filename: str = "graph.pickle"
    id_info_filename: str = "id_info.pickle"
    table_ids_filename: str = "table_ids.pickle"
    schema_sim_index_filename: str = "schema_sim_index.pkl"
    content_sim_index_filename: str = "content_sim_index.pkl"
