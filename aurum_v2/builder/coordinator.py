"""
Network‑building pipeline coordinator.

Orchestrates the full offline pipeline:

1. Connect to Elasticsearch and read all profiled fields.
2. Build graph skeleton (``init_meta_schema``).
3. Build schema‑similarity edges (TF‑IDF + NearPy LSH).
4. Build content‑similarity edges — text (MinHash LSH).
5. Build content‑similarity edges — numeric (distribution overlap).
6. Build PK/FK edges (cardinality).
7. Serialize the network and LSH indexes to disk.

Direct port of ``networkbuildercoordinator.py``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aurum_v2.builder import network_builder
from aurum_v2.graph.field_network import FieldNetwork, serialize_network
from aurum_v2.store.elastic_store import StoreHandler
from aurum_v2.utils.io_utils import serialize_object

if TYPE_CHECKING:
    from aurum_v2.config import AurumConfig

__all__ = ["build_network"]


def build_network(config: "AurumConfig", output_path: str) -> None:
    """Run the complete offline network‑building pipeline.

    Parameters
    ----------
    config : AurumConfig
        System configuration (ES connection, thresholds, etc.).
    output_path : str
        Directory where the serialised model artefacts will be written.
    """
    start_all = time.time()

    network = FieldNetwork()
    store = StoreHandler(config)

    # ── Stage 1: Read all fields ──────────────────────────────────────
    fields_gen = store.get_all_fields()

    # ── Stage 2: Build skeleton ───────────────────────────────────────
    _timed("meta_schema", lambda: network.init_meta_schema(fields_gen))

    # ── Stage 3: Schema similarity (TF‑IDF → NearPy LSH) ─────────────
    schema_sim_index = _timed(
        "schema_sim",
        lambda: network_builder.build_schema_sim_relation(network, config),
    )

    # ── Stage 4: Content similarity — text (MinHash LSH) ──────────────
    mh_sigs = store.get_all_mh_text_signatures()
    content_sim_index = _timed(
        "content_sim_text",
        lambda: network_builder.build_content_sim_mh_text(network, mh_sigs, config),
    )

    # ── Stage 5: Content similarity — numeric (overlap distr.) ────────
    num_sigs = store.get_all_fields_num_signatures()
    _timed(
        "content_sim_num",
        lambda: network_builder.build_content_sim_relation_num_overlap_distr(
            network, num_sigs, config
        ),
    )

    # ── Stage 6: PK / FK ─────────────────────────────────────────────
    _timed("pkfk", lambda: network_builder.build_pkfk_relation(network, config))

    # ── Stage 7: Serialize ────────────────────────────────────────────
    serialize_network(network, output_path)
    serialize_object(
        schema_sim_index, f"{output_path}/{config.schema_sim_index_filename}"
    )
    serialize_object(
        content_sim_index, f"{output_path}/{config.content_sim_index_filename}"
    )

    elapsed = time.time() - start_all
    print(f"Network build complete.  Total time: {elapsed:.2f}s")


# ── Timing helper ─────────────────────────────────────────────────────

def _timed(label: str, fn):
    """Run *fn*, print elapsed time, and return its result."""
    start = time.time()
    result = fn()
    elapsed = time.time() - start
    print(f"[{label}] {elapsed:.2f}s")
    return result
