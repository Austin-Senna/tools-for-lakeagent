"""
Public-facing API class + ``init_system`` convenience functions.

``API`` is a thin subclass of :class:`Algebra` that exposes the same
interface.  ``Helper`` provides reverse-lookup and path utilities.

Supports both DuckDB (zero-infrastructure) and Elasticsearch backends.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from aurum_v2.discovery.algebra import Algebra
from aurum_v2.graph.field_network import FieldNetwork, deserialize_network
from aurum_v2.utils.io_utils import deserialize_object

if TYPE_CHECKING:
    from aurum_v2.config import AurumConfig
    from aurum_v2.builder.network_builder import _LSHIndex

log = logging.getLogger(__name__)

__all__ = ["API", "Helper", "init_system", "init_system_duck"]


class Helper:
    """Utility class for reverse lookups and file-path resolution."""

    def __init__(self, network: FieldNetwork, store_client) -> None:
        self._network = network
        self._store_client = store_client

    def reverse_lookup(self, nid: str) -> list:
        """Return ``[(nid, db_name, source_name, field_name)]``."""
        return self._network.get_info_for([nid])

    def get_path_nid(self, nid: str) -> str:
        """Resolve the filesystem path for the data source containing *nid*."""
        if hasattr(self._store_client, "get_path_of"):
            return self._store_client.get_path_of(nid)
        return ""


class API(Algebra):
    """Top-level API — inherits all :class:`Algebra` methods.

    Also attaches a :class:`Helper` instance as ``self.helper``.
    """

    def __init__(self, network: FieldNetwork, store_client, schema_sim_index: _LSHIndex | None = None) -> None:
        super().__init__(network, store_client)
        self.helper = Helper(network, store_client)
        self._schema_sim_index = schema_sim_index

    def search_schema_sim(self, kw: str, max_results: int = 10) -> list[dict]:
        """Find columns with names similar to *kw* via LSH (TF-IDF cosine).

        Returns plain ``{nid, db, source, field}`` dicts so callers don't need
        to know about Hit internals.  Returns an empty list if the index was
        not loaded.
        """
        if self._schema_sim_index is None:
            return []
        try:
            raw = self._schema_sim_index.query_string(kw)
        except RuntimeError:
            return []

        results = []
        seen: set[str] = set()
        for _, nid, _dist in raw:
            if nid in seen:
                continue
            seen.add(nid)
            info = self._network.get_info_for([nid])
            if not info:
                continue
            _, db, source, field = info[0]
            results.append({"nid": nid, "db": db, "source": source, "field": field})
            if len(results) >= max_results:
                break
        return results


def _load_schema_sim_index(model_path: str, config: AurumConfig) -> _LSHIndex | None:
    """Deserialize the schema-sim LSH index if present, else return None."""
    index_path = os.path.join(model_path, config.schema_sim_index_filename)
    if not os.path.exists(index_path):
        log.debug("schema_sim_index not found at %s — LSH search disabled", index_path)
        return None
    try:
        idx = deserialize_object(index_path)
        log.debug("schema_sim_index loaded from %s", index_path)
        return idx
    except Exception as exc:
        log.warning("Failed to load schema_sim_index: %s", exc)
        return None


def init_system(model_path: str, config: AurumConfig | None = None) -> API:
    """Load model + Elasticsearch store.  Requires ``elasticsearch`` package."""
    if config is None:
        from aurum_v2.config import AurumConfig
        config = AurumConfig()

    network = deserialize_network(model_path)
    schema_sim_index = _load_schema_sim_index(model_path, config)
    from aurum_v2.store.elastic_store import StoreHandler
    store = StoreHandler(config)
    return API(network, store, schema_sim_index)


def init_system_duck(
    model_path: str,
    db_path: str = "aurum.db",
    config: AurumConfig | None = None,
) -> API:
    """Load model + DuckDB store.  No external services needed.

    Parameters
    ----------
    model_path : str
        Directory containing ``graph.pickle``, ``id_info.pickle``, etc.
    db_path : str
        Path to the DuckDB ``.db`` file.
    config : AurumConfig | None
        If ``None``, a default :class:`AurumConfig` is created.
    """
    if config is None:
        from aurum_v2.config import AurumConfig
        config = AurumConfig()

    network = deserialize_network(model_path)
    schema_sim_index = _load_schema_sim_index(model_path, config)
    from aurum_v2.store.duck_store import DuckStore
    store = DuckStore(config, db_path)
    return API(network, store, schema_sim_index)
