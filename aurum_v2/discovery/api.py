"""
Public-facing API class + ``init_system`` convenience functions.

``API`` is a thin subclass of :class:`Algebra` that exposes the same
interface.  ``Helper`` provides reverse-lookup and path utilities.

Supports both DuckDB (zero-infrastructure) and Elasticsearch backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aurum_v2.discovery.algebra import Algebra
from aurum_v2.graph.field_network import FieldNetwork, deserialize_network

if TYPE_CHECKING:
    from aurum_v2.config import AurumConfig

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

    def __init__(self, network: FieldNetwork, store_client) -> None:
        super().__init__(network, store_client)
        self.helper = Helper(network, store_client)


def init_system(model_path: str, config: AurumConfig | None = None) -> API:
    """Load model + Elasticsearch store.  Requires ``elasticsearch`` package."""
    if config is None:
        from aurum_v2.config import AurumConfig
        config = AurumConfig()

    network = deserialize_network(model_path)
    from aurum_v2.store.elastic_store import StoreHandler
    store = StoreHandler(config)
    return API(network, store)


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
    from aurum_v2.store.duck_store import DuckStore
    store = DuckStore(config, db_path)
    return API(network, store)
