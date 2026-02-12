"""
Public‑facing API class + ``init_system`` convenience function.

``API`` is a thin subclass of :class:`Algebra` that exposes the same
interface.  ``Helper`` provides reverse‑lookup and path utilities.

Direct port of ``algebra.py::API`` / ``Helper`` and ``main.py::init_system``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aurum_v2.discovery.algebra import Algebra
from aurum_v2.graph.field_network import FieldNetwork, deserialize_network
from aurum_v2.store.elastic_store import StoreHandler

if TYPE_CHECKING:
    from aurum_v2.config import AurumConfig

__all__ = ["API", "Helper", "init_system"]


class Helper:
    """Utility class for reverse lookups and file‑path resolution."""

    def __init__(self, network: FieldNetwork, store_client: StoreHandler) -> None:
        self._network = network
        self._store_client = store_client

    def reverse_lookup(self, nid: str) -> list:
        """Return ``[(nid, db_name, source_name, field_name)]``."""
        return self._network.get_info_for([nid])

    def get_path_nid(self, nid: str) -> str:
        """Resolve the filesystem path for the data source containing *nid*."""
        return self._store_client.get_path_of(nid)


class API(Algebra):
    """Top‑level API — inherits all :class:`Algebra` methods.

    Also attaches a :class:`Helper` instance as ``self.helper``.
    """

    def __init__(self, network: FieldNetwork, store_client: StoreHandler) -> None:
        super().__init__(network, store_client)
        self.helper = Helper(network, store_client)


def init_system(model_path: str, config: AurumConfig | None = None) -> API:
    """Load a serialised model and return a ready‑to‑use :class:`API`.

    Parameters
    ----------
    model_path : str
        Directory containing ``graph.pickle``, ``id_info.pickle``, etc.
    config : AurumConfig | None
        If ``None``, a default :class:`AurumConfig` is created.

    Returns
    -------
    API
        Fully initialised discovery API backed by the loaded network and
        an Elasticsearch connection.
    """
    if config is None:
        from aurum_v2.config import AurumConfig
        config = AurumConfig()

    network = deserialize_network(model_path)
    store = StoreHandler(config)
    return API(network, store)
