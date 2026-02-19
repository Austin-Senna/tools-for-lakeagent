"""
Aurum v2 — Data Discovery System.

A streamlined rewrite of the Aurum legacy codebase, preserving the original
algorithms and data‑flow while adopting modern Python practices.

Quick start::

    from aurum_v2 import init_system
    api = init_system("/path/to/serialized/model")
    drs = api.search_content("salary")
"""

from aurum_v2.discovery.api import API, init_system, init_system_duck

__all__ = ["API", "init_system", "init_system_duck"]
__version__ = "2.0.0"
