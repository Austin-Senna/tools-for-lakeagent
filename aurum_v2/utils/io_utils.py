"""
Serialisation helpers (pickle‑based).

Mirrors ``inputoutput/inputoutput.py`` from the legacy codebase.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

__all__ = ["serialize_object", "deserialize_object"]


def serialize_object(obj: Any, path: str) -> None:
    """Pickle *obj* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def deserialize_object(path: str) -> Any:
    """Unpickle an object from *path*."""
    with open(path, "rb") as f:
        return pickle.load(f)
