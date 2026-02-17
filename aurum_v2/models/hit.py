"""
Hit — the atomic column reference used everywhere in Aurum.

A Hit uniquely identifies one column in the data lake by its *nid*
(CRC32 hash of ``db_name + source_name + field_name``).

Identity semantics:
    * ``__hash__``  → ``hash(int(nid))``
    * ``__eq__``    → compare ``nid`` values

These match the legacy ``api/apiutils.py`` behaviour exactly.
"""

from __future__ import annotations

import binascii
from collections import namedtuple

__all__ = ["Hit", "compute_field_id"]


# ---------------------------------------------------------------------------
# ID helper
# ---------------------------------------------------------------------------

def compute_field_id(db_name: str, source_name: str, field_name: str) -> str:
    """Return the CRC32 hash of *db_name + source_name + field_name* as a string.

    This is the canonical node identifier used by the field‑network graph
    and by Elasticsearch's profiler index.
    """
    raw = db_name + source_name + field_name
    nid = binascii.crc32(raw.encode("utf-8"))
    return str(nid)


# ---------------------------------------------------------------------------
# Hit
# ---------------------------------------------------------------------------

_BaseHit = namedtuple("Hit", ["nid", "db_name", "source_name", "field_name", "score"])


class Hit(_BaseHit):
    """Lightweight reference to a single profiled column.

    Parameters
    ----------
    nid : str
        CRC32 string identifier (see :func:`compute_field_id`).
    db_name : str
        Logical database / data‑source group name.
    source_name : str
        Table or file name.
    field_name : str
        Column / attribute name.
    score : float
        Relevance score assigned by the operation that produced this Hit.
    """

    __slots__ = ()

    # Identity is determined *solely* by the column id.
    def __hash__(self) -> int:  # noqa: D105
        return int(self.nid)

    def __eq__(self, other: object) -> bool:  # noqa: D105
        if isinstance(other, int):
            return self.nid == other
        if isinstance(other, Hit):
            return self.nid == other.nid
        if other is not None and hasattr(other, "nid"):
            return self.nid == other.nid
        return False

    def __str__(self) -> str:  # noqa: D105
        return self.__repr__()
