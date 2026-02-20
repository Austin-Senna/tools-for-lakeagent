"""
Hit — the atomic column reference used everywhere in Aurum.

A Hit uniquely identifies one column in the data lake by its *nid*
(``source_name.column_name``).

Identity semantics:
    * ``__hash__``  → ``hash(nid)``
    * ``__eq__``    → compare ``nid`` values
"""

from __future__ import annotations

from collections import namedtuple

__all__ = ["Hit", "compute_field_id"]


# ---------------------------------------------------------------------------
# ID helper
# ---------------------------------------------------------------------------

def compute_field_id(db_name: str, source_name: str, field_name: str) -> str:
    """Return a human-readable node identifier: ``source_name.field_name``.

    Previous versions used CRC32 hashes (following the legacy ES store),
    but that was fragile (32-bit collisions, no separator between
    components).  String nids are unique by construction and readable
    in graph visualisations.
    """
    return f"{source_name}.{field_name}"


# ---------------------------------------------------------------------------
# Hit
# ---------------------------------------------------------------------------

_BaseHit = namedtuple("Hit", ["nid", "db_name", "source_name", "field_name", "score"])


class Hit(_BaseHit):
    """Lightweight reference to a single profiled column.

    Parameters
    ----------
    nid : str
        String identifier ``source_name.column_name`` (see :func:`compute_field_id`).
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
        return hash(self.nid)

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
