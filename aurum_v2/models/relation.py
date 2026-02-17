"""
Enumerations for edge types (``Relation``), provenance operations (``OP``),
DRS iteration mode (``DRSMode``), and the ``Operation`` carrier.

Values are identical to the legacy ``api/apiutils.py`` definitions.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Relation", "OP", "Operation", "DRSMode"]


# ---------------------------------------------------------------------------
# Relation — edge types in the FieldNetwork graph
# ---------------------------------------------------------------------------

class Relation(Enum):
    """Types of edges that can exist between two columns in the field‑network."""

    SCHEMA = 0                  # same‑table co‑occurrence
    SCHEMA_SIM = 1              # TF‑IDF + NearPy LSH on column names
    CONTENT_SIM = 2             # MinHash LSH (text) / distribution overlap (numeric)
    ENTITY_SIM = 3              # semantic entity similarity (disabled in current pipeline)
    PKFK = 5                    # primary‑key / foreign‑key candidate
    INCLUSION_DEPENDENCY = 6    # value‑set containment
    # ── User‑annotated metadata relations ──────────────────────────────
    MEANS_SAME = 10
    MEANS_DIFF = 11
    SUBCLASS = 12
    SUPERCLASS = 13
    MEMBER = 14
    CONTAINER = 15

    def is_metadata(self) -> bool:
        """Return ``True`` if this relation is a user‑annotated metadata type."""
        return self.value >= 10


# ---------------------------------------------------------------------------
# OP — provenance edge labels
# ---------------------------------------------------------------------------

class OP(Enum):
    """Labels placed on provenance‑graph edges to record *how* a result was derived."""

    NONE = 0
    ORIGIN = 1
    KW_LOOKUP = 2
    SCHNAME_LOOKUP = 3
    SCHEMA_SIM = 4
    TABLE = 5
    CONTENT_SIM = 6
    PKFK = 7
    ENTITY_SIM = 8
    ENTITY_LOOKUP = 9
    MEANS_SAME = 10
    MEANS_DIFF = 11
    SUBCLASS = 12
    SUPERCLASS = 13
    MEMBER = 14
    CONTAINER = 15


# ---------------------------------------------------------------------------
# Operation — (OP, params) carrier
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Operation:
    """Bundles an OP with optional parameters (e.g. the keyword or source Hit)."""
    op: OP
    params: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DRSMode — iteration mode for DRS
# ---------------------------------------------------------------------------

class DRSMode(Enum):
    """Controls whether iterating a DRS yields columns or table names."""

    FIELDS = 0
    TABLE = 1
