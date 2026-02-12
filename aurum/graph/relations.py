"""
Relation types for the column knowledge graph.

Ported from ``aurum/api/apiutils.py :: Relation`` and ``OP`` enums.
"""

from __future__ import annotations

from enum import Enum, auto


class Relation(Enum):
    """Edge types in the field network.

    Each value corresponds to a distinct kind of relationship between two
    columns that was discovered during the index-building phase.

    Origin mapping (Aurum ``apiutils.Relation``):
    - SCHEMA       → same-table membership (implicit in Aurum's ``source_ids``)
    - SCHEMA_SIM   → column names are similar (TF-IDF / embeddings)
    - CONTENT_SIM  → column *values* are similar (MinHash or numeric overlap)
    - PKFK         → primary-key / foreign-key candidate
    - INCLUSION_DEP→ one column's value range contained in another's
    """

    SCHEMA = auto()
    SCHEMA_SIM = auto()
    CONTENT_SIM = auto()
    ENTITY_SIM = auto()
    PKFK = auto()
    INCLUSION_DEPENDENCY = auto()

    # ── Metadata / Annotation relations (from Aurum's ontology matcher) ──
    MEANS_SAME = auto()
    MEANS_DIFF = auto()
    SUBCLASS = auto()
    SUPERCLASS = auto()
    MEMBER = auto()
    CONTAINER = auto()

    def is_metadata(self) -> bool:
        """Return *True* for relations that come from user annotations
        rather than automatic discovery."""
        return self in {
            Relation.MEANS_SAME,
            Relation.MEANS_DIFF,
            Relation.SUBCLASS,
            Relation.SUPERCLASS,
            Relation.MEMBER,
            Relation.CONTAINER,
        }


class OP(Enum):
    """Operation codes for provenance tracking.

    Each describes *how* a result was produced in the query algebra.
    Ported from ``apiutils.OP``.
    """

    NONE = auto()
    ORIGIN = auto()
    KW_LOOKUP = auto()
    SCHNAME_LOOKUP = auto()
    SCHEMA_SIM = auto()
    TABLE = auto()
    CONTENT_SIM = auto()
    PKFK = auto()
    ENTITY_SIM = auto()
    ENTITY_LOOKUP = auto()
