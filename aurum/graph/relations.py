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
    SCHEMA = 0
    SCHEMA_SIM = 1
    CONTENT_SIM = 2
    ENTITY_SIM = 3
    PKFK = 5
    INCLUSION_DEPENDENCY = 6
    MEANS_SAME = 10
    MEANS_DIFF = 11
    SUBCLASS = 12
    SUPERCLASS = 13
    MEMBER = 14
    CONTAINER = 15

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
    NONE = 0  # for initial creation of DRS
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
