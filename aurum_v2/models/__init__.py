"""Core data‑model classes used throughout Aurum v2."""

from aurum_v2.models.hit import Hit, compute_field_id
from aurum_v2.models.relation import Relation, OP, Operation, DRSMode
from aurum_v2.models.provenance import Provenance
from aurum_v2.models.drs import DRS

__all__ = [
    "Hit",
    "compute_field_id",
    "Relation",
    "OP",
    "Operation",
    "DRSMode",
    "Provenance",
    "DRS",
]
