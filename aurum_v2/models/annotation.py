"""
Metadata annotation types — MDHit, MDComment, MRS.

Exact port of legacy ``api/annotation.py``.  These types support the
optional metadata / annotation system where users can label columns
with semantic relationships (MEANS_SAME_AS, IS_SUBCLASS_OF, etc.)
and attach free-text comments.

Key types
---------
MDClass
    Annotation severity: WARNING, INSIGHT, QUESTION.
MDRelation
    Semantic relation between two columns: MEANS_SAME_AS, MEANS_DIFF_FROM,
    IS_SUBCLASS_OF, IS_SUPERCLASS_OF, IS_MEMBER_OF, IS_CONTAINER_OF.
MDHit
    A metadata annotation record (namedtuple with custom identity).
MDComment
    A free-text comment attached to an annotation.
MRS
    Metadata Result Set — iterable container analogous to DRS but for
    metadata records.
"""

from __future__ import annotations

from collections import namedtuple
from enum import Enum

__all__ = [
    "MDClass",
    "MDRelation",
    "MDHit",
    "MDComment",
    "MRS",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MDClass(Enum):
    """Classification of a metadata annotation."""

    WARNING = 0
    INSIGHT = 1
    QUESTION = 2


class MDRelation(Enum):
    """Semantic relationship between two columns (user-annotated)."""

    MEANS_SAME_AS = 0
    MEANS_DIFF_FROM = 1
    IS_SUBCLASS_OF = 2
    IS_SUPERCLASS_OF = 3
    IS_MEMBER_OF = 4
    IS_CONTAINER_OF = 5


# ---------------------------------------------------------------------------
# MDHit
# ---------------------------------------------------------------------------

_BaseMDHit = namedtuple(
    "MDHit",
    ["id", "author", "md_class", "text", "source", "target", "relation"],
)


class MDHit(_BaseMDHit):
    """A single metadata annotation record.

    Identity is determined solely by :attr:`id` (matching legacy exactly).

    Parameters
    ----------
    id : str
        Unique annotation identifier.
    author : str
        Who created this annotation.
    md_class : MDClass
        WARNING / INSIGHT / QUESTION.
    text : str
        Free-text body of the annotation.
    source : str | Hit
        Source column (nid or Hit).
    target : str | Hit | None
        Target column for relational annotations; ``None`` for unary.
    relation : MDRelation | None
        Semantic relation type; ``None`` for unary annotations.
    """

    __slots__ = ()

    def __hash__(self) -> int:  # noqa: D105
        return hash(self.id)

    def __eq__(self, other: object) -> bool:  # noqa: D105
        if isinstance(other, MDHit):
            return self.id == other.id
        if isinstance(other, str):
            return self.id == other
        return False

    def __repr__(self) -> str:  # noqa: D105
        if self.target is None:
            relation_str = f"{self.source}"
        else:
            relation_str = f"{self.source} {self.relation} {self.target}"
        return f"ID: {self.id:20} RELATION: {relation_str:30} TEXT: {self.text}"

    def __str__(self) -> str:  # noqa: D105
        return self.__repr__()


# ---------------------------------------------------------------------------
# MDComment
# ---------------------------------------------------------------------------

_BaseMDComment = namedtuple("MDComment", ["id", "author", "text", "ref_id"])


class MDComment(_BaseMDComment):
    """A free-text comment attached to a metadata annotation.

    Identity is determined solely by :attr:`id`.

    Parameters
    ----------
    id : str
        Unique comment identifier.
    author : str
        Who created this comment.
    text : str
        Free-text body.
    ref_id : str
        The :attr:`MDHit.id` this comment refers to.
    """

    __slots__ = ()

    def __hash__(self) -> int:  # noqa: D105
        return hash(self.id)

    def __eq__(self, other: object) -> bool:  # noqa: D105
        if isinstance(other, MDComment):
            return self.id == other.id
        if isinstance(other, str):
            return self.id == other
        return False

    def __repr__(self) -> str:  # noqa: D105
        return f"ID: {self.id:20} REF_ID: {self.ref_id:32} TEXT: {self.text}"

    def __str__(self) -> str:  # noqa: D105
        return self.__repr__()


# ---------------------------------------------------------------------------
# MRS — Metadata Result Set
# ---------------------------------------------------------------------------

class MRS:
    """Iterable container of metadata records (``MDHit`` or ``MDComment``).

    Analogous to :class:`DRS` for data results but for metadata.  Supports
    iteration, ``len()``, ``repr()``, and data replacement.

    Legacy equivalent: ``api/annotation.py::MRS``.
    """

    def __init__(self, data: list[MDHit | MDComment] | None = None) -> None:
        self._data: list[MDHit | MDComment] = data if data is not None else []
        self._idx: int = 0

    # ── Iteration ──────────────────────────────────────────────────────

    def __iter__(self) -> MRS:  # noqa: D105
        self._idx = 0
        return self

    def __next__(self) -> MDHit | MDComment:  # noqa: D105
        if self._idx < len(self._data):
            item = self._data[self._idx]
            self._idx += 1
            return item
        self._idx = 0
        raise StopIteration

    # ── Display ────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # noqa: D105
        return "\n".join(str(item) for item in self._data)

    def __str__(self) -> str:  # noqa: D105
        return self.__repr__()

    def __len__(self) -> int:  # noqa: D105
        return len(self._data)

    # ── Data access / mutation ─────────────────────────────────────────

    @property
    def data(self) -> list[MDHit | MDComment]:
        """The underlying list of metadata records."""
        return self._data

    def set_data(self, data: list[MDHit | MDComment]) -> MRS:
        """Replace the internal data and reset the iterator.

        Returns ``self`` for chaining.
        """
        self._data = list(data)
        self._idx = 0
        return self

    def size(self) -> int:
        """Return the number of records (alias for ``len()``)."""
        return len(self._data)
