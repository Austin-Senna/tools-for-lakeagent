"""
Materialised view analysis — 4C classification.

Compares pairs of materialised views (DataFrames) and classifies their
relationship as one of:

* **Equivalent** — identical cardinality, schema, and values.
* **Contained** — every value in the smaller view exists in the larger.
* **Complementary** — views contribute different key values.
* **Contradictory** — same key values but different non-key values.

Also provides utility functions for determining the most likely primary key,
per-column uniqueness ratios, and row-level conflict detection.

Direct port of legacy ``DoD/material_view_analysis.py``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd

__all__ = [
    "ViewClass",
    "most_likely_key",
    "uniqueness",
    "curate_view",
    "equivalent",
    "contained",
    "complementary",
    "contradictory",
    "inconsistent_value_on_key",
]


# ---------------------------------------------------------------------------
# Classification enum  (legacy: EQUI)
# ---------------------------------------------------------------------------

class ViewClass(Enum):
    """Classification result for view comparison.

    Legacy name: ``EQUI`` in ``DoD/material_view_analysis.py``.
    """

    EQUIVALENT = 1
    DIF_CARDINALITY = 2
    DIF_SCHEMA = 3
    DIF_VALUES = 4


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def most_likely_key(df: pd.DataFrame) -> tuple[str, float]:
    """Return ``(column_name, uniqueness_ratio)`` for the most-unique column.

    The column with the highest ``unique / total`` ratio is the best
    candidate primary key.

    Legacy equivalent: ``material_view_analysis.most_likely_key``.

    Parameters
    ----------
    df : pd.DataFrame
        A materialised view.

    Returns
    -------
    (str, float)
        Column name and its uniqueness ratio.
    """
    raise NotImplementedError


def uniqueness(df: pd.DataFrame) -> dict[str, float]:
    """Return a dictionary mapping each column to its uniqueness ratio.

    ``uniqueness = unique_values / total_values``

    Legacy equivalent: ``material_view_analysis.uniqueness``.
    """
    raise NotImplementedError


def curate_view(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a view before comparison: drop NaN, deduplicate, sort.

    Steps (matching legacy exactly):

    1. ``dropna()``
    2. ``drop_duplicates()``
    3. ``reset_index(drop=True)``
    4. ``sort_index(axis=1)`` — sort columns alphabetically.
    5. ``sort_index(axis=0)`` — sort rows by index.

    Legacy equivalent: ``material_view_analysis.curate_view``.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4C classification functions
# ---------------------------------------------------------------------------

def equivalent(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
) -> tuple[bool, ViewClass]:
    """Check if two views are exactly equivalent.

    Algorithm (matches legacy):

    1. Curate both views.
    2. If row counts differ → ``(False, DIF_CARDINALITY)``.
    3. If column counts differ → ``(False, DIF_SCHEMA)``.
    4. If column names differ → ``(False, DIF_SCHEMA)``.
    5. For each column, compare lowercased sorted values.
       If any mismatch → ``(False, DIF_VALUES)``.
    6. Otherwise → ``(True, EQUIVALENT)``.

    Legacy equivalent: ``material_view_analysis.equivalent``.
    """
    raise NotImplementedError


def contained(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
) -> tuple[bool, int] | bool:
    """Check if one view is contained within the other.

    For each column, every value in the smaller view must exist in the
    larger view (case-insensitive).

    Returns
    -------
    True
        If containment holds.
    (False, diff_count)
        If containment fails, with the number of non-contained values.

    Legacy equivalent: ``material_view_analysis.contained``.
    """
    raise NotImplementedError


def complementary(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
) -> tuple[bool, set[Any]] | bool:
    """Check if two views contribute different key values.

    Uses :func:`most_likely_key` to identify the key column in each view,
    then computes the symmetric difference of key sets.

    Returns
    -------
    (True, sdiff)
        If the symmetric difference is non-empty.
    False
        If key sets are identical.

    Legacy equivalent: ``material_view_analysis.complementary``.
    """
    raise NotImplementedError


def contradictory(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
) -> tuple[bool, int] | bool:
    """Check if two views have conflicting non-key values for the same keys.

    Groups both views by their most-likely key, then for each shared group
    checks :func:`equivalent`.  Any non-equivalent group is a contradiction.

    Returns
    -------
    (True, contradiction_count)
        If contradictions exist.
    False
        If no contradictions.

    Legacy equivalent: ``material_view_analysis.contradictory``.
    """
    raise NotImplementedError


def inconsistent_value_on_key(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    key: str | None = None,
) -> tuple[list, list, list, list[tuple]]:
    """Detailed row-level conflict detection between two views.

    For each key value, compares corresponding rows across *df1* and *df2*
    and reports:

    1. **missing_keys** — key values present in one but not the other.
    2. **non_unique_df1** — key values with duplicate rows in *df1*.
    3. **non_unique_df2** — key values with duplicate rows in *df2*.
    4. **conflicting_pairs** — ``(row1_values, row2_values)`` tuples where
       the same key maps to different non-key values.

    Legacy equivalent: ``material_view_analysis.inconsistent_value_on_key``.
    """
    raise NotImplementedError
