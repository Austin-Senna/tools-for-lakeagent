"""
4C view classification and analysis.

Port of ``aurum/DoD/material_view_analysis.py``.

Given two DataFrames (the *base* table and a *candidate* view),
classify their relationship as one of the **4C** categories:

- **Contained** — every row of the candidate is present in the base.
- **Complementary** — no row overlap at all.
- **Contradictory** — same key but different non-key values.
- **Equivalent** — identical content (within key projection).
"""

from __future__ import annotations

import polars as pl


# ---------------------------------------------------------------------------
# Utility: most-likely key
# ---------------------------------------------------------------------------

def uniqueness(series: pl.Series) -> float:
    """Fraction of unique values in a series."""
    total = series.len()
    if total == 0:
        return 0.0
    return series.n_unique() / total


def most_likely_key(df: pl.DataFrame, threshold: float = 0.7) -> str | None:
    """Return the column name most likely to be a primary key.

    Heuristic from ``material_view_analysis.most_likely_key``:
    choose the column with the highest uniqueness ≥ *threshold*.
    """
    best_col: str | None = None
    best_score = 0.0
    for col in df.columns:
        u = uniqueness(df[col])
        if u >= threshold and u > best_score:
            best_score = u
            best_col = col
    return best_col


# ---------------------------------------------------------------------------
# 4C classification
# ---------------------------------------------------------------------------

def equivalent(
    base: pl.DataFrame,
    candidate: pl.DataFrame,
    key: str | None = None,
) -> bool:
    """Return *True* if both DataFrames contain the same rows (order-independent).

    Ported from ``material_view_analysis.equivalent``.
    """
    if set(base.columns) != set(candidate.columns):
        return False
    if base.height != candidate.height:
        return False
    # Sort both by all columns and compare
    cols = sorted(base.columns)
    a = base.select(cols).sort(cols)
    b = candidate.select(cols).sort(cols)
    return a.frame_equal(b)


def contained(
    candidate: pl.DataFrame,
    base: pl.DataFrame,
    key: str | None = None,
) -> bool:
    """Return *True* if every row of *candidate* appears in *base*.

    Ported from ``material_view_analysis.contained``.
    """
    common_cols = sorted(set(base.columns) & set(candidate.columns))
    if not common_cols:
        return False
    a = candidate.select(common_cols)
    b = base.select(common_cols)
    # Anti-join: rows in a that are NOT in b
    leftover = a.join(b, on=common_cols, how="anti")
    return leftover.height == 0


def complementary(
    base: pl.DataFrame,
    candidate: pl.DataFrame,
    key: str | None = None,
) -> bool:
    """Return *True* if the two frames share no rows at all.

    Ported from ``material_view_analysis.complementary``.
    """
    if key is None:
        key = most_likely_key(base)
    if key is None or key not in base.columns or key not in candidate.columns:
        # Fall back to full-row comparison
        common_cols = sorted(set(base.columns) & set(candidate.columns))
        if not common_cols:
            return True
        overlap = base.select(common_cols).join(
            candidate.select(common_cols),
            on=common_cols,
            how="inner",
        )
        return overlap.height == 0

    overlap = base.select(key).join(candidate.select(key), on=key, how="inner")
    return overlap.height == 0


def contradictory(
    base: pl.DataFrame,
    candidate: pl.DataFrame,
    key: str | None = None,
    non_key_cols: list[str] | None = None,
) -> bool:
    """Return *True* if rows sharing the same key differ on non-key columns.

    Ported from ``material_view_analysis.contradictory``.
    """
    if key is None:
        key = most_likely_key(base)
    if key is None or key not in base.columns or key not in candidate.columns:
        return False

    common_cols = sorted(set(base.columns) & set(candidate.columns))
    if non_key_cols is None:
        non_key_cols = [c for c in common_cols if c != key]
    if not non_key_cols:
        return False

    # Inner-join on key, then check if non-key columns differ
    joined = base.select([key] + non_key_cols).join(
        candidate.select([key] + non_key_cols),
        on=key,
        how="inner",
        suffix="_cand",
    )
    if joined.height == 0:
        return False

    for col in non_key_cols:
        cand_col = f"{col}_cand"
        if cand_col in joined.columns:
            mismatches = joined.filter(pl.col(col) != pl.col(cand_col))
            if mismatches.height > 0:
                return True
    return False


# ---------------------------------------------------------------------------
# Convenience: classify
# ---------------------------------------------------------------------------

def classify_relationship(
    base: pl.DataFrame,
    candidate: pl.DataFrame,
    key: str | None = None,
) -> str:
    """Return the 4C label for the relationship between two DataFrames.

    Returns one of: ``"equivalent"``, ``"contained"``, ``"complementary"``,
    ``"contradictory"``, or ``"partial_overlap"``.
    """
    if equivalent(base, candidate, key):
        return "equivalent"
    if contained(candidate, base, key):
        return "contained"
    if complementary(base, candidate, key):
        return "complementary"
    if contradictory(base, candidate, key):
        return "contradictory"
    return "partial_overlap"
