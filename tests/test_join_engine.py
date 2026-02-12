"""Tests for aurum.materialization."""

import polars as pl
import pytest

from aurum.materialization.join_engine import (
    apply_filter,
    does_join_fit_in_memory,
    estimate_output_row_size,
    join_on_key,
)
from aurum.materialization.view_analysis import (
    classify_relationship,
    complementary,
    contained,
    contradictory,
    equivalent,
    most_likely_key,
    uniqueness,
)


# ── join_engine ──────────────────────────────────────────────────────

class TestJoinEngine:
    def test_estimate_row_size(self):
        a = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        b = pl.DataFrame({"x": [1, 2], "z": [10.0, 20.0]})
        size = estimate_output_row_size(a, b)
        assert size > 0

    def test_does_join_fit(self):
        a = pl.DataFrame({"x": list(range(100))})
        b = pl.DataFrame({"x": list(range(100))})
        # Small DataFrames should always fit
        assert does_join_fit_in_memory(a, b) is True

    def test_join_on_key(self):
        a = pl.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        b = pl.DataFrame({"id": [2, 3, 4], "dept": ["Sales", "Eng", "HR"]})
        result = join_on_key(a, b, "id", "id")
        assert result.height == 2  # ids 2 and 3
        assert "name" in result.columns
        assert "dept" in result.columns

    def test_join_different_keys(self):
        a = pl.DataFrame({"emp_id": [1, 2], "name": ["Alice", "Bob"]})
        b = pl.DataFrame({"worker_id": [1, 2], "salary": [50000, 60000]})
        result = join_on_key(a, b, "emp_id", "worker_id")
        assert result.height == 2

    def test_apply_filter(self):
        df = pl.DataFrame({"city": ["New York", "Los Angeles", "New Orleans"]})
        result = apply_filter(df, "city", "new")
        assert result.height == 2  # New York, New Orleans


# ── view_analysis ────────────────────────────────────────────────────

class TestViewAnalysis:
    def test_uniqueness(self):
        s = pl.Series([1, 2, 3, 4, 5])
        assert uniqueness(s) == pytest.approx(1.0)
        s2 = pl.Series([1, 1, 1, 1, 1])
        assert uniqueness(s2) == pytest.approx(0.2)

    def test_most_likely_key(self):
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "category": ["A", "A", "B", "B", "C"],
        })
        assert most_likely_key(df) == "id"

    def test_equivalent(self):
        a = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        b = pl.DataFrame({"x": [2, 1], "y": ["b", "a"]})
        assert equivalent(a, b) is True

    def test_not_equivalent(self):
        a = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        b = pl.DataFrame({"x": [1, 3], "y": ["a", "c"]})
        assert equivalent(a, b) is False

    def test_contained(self):
        base = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        candidate = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        assert contained(candidate, base) is True

    def test_not_contained(self):
        base = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        candidate = pl.DataFrame({"x": [1, 3], "y": ["a", "c"]})
        assert contained(candidate, base) is False

    def test_complementary(self):
        a = pl.DataFrame({"id": [1, 2], "v": ["a", "b"]})
        b = pl.DataFrame({"id": [3, 4], "v": ["c", "d"]})
        assert complementary(a, b) is True

    def test_not_complementary(self):
        a = pl.DataFrame({"id": [1, 2], "v": ["a", "b"]})
        b = pl.DataFrame({"id": [2, 3], "v": ["b", "c"]})
        assert complementary(a, b, key="id") is False

    def test_contradictory(self):
        a = pl.DataFrame({"id": [1, 2], "val": ["x", "y"]})
        b = pl.DataFrame({"id": [1, 2], "val": ["x", "z"]})
        assert contradictory(a, b, key="id") is True

    def test_classify_equivalent(self):
        a = pl.DataFrame({"x": [1, 2]})
        assert classify_relationship(a, a) == "equivalent"

    def test_classify_contained(self):
        base = pl.DataFrame({"x": [1, 2, 3]})
        cand = pl.DataFrame({"x": [1, 2]})
        assert classify_relationship(base, cand) == "contained"

    def test_classify_complementary(self):
        a = pl.DataFrame({"x": [1, 2]})
        b = pl.DataFrame({"x": [3, 4]})
        assert classify_relationship(a, b) == "complementary"
