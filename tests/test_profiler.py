"""Tests for aurum.profiler."""

import polars as pl
import pytest

from aurum.config import aurumConfig
from aurum.profiler.column_profiler import (
    ColumnId,
    ColumnProfile,
    NumericProfile,
    profile_column,
    profile_dataframe,
)
from aurum.profiler.text_utils import (
    camelcase_to_snakecase,
    curate_string,
    filter_term_vector,
    normalise_value,
    tokenize_name,
)


# ── text_utils ───────────────────────────────────────────────────────

class TestTextUtils:
    def test_camelcase(self):
        assert camelcase_to_snakecase("EmployeeName") == "employee_name"
        assert camelcase_to_snakecase("HTTPSConnection") == "https_connection"

    def test_curate_string(self):
        result = curate_string("Hello--World__Foo  bar")
        # Should split on delimiters and rejoin with spaces
        assert "hello" in result.lower()
        assert "world" in result.lower()

    def test_tokenize_name(self):
        tokens = tokenize_name("employeeFirstName")
        assert "employee" in tokens
        assert "first" in tokens
        assert "name" in tokens

    def test_normalise_value(self):
        assert normalise_value("  Hello World  ") == "hello world"
        assert normalise_value("123") == "123"

    def test_filter_stopwords(self):
        terms = {"the": 5, "hello": 3, "a": 10, "discovery": 2}
        filtered = filter_term_vector(terms)
        assert "the" not in filtered
        assert "a" not in filtered
        assert "hello" in filtered
        assert "discovery" in filtered


# ── column_profiler ──────────────────────────────────────────────────

class TestColumnProfiler:
    def test_column_id(self):
        cid = ColumnId("testdb", "test.csv", "age")
        assert cid.nid  # non-empty
        # Same inputs → same nid
        cid2 = ColumnId("testdb", "test.csv", "age")
        assert cid.nid == cid2.nid

    def test_profile_numeric_column(self):
        df = pl.DataFrame({"age": [25, 30, 35, 40, 45]})
        cfg = aurumConfig()
        prof = profile_column(df["age"], db_name="test", source_name="test.csv", field_name="age", cfg=cfg)
        assert prof is not None
        assert prof.total_values == 5
        assert prof.numeric is not None
        assert prof.numeric.median == 35.0

    def test_profile_text_column(self):
        df = pl.DataFrame({"name": ["Alice", "Bob", "Charlie", "Diana"]})
        cfg = aurumConfig()
        prof = profile_column(df["name"], db_name="test", source_name="test.csv", field_name="name", cfg=cfg)
        assert prof is not None
        assert prof.total_values == 4
        assert prof.minhash is not None

    def test_profile_dataframe(self):
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "value": [1.0, 2.0, 3.0],
        })
        cfg = LakeAgentConfig()
        profiles = profile_dataframe(df, db_name="test", source_name="test.csv", cfg=cfg)
        assert len(profiles) == 3
        col_names = {p.col_id.field_name for p in profiles}
        assert col_names == {"id", "name", "value"}
