"""Tests for aurum.discovery.algebra."""

import polars as pl
import pytest

from aurum.config import aurumConfig
from aurum.discovery.algebra import Algebra
from aurum.discovery.result_set import DRS, DRSMode, Operation
from aurum.graph.field_network import FieldNetwork, Hit
from aurum.graph.relations import OP, Relation
from aurum.profiler.column_profiler import profile_dataframe


@pytest.fixture
def algebra_env():
    """Build a small field network with 2 tables."""
    cfg = aurumConfig()
    df1 = pl.DataFrame({
        "employee_id": [1, 2, 3, 4, 5],
        "employee_name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "dept_id": [10, 20, 10, 30, 20],
    })
    df2 = pl.DataFrame({
        "department_id": [10, 20, 30],
        "department_name": ["Engineering", "Sales", "HR"],
    })
    p1 = profile_dataframe(df1, "hr", "employees.csv", cfg)
    p2 = profile_dataframe(df2, "hr", "departments.csv", cfg)
    profiles = p1 + p2
    net = FieldNetwork()
    net.init_from_profiles(profiles)

    # Manually add a PKFK edge between dept_id and department_id
    dept_id_prof = [p for p in profiles if p.col_id.field_name == "dept_id"][0]
    department_id_prof = [p for p in profiles if p.col_id.field_name == "department_id"][0]
    h1 = Hit(dept_id_prof.col_id.nid, "hr", "employees.csv", "dept_id", 0.8)
    h2 = Hit(department_id_prof.col_id.nid, "hr", "departments.csv", "department_id", 0.8)
    net.add_relation(h1, h2, Relation.PKFK, score=0.8)

    return Algebra(net, cfg), net, profiles


class TestAlgebra:
    def test_search_attribute(self, algebra_env):
        api, _, _ = algebra_env
        drs = api.search_attribute("employee")
        assert len(drs) >= 1
        # Should find employee_id and/or employee_name
        names = {h.field_name for h in drs}
        assert names & {"employee_id", "employee_name"}

    def test_search_exact_attribute(self, algebra_env):
        api, _, _ = algebra_env
        drs = api.search_exact_attribute("dept_id")
        assert len(drs) == 1
        assert list(drs)[0].field_name == "dept_id"

    def test_drs_from_table(self, algebra_env):
        api, _, _ = algebra_env
        drs = api.drs_from_table("employees.csv")
        assert len(drs) == 3  # employee_id, employee_name, dept_id

    def test_pkfk_of(self, algebra_env):
        api, _, profiles = algebra_env
        dept_id_prof = [p for p in profiles if p.col_id.field_name == "dept_id"][0]
        hit = Hit(dept_id_prof.col_id.nid, "hr", "employees.csv", "dept_id", 0.0)
        drs = api.pkfk_of(hit)
        assert len(drs) >= 1
        assert any(h.field_name == "department_id" for h in drs)

    def test_set_operations(self, algebra_env):
        api, _, _ = algebra_env
        a = api.search_attribute("employee")
        b = api.search_attribute("department")
        # Union
        u = a | b
        assert len(u) >= len(a)
        assert len(u) >= len(b)
        # Intersection
        i = a & b
        assert len(i) <= len(a)
        # Difference
        d = a - b
        assert len(d) <= len(a)


class TestDRS:
    def test_iteration(self):
        hits = [Hit("n1", "db", "t.csv", "c1", 1.0), Hit("n2", "db", "t.csv", "c2", 0.9)]
        drs = DRS(hits, Operation(OP.ORIGIN))
        assert len(drs) == 2
        assert list(drs) == hits

    def test_mode(self):
        drs = DRS([], Operation(OP.NONE))
        assert drs.mode == DRSMode.FIELDS
        drs.set_table_mode()
        assert drs.mode == DRSMode.TABLE
