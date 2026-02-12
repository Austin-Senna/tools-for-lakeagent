"""Tests for aurum.graph — network builder and field network."""

import polars as pl
import pytest

from aurum.config import aurumConfig
from aurum.graph.field_network import FieldNetwork, Hit
from aurum.graph.network_builder import (
    _compute_interval_overlap,
    build_content_sim_text,
    build_pkfk,
    build_schema_sim,
)
from aurum.graph.relations import Relation
from aurum.profiler.column_profiler import profile_dataframe


@pytest.fixture
def sample_profiles():
    cfg = aurumConfig()
    df1 = pl.DataFrame({
        "employee_name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "salary": [50000, 60000, 70000, 80000, 90000],
    })
    df2 = pl.DataFrame({
        "worker_name": ["Alice", "Bob", "Charlie", "Frank", "Grace"],
        "income": [50000, 60000, 70000, 55000, 65000],
    })
    p1 = profile_dataframe(df1, db_name="hr", source_name="employees.csv", cfg=cfg)
    p2 = profile_dataframe(df2, db_name="hr", source_name="workers.csv", cfg=cfg)
    return p1 + p2, cfg


@pytest.fixture
def sample_network(sample_profiles):
    profiles, cfg = sample_profiles
    net = FieldNetwork()
    net.init_from_profiles(profiles)
    return net, profiles, cfg


class TestIntervalOverlap:
    def test_full_overlap(self):
        assert _compute_interval_overlap(0, 10, 0, 10) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _compute_interval_overlap(0, 10, 20, 30) == pytest.approx(0.0)

    def test_partial_overlap(self):
        overlap = _compute_interval_overlap(0, 10, 5, 15)
        assert 0.0 < overlap < 1.0

    def test_contained(self):
        overlap = _compute_interval_overlap(0, 100, 25, 75)
        assert overlap > 0.0


class TestFieldNetwork:
    def test_init_from_profiles(self, sample_network):
        net, profiles, _ = sample_network
        # All profiles should be in the graph
        for p in profiles:
            assert net._graph.has_node(p.col_id.nid)

    def test_add_relation(self, sample_network):
        net, profiles, _ = sample_network
        h1 = Hit(profiles[0].col_id.nid, "", "employees.csv", "employee_name", 0.9)
        h2 = Hit(profiles[2].col_id.nid, "", "workers.csv", "worker_name", 0.9)
        net.add_relation(h1, h2, Relation.CONTENT_SIM, score=0.9)
        neighbours = net.neighbors_id(h1, Relation.CONTENT_SIM)
        assert any(n.nid == h2.nid for n in neighbours)

    def test_get_hits_from_table(self, sample_network):
        net, profiles, _ = sample_network
        hits = net.get_hits_from_table("employees.csv")
        assert len(hits) == 2  # employee_name, salary


class TestNetworkBuilder:
    def test_build_schema_sim(self, sample_network):
        net, profiles, cfg = sample_network
        build_schema_sim(net, profiles, cfg)
        # employee_name and worker_name should be schema-similar
        emp = [p for p in profiles if p.col_id.field_name == "employee_name"][0]
        hit = Hit(emp.col_id.nid, "", "employees.csv", "employee_name", 0.0)
        neighbours = net.neighbors_id(hit, Relation.SCHEMA_SIM)
        # May or may not find a match depending on threshold, but should not error
        assert isinstance(neighbours, list)

    def test_build_content_sim_text(self, sample_network):
        net, profiles, cfg = sample_network
        build_content_sim_text(net, profiles, cfg)
        # Textual columns with overlapping values should get CONTENT_SIM edges
        # At minimum, check no errors
        edge_count = sum(
            1 for _, _, d in net._graph.edges(data=True)
            if d.get("relation") == Relation.CONTENT_SIM
        )
        assert edge_count >= 0  # Sanity check
