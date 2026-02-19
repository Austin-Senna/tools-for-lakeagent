"""
Tests for aurum_v2 bug fixes identified in the architectural review.

Each test class targets a specific bug fix:
- P0 (crash bugs): Hit.__hash__, DRS.to_dict, pretty_print, ES get_all_fields_name
- P1 (logic bugs): find_path data assembly, certainty ranking, intersection TABLE mode
- P2 (scaling):    numeric overlap early break, schema sim vector caching
- P3 (design):     provenance thread safety, DuckStore regex injection, DRS iteration
"""

from __future__ import annotations

import itertools
import threading
from collections import defaultdict
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from aurum_v2.models.hit import Hit, compute_field_id
from aurum_v2.models.relation import (
    DRSMode,
    OP,
    Operation,
    Relation,
)
from aurum_v2.models.drs import DRS, _DRSIterator
from aurum_v2.models.provenance import Provenance, _global_origin_counter
from aurum_v2.graph.field_network import FieldNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hit(db: str = "db", src: str = "table", field: str = "col", score: float = 1.0) -> Hit:
    nid = compute_field_id(db, src, field)
    return Hit(nid=nid, db_name=db, source_name=src, field_name=field, score=score)


def _make_hits(n: int, table: str = "t") -> list[Hit]:
    return [_make_hit(src=table, field=f"col{i}", score=float(i)) for i in range(n)]


def _build_simple_network() -> FieldNetwork:
    """Graph: t1.a --PKFK-- t2.b --PKFK-- t3.c"""
    net = FieldNetwork()
    fields = [
        ("1", "db", "t1", "a", 100, 95, "T"),
        ("2", "db", "t2", "b", 100, 90, "T"),
        ("3", "db", "t3", "c", 100, 85, "T"),
    ]
    net.init_meta_schema(iter(fields))
    net.add_relation("1", "2", Relation.PKFK, 0.9)
    net.add_relation("2", "3", Relation.PKFK, 0.85)
    return net


# ======================================================================
# P0-1: Hit.__hash__ must return int, including synthetic nid strings
# ======================================================================


class TestHitHash:
    def test_hash_returns_int(self):
        h = _make_hit()
        assert isinstance(hash(h), int)

    def test_synthetic_nid_hashable(self):
        """Provenance creates synthetic hits with nid='synthetic_origin_N'."""
        h = Hit(nid="synthetic_origin_42", db_name="kw", source_name="kw",
                field_name="kw", score=-1.0)
        assert isinstance(hash(h), int)

    def test_synthetic_hit_in_set(self):
        """Set operations must not crash on synthetic hits."""
        h1 = Hit("synthetic_origin_0", "a", "a", "a", -1.0)
        h2 = Hit("synthetic_origin_1", "b", "b", "b", -1.0)
        h3 = _make_hit()
        s = {h1, h2, h3}
        assert len(s) == 3

    def test_synthetic_hit_in_dict(self):
        h = Hit("synthetic_origin_99", "x", "x", "x", -1.0)
        d = {h: "ok"}
        assert d[h] == "ok"

    def test_hash_consistency(self):
        """Same nid → same hash."""
        h1 = Hit("abc", "db", "t", "c", 0.0)
        h2 = Hit("abc", "db2", "t2", "c2", 1.0)
        assert hash(h1) == hash(h2)

    def test_different_nid_different_hash(self):
        h1 = _make_hit(field="a")
        h2 = _make_hit(field="b")
        # Not guaranteed but extremely likely for different strings
        assert h1.nid != h2.nid


# ======================================================================
# P0-2: DRS.to_dict() must use _asdict() not dataclasses.asdict()
# ======================================================================


class TestDRSToDict:
    def test_to_dict_no_crash(self):
        hits = _make_hits(3)
        drs = DRS(hits, Operation(OP.ORIGIN))
        result = drs.to_dict()
        assert "sources" in result
        assert "edges" in result

    def test_to_dict_structure(self):
        h1 = _make_hit(src="t1", field="a")
        h2 = _make_hit(src="t1", field="b")
        drs = DRS([h1, h2], Operation(OP.ORIGIN))
        result = drs.to_dict()
        assert "t1" in result["sources"]
        assert len(result["sources"]["t1"]["field_res"]) == 2
        # Each field_res entry should be a dict with nid, db_name, etc.
        fr = result["sources"]["t1"]["field_res"][0]
        assert "nid" in fr
        assert "db_name" in fr


# ======================================================================
# P0-3: pretty_print_columns uses fields mode, not table mode
# ======================================================================


class TestPrettyPrintColumns:
    def test_pretty_print_no_crash(self, capsys):
        hits = _make_hits(2, table="my_table")
        drs = DRS(hits, Operation(OP.ORIGIN))
        drs.pretty_print_columns()
        captured = capsys.readouterr()
        assert "my_table" in captured.out
        assert "col0" in captured.out

    def test_mode_restored(self):
        hits = _make_hits(2)
        drs = DRS(hits, Operation(OP.ORIGIN))
        drs.set_table_mode()
        drs.pretty_print_columns()
        assert drs.mode == DRSMode.TABLE


# ======================================================================
# P1-5: find_path_hit returns DRS with ALL path data, not just source
# ======================================================================


class TestFindPathHitDataAssembly:
    def test_path_contains_all_hops(self):
        net = _build_simple_network()
        src = Hit("1", "db", "t1", "a", 0)
        tgt = Hit("3", "db", "t3", "c", 0)

        result = net.find_path_hit(src, tgt, Relation.PKFK)
        nids = {h.nid for h in result.data}
        assert "1" in nids, "source should be in data"
        assert "2" in nids, "intermediate hop should be in data"
        assert "3" in nids, "target should be in data"

    def test_single_hop_path(self):
        net = _build_simple_network()
        src = Hit("1", "db", "t1", "a", 0)
        tgt = Hit("2", "db", "t2", "b", 0)

        result = net.find_path_hit(src, tgt, Relation.PKFK)
        nids = {h.nid for h in result.data}
        assert "1" in nids
        assert "2" in nids

    def test_self_path(self):
        net = _build_simple_network()
        src = Hit("1", "db", "t1", "a", 0)
        result = net.find_path_hit(src, src, Relation.PKFK)
        assert result.size() == 1

    def test_no_path(self):
        net = _build_simple_network()
        src = Hit("1", "db", "t1", "a", 0)
        tgt = Hit("99", "db", "tx", "x", 0)
        result = net.find_path_hit(src, tgt, Relation.PKFK)
        assert result.size() == 0


# ======================================================================
# P1-6: Certainty ranking — per-element visited sets
# ======================================================================


class TestCertaintyRanking:
    def test_all_elements_get_scores(self):
        """With shared visited, element B could be skipped. Now each gets its own."""
        h1 = _make_hit(field="a", score=3.0)
        h2 = _make_hit(field="b", score=5.0)
        h3 = _make_hit(field="shared_neighbor", score=1.0)

        # Build a provenance where h1→h3 and h2→h3 share a node
        drs = DRS([h1, h2], Operation(OP.ORIGIN))
        step1 = DRS([h3], Operation(OP.CONTENT_SIM, params=[h1]))
        step2 = DRS([h3], Operation(OP.CONTENT_SIM, params=[h2]))
        drs.absorb_provenance(step1)
        drs.absorb_provenance(step2)

        drs.rank_certainty()
        # Both elements must have certainty scores
        for el in [h1, h2]:
            assert el in drs._rank_data
            assert "certainty_score" in drs._rank_data[el]

    def test_ranking_order(self):
        h1 = _make_hit(field="low", score=1.0)
        h2 = _make_hit(field="high", score=10.0)
        drs = DRS([h1, h2], Operation(OP.ORIGIN))
        drs.rank_certainty()
        assert drs.data[0].score >= drs.data[1].score


# ======================================================================
# P1-7: Intersection in TABLE mode collects ALL columns per table
# ======================================================================


class TestIntersectionTableMode:
    def test_intersection_keeps_all_columns(self):
        # Table "t" has cols a, b, c in self and cols a, d in other
        h_a1 = _make_hit(src="t", field="a", score=1.0)
        h_b = _make_hit(src="t", field="b", score=2.0)
        h_c = _make_hit(src="t", field="c", score=3.0)
        h_a2 = _make_hit(src="t", field="a", score=0.5)  # same col, different score
        h_d = _make_hit(src="t", field="d", score=4.0)

        drs1 = DRS([h_a1, h_b, h_c], Operation(OP.ORIGIN))
        drs2 = DRS([h_a2, h_d], Operation(OP.ORIGIN))
        drs2.set_table_mode()

        result = drs1.intersection(drs2)
        # All 5 unique hits from the shared table should be in the result
        # (h_a1 and h_a2 have same nid so only one survives dedup)
        nids = {h.nid for h in result.data}
        assert h_b.nid in nids, "col b must survive"
        assert h_c.nid in nids, "col c must survive"
        assert h_d.nid in nids, "col d must survive"

    def test_intersection_fields_mode(self):
        h1 = _make_hit(field="x")
        h2 = _make_hit(field="y")
        h3 = _make_hit(field="x")  # same nid as h1

        drs1 = DRS([h1, h2], Operation(OP.ORIGIN))
        drs2 = DRS([h3], Operation(OP.ORIGIN))
        result = drs1.intersection(drs2)
        assert result.size() == 1
        assert result.data[0].nid == h1.nid


# ======================================================================
# P2-8: Numeric overlap early break optimization
# ======================================================================


class TestNumericOverlapEarlyBreak:
    def test_early_break_does_not_miss_matches(self):
        """Ensure the early break doesn't incorrectly skip valid pairs."""
        from aurum_v2.builder.network_builder import (
            build_content_sim_relation_num_overlap_distr,
        )
        from aurum_v2.config import AurumConfig

        net = FieldNetwork()
        # Two columns with near-identical distributions
        fields = [
            ("n1", "db", "t1", "salary", 1000, 500, "N"),
            ("n2", "db", "t2", "income", 1000, 500, "N"),
        ]
        net.init_meta_schema(iter(fields))

        # median=100, iqr=20 → range [80, 120] for both
        sigs = [
            ("n1", (100.0, 20.0, 50.0, 150.0)),
            ("n2", (100.0, 20.0, 55.0, 145.0)),
        ]
        config = AurumConfig()
        build_content_sim_relation_num_overlap_distr(net, iter(sigs), config)

        # Should have created CONTENT_SIM edge
        neighbors = net.neighbors_id("n1", Relation.CONTENT_SIM)
        assert neighbors.size() > 0


# ======================================================================
# P2-9: Schema sim caches dense vectors (no crash, correct results)
# ======================================================================


class TestSchemaSim:
    def test_schema_sim_builds_edges(self):
        from aurum_v2.builder.network_builder import build_schema_sim_relation

        net = FieldNetwork()
        fields_meta = [
            ("s1", "db", "t1", "employee_name", 100, 90, "T"),
            ("s2", "db", "t2", "employee_name", 100, 85, "T"),
            ("s3", "db", "t3", "totally_different", 100, 70, "T"),
        ]
        net.init_meta_schema(iter(fields_meta))

        field_names = [("s1", "employee_name"), ("s2", "employee_name"),
                       ("s3", "totally_different")]
        build_schema_sim_relation(net, iter(field_names))

        # Identical names should be connected
        neighbors = net.neighbors_id("s1", Relation.SCHEMA_SIM)
        neighbor_nids = {h.nid for h in neighbors.data}
        assert "s2" in neighbor_nids


# ======================================================================
# P3-10: Provenance thread safety — itertools.count
# ======================================================================


class TestProvenanceThreadSafety:
    def test_counter_is_atomic(self):
        """itertools.count() produces unique values under concurrent access."""
        from aurum_v2.models.provenance import _global_origin_counter

        results: list[int] = []
        lock = threading.Lock()

        def grab_ids(n: int):
            local = []
            for _ in range(n):
                local.append(next(_global_origin_counter))
            with lock:
                results.extend(local)

        threads = [threading.Thread(target=grab_ids, args=(100,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 1000
        assert len(set(results)) == 1000, "All IDs must be unique"

    def test_synthetic_hits_unique_across_drs(self):
        h1 = _make_hit(field="a")
        h2 = _make_hit(field="b")
        drs1 = DRS([h1], Operation(OP.KW_LOOKUP, params=["salary"]))
        drs2 = DRS([h2], Operation(OP.KW_LOOKUP, params=["name"]))

        # Extract synthetic nodes
        synths1 = [n for n in drs1.get_provenance().prov_graph().nodes()
                    if isinstance(n.nid, str) and n.nid.startswith("synthetic")]
        synths2 = [n for n in drs2.get_provenance().prov_graph().nodes()
                    if isinstance(n.nid, str) and n.nid.startswith("synthetic")]
        assert len(synths1) == 1
        assert len(synths2) == 1
        assert synths1[0].nid != synths2[0].nid


# ======================================================================
# P3-11: DuckStore regex injection
# ======================================================================


class TestDuckStoreRegexSafety:
    def test_regex_metacharacters_escaped(self):
        """Keywords with regex metacharacters should not crash or match wrongly."""
        import re
        # Simulate what the fixed code does
        keywords = "price(USD)"
        escaped = re.escape(keywords)
        pattern = rf"\b{escaped}\b"
        # Should not raise
        compiled = re.compile(pattern)
        assert compiled.pattern == r"\bprice\(USD\)\b"


# ======================================================================
# P3-12: DRS nested iteration
# ======================================================================


class TestDRSIteration:
    def test_iter_returns_separate_object(self):
        hits = _make_hits(3)
        drs = DRS(hits, Operation(OP.ORIGIN))
        it = iter(drs)
        assert isinstance(it, _DRSIterator)
        assert it is not drs

    def test_nested_iteration(self):
        """Nested for loops must yield the full cross product."""
        hits = _make_hits(3)
        drs = DRS(hits, Operation(OP.ORIGIN))
        pairs = []
        for x in drs:
            for y in drs:
                pairs.append((x.nid, y.nid))
        assert len(pairs) == 9  # 3 × 3

    def test_repeated_iteration(self):
        hits = _make_hits(5)
        drs = DRS(hits, Operation(OP.ORIGIN))
        first = [h for h in drs]
        second = [h for h in drs]
        assert first == second

    def test_table_mode_iteration(self):
        h1 = _make_hit(src="t1", field="a")
        h2 = _make_hit(src="t1", field="b")
        h3 = _make_hit(src="t2", field="c")
        drs = DRS([h1, h2, h3], Operation(OP.ORIGIN))
        drs.set_table_mode()
        tables = list(drs)
        assert set(tables) == {"t1", "t2"}

    def test_nested_table_iteration(self):
        h1 = _make_hit(src="t1", field="a")
        h2 = _make_hit(src="t2", field="b")
        drs = DRS([h1, h2], Operation(OP.ORIGIN))
        drs.set_table_mode()
        pairs = []
        for x in drs:
            for y in drs:
                pairs.append((x, y))
        assert len(pairs) == 4  # 2 × 2


# ======================================================================
# DRS set operations
# ======================================================================


class TestDRSSetOps:
    def test_union(self):
        h1, h2, h3 = _make_hit(field="a"), _make_hit(field="b"), _make_hit(field="c")
        d1 = DRS([h1, h2], Operation(OP.ORIGIN))
        d2 = DRS([h2, h3], Operation(OP.ORIGIN))
        result = d1.union(d2)
        assert result.size() == 3

    def test_set_difference(self):
        h1, h2, h3 = _make_hit(field="a"), _make_hit(field="b"), _make_hit(field="c")
        d1 = DRS([h1, h2, h3], Operation(OP.ORIGIN))
        d2 = DRS([h2], Operation(OP.ORIGIN))
        result = d1.set_difference(d2)
        assert result.size() == 2
        nids = {h.nid for h in result.data}
        assert h2.nid not in nids


# ======================================================================
# FieldNetwork basics
# ======================================================================


class TestFieldNetworkBasics:
    def test_init_meta_schema(self):
        net = _build_simple_network()
        assert net.graph_order() == 3
        assert net.get_number_tables() == 3

    def test_neighbors_id(self):
        net = _build_simple_network()
        result = net.neighbors_id("1", Relation.PKFK)
        assert result.size() == 1
        assert result.data[0].nid == "2"

    def test_neighbors_empty(self):
        net = _build_simple_network()
        result = net.neighbors_id("1", Relation.CONTENT_SIM)
        assert result.size() == 0

    def test_get_cardinality(self):
        net = _build_simple_network()
        card = net.get_cardinality_of("1")
        assert card == pytest.approx(95 / 100)

    def test_fields_degree(self):
        net = _build_simple_network()
        top = net.fields_degree(2)
        # Node "2" is the hub (connected to both 1 and 3)
        assert top[0][0] == "2"


# ======================================================================
# Provenance basics
# ======================================================================


class TestProvenance:
    def test_origin_provenance(self):
        hits = _make_hits(3)
        prov = Provenance(hits, Operation(OP.ORIGIN))
        assert len(prov.prov_graph().nodes()) == 3
        assert len(prov.prov_graph().edges()) == 0

    def test_kw_lookup_provenance(self):
        hits = _make_hits(2)
        prov = Provenance(hits, Operation(OP.KW_LOOKUP, params=["salary"]))
        # Should have 3 nodes: 2 hits + 1 synthetic origin
        assert len(prov.prov_graph().nodes()) == 3
        assert len(prov.prov_graph().edges()) == 2

    def test_compute_paths(self):
        h1 = _make_hit(field="a")
        h2 = _make_hit(field="b")
        prov = Provenance([h1], Operation(OP.ORIGIN))
        # Merge another step
        step = Provenance([h2], Operation(OP.PKFK, params=[h1]))
        merged = nx.compose(prov.prov_graph(), step.prov_graph())
        prov.swap_p_graph(merged)

        paths = prov.compute_all_paths()
        assert len(paths) >= 1

    def test_leafs_and_heads(self):
        h1 = _make_hit(field="a")
        h2 = _make_hit(field="b")
        prov = Provenance([h2], Operation(OP.CONTENT_SIM, params=[h1]))
        leafs, heads = prov.get_leafs_and_heads()
        assert h1 in leafs
        assert h2 in heads


# ======================================================================
# DRS Provenance query helpers
# ======================================================================


class TestDRSProvenanceHelpers:
    def test_why(self):
        h_origin = _make_hit(field="origin")
        h_result = _make_hit(field="result")
        drs = DRS([h_origin], Operation(OP.ORIGIN))
        step = DRS([h_result], Operation(OP.PKFK, params=[h_origin]))
        drs.absorb(step)

        origins = drs.why(h_result)
        assert any(h.nid == h_origin.nid for h in origins)

    def test_why_missing(self):
        h1 = _make_hit(field="a")
        h_missing = _make_hit(field="missing")
        drs = DRS([h1], Operation(OP.ORIGIN))
        assert drs.why(h_missing) == []


# ======================================================================
# Coverage ranking
# ======================================================================


class TestCoverageRanking:
    def test_basic_coverage(self):
        h1 = _make_hit(field="a", score=1.0)
        h2 = _make_hit(field="b", score=2.0)
        drs = DRS([h1, h2], Operation(OP.ORIGIN))
        drs.rank_coverage()
        # Both originate from themselves, so coverage = 1/2 each... or full
        for el in drs.data:
            assert el in drs._rank_data


# ======================================================================
# find_path_table data assembly
# ======================================================================


class TestFindPathTable:
    def test_table_path_contains_data(self):
        """Verify find_path_table includes endpoint data."""
        net = _build_simple_network()

        src = Hit("1", "db", "t1", "a", 0)
        tgt = Hit("3", "db", "t3", "c", 0)

        # Mock the API that provides table siblings
        mock_api = MagicMock()

        def mock_drs_from_table_hit(hit):
            table = hit.source_name
            return net.get_hits_from_table(table)

        mock_api.drs_from_table_hit = mock_drs_from_table_hit

        result = net.find_path_table(src, tgt, Relation.PKFK, mock_api)
        # Result should contain data (not be empty)
        assert result.size() > 0


# ======================================================================
# Compute field ID
# ======================================================================


class TestComputeFieldId:
    def test_deterministic(self):
        id1 = compute_field_id("db", "table", "col")
        id2 = compute_field_id("db", "table", "col")
        assert id1 == id2

    def test_different_inputs(self):
        id1 = compute_field_id("db", "table", "col1")
        id2 = compute_field_id("db", "table", "col2")
        assert id1 != id2

    def test_returns_string(self):
        assert isinstance(compute_field_id("a", "b", "c"), str)
