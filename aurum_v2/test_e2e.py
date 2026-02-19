#!/usr/bin/env python3
"""Quick E2E smoke test: profile → DuckDB → graph → algebra query."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurum_v2.config import AurumConfig
from aurum_v2.store.duck_store import DuckStore, KWType
from aurum_v2.profiler.column_profiler import profile_column
from aurum_v2.graph.field_network import FieldNetwork, serialize_network
from aurum_v2.builder import network_builder
from aurum_v2.discovery.api import API, init_system_duck

config = AurumConfig()
tmpdir = tempfile.mkdtemp()
db_path = os.path.join(tmpdir, "test.db")

# ── 1. Store profiles ────────────────────────────────────────────
duck = DuckStore(config, db_path)
duck.init_tables(recreate=True)

profiles = [
    profile_column("db1", "counties", "county_name", ["Ada", "Boise", "Canyon", "Gem", "Owyhee"], "T"),
    profile_column("db1", "census", "county", ["Ada", "Boise", "Canyon", "Gem", "Owyhee"], "T"),
    profile_column("db1", "census", "population", ["50000", "8000", "230000", "18000", "12000"], "N"),
    profile_column("db1", "va_comp", "county", ["Ada", "Boise", "Canyon", "Gem"], "T"),
    profile_column("db1", "va_comp", "recipients", ["100", "90", "110"], "N"),
]
duck.bulk_insert_profiles(profiles)
print(f"✓ Stored {len(profiles)} profiles")

# ── 2. Build graph ───────────────────────────────────────────────
net = FieldNetwork()
net.init_meta_schema(duck.get_all_fields())
network_builder.build_schema_sim_relation(net, duck.get_all_fields_name(), config)
network_builder.build_content_sim_mh_text(net, duck.get_all_mh_text_signatures(), config)
network_builder.build_content_sim_relation_num_overlap_distr(net, duck.get_all_fields_num_signatures(), config)
network_builder.build_pkfk_relation(net, config)

model_path = os.path.join(tmpdir, "model")
serialize_network(net, model_path)
G = net._get_underlying_repr_graph()
print(f"✓ Graph: {net.graph_order()} nodes, {net.get_number_tables()} tables, {G.number_of_edges()} edges")

# ── 3. Query via API ─────────────────────────────────────────────
api = init_system_duck(model_path, db_path, config)

drs = api.search_content("Ada")
print(f"\nsearch_content('Ada'): {drs.size()} hits")
for h in drs:
    print(f"  {h.source_name}.{h.field_name} (score={h.score:.2f})")

drs2 = api.search_attribute("county")
print(f"\nsearch_attribute('county'): {drs2.size()} hits")
for h in drs2:
    print(f"  {h.source_name}.{h.field_name}")

# ── 4. Neighbor search ───────────────────────────────────────────
if drs2.size() > 0:
    first = list(drs2)[0]
    nbrs = api.pkfk_of(first)
    print(f"\npkfk_of({first.source_name}.{first.field_name}): {nbrs.size()} hits")
    for h in nbrs:
        print(f"  {h.source_name}.{h.field_name}")

    content_sim = api.content_similar_to(first)
    print(f"\ncontent_similar_to({first.source_name}.{first.field_name}): {content_sim.size()} hits")
    for h in content_sim:
        print(f"  {h.source_name}.{h.field_name}")

# ── 5. Traverse ──────────────────────────────────────────────────
if drs2.size() > 0:
    from aurum_v2.models.relation import Relation
    first_hit = list(drs2)[0]
    traversed = api.traverse(first_hit, Relation.CONTENT_SIM, max_hops=2)
    print(f"\ntraverse(CONTENT_SIM, 2 hops): {traversed.size()} hits")

print("\n✅ E2E test PASSED")
