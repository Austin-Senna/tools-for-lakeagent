import logging
from aurum_v2.graph.field_network import FieldNetwork
from aurum_v2.builder import network_builder
from aurum_v2.discovery.algebra import Algebra
from aurum_v2.models.relation import Relation

# 1. Setup Logging to see what's happening
logging.basicConfig(level=logging.INFO)
print(">>> INITIALIZING LAKEAGENT INTEGRATION TEST...\n")

# 2. Initialize an Empty Network
network = FieldNetwork()

# 3. Create Dummy Data (The "Mock" Data Lake)
# We manually add columns as if the Profiler just read them from DuckDB.

# --- TABLE: USERS ---
# nid format: "db_name.source_name.field_name" (simplified for test)
u_id = network.add_field("lake.users.id", cardinality=1.0)        # PK
u_email = network.add_field("lake.users.email", cardinality=1.0)  # Text
network._id_names["lake.users.id"] = ("lake", "users", "id", "N")
network._id_names["lake.users.email"] = ("lake", "users", "email", "T")
network._source_ids["users"].extend([u_id, u_email])

# --- TABLE: ORDERS ---
o_id = network.add_field("lake.orders.id", cardinality=1.0)       # PK
o_uid = network.add_field("lake.orders.user_id", cardinality=0.1) # FK to users
o_pid = network.add_field("lake.orders.prod_id", cardinality=0.1) # FK to products
network._id_names["lake.orders.id"] = ("lake", "orders", "id", "N")
network._id_names["lake.orders.user_id"] = ("lake", "orders", "user_id", "N")
network._id_names["lake.orders.prod_id"] = ("lake", "orders", "prod_id", "N")
network._source_ids["orders"].extend([o_id, o_uid, o_pid])

# --- TABLE: PRODUCTS ---
p_id = network.add_field("lake.products.id", cardinality=1.0)     # PK
p_name = network.add_field("lake.products.name", cardinality=0.9) # Text
network._id_names["lake.products.id"] = ("lake", "products", "id", "N")
network._id_names["lake.products.name"] = ("lake", "products", "name", "T")
network._source_ids["products"].extend([p_id, p_name])

print(f"Graph Nodes Created: {network.graph_order()}")

# 4. Manually Add 'Simulated' Edges 
# (In real life, MinHash/LSH would build these. We mock them to test the logic.)

# SCHEMA SIM: users.id <-> orders.user_id (Names look similar)
network.add_relation(u_id, o_uid, Relation.SCHEMA_SIM, score=0.9)
network.add_relation(o_uid, u_id, Relation.SCHEMA_SIM, score=0.9)

# CONTENT SIM: users.id <-> orders.user_id (Data overlaps)
# (This simulates what the MinHash LSH would find)
network.add_relation(u_id, o_uid, Relation.INCLUSION_DEPENDENCY, score=0.8) 
network.add_relation(o_uid, u_id, Relation.INCLUSION_DEPENDENCY, score=0.8)

print(">>> BUILDING PKFK RELATIONSHIPS...")
# Run the actual builder logic you wrote!
network_builder.build_pkfk_relation(network)

# Check if it worked
pkfk_edges = list(network.enumerate_relation(Relation.PKFK))
print(f"PKFK Edges Found: {len(pkfk_edges)}")
for edge in pkfk_edges:
    print(f"  - {edge}")

# 5. Initialize API (The Bridge)
# We mock the 'ElasticStore' keyword search because we don't have ES running.
class MockStore:
    def search_keywords(self, keywords, **kwargs):
        # If user searches 'email', return the users.email column
        if "email" in keywords:
            return [network.get_hits_from_table("users")[1]] # return email hit
        return []

api = Algebra(network, es=MockStore())

# 6. Run a Semantic Query
print("\n>>> RUNNING USER QUERY: 'Find tables related to email'...")

# A. Search for "email"
drs_email = api.search("email", kw_type=None)
print(f"1. Search found: {drs_email.data[0].field_name}")

# B. Traverse to neighbors (Find tables joined to the email table)
# We look for PKFK neighbors of the 'users' table
print("2. Traversing PKFK links...")
drs_neighbors = api.traverse(drs_email, Relation.PKFK, max_hops=2)

print("\n>>> FINAL RESULTS (The AI's Answer):")
drs_neighbors.pretty_print_columns()

# 7. Verify Provenance (Did it track the path?)
print("\n>>> EXPLAINING THE PATH:")
target_hit = drs_neighbors.data[0] # Just pick one
paths = drs_neighbors.how(target_hit)
for p in paths:
    print(p)

print("\n>>> TEST COMPLETE. SUCCESS.")