# Aurum Legacy → New Code Audit & Architecture Summary

> Generated 2026-02-12. Covers the legacy codebase at `aurum_legacy/` and the
> Opus-generated rewrite at `aurum/`.

---

## 1  Feature-Reference Importance Verification

Every 🔴 IMPORTANT rating in `aurum_feature_reference.md` was checked against
the **actual** legacy source code.  
**Result:** all ratings confirmed TRUE.

| Area | Finding |
|------|---------|
| Algorithms & thresholds | Accurate throughout (MinHash 0.7/512, numeric overlap 0.85, inclusion 0.3, PKFK cardinality > 0.7, DBSCAN eps=0.1, join_overlap 0.4, max_hops 5) |
| Function signatures | Frequently inaccurate in the reference doc — param names simplified, `store` params added where they don't exist |
| `init_meta_schema` | Takes a **fields iterable**, not a store object |
| `find_path_table` | Requires an **api** param (not mentioned in reference) |
| `virtual_schema_iterative_search` | Doesn't have 5 cleanly named stages — only 2 are timed |
| `read_relation(path)` | Has **1** param, not 2 |
| `apply_filter` | First param is a **path string**, not a DataFrame |
| `materialize_join_graph` | Second param is **dod** instance, not filters |

---

## 2  New-Code Audit (aurum/ vs aurum_legacy/)

### 2.1  Verdict Table

| # | Feature | Verdict | Severity |
|---|---------|---------|----------|
| 1 | `compute_field_id` (CRC32) | 🟡 **DUAL ID CONFLICT** — `ColumnProfile` uses MD5, `Hit` uses CRC32 → IDs never match | 🔴 Critical |
| 2 | `Hit.__hash__` on `int(nid)` | 🟡 **CRASH** — `int()` on MD5 hex without `base=16` | 🔴 Critical |
| 3 | `Relation` enum values | ✅ Correct | — |
| 4 | DRS set ops + provenance | ✅ Correct | — |
| 5 | `init_from_profiles` (graph skeleton) | ✅ Correct | — |
| 6 | `add_relation` typed edges | ✅ Correct | — |
| 7 | `neighbors_id` return type | 🟡 Returns bare list instead of DRS (provenance at Algebra layer) | ⚠️ Minor |
| 8 | `find_path` provenance assembly | 🔴 **NO hop-by-hop provenance chain** — returns bare list | 🔴 Critical |
| 9 | `find_path_table` (table-level DFS) | 🔴 **MISSING** entirely | 🔴 Critical |
| 10 | Schema-sim (TF-IDF → LSH) | 🟡 O(n²) cosine instead of LSH; `schema_sim_threshold` commented out → crash | 🔴 Crash |
| 11 | MinHash LSH (0.7, 512) | ✅ Correct | — |
| 12 | Numeric overlap (0.85 / 0.3 / DBSCAN) | ✅ Correct | — |
| 13 | PKFK (cardinality > 0.7) | ✅ Correct | — |
| 14 | DoD `virtual_schema_iterative_search` | 🟡 Simplified greedy — no validation stage, no backup groups | ⚠️ Major |
| 15 | `joinable()` enumeration | 🟡 Simplified — no dedup, no unjoinable cache | ⚠️ Major |
| 16 | `is_join_graph_materializable` | 🔴 **MISSING** — no trial-join validation | 🔴 Critical |
| 17 | `join_ab_on_key_optimizer` timeout | 🟡 Polars chunks exist but 3-min timeout **never enforced**, no disk spill | ⚠️ Major |
| 18 | `materialize_join_graph` tree-fold | 🟡 Sequential left-to-right instead of tree-fold — breaks non-linear graphs | ⚠️ Major |
| 19 | Config values | ✅ Values correct but `schema_sim_threshold` commented out; `aurumConfig` missing | 🔴 Crash |

### 2.2  Critical Bugs

1. **Dual ID system** — `ColumnProfile.nid` is MD5 hex, `compute_field_id` is CRC32 → nodes indexed by one, looked up by the other → guaranteed mismatch.
2. **`Hit.__hash__`** calls `int(nid)` on MD5 hex → `ValueError` at runtime.
3. **`make_drs`** calls `compute_field_id` with 2 args, but it requires 3 → `TypeError`.
4. **`schema_sim_threshold`** is commented out → `AttributeError` when `build_schema_sim` runs.

---

## 3  Legacy Architecture Schema

### 3.1  Complete Data Flow

```
┌──────────────────────┐
│  Raw Data Sources     │  CSV files, databases, etc.
└──────────┬───────────┘
           │
    ┌──────▼──────────┐
    │  ddprofiler      │  Java — profiles every column
    │  → Elasticsearch │  index: 'profile'
    └──────┬──────────┘  per-column: nid, dbName, sourceName,
           │             columnName, dataType, minhash sigs,
           │             num sigs, totalValues, uniqueValues, path
    ┌──────▼──────────────────────────────────────────────┐
    │  networkbuildercoordinator.main(output_path)        │
    │  ────────────────────────────────────────────────── │
    │  1. store.get_all_fields()                          │
    │     → generator of (nid, db, src, field, total,     │
    │       unique, dataType)                             │
    │                                                      │
    │  2. network.init_meta_schema(fields)                │
    │     [fieldnetwork.py]                               │
    │     → graph nodes + id_names + source_ids           │
    │                                                      │
    │  3. build_schema_sim_relation(network, store)       │
    │     [dataanalysis.py]                               │
    │     → TF-IDF on column names → NearPy LSH           │
    │     → SCHEMA_SIM edges                               │
    │                                                      │
    │  4. build_content_sim_mh_text(network, mh_sigs)     │
    │     [dataanalysis.py]                               │
    │     → MinHashLSH(threshold=0.7, num_perm=512)       │
    │     → CONTENT_SIM edges (text columns)               │
    │                                                      │
    │  5. build_content_sim_num_overlap_distr(net, sigs)   │
    │     [dataanalysis.py]                               │
    │     → median±IQR overlap ≥ 0.85 → CONTENT_SIM       │
    │     → core overlap ≥ 0.3 → INCLUSION_DEPENDENCY      │
    │     → DBSCAN(eps=0.1) for single-point columns       │
    │                                                      │
    │  6. build_pkfk_relation(network, store)             │
    │     [dataanalysis.py]                               │
    │     → cardinality ratio > 0.7 → PKFK edges          │
    │                                                      │
    │  7. serialize_network(path)                         │
    │     → graph.pickle, id_info.pickle,                  │
    │       table_ids.pickle, lsh_indexes                  │
    └──────┬──────────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────────────────┐
    │  Serialized Model (pickle files on disk)            │
    └──────┬──────────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────────────────┐
    │  main.init_system(path)  [main.py]                  │
    │  → network = deserialize_network(path)              │
    │  → store   = StoreHandler(config)                   │
    │  → api     = API(network, store)   [ddapi.py]       │
    │    (API inherits Algebra)                            │
    └──────┬──────────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────────────────┐
    │  User Queries via Algebra  [algebra.py]             │
    │                                                      │
    │  api.search_content("salary")     → DRS             │
    │  api.content_similar_to(drs)      → DRS             │
    │  api.pkfk_of(drs)                → DRS             │
    │  api.paths(drs_a, drs_b, PKFK)   → DRS             │
    │  api.intersection(a, b)           → DRS             │
    │  drs.why(hit)  /  drs.how(hit)   → provenance      │
    │  drs.rank_certainty()  /  rank_coverage()           │
    └──────┬──────────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────────────────┐
    │  Data on Demand  [DoD/dod.py]                       │
    │                                                      │
    │  Input: list_attributes + list_values               │
    │                                                      │
    │  Stage 1 — Search filters                           │
    │    api.search_exact_attribute(attr)  → DRS          │
    │    api.search_content(value)         → DRS          │
    │    intersect where both specified                    │
    │                                                      │
    │  Stage 2 — Candidate group formation                │
    │    group tables by filter coverage                   │
    │    greedy enumeration w/ pivot exploration            │
    │                                                      │
    │  Stage 3 — Join graph discovery                     │
    │    api.paths(t1, t2, PKFK)  per pair                │
    │    itertools.product → covering join graphs          │
    │                                                      │
    │  Stage 4 — Materializability check                  │
    │    is_join_graph_materializable()                    │
    │    trial join per hop; reject if 0 rows              │
    │                                                      │
    │  Stage 5 — Materialization                          │
    │    [data_processing_utils.py]                        │
    │    materialize_join_graph (tree-fold)                │
    │    join_ab_on_key_optimizer (3-min timeout)          │
    │    project requested columns                         │
    │    yield (materialized_view, metadata)               │
    └─────────────────────────────────────────────────────┘
```

### 3.2  Key Data Structures

#### FieldNetwork  (`knowledgerepr/fieldnetwork.py`)

| Component | Type | Description |
|-----------|------|-------------|
| `graph` | `nx.MultiGraph` | Nodes = column nids (CRC32). Node attr `cardinality` = unique/total. Edges keyed by `Relation`, carry `score`. |
| `id_names` | `dict[str, tuple]` | nid → `(db_name, source_name, field_name, data_type)` |
| `source_ids` | `defaultdict(list)` | `source_name` → `[nid, ...]` |

#### Hit  (`api/apiutils.py`)

```python
Hit = namedtuple('Hit', 'nid, db_name, source_name, field_name, score')
# Identity: hash on int(nid), equality on nid
# nid = str(binascii.crc32(bytes(db + source + field, 'utf8')))
```

#### DRS — Domain Result Set  (`api/apiutils.py`)

| Property | Type | Description |
|----------|------|-------------|
| `data` | `set[Hit]` | The result set |
| `provenance` | `Provenance` | DAG tracking derivation |
| `operation` | `OP` | The op that created this DRS |
| `score` | `dict[Hit, float]` | Per-element ranking scores |

#### Provenance  (`api/apiutils.py`)

- Backed by `nx.MultiDiGraph`
- **Nodes** = `Hit` objects (including synthetic origin Hits for keyword searches)
- **Edges** = labeled with `OP` enum values
- **Leafs** = origin nodes (no predecessors)
- **Heads** = terminal nodes (no successors)

#### Relation Enum  (`api/apiutils.py`)

| Name | Value | Built By |
|------|-------|----------|
| `SCHEMA` | 0 | `init_meta_schema` (same-table) |
| `SCHEMA_SIM` | 1 | `build_schema_sim_relation` (TF-IDF + NearPy) |
| `CONTENT_SIM` | 2 | `build_content_sim_mh_text` / `build_content_sim_num_overlap_distr` |
| `ENTITY_SIM` | 3 | *(disabled)* |
| `PKFK` | 5 | `build_pkfk_relation` |
| `INCLUSION_DEPENDENCY` | 6 | `build_content_sim_num_overlap_distr` |

### 3.3  Entry Points

| Task | File | Command |
|------|------|---------|
| Build network | `networkbuildercoordinator.py` | `python networkbuildercoordinator.py --opath /output/` |
| Query interactively | `main.py` | `python main.py --path_to_model /model/` |
| Run DoD | `run_dod.py` | `python run_dod.py --model_path /model/ --list_attributes "A;B" --list_values "v1;v2"` |

### 3.4  External Dependencies

| Dependency | Role |
|------------|------|
| **Elasticsearch** | Persistent column-profile store + keyword search |
| **NetworkX** | MultiGraph (field network), MultiDiGraph (provenance) |
| **scikit-learn** | `TfidfVectorizer` for schema similarity |
| **NearPy** | LSH (RandomBinaryProjections) for schema/content sim |
| **datasketch** | MinHash + MinHashLSH for text content similarity |
| **pandas** | DataFrame operations for join materialization |
| **psutil** | Memory-limit estimation during joins |
| **NumPy / SciPy** | Numerical analysis, distribution comparison |
