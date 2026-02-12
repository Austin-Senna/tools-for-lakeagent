# Aurum Legacy → New Code Audit & Architecture Summary

> Generated 2026-02-12. Updated 2026-02-13.  
> Covers the legacy codebase at `aurum_legacy/`, the Opus-generated rewrite at
> `aurum/`, and the clean skeleton at `aurum_v2/`.

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

---

## 4  ddprofiler Analysis (Java → Python Replacement Needed)

### 4.1  What It Is

The `ddprofiler` is a **Java 8 application** located at `aurum_legacy/ddprofiler/`.
It is the **first stage of the entire Aurum pipeline** — without it, Elasticsearch
has no data, and none of the Python code can function.

### 4.2  Architecture

```
┌────────────────────────────────────────────────────────┐
│  Main.java  →  startProfiler(ProfilerConfig)           │
│  ────────────────────────────────────────────────────  │
│  1. Creates Store (NativeElasticStore)                 │
│  2. Creates Conductor (thread pool + N Workers)        │
│  3. Parses YAML config → Source objects                │
│     (CSVSource, PostgresSource, HiveSource, etc.)      │
│  4. Submits each Source to Conductor queue              │
│  5. Waits for completion → teardown                    │
└────────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│  Worker.java  (per-table/file processing)              │
│  ────────────────────────────────────────────────────  │
│  For each column:                                      │
│   1. PreAnalyzer.readRows(chunk) → detect type         │
│      (FLOAT, INT, STRING)                              │
│   2. AnalyzerFactory → TextualAnalysis or              │
│      NumericalAnalysis                                 │
│   3. Feed data chunks iteratively to analyzers         │
│   4. FilterAndBatchDataIndexer → index raw text to ES  │
│      (ES index: "text", type: "column")                │
│   5. Wrap results → WorkerTaskResult                   │
│   6. store.storeDocument(wtr) → profile to ES          │
│      (ES index: "profile", type: "column")             │
└────────────────────────────────────────────────────────┘
```

### 4.3  What Gets Stored in Elasticsearch

#### ES Index: `profile` (one document per column)

| Field | Type | Source | Used By |
|-------|------|--------|---------|
| `id` | long | CRC32 of db+source+column | Node ID throughout system |
| `dbName` | keyword | Source config | Hit.db_name |
| `path` | keyword | Source config | File path |
| `sourceName` | text (aurum_analyzer) | Table/file name | Hit.source_name, `search_keywords` |
| `sourceNameNA` | keyword | Table/file name (not analyzed) | Exact match lookups |
| `columnName` | text (aurum_analyzer) | Column name | `search_keywords` |
| `columnNameNA` | keyword | Column name (not analyzed) | Exact match lookups |
| `dataType` | keyword | "T" (text) or "N" (numeric) | Type filtering |
| `totalValues` | long | Row count | Cardinality ratio |
| `uniqueValues` | long | HyperLogLog estimate | Cardinality ratio, PKFK |
| `entities` | keyword | OpenNLP NER labels | Entity similarity |
| `minhash` | long[] | KMinHash (K=512, Mersenne prime) | `build_content_sim_mh_text` |
| `minValue` | double | Column min | Numeric overlap |
| `maxValue` | double | Column max | Numeric overlap |
| `avgValue` | double | Column average | Numeric overlap |
| `median` | long | Column median | `build_content_sim_num_overlap_distr` |
| `iqr` | long | Interquartile range | `build_content_sim_num_overlap_distr` |

#### ES Index: `text` (one document per column, raw values)

| Field | Type | Source | Used By |
|-------|------|--------|---------|
| `id` | long | Column ID | Join back to profile |
| `dbName` | keyword | Source config | — |
| `path` | keyword | Source config | — |
| `sourceName` | keyword | Table/file name | — |
| `columnName` | keyword | Column name | — |
| `columnNameSuggest` | completion | Column name | Auto-suggest |
| `text` | text (english analyzer) | Raw column values | `search_content`, keyword search |

### 4.4  Key Profiling Algorithms

| Algorithm | Java Class | Output | Python Equivalent |
|-----------|-----------|--------|-------------------|
| **KMinHash** | `analysis.modules.KMinHash` | `long[512]` — 512 min-hash signatures using Mersenne prime `(2^61-1)` | `datasketch.MinHash` |
| **Cardinality** | `analysis.modules.CardinalityAnalyzer` | `uniqueValues` via HyperLogLog (stream lib) | `hyperloglog` or `datasketch.HyperLogLog` |
| **Range** | `analysis.modules.RangeAnalyzer` | min, max, avg, median, IQR | `numpy`/`pandas` describe |
| **NER Entities** | `analysis.modules.EntityAnalyzer` | Entity type labels via OpenNLP | `spacy` or `transformers` NER |
| **Type Detection** | `preanalysis.PreAnalyzer` | FLOAT / INT / STRING per column | `pandas.api.types` |

### 4.5  Recommendation

**YES, aurum_v2 needs a Python-based profiler module.** The ddprofiler is the
critical first stage — it creates the ES indices that everything else depends on.
A Python replacement using `pandas` + `datasketch` + `spacy` (optional NER) can
replicate all essential functionality. Proposed module: `aurum_v2/profiler/`.

---

## 5  EKG / Neo4j Analysis

### 5.1  What It Is

The EKG (Enterprise Knowledge Graph) subsystem consists of:
- `EKGapi.py` — Abstract graph backend with pluggable backends (IN_MEMORY, POSTGRES, JANUS, NEO, VIRTUOSO, G_INDEX)
- `inmemoryekg.py` — `InMemoryEKG(EKGapi)` — stub, no methods implemented
- `gindexekg.py` — `GIndexEKG(EKGapi)` — uses C shared library (`graph_index.so`) + PostgreSQL store
- `export_network_2_neo4j.py` — One-way export script: serialized FieldNetwork → Neo4j
- `ekgstore/neo4j_store.py` — Neo4j driver for the export

### 5.2  Is It Used by the Core Pipeline?

**NO.** Confirmed by grep:
- `algebra.py` — zero references to EKG/Neo4j
- `ddapi.py` — zero references to EKG/Neo4j
- `main.py` — zero references to EKG/Neo4j
- `networkbuildercoordinator.py` — zero references to EKG/Neo4j
- `DoD/dod.py` — zero references to EKG/Neo4j

The EKG is a **separate experimental subsystem** — an alternative graph backend
that was never integrated into the main query pipeline. The core pipeline uses
`FieldNetwork` (NetworkX in-memory graph) exclusively.

The Neo4j export is a **one-way visualization tool** — it exports the pickle-based
network to Neo4j for browsing, but the query engine never reads from Neo4j.

### 5.3  Recommendation

**Not needed in aurum_v2.** The EKG is an experimental add-on with incomplete
implementations (InMemoryEKG is empty, GIndexEKG.neighbors_id returns None).
If Neo4j visualization is desired later, it can be added as an optional export
script outside the core library.

---

## 6  aurum_v2 Completeness Audit

### 6.1  Coverage Summary

| Category | Legacy Modules | In aurum_v2? |
|----------|---------------|--------------|
| **Config** | `config.py` | ✅ `config.py` (AurumConfig dataclass) |
| **Data Models** | `apiutils.py` (Hit, DRS, Relation, OP, Provenance) | ✅ `models/` (4 files) |
| **ES Store** | `inputoutput.py` (StoreHandler) | ✅ `store/elastic_store.py` |
| **Field Network** | `fieldnetwork.py` (FieldNetwork) | ✅ `graph/field_network.py` |
| **Network Builder** | `networkbuildercoordinator.py` + `dataanalysis.py` | ⚠️ `builder/` (2 files) — missing `dataanalysis.py` analysis functions |
| **Algebra** | `algebra.py` | ✅ `discovery/algebra.py` |
| **API** | `ddapi.py` (API + Helper) | ⚠️ `discovery/api.py` — missing ~20 convenience methods |
| **DoD** | `DoD/dod.py` | ✅ `dod/dod.py` |
| **Join Materialization** | `DoD/data_processing_utils.py` | ⚠️ `dod/join_utils.py` — missing 7 functions |
| **View Analysis** | `DoD/material_view_analysis.py` | ❌ **MISSING** — ViewClass enum + 8 functions |
| **Text Utils** | `dataanalysis/nlp_utils.py` | ⚠️ `utils/text_utils.py` — missing 4 NLP functions |
| **IO Utils** | `inputoutput/inputoutput.py` (pickle) | ✅ `utils/io_utils.py` |
| **Profiler** | `ddprofiler/` (Java) | ❌ **MISSING** — no Python profiler |
| **Annotation/Metadata** | `api/annotation.py` (MRS, MDHit, MDComment) | ❌ **MISSING** — entire metadata type system |
| **Reporting** | `api/reporting.py` (Report class) | ❌ **MISSING** — graph statistics |
| **Sugar** | `sugar.py` (interactive shortcuts) | ❌ Not needed — UX layer |
| **Data Analysis** | `dataanalysis/dataanalysis.py` (25+ functions) | ❌ **MISSING** — TF-IDF, KS test, cosine similarity, etc. |
| **EKG/Neo4j** | `knowledgerepr/EKGapi.py` + neo4j export | ❌ Not needed — experimental, unused |

### 6.2  Critical Missing Modules (blocks functionality)

| # | Module | What It Does | Priority |
|---|--------|-------------|----------|
| 1 | **Profiler** (`profiler/`) | Reads CSV/DB → computes per-column stats → populates ES | 🔴 CRITICAL — without this, no data exists |
| 2 | **Data Analysis** (`builder/analysis.py`) | TF-IDF vectorization, cosine similarity, KS test, distribution overlap — called by network builder | 🔴 CRITICAL — builder stubs call these |
| 3 | **Annotation** (`models/annotation.py`) | MDClass, MDRelation, MDHit, MDComment, MRS — metadata type system | 🟡 IMPORTANT — needed for metadata features |
| 4 | **View Analysis** (`dod/view_analysis.py`) | ViewClass enum, 4C classification (equivalent/contained/complementary/contradictory) | 🟡 IMPORTANT — DoD output classification |

### 6.3  Missing Functions in Existing Modules

#### `discovery/api.py` — Missing ~20 Convenience Methods

```
make_drs(db, source, field)     drs_from_hit(hit)
drs_from_hits(hits)             drs_from_table(source)
drs_expand_to_table(drs)        search_content(kw)
search_attribute(kw)            search_exact_attribute(kw)
search_exact_source(kw)         search_entity(entity)
similar_content_to(drs)         similar_schema_to(drs)
pkfk_of(drs)                    inclusion_dependency_of(drs)
neighbor_of(drs, rel)           paths_between(a, b, rel)
traverse(a, b, rel)             display_drs(drs)
print_drs(drs)                  Helper class (web formatting)
```

#### `store/elastic_store.py` — Missing Methods

```
get_all_fields_of_source(source)    search_fuzzy(kw, type)
get_column_entities(nid)            get_text_signatures(nid)
write_annotation(...)               write_comment(...)
search_annotations(...)             read_annotations(nid)
read_comments(nid)                  sample_col_values(nid)
```

#### `dod/join_utils.py` — Missing Functions

```
join_dfs_on_key(df_a, df_b, key)     join_ab_on_key_disk(...)
join_ab_on_key_nan_safe(...)         compute_join_selectivity(...)
filter_by_values(df, filter)         project_columns_alt(...)
estimate_cartesian_memory(...)       profile_column_quality(...)
```

#### `builder/network_builder.py` — Missing Analysis Functions

These are in `dataanalysis/dataanalysis.py` in legacy and need to be either
inlined or extracted into a separate analysis module:

```
build_schema_sim_relation()  — calls: tf_idf_vectorize(), cosine_sim()
build_content_sim_mh_text()  — calls: compute_minhash()
build_content_sim_num()      — calls: compute_overlap(), ks_test()
build_entity_sim()           — calls: entity_overlap()
build_content_sim_lsa()      — alternative: SVD-based
build_schema_sim_lsa()       — alternative: LSA schema matching
```

### 6.4  Complete Pipeline Data Flow (Updated)

```
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 0: DATA INGESTION  (❌ MISSING from aurum_v2)         │
 │─────────────────────────────────────────────────────────────│
 │                                                             │
 │  ┌─────────────────┐                                        │
 │  │  Data Sources    │  CSV files, PostgreSQL, Hive, etc.    │
 │  └────────┬────────┘                                        │
 │           │                                                  │
 │  ┌────────▼────────────────────────────────────────────┐    │
 │  │  Profiler  (legacy: ddprofiler Java)                │    │
 │  │  For each source → for each column:                 │    │
 │  │   • Detect type (FLOAT/INT/STRING)                  │    │
 │  │   • Compute KMinHash[512] (text columns)            │    │
 │  │   • Compute HyperLogLog cardinality                 │    │
 │  │   • Compute Range stats (min/max/avg/median/IQR)    │    │
 │  │   • Run NER (date/location/money/org/person/time)   │    │
 │  │   • Index raw text values for keyword search         │    │
 │  │  Store to ES: "profile" index + "text" index        │    │
 │  └────────┬────────────────────────────────────────────┘    │
 │           │                                                  │
 │  ┌────────▼────────────────────────────────────────────┐    │
 │  │  Elasticsearch                                      │    │
 │  │  ├── "profile" index: 1 doc/column (stats + sigs)   │    │
 │  │  └── "text" index: 1 doc/column (raw values)        │    │
 │  └────────┬────────────────────────────────────────────┘    │
 └───────────┼─────────────────────────────────────────────────┘
             │
 ┌───────────▼─────────────────────────────────────────────────┐
 │ STAGE 1: NETWORK BUILDING  (✅ aurum_v2/builder/)           │
 │─────────────────────────────────────────────────────────────│
 │                                                             │
 │  coordinator.build_network(config):                         │
 │   1. store.get_all_fields()                                 │
 │      → generator: (nid, db, src, field, total, unique, dt)  │
 │   2. network.init_meta_schema(fields)                       │
 │      → graph nodes + id_names + source_ids + SCHEMA edges   │
 │   3. build_schema_sim(network, store)                       │
 │      → TF-IDF on column names → NearPy LSH → SCHEMA_SIM    │
 │      ⚠️ NEEDS: dataanalysis.tf_idf_vectorize, cosine_sim   │
 │   4. build_content_sim_mh_text(network, mh_sigs)            │
 │      → datasketch MinHashLSH(0.7, 512) → CONTENT_SIM       │
 │   5. build_content_sim_num_overlap(network, num_sigs)       │
 │      → median±IQR overlap ≥ 0.85 → CONTENT_SIM             │
 │      → core overlap ≥ 0.3 → INCLUSION_DEPENDENCY            │
 │      → DBSCAN(eps=0.1) for single-point columns             │
 │   6. build_pkfk(network, store)                             │
 │      → cardinality ratio > 0.7 → PKFK edges                │
 │   7. serialize_network(path) → pickle files                  │
 │                                                             │
 └───────────┬─────────────────────────────────────────────────┘
             │
 ┌───────────▼─────────────────────────────────────────────────┐
 │ STAGE 2: SYSTEM INIT  (✅ aurum_v2/discovery/api.py)        │
 │─────────────────────────────────────────────────────────────│
 │                                                             │
 │  init_system(config):                                       │
 │   → network = deserialize_network(path)                     │
 │   → store   = StoreHandler(config)                          │
 │   → api     = API(network, store)  [API extends Algebra]    │
 │                                                             │
 └───────────┬─────────────────────────────────────────────────┘
             │
 ┌───────────▼─────────────────────────────────────────────────┐
 │ STAGE 3: QUERY  (✅ aurum_v2/discovery/algebra.py)          │
 │─────────────────────────────────────────────────────────────│
 │                                                             │
 │  Algebra operations:                                        │
 │   search:   keyword_search(kw, type) → DRS                 │
 │   navigate: neighbor_search(drs, rel) → DRS                │
 │   paths:    find_path(a, b, rel) → DRS (w/ provenance)     │
 │   set ops:  intersection, union, difference                 │
 │   ranking:  rank_certainty, rank_coverage                   │
 │   prov:     drs.why(hit), drs.how(hit)                     │
 │                                                             │
 │  ⚠️ MISSING: ~20 API convenience wrappers (search_content, │
 │              similar_schema_to, pkfk_of, etc.)              │
 │                                                             │
 └───────────┬─────────────────────────────────────────────────┘
             │
 ┌───────────▼─────────────────────────────────────────────────┐
 │ STAGE 4: DATA ON DEMAND  (✅ aurum_v2/dod/)                 │
 │─────────────────────────────────────────────────────────────│
 │                                                             │
 │  dod.virtual_schema_iterative_search(attrs, values):        │
 │   1. Search filters → DRS per attribute/value               │
 │   2. Group tables by filter coverage → candidate groups      │
 │   3. Join graph discovery → api.find_path(t1,t2,PKFK)      │
 │   4. is_join_graph_materializable() → trial joins           │
 │   5. materialize_join_graph() → tree-fold join pipeline     │
 │      → join_ab_on_key_optimizer (3-min timeout, chunked)    │
 │      → project requested columns                            │
 │      → yield (materialized_view, metadata)                  │
 │                                                             │
 │  ⚠️ MISSING: view_analysis.py — ViewClass 4C classification │
 │  ⚠️ MISSING: 7 join utility functions                       │
 │                                                             │
 └─────────────────────────────────────────────────────────────┘
```

---

## 7  Action Items for aurum_v2 Completion

### Priority 1 — Required for Functional System

| # | Action | Files to Create/Modify |
|---|--------|----------------------|
| 1 | **Add Python profiler module** | `aurum_v2/profiler/__init__.py`, `profiler/column_profiler.py`, `profiler/source_readers.py` |
| 2 | **Add data analysis module** | `aurum_v2/builder/analysis.py` — TF-IDF vectorize, cosine sim, overlap functions |
| 3 | **Add annotation/metadata types** | `aurum_v2/models/annotation.py` — MDClass, MDRelation, MDHit, MDComment, MRS |
| 4 | **Add view analysis module** | `aurum_v2/dod/view_analysis.py` — ViewClass enum + 4C classification functions |

### Priority 2 — Required for Feature Parity

| # | Action | Files to Modify |
|---|--------|----------------|
| 5 | **Add API convenience methods** | `aurum_v2/discovery/api.py` — ~20 user-facing wrappers |
| 6 | **Add missing store methods** | `aurum_v2/store/elastic_store.py` — annotation CRUD, fuzzy search, etc. |
| 7 | **Add missing join functions** | `aurum_v2/dod/join_utils.py` — disk join, NaN-safe join, selectivity |
| 8 | **Add missing text utils** | `aurum_v2/utils/text_utils.py` — POS tagging, lemmatization |
| 9 | **Add reporting module** | `aurum_v2/graph/reporting.py` — graph statistics |

### Priority 3 — Optional Enhancements

| # | Action | Notes |
|---|--------|-------|
| 10 | Alt. builder algorithms (LSA, SVD) | Legacy had experimental variants; not needed initially |
| 11 | Interactive sugar module | Jupyter convenience; can be added when needed |
| 12 | Neo4j export script | One-way visualization tool; optional |
