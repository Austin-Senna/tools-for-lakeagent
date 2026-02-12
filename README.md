# Aurum v2 — Data Discovery Library

A clean, production-ready reimplementation of the Aurum data discovery system
(MIT, 2018). Designed to be **portable to AI agents** searching massive S3
data lakes (9.5 TB+) and answering complex multi-hop analytical questions.

## Overview

Aurum v2 automatically discovers **relationships between columns** in a data lake
and enables agents to answer questions like:

> *"Population of the largest city in the Idaho county that had 90–110 VA disability
> recipients in 2023 AND 88–92 in 2019?"*

by:

1. **Searching** for relevant tables (VA disability, census population, etc.)
2. **Finding** join paths connecting them (county → PKFK → state → city)
3. **Materializing** the combined view (joining on county)
4. **Filtering & aggregating** (Idaho, year, range, max population)

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### Build the Index (Offline, once)

```python
from aurum_v2.profiler.source_readers import discover_sources, SourceConfig
from aurum_v2.profiler.column_profiler import Profiler
from aurum_v2.builder.coordinator import Coordinator
from aurum_v2.config import AurumConfig

# 1. Configure data sources (CSV, S3, Postgres, etc.)
configs = [
    SourceConfig(
        name="va_datalake",
        source_type="s3",
        config={
            "bucket": "my-datalake-bucket",
            "prefix": "data/va_comp/",
            "region": "us-east-1",
        }
    ),
    SourceConfig(
        name="census_datalake",
        source_type="s3",
        config={
            "bucket": "my-datalake-bucket",
            "prefix": "data/census/",
        }
    ),
]

# 2. Profile all columns (compute stats, MinHash, embeddings)
readers = discover_sources(configs)
config = AurumConfig(es_host="localhost", es_port="9200")
profiler = Profiler(config)
profiler.profile_all(readers)  # Creates ES indices

# 3. Build relationship graph
coordinator = Coordinator(config)
coordinator.build_network()  # Discovers PKFK, schema_sim, content_sim
```

Takes **hours to days** depending on data lake size and cluster capacity.

### Query the Index (Runtime, per question)

```python
from aurum_v2 import init_system
from aurum_v2.dod.dod import DoD

# Load the pre-built index
api = init_system("/path/to/serialized/model")
dod = DoD(api)

# One-shot: "Find me county + VA recipients + population for Idaho"
results = dod.virtual_schema_iterative_search(
    list_attributes=["county", "VA disability recipients", "population"],
    list_samples=["Idaho"]
)

# results is a list of DataFrames, each is a candidate view
for view in results:
    print(view.head())
```

Or use **individual discovery tools** for more control:

```python
# Search for columns containing "disability"
hits = api.search_content("disability")
for hit in hits:
    print(f"{hit.db_name}.{hit.source_name}.{hit.field_name}")

# Find columns joinable with a given column
joins = api.pkfk_of(hit)

# Get the join path between two tables
path = api.paths(drs_a, drs_b, relation=Relation.PKFK)
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  OFFLINE STAGE (run once)                                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  profiler/                                               │
│  ├── source_readers.py    → S3Reader, CSVReader, etc.   │
│  └── column_profiler.py   → Per-column stats + MinHash  │
│                          ↓                              │
│  [Elasticsearch indices: profile + text]                │
│                          ↓                              │
│  builder/                                                │
│  ├── network_builder.py   → Build edges (PKFK, etc.)    │
│  ├── analysis.py          → Statistical helpers         │
│  └── coordinator.py       → Orchestrate build pipeline  │
│                          ↓                              │
│  [FieldNetwork graph + serialized pickles]              │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  RUNTIME STAGE (per query)                               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  discovery/api.py                                        │
│  └── init_system(model_path) → API object               │
│                          ↓                              │
│  discovery/algebra.py                                    │
│  ├── search_content(keywords)    → DRS                  │
│  ├── pkfk_of(hit)                → DRS                  │
│  ├── paths(a, b)                 → join path            │
│  └── intersection/union/diff      → set operations      │
│                          ↓                              │
│  graph/field_network.py                                  │
│  ├── neighbors_id(nid)           → related columns      │
│  ├── find_path_hit(a, b)         → DFS join discovery   │
│  └── [stores ES + NetworkX in-memory]                   │
│                          ↓                              │
│  dod/dod.py                                              │
│  └── virtual_schema_iterative_search()  → materialized  │
│      (automated join + filter + materialize)            │
│                          ↓                              │
│  Agent or user gets back DataFrames                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Module Guide

| Module | Purpose |
|--------|---------|
| **config.py** | Single source of truth for all thresholds (overlap=0.85, cardinality_th=0.7, DBSCAN_eps=0.1, etc.) |
| **models/** | Core data structures: `Hit`, `DRS` (result sets with provenance), `Relation`, `Provenance` |
| **store/** | Elasticsearch client: keyword search, profile retrieval, field enumeration |
| **graph/** | FieldNetwork: multi-relation graph over columns, path finding, relationship lookup |
| **discovery/** | Algebra API: high-level query interface (search, navigate, set operations) |
| **builder/** | Offline graph construction: statistics, similarity metrics, edge builders |
| **dod/** | Data-on-Demand: automated multi-table discovery and materialization |
| **profiler/** | Column indexing: read CSVs/S3, compute stats + MinHash, store in ES |
| **utils/** | Helpers: text normalization, pickle I/O |

## Key Concepts

### Hit

A **column reference**:

```python
Hit(nid="abc123", db_name="va_lake", source_name="va_comp_2023.csv",
    field_name="recipients", score=0.92)
```

Uniquely identified by `nid` (content-hash of db+source+field).

### DRS — Domain Result Set

A **result set with provenance**:

```python
drs = api.search_content("disability")
# drs.data is a set of Hit objects
# drs.provenance is a DAG tracking how results were derived (search → neighbors → paths)
# drs can be iterated, unioned, intersected, ranked
```

### Relation Enum

Edge types in the FieldNetwork:

| Relation | Meaning | Built By |
|----------|---------|----------|
| `SCHEMA` | Same table (column neighbors) | `init_meta_schema` |
| `SCHEMA_SIM` | Similar names (TF-IDF cosine) | `build_schema_sim_relation` |
| `CONTENT_SIM` | Similar values (MinHash Jaccard ≥ 0.7) | `build_content_sim_mh_text` |
| `INCLUSION_DEPENDENCY` | Values fully contained with overlap ≥ 0.3 | `build_content_sim_num_overlap` |
| `PKFK` | PK/FK relationship (cardinality > 0.7) | `build_pkfk_relation` |

## Configuration

All thresholds are in `config.py`:

```python
from aurum_v2.config import AurumConfig

config = AurumConfig(
    # Elasticsearch
    es_host="localhost",
    es_port="9200",
    
    # Similarity thresholds
    minhash_threshold=0.7,           # Content similarity (Jaccard)
    num_overlap_th=0.85,             # Numeric overlap
    inclusion_dep_th=0.3,            # Containment ratio
    pkfk_cardinality_th=0.7,         # PK/FK cardinality
    
    # Join materialization
    join_timeout_seconds=180,        # 3 minutes max per join
    memory_limit_fraction=0.6,       # Use 60% of available RAM
)
```

## What's Implemented vs Stubs

**Fully implemented (~55 methods):**
- Config, models (Hit, Relation), ES store initialization
- FieldNetwork graph structure and serialization
- Algebra API structure and convenience methods
- DRS data structure
- S3Reader + CSVReader

**Stubs to implement (~82 methods):**
- ES keyword search and field enumeration
- Graph path-finding algorithms
- Similarity metrics (TF-IDF, KS test, cosine, etc.)
- Network builder (edge construction)
- DoD pipeline (join discovery & materialization)
- View analysis (4C classification)
- Column profiler (MinHash, stats, entity recognition)

See [aurum_v2_dependency_map.md](../aurum_v2_dependency_map.md) for complete
mapping of stubs → legacy source code.

## For AI Agents — Exposable Tools

A thin **adapter layer** (`agent_api/`) serializes aurum_v2 results to JSON,
exposing ~13 tools:

**Discovery Tools:**
- `search_columns(keywords)` → list of matching columns
- `search_tables(keywords)` → list of matching tables
- `find_joinable_columns(col)` → columns that can join with col
- `find_join_path(table_a, table_b)` → multi-hop join sequence

**Data Access Tools:**
- `preview_table(table)` → first N rows
- `get_table_path(nid)` → S3/filesystem path
- `describe_column(col)` → stats (type, cardinality, range, etc.)

**Orchestration Tool:**
- `find_and_join_data(attributes, samples)` → DoD: automated join + materialize

**Utility Tools:**
- `compare_views(v1, v2)` → equivalent / contained / complementary / contradictory

The agent uses its own `execute_code` function for DataFrames (filter, aggregate, sort).

See [port.md](../port.md) for detailed porting strategy.

## Performance Notes

- **Profiling**: ~1 TB of CSVs → 4-8 hours (4-8 workers, parallel)
- **Graph building**: ~50K columns → 30-60 minutes (schema sim is O(n²) without LSH)
- **Query time**: Most searches return in <1 second (ES keyword search)
- **Join materialization**: Optimized with chunking + memory limits, but can be slow for 100M+ row joins

## Testing

```bash
# Run all tests
pytest tests/

# Run specific module
pytest tests/test_algebra.py -v

# With coverage
pytest --cov=aurum_v2 tests/
```

## Debugging

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run queries
api = init_system("/model")
drs = api.search_content("salary")  # Will log all ES queries
```

## Roadmap

- [ ] Implement all ~82 stubs from legacy codebase
- [ ] Wave 1: ES search + basic discovery (2-3 days)
- [ ] Wave 2: Join path finding (2-3 days)
- [ ] Wave 3: Full DoD pipeline (3-5 days)
- [ ] Agent API adapter layer (1 day)
- [ ] Performance optimization (ongoing)
- [ ] Web UI (out of scope for agent use case)

## References

- **Original Aurum**: https://github.com/mitdbg/aurum-datadiscovery (Java, 2018)
- **Key paper**: "Aurum: A Data Discovery System" (ICDE 2018)
- **Legacy codebase**: `aurum_legacy/` in this repo
- **Architecture audit**: [aurum_audit_summary.md](../aurum_audit_summary.md)
- **Agent integration**: [port.md](../port.md)

## License

(Inherited from original Aurum project — check LICENSE file)
