# Porting Aurum v2 to an Agent-Exposable API

> How does `aurum_v2/` become a set of callable tools for an AI agent
> searching a 9.5 TB S3 data lake?

---

## 1  Current Architecture vs What the Agent Needs

### What exists today

```≠
aurum_v2/
├── discovery/
│   ├── api.py          ← init_system() → API object (entry point)
│   └── algebra.py      ← Algebra class: search, paths, set ops
├── store/
│   └── elastic_store.py ← StoreHandler: ES keyword search
├── graph/
│   └── field_network.py ← FieldNetwork: relationship graph
├── builder/
│   ├── coordinator.py   ← Build pipeline orchestration
│   ├── network_builder.py ← Edge builders (schema_sim, content_sim, pkfk)
│   └── analysis.py      ← Statistical analysis (KS test, TF-IDF, overlap)
├── dod/
│   ├── dod.py           ← DoD: virtual schema search + join orchestration
│   ├── join_utils.py    ← DataFrame join/materialize utilities
│   └── view_analysis.py ← 4C view classification
├── profiler/
│   ├── column_profiler.py ← Column stats + MinHash + ES indexing
│   └── source_readers.py  ← CSVReader, S3Reader, DatabaseReader
├── models/
│   ├── hit.py           ← Hit (column reference)
│   ├── drs.py           ← DRS (result set with provenance)
│   ├── relation.py      ← Relation/OP enums
│   ├── provenance.py    ← Provenance graph (why/how)
│   └── annotation.py    ← MDHit/MDComment (not needed)
├── utils/
│   ├── text_utils.py    ← CamelCase normalization
│   └── io_utils.py      ← Pickle helpers
└── config.py            ← AurumConfig dataclass
```

### What the agent needs

The agent is an LLM with `execute_code` capability. It doesn't need the raw
Python objects — it needs **functions it can call by name** with **JSON-serializable
inputs and outputs**. The agent's framework likely provides tools as:

```python
@tool
def search_columns(keywords: str) -> list[dict]:
    """Find columns matching keywords across the data lake."""
    ...
```

The gap: **aurum_v2 returns `DRS` and `Hit` objects**, which are Python classes
with NetworkX provenance graphs, not JSON. We need a **thin adapter layer**.

---

## 2  The Adapter Layer — `aurum_v2/agent_api/`

The port strategy is a **thin wrapper module** that:

1. Imports `aurum_v2` internals
2. Calls the real Algebra/DoD methods
3. Serializes results to dicts/lists the agent can consume
4. Handles errors gracefully (returns error strings, never crashes)

### Proposed structure

```
aurum_v2/agent_api/
├── __init__.py          ← register_tools(api, dod) → tool dict
├── discovery_tools.py   ← search_columns, search_tables, find_joinable, find_path
├── data_tools.py        ← preview_table, read_table (from S3/filesystem)
├── orchestration_tools.py ← find_and_join_data (DoD wrapper)
└── util_tools.py        ← describe_column, get_table_path, compare_views
```

### Why a separate module?

- **aurum_v2 core stays clean** — pure data discovery library, testable without agent dependencies
- **Agent framework agnostic** — the adapter returns plain dicts; whichever agent framework is used (LangChain, custom, etc.) just wraps them as tools
- **Serialization boundary** — all Hit/DRS/DataFrame → dict/list conversion happens here
- **Error boundary** — all try/except handling for the agent happens here

---

## 3  Tool Definitions — Exact Signatures

### 3.1  Discovery Tools (`discovery_tools.py`)

These wrap `Algebra` methods. They are the most-used tools — the agent calls
them to find relevant data before doing anything else.

```python
def search_columns(keywords: str, max_results: int = 15) -> list[dict]:
    """
    Search for columns whose VALUES match the keywords.
    
    Wraps: api.search_content(keywords)
    
    Input:  keywords = "VA disability compensation"
    Output: [
        {"nid": "a1b2c3", "db": "va_lake", "table": "va_comp_2023.csv",
         "column": "recipients", "score": 0.85},
        ...
    ]
    """
```

```python
def search_columns_fuzzy(keywords: str, max_results: int = 15) -> list[dict]:
    """
    Fuzzy search tolerating typos (edit-distance matching).
    
    Wraps: store.fuzzy_keyword_match(keywords)
    
    Input:  keywords = "populaton" (typo)
    Output: same format as search_columns
    """
```

```python
def search_tables(keywords: str, max_results: int = 15) -> list[dict]:
    """
    Search for TABLES whose names match the keywords.
    
    Wraps: api.search_table(keywords)
    
    Input:  keywords = "census 2020"
    Output: [
        {"nid": "d4e5f6", "db": "census_lake", "table": "census_2020.csv",
         "column": "__table__", "score": 0.9},
        ...
    ]
    """
```

```python
def search_attribute(column_name: str) -> list[dict]:
    """
    Find columns with this exact name across all tables.
    
    Wraps: api.search_exact_attribute(column_name)
    
    Input:  column_name = "county"
    Output: [
        {"nid": "...", "db": "...", "table": "va_comp_2023.csv", "column": "county"},
        {"nid": "...", "db": "...", "table": "census_2020.csv", "column": "county"},
        ...
    ]
    """
```

```python
def find_joinable_columns(table: str, column: str) -> list[dict]:
    """
    Find columns that can be joined with the given column (PK/FK relationships).
    
    Wraps: api.pkfk_of(hit)
    
    Input:  table = "va_comp_2023.csv", column = "county"
    Output: [
        {"nid": "...", "table": "census_2020.csv", "column": "county_name",
         "relation": "PKFK", "score": 0.92},
        ...
    ]
    """
```

```python
def find_similar_columns(table: str, column: str) -> list[dict]:
    """
    Find columns with similar content (MinHash / content similarity).
    
    Wraps: api.content_similar_to(hit)
    """
```

```python
def find_join_path(
    table_a: str, column_a: str,
    table_b: str, column_b: str,
    max_hops: int = 3
) -> list[dict]:
    """
    Find a multi-hop join path connecting two columns.
    
    Wraps: api.paths(drs_a, drs_b, Relation.PKFK)
    
    Input:  table_a="va_comp_2023.csv", col_a="county",
            table_b="census_2020.csv", col_b="county_name"
    Output: [
        {"hop": 1, "from_table": "va_comp_2023.csv", "from_col": "county",
         "to_table": "bridge_table.csv", "to_col": "county_id"},
        {"hop": 2, "from_table": "bridge_table.csv", "from_col": "county_id",
         "to_table": "census_2020.csv", "to_col": "county_name"},
    ]
    """
```

### 3.2  Data Access Tools (`data_tools.py`)

These interact with the raw CSV files on S3/disk. Since the agent already has
`execute_code`, these are **convenience wrappers** that know how to resolve
Aurum nids to actual file paths and read them.

```python
def preview_table(table: str, n_rows: int = 5) -> dict:
    """
    Show the first N rows of a table.
    
    Wraps: api.helper.get_path_nid(nid) → pd.read_csv(path).head(n)
    
    Output: {
        "table": "va_comp_2023.csv",
        "path": "s3://my-bucket/data/va_comp_2023.csv",
        "columns": ["county", "state", "year", "recipients"],
        "rows": [
            {"county": "Ada", "state": "Idaho", "year": "2023", "recipients": "105"},
            ...
        ],
        "total_rows": 3142
    }
    """
```

```python
def get_table_path(table: str) -> str:
    """
    Get the S3/filesystem path for a table.
    
    Wraps: api.helper.get_path_nid(nid)
    """
```

```python
def describe_column(table: str, column: str) -> dict:
    """
    Get profiling stats for a column.
    
    Wraps: store.get_profile(nid)
    
    Output: {
        "table": "va_comp_2023.csv", "column": "recipients",
        "data_type": "N", "total_values": 3142, "unique_values": 287,
        "min": 0, "max": 1205, "avg": 94.3, "median": 78
    }
    """
```

### 3.3  Orchestration Tools (`orchestration_tools.py`)

The "power tool" — wraps the full DoD pipeline.

```python
def find_and_join_data(
    desired_attributes: list[str],
    sample_values: list[str] | None = None,
    max_results: int = 5
) -> list[dict]:
    """
    Automatically discover, join, and return tables matching desired attributes.
    
    This is the one-shot tool for complex multi-hop queries.
    
    Wraps: DoD.virtual_schema_iterative_search(attrs, samples)
    
    Input:
        desired_attributes = ["county", "VA disability recipients", "population"]
        sample_values = ["Idaho"]
    
    Output: [
        {
            "view_id": "jg_001",
            "tables_used": ["va_comp_2023.csv", "census_2020.csv"],
            "join_path": [{"from": "va_comp..county", "to": "census..county_name"}],
            "columns": ["county", "recipients", "population"],
            "preview_rows": [...],
            "total_rows": 44
        },
        ...
    ]
    """
```

### 3.4  Utility Tools (`util_tools.py`)

```python
def compare_views(view_a: list[dict], view_b: list[dict]) -> dict:
    """
    Classify the relationship between two result views.
    
    Wraps: view_analysis.equivalent(), contained(), complementary(), contradictory()
    
    Output: {"relationship": "complementary", "details": "Views share key 'county'
             but have different value sets (symmetric diff = 12 values)"}
    """
```

```python
def get_hub_columns(top_k: int = 10) -> list[dict]:
    """
    Find the most-connected columns in the graph (likely join hubs).
    
    Wraps: network.fields_degree(top_k)
    
    Output: [
        {"nid": "...", "table": "counties.csv", "column": "fips_code", "degree": 47},
        ...
    ]
    """
```

---

## 4  Serialization Strategy — Hit/DRS → dict

Every tool in the adapter layer needs to convert internal objects to dicts.
Here's the canonical mapping:

```python
# Hit → dict
def _hit_to_dict(hit: Hit, api: API) -> dict:
    info = api.helper.reverse_lookup(hit.nid)
    # info = [(nid, db_name, source_name, field_name)]
    _, db, source, field = info[0]
    return {
        "nid": hit.nid,
        "db": db,
        "table": source,
        "column": field,
        "score": getattr(hit, "score", None),
    }

# DRS → list[dict]
def _drs_to_list(drs: DRS, api: API) -> list[dict]:
    return [_hit_to_dict(hit, api) for hit in drs]

# DataFrame → dict (for preview)
def _df_to_dict(df: pd.DataFrame, max_rows: int = 50) -> dict:
    return {
        "columns": list(df.columns),
        "rows": df.head(max_rows).to_dict(orient="records"),
        "total_rows": len(df),
    }
```

---

## 5  Initialization — What Happens Before the Agent Starts

The agent **does not** run profiling or graph building. Those are offline
batch jobs. The agent only needs the runtime query path:

```
OFFLINE (run once, hours/days):
  1. S3Reader → Profiler → Elasticsearch indices
  2. Coordinator → NetworkBuilder → FieldNetwork graph (pickled)

AGENT STARTUP (seconds):
  3. init_system(model_path, config) → API object
  4. DoD(api) → DoD object
  5. register_tools(api, dod) → {tool_name: callable}

AGENT RUNTIME (per-query, milliseconds-seconds):
  6. Agent calls tools by name → adapter layer → aurum_v2 → results
```

### Startup code

```python
from aurum_v2 import init_system
from aurum_v2.config import AurumConfig
from aurum_v2.dod.dod import DoD
from aurum_v2.agent_api import register_tools

config = AurumConfig(
    es_host="localhost",
    es_port="9200",
)
api = init_system("/path/to/serialized/model", config)
dod = DoD(api)
tools = register_tools(api, dod)

# tools = {
#     "search_columns": <function>,
#     "search_tables": <function>,
#     "find_joinable_columns": <function>,
#     "find_join_path": <function>,
#     "preview_table": <function>,
#     "find_and_join_data": <function>,
#     ...
# }
```

---

## 6  What Must Be Implemented (Dependency Chain)

The adapter layer is **thin** — it's just serialization + error handling.
The real work is implementing the stubs it calls. Here's the dependency
chain from agent tool → v2 method → stubs that need filling:

### `search_columns` → `Algebra.search()` → needs:

```
Algebra.search()                          ← STUB (algebra.py)
  └── StoreHandler.search_keywords()      ← STUB (elastic_store.py)
  └── Algebra._general_to_drs()           ← STUB (algebra.py)
  └── DRS.__init__()                      ← IMPL ✅
  └── Hit.__init__()                      ← IMPL ✅
```

**Minimum to make `search_columns` work: 2 stubs** (Algebra.search + StoreHandler.search_keywords)

### `find_joinable_columns` → `Algebra.pkfk_of()` → needs:

```
Algebra.pkfk_of()                         ← IMPL ✅ (delegates to _neighbor_search)
  └── Algebra._neighbor_search()          ← STUB (algebra.py)
      └── Algebra._general_to_drs()       ← STUB (algebra.py)
      └── FieldNetwork.neighbors_id()     ← STUB (field_network.py)
      └── DRS operations                  ← STUB (drs.py)
```

**Minimum: 3 stubs** (_neighbor_search + _general_to_drs + neighbors_id)

### `find_join_path` → `Algebra.paths()` → needs:

```
Algebra.paths()                           ← STUB (algebra.py)
  └── Algebra._general_to_drs()           ← STUB (already counted)
  └── FieldNetwork.find_path_hit()        ← STUB (field_network.py)
      └── FieldNetwork._find_path()       ← internal
      └── FieldNetwork._find_path_aux()   ← internal
```

**Additional: 2 stubs** (Algebra.paths + find_path_hit)

### `preview_table` → needs:

```
Helper.get_path_nid()                     ← IMPL ✅ (calls store.get_path_of)
  └── StoreHandler.get_path_of()          ← STUB (elastic_store.py)
pd.read_csv(path)                         ← pandas (no stub)
```

**Additional: 1 stub** (get_path_of)

### `find_and_join_data` → `DoD.virtual_schema_iterative_search()` → needs:

```
DoD.virtual_schema_iterative_search()     ← STUB (dod.py)
  └── DoD.joint_filters()                 ← STUB
  └── DoD.individual_filters()            ← STUB
  └── DoD._eager_candidate_exploration()  ← STUB
  └── DoD.joinable()                      ← STUB
      └── Algebra.paths()                 ← STUB (already counted)
  └── DoD.is_join_graph_materializable()  ← STUB
      └── join_utils.read_relation()      ← STUB (join_utils.py)
      └── join_utils.apply_filter()       ← STUB
      └── join_utils.join_ab_on_key()     ← STUB
  └── DoD.materialize_join_graphs()       ← STUB
      └── join_utils.materialize_join_graph() ← STUB
```

**Additional: ~12 stubs** (the entire DoD + join_utils chain)

---

## 7  Implementation Waves Mapped to Agent Capability

| Wave | Stubs to Fill | Agent Gains | Calendar Estimate |
|------|---------------|-------------|-------------------|
| **Wave 1** | `StoreHandler.search_keywords`, `StoreHandler.exact_search_keywords`, `StoreHandler.get_path_of`, `Algebra.search`, `Algebra.exact_search`, `Algebra._general_to_drs` | `search_columns`, `search_tables`, `search_attribute`, `preview_table` — **the agent can find and look at tables** | 1-2 days |
| **Wave 2** | `FieldNetwork.neighbors_id`, `FieldNetwork.find_path_hit`, `Algebra._neighbor_search`, `Algebra.paths`, `DRS.__iter__`, `DRS.intersection/union/difference` | `find_joinable_columns`, `find_join_path`, `find_similar_columns` — **the agent can navigate relationships** | 2-3 days |
| **Wave 3** | `join_utils.*` (read_relation, apply_filter, join_ab_on_key, materialize_join_graph), `DoD.*` (individual_filters through materialize_join_graphs) | `find_and_join_data` — **the agent can do automated multi-hop joins** | 3-5 days |
| **Wave 4** | `view_analysis.*`, ranking helpers, provenance | `compare_views`, better result quality, explainability | 1-2 days |
| **Adapter layer** | `agent_api/` module (serialization, error handling, tool registration) | All tools exposed to agent framework | 1 day |

**Total estimate: ~8-13 days of implementation** to go from current stubs to a fully
functional agent.

---

## 8  What the Agent Does NOT Need from Aurum

| Module/Concept | Why the Agent Doesn't Need It |
|---|---|
| **Profiler** | Runs offline. Agent never calls it. |
| **Network Builder** | Runs offline. Agent never calls it. |
| **Coordinator** | Orchestrates offline build. Agent never calls it. |
| **Provenance (why/how)** | Nice for explainability, but the LLM can explain its own reasoning. Wave 4 at best. |
| **Annotation (MDHit/MDComment)** | Human annotation workflow. The LLM IS the semantic reasoner. |
| **DRS printing (print_tables, print_columns)** | Debug output. The adapter layer formats results as JSON. |
| **DRS ranking (rank_certainty, rank_coverage)** | The LLM can sort results itself via `execute_code`. |
| **Serialization (serialize/deserialize)** | Only used at startup by `init_system`. Already implemented. |

---

## 9  Alternative: Skip the Adapter, Use `execute_code` Directly

Since the agent has `execute_code`, an alternative approach is to **not build
a tool layer at all** and let the agent write Python directly:

```python
# Agent generates this code:
from aurum_v2 import init_system
api = init_system("/model")
results = api.search_content("VA disability")
for hit in results:
    info = api.helper.reverse_lookup(hit.nid)
    print(info)
```

### Tradeoffs

| Approach | Pros | Cons |
|---|---|---|
| **Adapter tools** | Structured I/O, consistent error handling, agent can't misuse internal APIs, easier to test | Extra code to maintain, more rigid |
| **Raw execute_code** | Zero adapter code, maximum flexibility, agent can chain operations freely | Agent needs to know aurum_v2 internals, error handling is ad-hoc, outputs are unstructured |
| **Hybrid** (recommended) | Tools for common operations + execute_code for custom pandas work | Best of both worlds |

### Recommended: Hybrid Approach

Expose **6-8 discovery tools** as structured tools (search, joinable, path,
preview, find_and_join_data). Let the agent use `execute_code` for:

- Custom DataFrame filtering/aggregation (it already can)
- Ad-hoc pandas operations on loaded tables
- Chaining multiple operations in a single code block

This means the adapter layer is **~200 lines of code**, not a major project.

---

## 10  File-Level Port Plan

```
NEW FILES TO CREATE:
  aurum_v2/agent_api/__init__.py       ← register_tools(), tool registry
  aurum_v2/agent_api/discovery_tools.py ← 7 search/navigate tools
  aurum_v2/agent_api/data_tools.py     ← 3 data access tools  
  aurum_v2/agent_api/orchestration_tools.py ← 1 DoD tool
  aurum_v2/agent_api/util_tools.py     ← 2 utility tools
  aurum_v2/agent_api/serializers.py    ← _hit_to_dict, _drs_to_list, _df_to_dict

STUBS TO IMPLEMENT (in dependency order):
  Wave 1: elastic_store.py    → 3 methods
           algebra.py         → 3 methods  
  Wave 2: field_network.py    → 3 methods
           algebra.py         → 2 more methods
           drs.py             → 4 methods
  Wave 3: join_utils.py       → 10 functions
           dod.py             → 9 methods
  Wave 4: view_analysis.py    → 8 functions
           provenance.py      → 6 methods (optional)

OFFLINE PIPELINE (separate from agent):
  profiler/column_profiler.py → 9 functions (run once)
  builder/network_builder.py  → 4 functions (run once)
  builder/analysis.py         → 11 functions (run once)
  builder/coordinator.py      → already orchestrated
```

---

## 11  Summary

The porting strategy is:

1. **aurum_v2 core stays as-is** — a Python library with clean OOP
2. **`agent_api/`** is a thin adapter layer (~200 lines) that serializes
   Hit/DRS/DataFrame → JSON-friendly dicts
3. **13 tools** exposed to the agent (7 discovery + 3 data + 1 orchestration + 2 utility)
4. **`execute_code`** handles all DataFrame manipulation (filter, aggregate, sort)
5. **82 stubs** need implementing across 4 waves, with legacy Python sources
   as reference for all except the profiler (Java-only)
6. **Wave 1 alone** (6 stubs, 1-2 days) gives the agent basic search + preview capability
