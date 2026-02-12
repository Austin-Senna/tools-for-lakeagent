# Aurum v2 — Agent-Readiness Analysis for S3 Data Lake Discovery

> **Context**: An AI Agent with tool access needs to search a **9.5 TB S3 data lake
> of CSV files** and answer complex multi-hop analytical questions such as:
>
> *"Can you tell me the population of the largest city in the Idaho county reported
> in the 2020 census that had between 90 and 110 U.S. Department of Veterans Affairs
> disability compensation recipients in 2023 AND had between 88 and 92 recipients
> in 2019?"*

---

## 1  Query Decomposition — What the Agent Actually Needs

Let's break the example question into the operations an agent must perform:

| Step | Operation | What's needed |
|------|-----------|---------------|
| 1 | **Find** the table(s) containing "VA disability compensation recipients" | Keyword/content search across 9.5 TB of CSVs |
| 2 | **Filter** rows for 2023 + range 90–110 recipients | Read CSV, apply predicate |
| 3 | **Find** the same table(s) for 2019 + range 88–92 | Keyword search + filter |
| 4 | **Join/Intersect** the 2023 and 2019 results on county | Join two filtered result sets |
| 5 | **Find** a table with "Idaho county" + "2020 census" + "population" | Keyword/attribute search |
| 6 | **Join** the census table to the VA results on county | Cross-table join |
| 7 | **Filter** to Idaho, then find the "largest city" | Aggregate/sort |
| 8 | **Return** the population value | Point lookup |

**Core agent needs**: discover relevant tables → filter rows → join across tables → aggregate.

---

## 2  Module-by-Module Essentiality Analysis

### 🟢 ESSENTIAL — The Agent Cannot Function Without These

| Module | File(s) | Why Essential | Agent Tool? |
|--------|---------|---------------|-------------|
| **Profiler** | `profiler/column_profiler.py`, `profiler/source_readers.py` | **Without profiling, no data exists in ES.** The agent can't search 9.5 TB of CSVs without an index. This is Stage 0 — it creates the `profile` + `text` ES indices. Must run once offline over the S3 bucket. | ❌ Offline (run once before agent starts) |
| **Store (ES Client)** | `store/elastic_store.py` | The gateway to all indexed data. `search_keywords`, `exact_search_keywords`, `get_all_fields` — this is how the agent finds tables/columns matching natural language terms like "VA disability" or "population". | ✅ **Yes — `search`** |
| **Network Builder** | `builder/network_builder.py`, `builder/analysis.py`, `builder/coordinator.py` | Builds the relationship graph (which columns are joinable, which are similar). Without PKFK/CONTENT_SIM edges, the agent can't discover join paths between tables. Must run once offline. | ❌ Offline (run once) |
| **Field Network** | `graph/field_network.py` | The in-memory graph the agent queries at runtime. `neighbors_id` (find joinable columns), `find_path_hit`/`find_path_table` (discover multi-hop join paths) are critical for connecting tables. | ✅ **Yes — `neighbors`, `find_path`** |
| **Algebra** | `discovery/algebra.py` | The agent's primary query interface: `search_content("VA disability")`, `pkfk_of(drs)`, `paths(drs_a, drs_b)`, `intersection(a, b)`. Every step of the decomposed question maps to an Algebra operation. | ✅ **Yes — all methods become tools** |
| **API + init_system** | `discovery/api.py` | Entry point: loads the model, creates the `Algebra` instance. `Helper.reverse_lookup` and `Helper.get_path_nid` let the agent resolve column IDs back to file paths. | ✅ **Yes — `init_system`, `reverse_lookup`** |
| **Hit / DRS / Relation** | `models/hit.py`, `models/drs.py`, `models/relation.py` | Core data structures. Every tool input/output is a Hit or DRS. The agent needs to understand what it gets back. | 🔧 Internal (not directly exposed, but all tool results use these) |
| **Config** | `config.py` | Holds all thresholds. Essential for consistent behavior. | ❌ Internal |

### 🟡 IMPORTANT — Needed for the Full Join Materialization Pipeline

| Module | File(s) | Why Important | Agent Tool? |
|--------|---------|---------------|-------------|
| **DoD (Data on Demand)** | `dod/dod.py` | `virtual_schema_iterative_search` is the **killer feature** for the example question. Given `list_attributes=["county", "VA disability recipients", "population"]` and `list_samples=["Idaho"]`, it automatically discovers tables, finds join paths, validates them, and materialises the combined view. This is the "one-shot" tool for complex multi-hop queries. | ✅ **Yes — `virtual_schema_iterative_search`** |
| **Join Utils** | `dod/join_utils.py` | Materialises join graphs into actual DataFrames. `join_ab_on_key_optimizer` (3-min timeout, chunked) handles 9.5 TB scale. `materialize_join_graph` (tree-fold) handles multi-way joins. Without this, DoD can't produce results. | 🔧 Internal (called by DoD) |
| **View Analysis** | `dod/view_analysis.py` | When DoD produces multiple candidate views, `equivalent()`/`contained()`/`complementary()`/`contradictory()` help the agent deduplicate and choose the best one. Important for quality but not strictly blocking. | ✅ **Yes — `equivalent`, `contradictory`** (for dedup) |

### 🔵 NICE TO HAVE — Useful But Not Critical

| Module | File(s) | Purpose | Agent Tool? |
|--------|---------|---------|-------------|
| **Provenance** | `models/provenance.py` | Explains *how* a result was derived. `drs.why(hit)` / `drs.how(hit)`. Useful for agent reasoning ("I found this via PKFK through table X"). Not needed for correctness. | 🟡 Optional — `why`, `how` |
| **Annotation** | `models/annotation.py` | MDHit/MDComment/MRS — user can label "these columns mean the same thing." Useful for a human-curated knowledge layer but the agent can reason about this itself. | ❌ Not needed for agent |
| **Text Utils** | `utils/text_utils.py` | CamelCase normalization for column names. Used internally by the profiler and network builder. | ❌ Internal |
| **IO Utils** | `utils/io_utils.py` | Pickle serialization. Used by coordinator and init_system. | ❌ Internal |

### ⚪ NOT NEEDED for This Use Case

| Module | Why Not Needed |
|--------|---------------|
| **Annotation system** (MDHit, MDComment, MRS, metadata relations) | An AI agent doesn't need human annotation workflows. It can reason about column semantics from names/values. |
| **Reporting** (not yet created) | Graph statistics are for human debugging, not agent queries. |
| **EKG / Neo4j** (not ported, correctly) | The agent doesn't need a second graph backend. |
| **Sugar** (not ported, correctly) | Interactive REPL shortcuts. The agent uses tools, not a REPL. |
| **Ontomatch** (not ported, correctly) | GloVe-based ontology matching. The LLM agent IS the semantic matcher. |

---

## 3  Tools to Expose to the Agent

The agent needs a focused set of tools — not the entire codebase. Here's the recommended tool interface:

### 3.1  Discovery Tools (Search & Navigate)

| Tool Name | Maps To | Input | Output | When to Use |
|-----------|---------|-------|--------|-------------|
| **`search_columns`** | `Algebra.search_content(kw)` | keyword string | List of matching columns (table, column, score) | "Find tables about VA disability" |
| **`search_tables`** | `Algebra.search_table(kw)` | keyword string | List of matching tables | "Find census tables" |
| **`search_attribute`** | `Algebra.search_exact_attribute(kw)` | column name | List of columns with that exact name | "Find columns named 'population'" |
| **`find_similar_columns`** | `Algebra.content_similar_to(col)` | column reference | List of content-similar columns | "What other columns have similar values to this one?" |
| **`find_joinable_columns`** | `Algebra.pkfk_of(col)` | column reference | List of PK/FK-related columns | "What can I join this column with?" |
| **`find_join_path`** | `Algebra.paths(a, b, PKFK)` | two table/column references | Join path (sequence of hops) | "How do I connect table A to table B?" |

### 3.2  Data Access Tools (Read & Filter)

| Tool Name | Maps To | Input | Output | When to Use |
|-----------|---------|-------|--------|-------------|
| **`read_table`** | `join_utils.read_relation(path)` | table path (from reverse_lookup) | DataFrame (or preview) | "Show me what's in this table" |
| **`filter_table`** | `join_utils.apply_filter(path, attr, value)` | path + column + value | Filtered DataFrame | "Filter rows where state = Idaho" |
| **`join_tables`** | `join_utils.join_ab_on_key_optimizer(a, b, key_a, key_b)` | two DataFrames + join keys | Joined DataFrame | "Join census with VA data on county" |
| **`preview_table`** | `read_relation(path).head(n)` | table path + n rows | Sample rows | "Show me first 5 rows" |

### 3.3  High-Level Orchestration Tool (The Power Tool)

| Tool Name | Maps To | Input | Output | When to Use |
|-----------|---------|-------|--------|-------------|
| **`find_and_join_data`** | `DoD.virtual_schema_iterative_search(attrs, values)` | list of desired attributes + sample values | Materialised DataFrame(s) | "I need county + VA recipients + population, filtered to Idaho" |

### 3.4  Utility Tools

| Tool Name | Maps To | Input | Output | When to Use |
|-----------|---------|-------|--------|-------------|
| **`describe_column`** | `Helper.reverse_lookup(nid)` | column ID | (db, table, column, type) | "What table/column is this ID?" |
| **`get_table_path`** | `Helper.get_path_nid(nid)` | column ID | S3/filesystem path | "Where is this table stored?" |
| **`compare_views`** | `view_analysis.equivalent(v1, v2)` | two DataFrames | Equivalence classification | "Are these two result sets the same?" |

---

## 4  Critical Gaps for the 9.5 TB S3 Use Case

### ✅ ~~Gap 1: S3 Source Reader~~ — IMPLEMENTED

The `S3Reader` class has been added to `aurum_v2/profiler/source_readers.py`.
It uses boto3 to paginate `ListObjectsV2`, streams each CSV via `get_object()`,
supports prefix filtering, optional row-sampling (`sample_rows=1000`), and
configurable AWS profiles/regions. The `discover_sources()` factory now accepts
`source_type: "s3"` configs.

### 🔴 Gap 2: Scale — ES Won't Index 9.5 TB of Raw Values

The legacy profiler indexes **raw text values** into the ES `text` index
(one document per column with ALL values). At 9.5 TB this is impractical.

**Recommendations:**
- Profile only **metadata** (column names, types, stats, MinHash sigs) — this is small.
- For the `text` index, store only **sampled values** (e.g., 1000 representative values per column) rather than all values.
- Consider using the agent's ability to read CSVs on-demand from S3 instead of pre-indexing all values.

### 🟡 Gap 3: Missing DataFrame Operations for Agent

The agent needs to do things like:
- `df.groupby("county").sum()` (aggregation)
- `df.sort_values("population", ascending=False).head(1)` (find max)
- `df[(df["year"] == 2023) & (df["recipients"].between(90, 110))]` (range filter)

None of these exist as tools. The agent needs a **`run_pandas_query`** tool or
equivalent that lets it execute arbitrary DataFrame operations on loaded tables.

### 🟡 Gap 4: Agent Response Formatting

The tools return DataFrames and DRS objects. The agent needs to:
- Extract a single scalar value from a DataFrame
- Format it as `[Answer]`
- Handle cases where multiple views are returned

This is a thin formatting layer, not an Aurum issue, but it matters for the end-to-end flow.

---

## 5  Recommended Agent Architecture

```
┌──────────────────────────────────────────────────────────┐
│  OFFLINE (run once)                                      │
│                                                          │
│  1. S3Reader scans all 9.5 TB CSVs                      │
│  2. Profiler computes per-column stats + MinHash sigs    │
│  3. Profiles stored in Elasticsearch                     │
│  4. Network Builder creates relationship graph           │
│  5. Graph + indexes serialized to disk                   │
│                                                          │
│  Time: hours to days (depends on cluster size)           │
│  Output: ES indices + pickle files                       │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  RUNTIME (agent query time)                              │
│                                                          │
│  API = init_system(model_path, config)                   │
│  ↓                                                       │
│  Agent receives question from user                       │
│  ↓                                                       │
│  Agent has access to TOOLS:                              │
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │  DISCOVERY TOOLS                               │      │
│  │  • search_columns(kw) → matching columns       │      │
│  │  • search_tables(kw) → matching tables         │      │
│  │  • search_attribute(kw) → exact column match   │      │
│  │  • find_joinable_columns(col) → PKFK neighbors │      │
│  │  • find_join_path(t1, t2) → join hops          │      │
│  └────────────────────────────────────────────────┘      │
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │  DATA ACCESS TOOLS                             │      │
│  │  • preview_table(table) → first N rows         │      │
│  │  • filter_table(table, col, val) → filtered df │      │
│  │  • join_tables(t1, t2, key) → joined df        │      │
│  │  • run_query(df, pandas_expr) → result         │      │
│  └────────────────────────────────────────────────┘      │
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │  ORCHESTRATION TOOL                            │      │
│  │  • find_and_join_data(attrs, values) → view(s) │      │
│  │    (DoD — automated multi-table discovery)     │      │
│  └────────────────────────────────────────────────┘      │
│                                                          │
│  Agent reasoning loop:                                   │
│  1. Decompose question into sub-queries                  │
│  2. Use search tools to find relevant tables             │
│  3. Use data tools to read/filter/join                   │
│  4. Use pandas operations to aggregate                   │
│  5. Extract and format the answer                        │
└──────────────────────────────────────────────────────────┘
```

---

## 6  Example: Solving the VA Question Step by Step

```
User: "Population of largest city in Idaho county with 90-110 VA
       disability recipients in 2023 AND 88-92 in 2019?"

Agent reasoning:

1. search_columns("VA disability compensation recipients")
   → finds: table="va_comp_by_county.csv", col="recipients"

2. preview_table("va_comp_by_county.csv")
   → sees columns: [county, state, year, recipients, ...]

3. filter_table("va_comp_by_county.csv", "year", "2023")
   → df_2023

4. run_query(df_2023, 'df[(df["recipients"]>=90) & (df["recipients"]<=110)]')
   → df_2023_filtered

5. filter_table("va_comp_by_county.csv", "year", "2019")
   → df_2019

6. run_query(df_2019, 'df[(df["recipients"]>=88) & (df["recipients"]<=92)]')
   → df_2019_filtered

7. join_tables(df_2023_filtered, df_2019_filtered, "county", "county")
   → df_both_years (counties meeting BOTH criteria)

8. run_query(df_both_years, 'df[df["state"]=="Idaho"]')
   → idaho_counties

9. search_columns("2020 census population city")
   → finds: table="census_2020_cities.csv", col="population"

10. find_join_path("va_comp_by_county.csv", "census_2020_cities.csv")
    → join via "county" column

11. join_tables(idaho_counties, census_2020_cities, "county", "county")
    → combined_df

12. run_query(combined_df, 'df.sort_values("population", ascending=False).head(1)')
    → largest city

13. Return: [Answer] {population_value}
```

---

## 7  Summary Verdict

| Aspect | Status |
|--------|--------|
| **Is aurum_v2 the right foundation?** | ✅ Yes — the discover→build→query→join pipeline is exactly what's needed |
| **Is it complete enough to use today?** | ❌ No — all core methods are `raise NotImplementedError` stubs |
| **What's critically missing?** | S3Reader, implemented methods (search/join/filter/path), DataFrame query tool |
| **What's unnecessary?** | Annotation system, reporting, EKG/Neo4j, sugar, ontomatch |
| **Biggest risk at 9.5 TB scale?** | ES text index size (need sampling), join materialization memory (need chunking — designed but unimplemented), profiling time (need parallelism) |
| **Modules to implement first?** | 1) Profiler (S3 variant) 2) StoreHandler 3) FieldNetwork graph ops 4) Algebra search+paths 5) join_utils 6) DoD pipeline |

### Priority Implementation Order

| Priority | Module | Effort | Impact |
|----------|--------|--------|--------|
| **P0** | `profiler/` — implement + add S3Reader | Large | Without this, no data in ES |
| **P0** | `store/elastic_store.py` — implement all methods | Medium | Agent can't search without this |
| **P0** | `graph/field_network.py` — implement init_meta_schema, add_relation, neighbors_id, find_path_hit | Medium | Agent can't navigate relationships |
| **P0** | `discovery/algebra.py` — implement search, _neighbor_search, paths, make_drs | Medium | Agent's primary interface |
| **P1** | `builder/` — implement all 4 build functions + analysis.py | Large | Graph edges won't exist without this |
| **P1** | `dod/join_utils.py` — implement read_relation, apply_filter, join_ab_on_key_optimizer, materialize_join_graph | Medium | Agent can't materialize joins |
| **P1** | `dod/dod.py` — implement virtual_schema_iterative_search | Large | The "one-shot" power tool |
| **P2** | `models/drs.py` — implement set ops, provenance | Medium | Needed for intersection/union queries |
| **P2** | `dod/view_analysis.py` — implement 4C classification | Small | Dedup multiple results |
| **P3** | New: `run_pandas_query` tool | Small | Agent needs arbitrary DataFrame ops |
| **P3** | New: Agent tool wrapper layer | Medium | Translate Aurum APIs into clean tool interfaces |
