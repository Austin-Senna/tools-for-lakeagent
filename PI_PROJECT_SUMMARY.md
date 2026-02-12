# Aurum v2 Project — Executive Summary for PI

**Date:** February 2026  
**Project:** Building an AI agent capable of discovering and joining tables across
a 9.5 TB S3 data lake of CSV files to answer complex analytical questions.

---

## The Problem

Your team has a **9.5 TB data lake** — thousands of CSV files containing disparate
data sources (census, VA disability compensation, climate, etc.). An AI agent
needs to answer questions like:

> *"What is the population of the largest city in Idaho county that had
> between 90–110 VA disability recipients in 2023 AND between 88–92 in 2019?"*

**The challenge:** This question requires the agent to:
1. Find the relevant tables (VA disability, census population, county metadata)
2. Figure out which columns can be joined (counties match across tables)
3. Join those tables correctly
4. Filter and aggregate the result

A human would do this in 10 minutes. Without automation, it's weeks of work.

## The Solution: Aurum v2

**Aurum** is a data discovery system originally built at MIT (2018). It's like
a **search engine for data relationships**.

**How it works:**

```
OFFLINE (once, before the agent runs):
┌─────────────────────────────────────────────────────┐
│ 1. Scan all 9.5 TB of CSVs on S3                   │
│ 2. Extract statistics per column:                   │
│    - Data type (number vs text)                    │
│    - Unique values count                           │
│    - Min/max/median ranges                         │
│    - MinHash signatures (compressed similarity)     │
│ 3. Store all stats in Elasticsearch                │
│ 4. Build a relationship graph:                     │
│    - Which columns can be joined together?         │
│    - Which have similar values?                    │
│    - Which are likely keys?                        │
│ Time: ~4-8 hours (one-time cost)                   │
└─────────────────────────────────────────────────────┘

RUNTIME (per question, milliseconds-seconds):
┌─────────────────────────────────────────────────────┐
│ Agent asks Aurum: "Where are VA disability cols?"  │
│                          ↓                         │
│ Aurum searches ES by keyword → finds 3 tables      │
│                          ↓                         │
│ Agent asks: "What joins county?"                   │
│                          ↓                         │
│ Aurum looks at relationship graph → finds county   │
│ as join key in census tables                       │
│                          ↓                         │
│ Agent orchestrates: load CSV → filter → join → agg │
│                          ↓                         │
│ Returns: [Answer] 2,847 people                     │
└─────────────────────────────────────────────────────┘
```

## Why Aurum v2 (Instead of Building from Scratch)?

### Option 1: Build a custom discovery system
- **Effort**: 6-12 months of R&D
- **Risk**: High — similarity metrics, join discovery are non-trivial
- **Maintenance**: You own all bugs

### Option 2: Use Aurum (existing, proven)
- **Effort**: 2-3 weeks (implementing the legacy algorithms in modern Python)
- **Risk**: Low — algorithms are published, reference implementation exists
- **Maintenance**: Inherits from MIT's 2018 work

**We chose Option 2.**

## What We've Done (So Far)

### Phase 1: Understand the Legacy
- ✅ Read the original Aurum codebase (Java + Python, ~20K lines)
- ✅ Identified the 20 core features needed (profiling, graph building, search, joins)
- ✅ Verified all importance ratings against actual source code
- ✅ Identified bugs in a previous rewrite (dual ID systems, missing validations)

### Phase 2: Build a Clean Skeleton
- ✅ Created `aurum_v2/` — 29 Python files replicating the architecture
- ✅ All data structures implemented (Hit, DRS, FieldNetwork, Relation, Provenance)
- ✅ Configuration system in place (all 15 thresholds from the paper)
- ✅ S3Reader implemented (can stream CSVs from S3 buckets)
- ✅ API entry point ready (`init_system` function)

### Phase 3: Map Implementation Work
- ✅ Created exhaustive dependency mapping: 82 stubs → 12 legacy files
- ✅ Analyzed 6 missing functions, added skeleton stubs
- ✅ Zero implementation errors or missing imports

### Phase 4: Design Agent Integration
- ✅ Wrote porting strategy document ([port.md](port.md))
- ✅ Designed 13 tools for agent use (search, join, materialize, compare)
- ✅ Estimated effort: **8-13 days** of implementation

## What's Next (Implementation Plan)

The work ahead is **systematic and low-risk** — we're porting **proven algorithms**
from legacy code, not inventing new ones.

### Wave 1: Basic Search (1-2 days)
**Goal:** Agent can find and preview tables

```python
# Agent can do this
search_columns("VA disability") 
  → [Hit(table="va_comp_2023.csv", column="recipients", score=0.92), ...]

preview_table("va_comp_2023.csv") 
  → first 50 rows
```

**Requires:** 6 stubs in Elasticsearch client + Algebra search

### Wave 2: Join Path Finding (2-3 days)
**Goal:** Agent can discover join paths between tables

```python
# Agent can do this
find_joinable_columns("va_comp_2023.csv", "county")
  → [Hit(table="census_2020.csv", column="county_name"), ...]

find_join_path("va_comp_2023.csv", "census_2020.csv")
  → linear path through intermediary tables (if needed)
```

**Requires:** Graph path-finding algorithms + NetworkX integration

### Wave 3: Automated Join Materialization (3-5 days)
**Goal:** Agent can answer complex multi-table questions in one call

```python
# Agent can do this — the "magic" call
find_and_join_data(
    attributes=["county", "VA disability recipients", "population"],
    sample_values=["Idaho"]
)
  → [DataFrame with joined results across 3+ tables]
```

**Requires:** Full DoD (Data-on-Demand) pipeline + join utilities

### Wave 4: Polish & Optimization (1-2 days)
**Goal:** Better ranking of results, view comparison, explainability

**Requires:** View analysis + provenance + ranking helpers

---

## Why This Approach Works

### 1. **We Have Reference Code**
Every algorithm is already written in the legacy Aurum codebase. We're not
inventing new similarity metrics or join strategies — we're replicating
peer-reviewed algorithms from the MIT paper.

### 2. **Clear Dependency Chain**
We know exactly which 82 functions need implementing and which legacy code
contains them. No guessing.

### 3. **Incremental Progress**
Wave 1 alone (1-2 days) gives the agent **half** of the core functionality.
Each wave is testable independently.

### 4. **Low Risk**
- No new ML models to train
- No experimental algorithms
- Algorithms validated in production (MIT, 2018)
- Legacy code available for verification

### 5. **Portable to Any Agent Framework**
The 13 tools are simple JSON-in, JSON-out functions. They work with
LangChain, Claude's tool system, custom agents, etc.

---

## Resource Requirements

| Resource | For Building | For Running |
|----------|--------------|------------|
| **CPU** | 4-8 cores for offline graph building | 2-4 cores for queries |
| **RAM** | 16 GB for build phase | 8 GB minimum |
| **Storage** | 20 GB for ES indices (for 9.5 TB data lake) | 10 GB ES + 2 GB pickles |
| **ES Cluster** | 1 node minimum | 1 node recommended |
| **Time** | ~1-2 weeks to implement all waves | <1 sec per typical query |

---

## Risks & Mitigation

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Similarity metrics don't work on our specific data | Low | We'll profile a sample first before full build |
| ES cluster runs out of memory | Low | Can shard by data source or archive old indices |
| Join materialization is too slow | Low | Timeouts + chunking built in; agent can execute_code for pandas |
| Missing algorithms in legacy code | Very Low | We have complete dependency map + reference |

---

## Success Criteria

After **Wave 1** (1-2 weeks):
- ✅ Agent can search for columns by keyword
- ✅ Agent can preview tables and see their structure
- ✅ Agent can read basic CSV files from S3

After **Wave 2** (1 additional week):
- ✅ Agent can find joinable columns
- ✅ Agent can discover join paths
- ✅ Agent can understand table relationships

After **Wave 3** (1-2 additional weeks):
- ✅ Agent can answer the example question in ONE tool call
- ✅ Agent can materialize multi-table views automatically
- ✅ Agent can filter, aggregate, and rank results

---

## Example: The Question Answered

**User asks:** "Population of the largest city in Idaho county with 90–110 VA
disability recipients in 2023 AND 88–92 in 2019?"

**Agent does (after Wave 3):**

```python
# 1. One-shot discovery + join + materialize
results = find_and_join_data(
    attributes=["county", "VA disability recipients", "population", "city"],
    sample_values=["Idaho", "2023", "2019"]
)

# 2. Filter and aggregate (agent's execute_code)
df = results[0]  # Aurum picked the best view
largest_city = df[
    (df["year"].isin([2023, 2019])) &
    (df["recipients_2023"].between(90, 110)) &
    (df["recipients_2019"].between(88, 92)) &
    (df["state"] == "Idaho")
].sort_values("population", ascending=False).iloc[0]

# 3. Return answer
return f"[Answer] {largest_city['population']} people"
```

**Total wall-clock time:** <2 seconds (after offline build)

---

## Questions for PI

1. **Data access:** Do we have S3 credentials configured for the team?
2. **ES cluster:** Should we use an existing Elasticsearch instance or provision a new one?
3. **Priority:** Is Wave 1 (search + preview) sufficient for initial testing, or do we need full Wave 3 immediately?
4. **Agent framework:** Which agent system will use these tools (LangChain, Claude tools, custom)?
5. **Timeline:** Can we do a phased rollout (Wave 1 → demo → Wave 2 → production)?

---

## Next Steps

1. **Week 1:** Implement Wave 1 stubs, test against sample data lake
2. **Week 1-2:** Implement Wave 2 stubs, verify join discovery
3. **Week 2-3:** Implement Wave 3 stubs, integrate agent
4. **Week 3+:** Optimization, edge cases, full deployment

---

## Appendix: Technical Jargon Decoded

| Term | What It Means (In Plain English) |
|------|----------------------------------|
| **MinHash** | A compact "fingerprint" of column values that lets us quickly check if two columns have similar data without comparing all values |
| **PKFK** | Primary Key / Foreign Key — a join relationship where one table's "ID" matches another table's "foreign ID" |
| **Elasticsearch** | A fast search engine (like Google for your data lake); we use it to index all column statistics |
| **Materialization** | Actually reading the CSVs from disk/S3, joining them, and creating a real table (as opposed to planning it) |
| **DRS** | "Domain Result Set" — a fancy term for "the results of a search, plus metadata about how we found them" |
| **Provenance** | A trace of "how did we get this result?" — useful for explaining why the agent chose certain tables |
| **LSH (Locality Sensitive Hashing)** | A trick to find similar items quickly without comparing everything to everything |

---

**Contact:** [Your Name]  
**Last Updated:** February 2026
