# 🏔️ aurum

**Modern Data Lake Discovery** — a from-scratch reimplementation of the core
algorithms from [Aurum (MIT, 2018)](https://github.com/mitdbg/aurum-datadiscovery),
rebuilt for Python 3.12+ with Polars, Sentence-Transformers, and MinHash LSH.

## What It Does

aurum automatically discovers **relationships between columns** across
hundreds or thousands of CSV / Parquet files in a data lake:

| Relationship        | How It's Found                                          |
|---------------------|---------------------------------------------------------|
| **Schema Similarity**   | Column names embedded via Sentence-Transformers, cosine similarity |
| **Content Similarity**  | Text columns → MinHash LSH (Jaccard ≥ 0.7)            |
| **Numeric Overlap**     | Median ± IQR interval overlap ≥ 0.85                   |
| **PK / FK**             | High-cardinality columns whose values include another's |
| **Inclusion Dependency**| Full value-range containment with overlap ≥ 0.3        |

Once the index is built, you can:

- **Search** for columns by keyword or content
- **Discover** join paths between tables
- **Synthesize** virtual schemas (Data-on-Demand)
- **Classify** materialized views (equivalent / contained / complementary / contradictory)

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Build an index over a folder of CSVs
aurum build-index ./my_data_lake/ --output .lake_index/

# Search for columns
aurum search "employee name" --index .lake_index/

# Find join paths between two tables
aurum discover "employees.csv" "departments.csv" --index .lake_index/
```

## Architecture

```
aurum/
├── config.py               # All thresholds as a frozen dataclass
├── profiler/
│   ├── column_profiler.py  # Per-column stats, MinHash, embeddings
│   └── text_utils.py       # camelCase→snake, stopword removal, tokenisation
├── graph/
│   ├── relations.py        # Relation enum (SCHEMA_SIM, CONTENT_SIM, PKFK, …)
│   ├── field_network.py    # NetworkX multi-graph of column relationships
│   └── network_builder.py  # Builds edges: schema-sim, content-sim, PK/FK
├── discovery/
│   ├── result_set.py       # Hit, DRS (Domain Result Set), Provenance
│   ├── algebra.py          # Query algebra: search, neighbor, union, intersect, paths
│   └── data_on_demand.py   # Virtual schema synthesis (greedy set cover → join graphs)
├── materialization/
│   ├── join_engine.py      # Memory-aware chunked joins with Polars
│   └── view_analysis.py    # 4C view classification
└── cli.py                  # Click CLI
```

## Key Algorithmic Decisions (ported from Aurum)

| Parameter               | Value | Origin                                  |
|--------------------------|-------|-----------------------------------------|
| MinHash perms            | 256   | Aurum used 512; 256 is sufficient in 2026 |
| Jaccard threshold        | 0.7   | `networkbuilder.build_content_sim_mh_text` |
| Numeric overlap threshold| 0.85  | `networkbuilder.build_content_sim_relation_num_overlap_distr` |
| PK cardinality cutoff    | 0.7   | `networkbuilder.build_pkfk_relation`    |
| Inclusion dep. overlap   | 0.3   | `networkbuilder.build_content_sim_relation_num_overlap_distr` |
| Join overlap             | 0.4   | `config.join_overlap_th`                |
| Schema sim: sublinear TF | True  | `dataanalysis.vect` global TfidfVectorizer |

## License

MIT
