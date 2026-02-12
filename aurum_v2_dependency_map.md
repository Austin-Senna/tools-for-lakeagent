# Aurum Legacy → v2 Dependency Map

> Complete mapping of every `raise NotImplementedError` stub in `aurum_v2/` to
> the legacy file and function that contains the real implementation.

---

## Key

| Symbol | Meaning |
|--------|---------|
| ✅ | Already implemented in v2 |
| 🔴 | Stub — needs porting from legacy Python |
| 🟣 | Stub — needs reimplementation from legacy Java (no Python source) |
| ⚪ | Not needed for the agent use case |

---

## 1  `aurum_v2/store/elastic_store.py`

> **Legacy source**: `aurum_legacy/modelstore/elasticstore.py` (class `StoreHandler`)

| v2 Stub Method | Legacy Method | What It Does |
|---|---|---|
| `get_all_fields()` | `StoreHandler.get_all_fields_of_source()` | ES scroll over `profile` index; yields (db, source, field, nid) |
| `get_all_fields_of_source()` | `StoreHandler.get_all_fields_of_source()` | Same, filtered by `sourceName` |
| `get_all_fields_of_source_and_field()` | `StoreHandler.get_all_fields_of_source_and_field()` | Filtered by source + field name |
| `search_keywords()` | `StoreHandler.search_keywords()` | Fuzzy ES search against `text` or `profile` index |
| `exact_search_keywords()` | `StoreHandler.exact_search_keywords()` | ES `term` query variant |
| `get_profile()` | `StoreHandler.peek_values()` | Point query on `profile` index for minhash/stats |
| `suggest_field_names()` | `StoreHandler.suggest_field_names()` | ES completion suggester on `columnNameSuggest` |
| `peek_values()` | `StoreHandler.peek_values()` | Currently a placeholder |

---

## 2  `aurum_v2/discovery/algebra.py`

> **Legacy source**: `aurum_legacy/algebra.py` (class `Algebra`)

| v2 Stub Method | Legacy Method | What It Does |
|---|---|---|
| `search()` | `Algebra.search()` | Calls `store.search_keywords()`, wraps hits in DRS with PKFK provenance |
| `exact_search()` | `Algebra.exact_search()` | Same pattern with `exact_search_keywords()` |
| `_neighbor_search()` | `Algebra.neighbor_search()` | Converts input→DRS, creates provenance carrier, expands table-mode, iterates `network.neighbors_id()` |
| `traverse()` | `Algebra.traverse()` | BFS loop: fringe expansion via `_neighbor_search`, unioned into accumulator |
| `paths()` | `Algebra.paths()` | Cartesian product of drs_a × drs_b; dispatches to `find_path_hit` or `find_path_table` |
| `intersection()` | `Algebra.intersection()` | Delegates to `DRS.intersection()` |
| `union()` | `Algebra.union()` | Delegates to `DRS.union()` |
| `difference()` | `Algebra.difference()` | Delegates to `DRS.difference()` |
| `make_drs()` | `Algebra.make_drs()` | Handles list input (union fold) or single Hit |
| `drs_from_table_hit()` | `Algebra.drs_from_table_hit()` | Gets all hits from table, wraps with provenance |
| `drs_from_table()` | `Algebra.drs_from_table()` | Same but nid-based input |
| `_general_to_drs()` | `Algebra._general_to_drs()` | Type-dispatch: DRS/None/int/str/tuple/Hit → DRS |
| `_hit_to_drs()` | Inline in `Algebra._general_to_drs()` | Wraps Hit in DRS; if table_mode, expands all columns |
| `_nid_to_hit()` | `Algebra._nid_to_hit()` | nid → Hit via `network.get_info_for()` |
| `_node_to_hit()` | `Algebra._node_to_hit()` | (source, field) → Hit |

---

## 3  `aurum_v2/graph/field_network.py`

> **Legacy source**: `aurum_legacy/knowledgerepr/fieldnetwork.py` (class `FieldNetwork`)

| v2 Stub Method | Legacy Method | What It Does |
|---|---|---|
| `init_meta_schema()` | `FieldNetwork.init_meta_schema()` | Populates `id_to_info`, `source_to_fields`, adds nodes with cardinality |
| `add_relation()` | `FieldNetwork.add_field_relation()` | Adds weighted edge to NetworkX graph under relation key |
| `get_info_for()` | `FieldNetwork.get_info_for()` | nid → (db, source, field, type) |
| `neighbors_id()` | `FieldNetwork.neighbors_id()` | Iterates `G[nid]`, filters by relation key, builds Hit list → DRS |
| `find_path_hit()` | `FieldNetwork.find_path_hit()` | DFS with `_find_path` + `_find_path_aux`; max_hops=5 |
| `find_path_table()` | `FieldNetwork.find_path_table()` | Table-level DFS with table→fields expansion |
| `enumerate_relation()` | `FieldNetwork.enumerate_relation()` | Iterates all node pairs with a given relation type |
| `serialize()` | `FieldNetwork.serialize()` | Pickles graph + id_info + table_ids via `io_utils` |
| `deserialize()` (classmethod) | `FieldNetwork.deserialize()` | Unpickle × 3, reconstructs FieldNetwork |

---

## 4  `aurum_v2/builder/network_builder.py`

> **Legacy source**: `aurum_legacy/knowledgerepr/networkbuilder.py` (module-level functions)

| v2 Stub Method | Legacy Function | What It Does |
|---|---|---|
| `build_schema_sim()` | `networkbuilder.build_schema_sim_relation()` | TF-IDF on field names → NearPy LSH → connect SCHEMA_SIM edges |
| `build_content_sim_mh_text()` | `networkbuilder.build_content_sim_mh_text()` | MinHash objects → DataSketch LSH → connect CONTENT_SIM edges |
| `build_content_sim_num_overlap()` | `networkbuilder.build_content_sim_num_overlap()` | IQR overlap ≥ 0.85 + inclusion dependency + DBSCAN clustering |
| `build_pkfk()` | `networkbuilder.build_pkfk_relation()` | For each node with cardinality > 0.7, find INCLUSION_DEP or CONTENT_SIM neighbors → add PKFK edge |

---

## 5  `aurum_v2/builder/analysis.py`

> **Legacy source**: `aurum_legacy/dataanalysis/dataanalysis.py` (module-level functions)

| v2 Stub Method | Legacy Function | What It Does |
|---|---|---|
| `compute_numeric_signature()` | `dataanalysis.get_num_dist()` | KDE fit → sample S points from distribution |
| `compute_text_signature()` | `dataanalysis.get_text_dist()` | TF-IDF → top S terms by weight |
| `overlap()` | `dataanalysis.overlap()` | Early-termination sorted-list overlap check |
| `compute_value_frequencies()` | `dataanalysis.get_column_frequencies()` | Value → frequency dict |
| `compute_overlap()` | `dataanalysis.compute_overlap()` | Builds frequency dicts, computes thresholds, calls `overlap()` |
| `compute_containment()` | `dataanalysis.compute_containment()` | Unidirectional containment ratio |
| `ks_test()` | `dataanalysis.ks_test()` | Kolmogorov-Smirnov test + threshold using numeric signatures |
| `cosine_text_sim()` | `dataanalysis.cosine_text_sim()` | Truncate to 4000 chars, TF-IDF vectorize, cosine threshold |
| `tf_idf_cosine_sim()` | `dataanalysis.tf_idf_relation()` | TF-IDF on doc pair → cosine similarity scalar |
| `pairwise_ks_matrix()` | `dataanalysis.pairwise_ks()` | Pairwise KS matrix across all numeric columns |
| `pairwise_text_matrix()` | `dataanalysis.pairwise_text()` | Pairwise TF-IDF cosine matrix across text columns |

---

## 6  `aurum_v2/dod/dod.py`

> **Legacy source**: `aurum_legacy/DoD/dod.py` (class `DoD`)

| v2 Stub Method | Legacy Method | What It Does |
|---|---|---|
| `individual_filters()` | `DoD.individual_filters()` | `search_exact_attribute` for ATTR, `search_content` for CELL |
| `joint_filters()` | `DoD.joint_filters()` | If cell empty → attr only; else intersect attr∩content DRS |
| `virtual_schema_iterative_search()` | `DoD.virtual_schema_iterative_search()` | 5-stage pipeline: joint_filters → group tables → joinable → is_materializable → materialize (~440 lines) |
| `_eager_candidate_exploration()` | Nested function inside `virtual_schema_iterative_search` | Greedy filter-coverage enumeration |
| `joinable()` | `DoD.joinable()` | Pairwise `paths()` → enumerate product → combine → dedup → filter covering → sort by joins |
| `transform_join_path_to_pair_hop()` | `DoD.transform_join_path_to_pair_hop()` | Converts linear path to (src,trg) pairs, removes same-table |
| `compute_join_graph_id()` | `DoD.compute_join_graph_id()` | Sum of `nid` for all hops |
| `is_join_graph_materializable()` | `DoD.is_join_graph_materializable()` | Per-hop: read CSV, apply filters, join, check > 0 rows |
| `materialize_join_graphs()` | `DoD.materialize_join_graphs()` | For each jg: materialize + format |
| `format_join_graph_into_nodes_edges()` | `DoD.format_join_graph_into_nodes_edges()` | Builds `{nodes: [...], edges: [...]}` dict |

---

## 7  `aurum_v2/dod/join_utils.py`

> **Legacy source**: `aurum_legacy/DoD/utils.py` (module-level functions)

| v2 Stub Method | Legacy Function | What It Does |
|---|---|---|
| `read_relation()` | `utils.get_dataframe()` | `pd.read_csv()` with caching |
| `read_relation_on_copy()` | `utils.get_dataframe_copy()` | Cache read, return `.copy()` |
| `read_relation_no_cache()` | `utils.get_dataframe_nocache()` | Simple `pd.read_csv()`, no cache |
| `apply_filter()` | `utils.get_dataframe_with_filter()` | Read copy, lowercase+strip attribute, filter rows where col contains value |
| `get_filter_columns()` | `utils.get_filter_columns()` | Extracts column names from filter set by FilterType |
| `normalize_key()` | `utils.normalize_for_join_spec()` | `s.lower().strip()` |
| `join_ab_on_key()` | `utils.join_ab_on_key()` | `pd.merge()` with optional key normalization |
| `estimate_row_size()` | `utils.estimate_row_size()` | Memory usage / rows → per-row byte estimate |
| `estimate_join_memory()` | `utils.estimate_join_memory()` | Estimated rows × row_size vs memory_limit |
| `join_ab_on_key_optimizer()` | `utils.join_ab_on_key_optimizer()` | Chunked join with memory estimation, 3-min timeout, spill-to-disk |
| `_build_tree()` | Nested function inside `utils.materialize_join_graph()` | Builds parent-child tree from join hops |
| `_fields_for_hop()` | Nested function inside `utils.materialize_join_graph()` | Resolves field names for (l_table, r_table) pair |
| `materialize_join_graph()` | `utils.materialize_join_graph()` | Tree-fold: build in-tree → iteratively join leaves → ancestor until root |
| `materialize_join_graph_filtered()` | `utils.materialize_join_graph_with_filters()` | Same tree-fold but applies `apply_filter()` before each join |
| `sample_by_key()` | `utils.sample_by_key()` | Deterministic hash-based ID sampling |

---

## 8  `aurum_v2/dod/view_analysis.py`

> **Legacy source**: `aurum_legacy/DoD/material_view_analysis.py` (module-level functions)

| v2 Stub Method | Legacy Function | What It Does |
|---|---|---|
| `most_likely_key()` | `material_view_analysis.most_likely_key()` | Unique ratio sorted descending, return top column |
| `unique_ratio()` | `material_view_analysis.unique_ratio()` | `nunique/len` per column |
| `curate()` | `material_view_analysis.curate()` | dropna → drop_duplicates → sort axes |
| `equivalent()` | `material_view_analysis.equivalent()` | Curate both, compare cardinality → schema → per-column values |
| `contained()` | `material_view_analysis.contained()` | Set difference of lowered values per column |
| `complementary()` | `material_view_analysis.complementary()` | Symmetric difference of most-likely-key value sets |
| `contradictory_value_check()` | `material_view_analysis.contradictory()` | Group by key, check nunique > 1 per group |
| `contradictory()` | Inline in `material_view_analysis.contradictory()` | Row-level conflict detection: missing keys, non-unique keys, conflicting pairs |

---

## 9  `aurum_v2/models/drs.py`

> **Legacy source**: `aurum_legacy/algebra.py` (class `DRS`)

| v2 Stub Method | Legacy Method | What It Does |
|---|---|---|
| `__iter__()` | `DRS.__iter__()` | FIELDS mode: iterate data; TABLE mode: lazy-init table_view |
| `absorb()` | `DRS.absorb()` | Replace data, reset table_view/indices/mode/ranking |
| `absorb_provenance()` | `DRS.absorb_provenance()` | Merge provenance graphs |
| `intersection()` | `DRS.intersection()` | TABLE mode: match on source; FIELDS mode: set intersection |
| `union()` | `DRS.union()` | Set union + composed provenance |
| `difference()` | `DRS.difference()` | Set difference + composed provenance |
| `why()` | `DRS.why()` | Delegates to `Provenance.why()` |
| `how()` | `DRS.how()` | Delegates to `Provenance.how()` |
| `paths()` | `DRS.paths()` | Calls `Provenance.paths()` |
| `why_provenance()` | `DRS.why_provenance()` | Find hit by nid, call `Provenance.why` |
| `how_provenance()` | `DRS.how_provenance()` | Find hit by nid, call `Provenance.how` |
| `set_ranking()` | `DRS.set_ranking()` | Store ranking function |
| `rank_certainty()` | `DRS.rank_certainty()` | Sort by descending certainty score |
| `rank_coverage()` | `DRS.rank_coverage()` | Sort by descending coverage score |
| `to_json()` | `DRS.__repr__()` override | Groups fields under sources, converts prov edges to dict |
| `print_tables()` | `DRS.print_tables()` | Set table mode, iterate+print, restore |
| `print_columns()` | `DRS.print_columns()` | Set fields mode, iterate+print (deduped), restore |
| `rank_by_table()` | `DRS.rank_per_table()` | Aggregates by table (certainty sum or coverage bitset union) |

---

## 10  `aurum_v2/models/provenance.py`

> **Legacy source**: `aurum_legacy/algebra.py` (class `Provenance`)

| v2 Stub Method | Legacy Method | What It Does |
|---|---|---|
| `record()` | `Provenance.record()` | Type-dispatch: NONE→skip, SEARCH→standalone, ALGEBRA→synthetic origin+edges, else→hit+edges |
| `identify_leafs_and_heads()` | `Provenance.identify_leafs_and_heads()` | Iterate nodes; no predecessors→leaf; no successors→head |
| `why()` | `Provenance.why()` | `all_simple_paths` for all leafs |
| `how()` | `Provenance.how()` | For each head, call `all_simple_paths` |
| `paths()` | `Provenance.paths()` | If a in leafs→paths to heads; if a in heads→paths from leafs; else→stitch |
| `explain()` | `Provenance.explain()` | Traverse pairs, format as human-readable string |

---

## 11  `aurum_v2/profiler/column_profiler.py`

> **Legacy source**: Java `ddprofiler/src/main/java/` — **NO Python legacy exists**
> Must be reimplemented from Java source.

| v2 Stub Method | Legacy Java Class | What It Does |
|---|---|---|
| `detect_column_type()` | `PreAnalyzer.readRows()` | >50% parseable as float → `"N"`, else `"T"` |
| `compute_kmin_hash()` | `KMinHash` class | Polynomial rolling hash with MERSENNE_PRIME, k=512 permutations |
| `compute_cardinality()` | `CardinalityAnalyzer` | Legacy used HyperLogLog; v2 uses Python `set()` |
| `compute_numeric_stats()` | `Range` + `RangeAnalyzer` | min/max/avg/median/iqr |
| `compute_entities()` | `EntityAnalyzer` | Legacy: OpenNLP; v2 plans spaCy |
| `profile_column()` | `Worker` pipeline | Combines type detection + cardinality + minhash/stats/NER |
| `create_es_indices()` | `NativeElasticStore.initStore()` | Creates `profile` + `text` indices with mappings |
| `run()` / `profile_all()` | `Conductor` + `Main` orchestration | Iterates sources, dispatches to Workers |
| `index_profile()` | `NativeElasticStore` bulk indexing | Bulk-indexes profile + text docs |

---

## 12  Functions Missing from v2 Entirely

These exist in the legacy but have no v2 stub. Only items useful for the agent use case are listed:

| Legacy File | Function | Purpose | Port? |
|---|---|---|---|
| `modelstore/elasticstore.py` | `search_keywords_fuzzily()` | ES fuzzy match with `"fuzziness": "AUTO"` | **Yes** — useful search variant |
| `DoD/dod.py` | `rank_join_graphs_by_key_likelihood()` | Score join graphs by key quality | **Yes** — improves DoD output ranking |
| `DoD/dod.py` | `rank_fields_in_join_path()` | Per-hop field ranking | **Yes** — companion to above |
| `DoD/dod.py` | `get_paths_for_tables()` | Resolve filesystem paths for table nids | **Yes** — helper used by DoD pipeline |
| `modelstore/elasticstore.py` | `get_all_fields_with(attr)` | Generic scroll with arbitrary attribute filter | **Maybe** — utility for extensibility |
| `knowledgerepr/fieldnetwork.py` | `get_degree_for()` | Top-k nodes by graph degree | **Maybe** — diagnostics |

---

## 13  Implementation Priority (What to Port First)

For the AI agent data-discovery use case, implement in this order:

### Wave 1 — Core Infrastructure (agent can't start without these)

| Priority | File | Effort | Depends On |
|---|---|---|---|
| **P0** | `profiler/column_profiler.py` | Large | Java source (no Python ref) |
| **P0** | `store/elastic_store.py` | Medium | `modelstore/elasticstore.py` |
| **P0** | `graph/field_network.py` | Medium | `knowledgerepr/fieldnetwork.py` |
| **P0** | `models/drs.py` | Medium | `algebra.py` (DRS class) |

### Wave 2 — Query Engine (agent can search but not join)

| Priority | File | Effort | Depends On |
|---|---|---|---|
| **P1** | `discovery/algebra.py` | Medium | `algebra.py` (Algebra class) |
| **P1** | `builder/network_builder.py` | Large | `knowledgerepr/networkbuilder.py` |
| **P1** | `builder/analysis.py` | Medium | `dataanalysis/dataanalysis.py` |

### Wave 3 — Join & Materialization (agent can answer multi-hop questions)

| Priority | File | Effort | Depends On |
|---|---|---|---|
| **P2** | `dod/join_utils.py` | Medium | `DoD/utils.py` |
| **P2** | `dod/dod.py` | Large | `DoD/dod.py` |
| **P2** | `dod/view_analysis.py` | Small | `DoD/material_view_analysis.py` |

### Wave 4 — Polish (nice to have)

| Priority | File | Effort | Depends On |
|---|---|---|---|
| **P3** | `models/provenance.py` | Small | `algebra.py` (Provenance class) |
| **P3** | Builder `coordinator.py` | — | Already orchestrated, just needs sub-functions |
