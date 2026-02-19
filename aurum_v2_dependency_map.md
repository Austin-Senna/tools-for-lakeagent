# Aurum Legacy → v2 Dependency Map

> Complete mapping of every `raise NotImplementedError` stub in `aurum_v2/` to
> the legacy file and function that contains the real implementation.
>
> **Last updated:** 2026-02-18

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

| v2 Method | Legacy Method | Status | Notes |
|---|---|---|---|
| `get_all_fields()` | `StoreHandler.get_all_fields_of_source()` | ✅ | ES scroll over `profile` index; yields (db, source, field, nid) |
| `get_all_fields_name()` | `StoreHandler.get_all_fields_name()` | ✅ | Added in bug-fix pass |
| `get_all_fields_of_source()` | `StoreHandler.get_all_fields_of_source()` | ✅ | Filtered by `sourceName` |
| `search_keywords()` | `StoreHandler.search_keywords()` | ✅ | Fuzzy ES search against `text` or `profile` index |
| `exact_search_keywords()` | `StoreHandler.exact_search_keywords()` | ✅ | ES `term` query variant |
| `bulk_insert_profiles()` | (new) | ✅ | ES `helpers.bulk` insertion |
| `get_profile()` | `StoreHandler.peek_values()` | ✅ | Point query on `profile` index |
| `suggest_field_names()` | `StoreHandler.suggest_field_names()` | ✅ | ES completion suggester |
| `StoreHandler` alias | — | ✅ | Backward-compat alias for api.py import |

---

## 2  `aurum_v2/discovery/algebra.py`

> **Legacy source**: `aurum_legacy/algebra.py` (class `Algebra`)

| v2 Method | Legacy Method | Status | Notes |
|---|---|---|---|
| `search()` | `Algebra.search()` | ✅ | Calls `store.search_keywords()`, wraps hits in DRS |
| `exact_search()` | `Algebra.exact_search()` | ✅ | Same pattern with `exact_search_keywords()` |
| `search_content()` | `Algebra.search_content()` | ✅ | Convenience wrapper |
| `search_attribute()` | `Algebra.search_attribute()` | ✅ | Convenience wrapper |
| `search_exact_attribute()` | `Algebra.search_exact_attribute()` | ✅ | Convenience wrapper |
| `search_table()` | `Algebra.search_table()` | ✅ | Convenience wrapper |
| `_neighbor_search()` | `Algebra.__neighbor_search()` | ✅ | General→DRS, provenance carrier, expand table-mode, iterate neighbors |
| `content_similar_to()` | `Algebra.content_similar_to()` | ✅ | Wraps _neighbor_search(CONTENT_SIM) |
| `schema_similar_to()` | `Algebra.schema_similar_to()` | ✅ | Wraps _neighbor_search(SCHEMA_SIM) |
| `pkfk_of()` | `Algebra.pkfk_of()` | ✅ | Wraps _neighbor_search(PKFK) |
| `traverse()` | `Algebra.__traverse()` | ✅ | BFS with max_hops, proper frontier tracking |
| `paths()` | `Algebra.paths()` | ✅ | Cartesian product → find_path_hit/find_path_table |
| `intersection()` | `Algebra.intersection()` | ✅ | Delegates to DRS.intersection() |
| `union()` | `Algebra.union()` | ✅ | Delegates to DRS.union() |
| `difference()` | `Algebra.difference()` | ✅ | Delegates to DRS.set_difference() |
| `make_drs()` | `Algebra.make_drs()` | ✅ | Handles list input (union fold) or single |
| `drs_from_table_hit()` | `Algebra.drs_from_table_hit()` | ✅ | Gets all hits from table + provenance |
| `_general_to_drs()` | `Algebra._general_to_drs()` | ✅ | Type-dispatch: DRS/None/int/str/tuple/Hit → DRS |
| `_nid_to_hit()` | `Algebra._nid_to_hit()` | ✅ | nid → Hit via network.get_info_for() |
| `_node_to_hit()` | `Algebra._node_to_hit()` | ✅ | (db, source, field) → Hit |
| `_hit_to_drs()` | Inline in legacy | ✅ | Wraps Hit in DRS |
| `suggest_schema()` | `Algebra.suggest_schema()` | ⚪ | Not ported — low-value for agent |
| Metadata API | `Algebra.__annotate()`, `__md_search()`, etc. | ⚪ | Not needed — agent reasons semantically |

---

## 3  `aurum_v2/graph/field_network.py`

> **Legacy source**: `aurum_legacy/knowledgerepr/fieldnetwork.py` (class `FieldNetwork`)

| v2 Method | Legacy Method | Status | Notes |
|---|---|---|---|
| `init_meta_schema()` | `FieldNetwork.init_meta_schema()` | ✅ | Populates id_to_info, source_to_fields, adds nodes with cardinality |
| `add_relation()` | `FieldNetwork.add_field_relation()` | ✅ | Adds weighted edge to NetworkX graph under relation key |
| `get_info_for()` | `FieldNetwork.get_info_for()` | ✅ | nid → (db, source, field, type) |
| `neighbors_id()` | `FieldNetwork.neighbors_id()` | ✅ | Iterates G[nid], filters by relation key, builds Hit list → DRS |
| `find_path_hit()` | `FieldNetwork.find_path_hit()` | ✅ | DFS with max_hops; bug-fixed to use absorb() not absorb_provenance() |
| `find_path_table()` | `FieldNetwork.find_path_table()` | ✅ | Table-level DFS with table→fields expansion; bug-fixed |
| `enumerate_relation()` | `FieldNetwork.enumerate_relation()` | ✅ | Iterates all node pairs with a given relation type |
| `get_hits_from_table()` | `FieldNetwork.get_hits_from_table()` | ✅ | Returns all Hits for a table name |
| `get_cardinality()` | `FieldNetwork.get_cardinality()` | ✅ | Node attribute lookup |
| `fields_degree()` | `FieldNetwork.fields_degree()` | ✅ | Top-k nodes by degree |
| `graph_order()` | `FieldNetwork.graph_order()` | ✅ | Number of nodes |
| `get_number_tables()` | (derived) | ✅ | len(source_to_fields) |
| `serialize()` | `FieldNetwork.serialize()` | ✅ | Pickles graph + dicts via io_utils |
| `deserialize_network()` | `FieldNetwork.deserialize()` | ✅ | Unpickle, reconstruct FieldNetwork |
| `iterate_ids()` / `iterate_values()` | `FieldNetwork.iterate_ids()` | ✅ | Generators yielding (db, source, field, type) |
| `md_neighbors_id()` | `FieldNetwork.md_neighbors_id()` | ⚪ | Metadata-relation traversal — not needed for agent |

---

## 4  `aurum_v2/builder/network_builder.py`

> **Legacy source**: `aurum_legacy/knowledgerepr/networkbuilder.py` (module-level functions)

| v2 Method | Legacy Function | Status | Notes |
|---|---|---|---|
| `build_schema_sim_relation()` | `networkbuilder.build_schema_sim_relation()` | ✅ | TF-IDF on field names → NearPy LSH → SCHEMA_SIM edges. Bug-fixed: cached dense vectors. |
| `build_content_sim_mh_text()` | `networkbuilder.build_content_sim_mh_text()` | ✅ | MinHash objects → DataSketch LSH → CONTENT_SIM edges |
| `build_content_sim_num_overlap()` | `networkbuilder.build_content_sim_num_overlap()` | ✅ | IQR overlap + DBSCAN clustering. Bug-fixed: early break optimization. |
| `build_pkfk_relation()` | `networkbuilder.build_pkfk_relation()` | ✅ | Cardinality > threshold → INCLUSION_DEP/CONTENT_SIM neighbors → PKFK edge |

---

## 5  `aurum_v2/builder/analysis.py`

> **Legacy source**: `aurum_legacy/dataanalysis/dataanalysis.py` (module-level functions)
>
> **Status**: 🔴 ALL 13 FUNCTIONS ARE STUBS. Signatures + docstrings are correct; bodies all `raise NotImplementedError`.

| v2 Stub Method | Legacy Function | Status |
|---|---|---|
| `get_tfidf_docs()` | `dataanalysis.get_tfidf_docs()` | 🔴 |
| `cosine_similarity_matrix()` | `dataanalysis.cosine_similarity_matrix()` | 🔴 |
| `build_dict_values()` | `dataanalysis.build_dict_values()` | 🔴 |
| `compute_overlap()` | `dataanalysis.compute_overlap()` | 🔴 |
| `compute_overlap_of_columns()` | `dataanalysis.compute_overlap_of_columns()` | 🔴 |
| `compare_num_columns_dist_ks()` | `dataanalysis.compare_num_columns_dist_ks()` | 🔴 |
| `compare_pair_num_columns()` | `dataanalysis.compare_pair_num_columns()` | 🔴 |
| `compare_pair_text_columns()` | `dataanalysis.compare_pair_text_columns()` | 🔴 |
| `compare_text_columns_cosine()` | `dataanalysis.compare_text_columns_dist()` | 🔴 |
| `get_numerical_signature()` | `dataanalysis.get_numerical_signature()` | 🔴 |
| `get_textual_signature()` | `dataanalysis.get_textual_signature()` | 🔴 |
| `get_sim_matrix_numerical()` | `dataanalysis.get_sim_matrix_numerical()` | 🔴 |
| `get_sim_matrix_text()` | `dataanalysis.get_sim_matrix_text()` | 🔴 |

> **Note**: `network_builder.py` is fully implemented but does NOT currently call these analysis.py functions — it inlines its own TF-IDF/MinHash/IQR logic. These functions are for future DoD-level column comparison and diagnostics.

---

## 6  `aurum_v2/dod/dod.py`

> **Legacy source**: `aurum_legacy/DoD/dod.py` (class `DoD`)

| v2 Method | Legacy Method | Status | Notes |
|---|---|---|---|
| `FilterType` enum | `FilterType` | ✅ | ATTR / CELL |
| `ViewSearchPredicate` | `FilterItem` | ✅ | NamedTuple with filter, col_name, keyword |
| `individual_filters()` | `DoD.individual_filters()` | 🔴 | `search_exact_attribute` for ATTR, `search_content` for CELL |
| `joint_filters()` | `DoD.joint_filters()` | 🔴 | If cell empty → attr only; else intersect attr∩content DRS |
| `virtual_schema_iterative_search()` | `DoD.virtual_schema_iterative_search()` | 🔴 | 5-stage pipeline (~440 lines in legacy) |
| `_eager_candidate_exploration()` | Nested in `virtual_schema_iterative_search` | 🔴 | Greedy filter-coverage enumeration |
| `joinable()` | `DoD.joinable()` | 🔴 | Pairwise paths → product → dedup → sort by joins |
| `transform_join_path_to_pair_hop()` | `DoD.transform_join_path_to_pair_hop()` | 🔴 | Linear path → (src,trg) pairs |
| `compute_join_graph_id()` | `DoD.compute_join_graph_id()` | 🔴 | Sum of nid for all hops |
| `is_join_graph_materializable()` | `DoD.is_join_graph_materializable()` | 🔴 | Per-hop CSV read + filter + join + row check |
| `materialize_join_graphs()` | `DoD.materialize_join_graphs()` | 🔴 | For each jg: materialize + format |
| `format_join_graph_into_nodes_edges()` | `DoD.format_join_graph_into_nodes_edges()` | 🔴 | Builds `{nodes, edges}` dict |
| `rank_join_graphs_by_key()` | Module-level | 🔴 | Score join graphs by key quality |
| `rank_fields_in_join_path()` | Module-level | 🔴 | Per-hop field ranking |
| `get_paths_for_tables()` | Module-level | 🔴 | Resolve filesystem paths for table nids |

---

## 7  `aurum_v2/dod/join_utils.py`

> **Legacy source**: `aurum_legacy/DoD/utils.py` + `aurum_legacy/DoD/data_processing_utils.py`

| v2 Method | Legacy Function | Status | Notes |
|---|---|---|---|
| `InTreeNode` class | `InTreeNode` | ✅ | Identical to legacy |
| `configure_csv_separator()` | Module state | ✅ | Sets `SEP` / `LINES_TO_READ` globals |
| `read_relation()` | `utils.get_dataframe()` | 🔴 | `pd.read_csv()` with caching |
| `read_relation_on_copy()` | `utils.get_dataframe_copy()` | 🔴 | Cache read, return `.copy()` |
| `read_relation_no_cache()` | `utils.get_dataframe_nocache()` | 🔴 | Simple `pd.read_csv()`, no cache |
| `apply_filter()` | `utils.get_dataframe_with_filter()` | 🔴 | Read + lowercase/strip + filter rows |
| `get_filter_columns()` | `utils.get_filter_columns()` | 🔴 | Extract column names by FilterType |
| `normalize_key()` | `utils.normalize_for_join_spec()` | 🔴 | `s.lower().strip()` |
| `join_ab_on_key()` | `utils.join_ab_on_key()` | 🔴 | `pd.merge()` + key normalization |
| `estimate_row_size()` | `utils.estimate_row_size()` | 🔴 | Memory usage / rows |
| `estimate_join_memory()` | `utils.estimate_join_memory()` | 🔴 | Estimated rows × row_size vs limit |
| `join_ab_on_key_optimizer()` | `utils.join_ab_on_key_optimizer()` | 🔴 | Chunked join, 3-min timeout, spill-to-disk |
| `_build_tree()` | Nested in `materialize_join_graph()` | 🔴 | Parent-child tree from join hops |
| `_fields_for_hop()` | Nested in `materialize_join_graph()` | 🔴 | Resolve field names for (l,r) pair |
| `materialize_join_graph()` | `utils.materialize_join_graph()` | 🔴 | Tree-fold: build in-tree → join leaves → root |
| `materialize_join_graph_filtered()` | `utils.materialize_join_graph_with_filters()` | 🔴 | Same + `apply_filter()` before each join |
| `sample_by_key()` | `utils.sample_by_key()` | 🔴 | Deterministic hash-based ID sampling |

---

## 8  `aurum_v2/dod/view_analysis.py`

> **Legacy source**: `aurum_legacy/DoD/material_view_analysis.py` (module-level functions)

| v2 Method | Legacy Function | Status | Notes |
|---|---|---|---|
| `ViewClass` enum | `ViewClass` | ✅ | EQUIVALENT / CONTAINED / COMPLEMENTARY / CONTRADICTORY |
| `most_likely_key()` | `most_likely_key()` | 🔴 | Unique ratio sorted desc, return top column |
| `unique_ratio()` | `unique_ratio()` | 🔴 | `nunique/len` per column |
| `curate()` | `curate()` | 🔴 | dropna → drop_duplicates → sort axes |
| `equivalent()` | `equivalent()` | 🔴 | Curate both, compare cardinality → schema → values |
| `contained()` | `contained()` | 🔴 | Set difference of lowered values per column |
| `complementary()` | `complementary()` | 🔴 | Symmetric diff of most-likely-key values |
| `contradictory_value_check()` | `contradictory()` | 🔴 | Group by key, check nunique > 1 |
| `contradictory()` | `contradictory()` | 🔴 | Row-level conflict detection |

---

## 9  `aurum_v2/models/drs.py`

> **Legacy source**: `aurum_legacy/api/apiutils.py` (class `DRS`)

| v2 Method | Legacy Method | Status | Notes |
|---|---|---|---|
| `__iter__()` / `__next__()` | `DRS.__iter__()` | ✅ | Bug-fixed: separate `_DRSIterator` class for safe nested iteration |
| `to_dict()` | `DRS.__dict__()` | ✅ | Bug-fixed: uses `_asdict()` not `asdict()` |
| `absorb()` | `DRS.absorb()` | ✅ | Set union + provenance merge |
| `absorb_provenance()` | `DRS.absorb_provenance()` | ✅ | Merge provenance graphs via nx.compose() |
| `intersection()` | `DRS.intersection()` | ✅ | Bug-fixed: TABLE mode now keeps all columns per table |
| `union()` | `DRS.union()` | ✅ | Set union + composed provenance |
| `set_difference()` | `DRS.set_difference()` | ✅ | Set diff + composed provenance |
| `why()` | `DRS.why()` | ✅ | Delegates to Provenance |
| `how()` | `DRS.how()` | ✅ | Delegates to Provenance |
| `paths()` | `DRS.paths()` | ✅ | Delegates to Provenance |
| `_compute_certainty_scores()` | `DRS._compute_certainty_scores()` | ✅ | Bug-fixed: per-element visited sets |
| `_compute_coverage_scores()` | `DRS._compute_coverage_scores()` | ✅ | Bitarray-based coverage |
| `rank_certainty()` | `DRS.rank_certainty()` | ✅ | Sort by descending certainty |
| `rank_coverage()` | `DRS.rank_coverage()` | ✅ | Sort by descending coverage |
| `print_tables()` | `DRS.print_tables()` | ✅ | Mode save/restore |
| `print_columns()` | `DRS.print_columns()` | ✅ | Deduped iteration |
| `pretty_print_columns()` | `DRS.pretty_print_columns()` | ✅ | Bug-fixed: added seen-set dedup |
| `visualize_provenance()` | `DRS.visualize_provenance()` | 🔴 | Matplotlib, low priority |
| `print_tables_with_scores()` | `DRS.print_tables_with_scores()` | 🔴 | Display helper, low priority |
| `print_columns_with_scores()` | `DRS.print_columns_with_scores()` | 🔴 | Display helper, low priority |

---

## 10  `aurum_v2/models/provenance.py`

> **Legacy source**: `aurum_legacy/algebra.py` (class `Provenance`)

| v2 Method | Legacy Method | Status | Notes |
|---|---|---|---|
| `record()` | `Provenance.record()` | ✅ | Type-dispatch: NONE→skip, SEARCH→standalone, ALGEBRA→synthetic origin+edges, else→hit+edges. Bug-fixed: `Hit` import added |
| `identify_leafs_and_heads()` | `Provenance.identify_leafs_and_heads()` | ✅ | Iterate nodes; no predecessors→leaf; no successors→head |
| `why()` | `Provenance.why()` | ✅ | `all_simple_paths` for all leafs |
| `how()` | `Provenance.how()` | ✅ | For each head, call `all_simple_paths` |
| `paths()` | `Provenance.paths()` | ✅ | If a in leafs→paths to heads; if a in heads→paths from leafs; else→stitch |
| `explain()` | `Provenance.explain()` | ✅ | Traverse pairs, format as human-readable string |

---

## 11  `aurum_v2/profiler/column_profiler.py`

> **Legacy source**: Java `ddprofiler/src/main/java/` — **NO Python legacy exists**
> Reimplemented from Java source in prior sessions.

| v2 Method | Legacy Java Class | Status | Notes |
|---|---|---|---|
| `detect_column_type()` | `PreAnalyzer.readRows()` | ✅ | >50% parseable as float → `"N"`, else `"T"` |
| `compute_kmin_hash()` | `KMinHash` class | ✅ | Polynomial rolling hash, MERSENNE_PRIME, k=512 |
| `compute_cardinality()` | `CardinalityAnalyzer` | ✅ | Python `set()` (simpler than legacy HyperLogLog) |
| `compute_numeric_stats()` | `Range` + `RangeAnalyzer` | ✅ | min/max/avg/median/iqr via numpy |
| `compute_entities()` | `EntityAnalyzer` | ✅ | Uses spaCy NER (legacy used OpenNLP) |
| `profile_column()` | `Worker` pipeline | ✅ | Combines type + cardinality + minhash/stats/NER |
| `create_es_indices()` | `NativeElasticStore.initStore()` | ✅ | Creates `profile` + `text` indices with mappings |
| `run()` / `profile_all()` | `Conductor` + `Main` | ✅ | Iterates sources, dispatches workers |
| `index_profile()` | `NativeElasticStore` bulk | ✅ | Bulk-indexes profile + text docs |

> **Also**: `aurum_v2/profiler/source_readers.py` — ✅ fully implemented (CSV + JSON + DB readers)

---

## 12  Functions Missing from v2 Entirely

These exist in the legacy but have no v2 stub. Only items useful for the agent use case are listed:

| Legacy File | Function | Purpose | Port? |
|---|---|---|---|
| `modelstore/elasticstore.py` | `search_keywords_fuzzily()` | ES fuzzy match with `"fuzziness": "AUTO"` | **Yes** — useful search variant |
| `modelstore/elasticstore.py` | `get_all_fields_with(attr)` | Generic scroll with arbitrary attribute filter | **Maybe** — utility for extensibility |
| `algebra.py` | `suggest_schema()` | Feed columns through traverse→union | ⚪ Low value for agent |
| `ddapi.py` | `keywords_search()` / batch variants | Batch keyword, schema, table name searches | **Maybe** — trivially built from existing `search()` |
| `ddapi.py` | `entity_search()` | Search by `KW_ENTITIES` | **Maybe** — depends on entity profiling |
| `ddapi.py` | `inclusion_dependency_to()` | `Relation.INCLUSION_DEPENDENCY` neighbor search | **Maybe** — _neighbor_search handles it already |

> **Note**: DoD ranking functions (`rank_join_graphs_by_key_likelihood`, `rank_fields_in_join_path`, `get_paths_for_tables`) now have stubs in `dod/dod.py` (Section 6).

---

## 13  Implementation Priority

### Wave 1 — Core Infrastructure ✅ DONE

| File | Status |
|---|---|
| `profiler/column_profiler.py` | ✅ All 9 methods implemented |
| `profiler/source_readers.py` | ✅ CSV + JSON + DB readers |
| `store/elastic_store.py` | ✅ All 9 methods + StoreHandler alias |
| `store/duck_store.py` | ✅ DuckDB alternative store |
| `graph/field_network.py` | ✅ All 20+ methods |
| `models/drs.py` | ✅ 17/20 methods (3 display stubs, low priority) |
| `models/hit.py` | ✅ NamedTuple |
| `models/relation.py` | ✅ Enum |
| `models/annotation.py` | ✅ Dataclass |

### Wave 2 — Query Engine ✅ DONE

| File | Status |
|---|---|
| `discovery/algebra.py` | ✅ All methods (search, traverse, set ops, convenience wrappers) |
| `discovery/api.py` | ✅ `init_system()` + `Helper` + `API(Algebra)` |
| `builder/network_builder.py` | ✅ All 4 build functions (TF-IDF, MinHash, content sim, schema sim). Bug-fixed. |
| `builder/coordinator.py` | ✅ Orchestration |
| `models/provenance.py` | ✅ All 6 methods (record, why, how, paths, explain, identify_leafs_and_heads) |
| `config.py` | ✅ |

### Wave 3 — Statistical Analysis (needed for DoD column comparison)

| File | Effort | Status |
|---|---|---|
| `builder/analysis.py` | Medium | 🔴 13 stubs. Not blocking — `network_builder.py` inlines its own logic. |

### Wave 4 — Join & Materialization (agent can answer multi-hop questions)

| File | Effort | Status |
|---|---|---|
| `dod/join_utils.py` | Medium | 🔴 15 stubs (InTreeNode ✅) |
| `dod/dod.py` | Large | 🔴 15 stubs (enums ✅) |
| `dod/view_analysis.py` | Small | 🔴 8 stubs (ViewClass enum ✅) |

### Summary: 52 stubs remain, all in Waves 3-4

The discovery/search pipeline is **fully operational**. Remaining work is the materialization layer (DoD) and standalone analysis functions.
