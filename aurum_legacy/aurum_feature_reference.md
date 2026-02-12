# Aurum Feature Reference — Debugging Cross-Reference

> **Purpose:** Per-file inventory of every feature in the legacy Aurum codebase.  
> Use this while debugging the new **LakeAgent** project.  
>
> **Legend:**
> - 🔴 **IMPORTANT** — Core algorithm / ported to LakeAgent / you will likely need to cross-reference this when debugging.
> - 🟡 **MODERATE** — Supporting logic that affects behavior but is not a core algorithm.
> - ⚪ **NOT IMPORTANT** — Legacy infrastructure, deprecated, test-only, or debug-only code.

---

## Table of Contents

1. [api/apiutils.py](#apiapiutilspy) — Hit, DRS, Provenance, Relation, OP
2. [knowledgerepr/fieldnetwork.py](#knowledgerepfieldnetworkpy) — FieldNetwork graph wrapper
3. [knowledgerepr/networkbuilder.py](#knowledgerepnetworkbuilderpy) — Edge-building algorithms
4. [networkbuildercoordinator.py](#networkbuildercoordinatorpy) — Pipeline orchestrator
5. [modelstore/elasticstore.py](#modelstoreelasticstorepy) — Elasticsearch client
6. [algebra.py](#algebrapy) — Newer query algebra API
7. [ddapi.py](#ddapipy) — Older query API
8. [DoD/dod.py](#doddodpy) — Data-on-Demand view search
9. [DoD/data_processing_utils.py](#doddata_processing_utilspy) — Join engine & CSV I/O
10. [DoD/material_view_analysis.py](#dodmaterial_view_analysispy) — View comparison
11. [DoD/utils.py](#dodutilspy) — FilterType enum
12. [DoD/experimental.py](#dodexperimentalpy) — Exhaustive search variant
13. [dataanalysis/dataanalysis.py](#dataanalysisdataanalysispy) — Column comparison analytics
14. [dataanalysis/nlp_utils.py](#dataanalysisnlp_utilspy) — Text preprocessing
15. [config.py](#configpy) — Global configuration
16. [ontomatch/ss_api.py](#ontomatchss_apipy) — Semantic schema matching API
17. [ontomatch/ss_utils.py](#ontomatchss_utilspy) — Semantic similarity utilities
18. [ontomatch/matcher_lib.py](#ontomatchmatcher_libpy) — Matching library
19. [ontomatch/glove_api.py](#ontomatchglove_apipy) — GloVe embeddings
20. [ontomatch/onto_parser.py](#ontomatchonto_parserpy) — Ontology parser
21. [ontomatch/no_matcher.py](#ontomatchno_matcherpy) — Wikipedia text matcher
22. [knowledgerepr/lite_graph.py](#knowledgereprlite_graphpy) — Bitarray graph
23. [inputoutput/inputoutput.py](#inputoutputinputoutputpy) — Pickle serialization
24. [api/annotation.py](#apiannotationpy) — Metadata annotation types
25. [api/reporting.py](#apireportingpy) — Network statistics
26. [sugar.py](#sugarpy) — Convenience REPL shortcuts
27. [main.py](#mainpy) — System init & IPython shell
28. [run_dod.py](#run_dodpy) — DoD CLI entry point
29. [server_config.py](#server_configpy) — Server path config
30. [server-api/app.py](#server-apiapppy) — Flask web API
31. [aurum_cli.py](#aurum_clipy) — Fire CLI wrapper
32. [export_network_2_neo4j.py](#export_network_2_neo4jpy) — Neo4j exporter

---

## api/apiutils.py
*904 lines — Core data structures used everywhere*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `compute_field_id(db, src, field)` | ~15 | 🔴 **IMPORTANT** | `result_set.py` | Uses `binascii.crc32` to produce integer IDs. Your new code may use a different hash; if IDs mismatch, nothing links. |
| 2 | `Hit` namedtuple | ~25 | 🔴 **IMPORTANT** | `result_set.py` | `(nid, db_name, source_name, field_name, score)` — hash on `int(nid)`. Custom `__hash__` and `__eq__` drive all set operations and graph lookups. |
| 3 | `Relation` enum | ~50 | 🔴 **IMPORTANT** | `graph/relations.py` | SCHEMA=0, SCHEMA_SIM=1, CONTENT_SIM=2, ENTITY_SIM=3, PKFK=5, INCLUSION_DEPENDENCY=6, plus metadata relations 10-15. `.from_metadata()` maps ints. |
| 4 | `OP` enum | ~70 | 🟡 MODERATE | `result_set.py` | Operation provenance codes. Mirrors `Relation` values. Used in provenance graph edges. |
| 5 | `DRSMode` enum | ~80 | 🟡 MODERATE | `result_set.py` | FIELDS=0, TABLE=1. Controls iteration behavior of DRS. |
| 6 | `Operation` class | ~85 | 🟡 MODERATE | — | Wraps `op` + `params` for provenance tracking. |
| 7 | `Provenance` class | ~90-200 | 🟡 MODERATE | Simplified in LakeAgent | nx.MultiDiGraph-based DAG. Methods: `populate_provenance()`, `get_leafs_and_heads()`, `compute_paths_from_origin_to()`, `compute_all_paths()`, `compute_paths_with()`, `explain_path()`. |
| 8 | `DRS` class — core container | ~200-400 | 🔴 **IMPORTANT** | `discovery/result_set.py` | The main result object. Holds `self.data` (set of Hits), provenance graph, mode (field/table). Iterator switches between field-level and table-level output. |
| 9 | `DRS.absorb_provenance()` | ~350 | 🟡 MODERATE | — | Merges provenance graphs via `nx.compose()` with AND/OR edge annotations. |
| 10 | `DRS.absorb()` | ~370 | 🟡 MODERATE | — | Set union + provenance merge. |
| 11 | `DRS.intersection()` / `union()` / `set_difference()` | ~400-450 | 🔴 **IMPORTANT** | `result_set.py` | Set algebra on result sets. These drive the compositional query model. |
| 12 | `DRS.paths()` / `path(a)` | ~460 | 🟡 MODERATE | — | Provenance path enumeration. |
| 13 | `DRS.why(a)` / `how(a)` | ~500 | ⚪ NOT IMPORTANT | — | Provenance explanation — nice-to-have, not core. |
| 14 | `DRS._compute_certainty_scores()` | ~600 | 🟡 MODERATE | — | Recursive graph traversal to compute certainty ranking scores. |
| 15 | `DRS._compute_coverage_scores()` | ~650 | 🟡 MODERATE | — | Bitarray-based coverage scoring. |
| 16 | `DRS.rank_certainty()` / `rank_coverage()` | ~700 | 🟡 MODERATE | — | Sort data by scores. |
| 17 | `DRS.print_tables()` / `print_columns()` / `pretty_print_columns()` | ~750 | ⚪ NOT IMPORTANT | — | Display helpers. |
| 18 | `DRS.__dict__()` | ~850 | ⚪ NOT IMPORTANT | — | JSON serialization for Flask web API. |

---

## knowledgerepr/fieldnetwork.py
*482 lines — Central graph that stores the knowledge network*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `FieldNetwork.__init__(G, id_to_info, table_to_ids)` | ~30 | 🔴 **IMPORTANT** | `graph/field_network.py` | Wraps a `nx.MultiGraph` + two dicts. Everything queries through this. |
| 2 | `init_meta_schema(self, store)` | ~50 | 🔴 **IMPORTANT** | `graph/network_builder.py` | Populates graph nodes from ES (all fields with cardinality). This is the schema skeleton before edges. |
| 3 | `add_field()` / `add_fields()` / `add_relation()` | ~80 | 🔴 **IMPORTANT** | `field_network.py` | Graph mutation primitives. `add_relation` creates typed edges. |
| 4 | `iterate_ids()` / `iterate_ids_text()` / `iterate_values()` | ~120 | 🔴 **IMPORTANT** | `field_network.py` | Generators that yield `(db_name, source_name, field_name, data_type)` from graph nodes. Used everywhere during edge building. |
| 5 | `neighbors_id(hit, relation)` → DRS | ~180 | 🔴 **IMPORTANT** | `field_network.py` | Core traversal: given a Hit and relation type, return all neighbor Hits as a DRS. This is the fundamental graph query. |
| 6 | `md_neighbors_id()` | ~220 | ⚪ NOT IMPORTANT | — | Metadata-relation neighbor traversal. Only used with annotation system. |
| 7 | `find_path_hit(source, target, relation, max_hops)` | ~250 | 🔴 **IMPORTANT** | `field_network.py` | DFS path finding between two Hits with provenance assembly. Core to `paths()` in algebra. |
| 8 | `find_path_table(source, target, relation, max_hops)` | ~300 | 🔴 **IMPORTANT** | `field_network.py` | Table-level DFS with sibling tracking. Complex provenance assembly with `sources` list tracking same-table attributes. |
| 9 | `enumerate_relation(relation)` | ~380 | 🟡 MODERATE | — | Yields all edges of a given relation type. Used in reporting/stats. |
| 10 | `get_op_from_relation()` | ~400 | 🟡 MODERATE | — | Maps Relation enum → OP enum for provenance. |
| 11 | `fields_degree(topk)` | ~420 | ⚪ NOT IMPORTANT | — | Returns top-k nodes by degree. Diagnostic only. |
| 12 | `serialize_network()` / `deserialize_network()` | ~450 | 🟡 MODERATE | `field_network.py` | Pickle-based serde via `nx.write_gpickle`/`nx.read_gpickle`. Your new code replaces this. |
| 13 | `serialize_network_to_csv()` | ~440 | ⚪ NOT IMPORTANT | — | Debug CSV export. |

---

## knowledgerepr/networkbuilder.py
*672 lines — The algorithms that create graph edges (the "secret sauce")*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `LSHRandomProjectionsIndex` class | ~30 | 🔴 **IMPORTANT** | `graph/network_builder.py` (uses datasketch instead) | Wraps NearPy `RandomBinaryProjections`. Your code uses datasketch MinHashLSH instead—same concept, different library. |
| 2 | `build_schema_sim_relation(network, store)` | ~80 | 🔴 **IMPORTANT** | `network_builder.py::_build_schema_sim()` | TF-IDF on column names → LSH → SCHEMA_SIM edges. **Key thresholds:** uses NearPy LSH with RandomBinaryProjections (10 bits). |
| 3 | `build_schema_sim_relation_lsa()` | ~120 | ⚪ NOT IMPORTANT | — | LSA variant. Not called in coordinator. Dead code. |
| 4 | `build_entity_sim_relation()` | ~160 | ⚪ NOT IMPORTANT | — | TF-IDF on entities → ENTITY_SIM. Commented out in coordinator. |
| 5 | `build_content_sim_relation_text()` | ~200 | ⚪ NOT IMPORTANT | — | TF-IDF+LSH on text values. Superseded by MinHash variant. |
| 6 | `build_content_sim_mh_text(network, store)` | ~280 | 🔴 **IMPORTANT** | `network_builder.py::_build_content_sim_text()` | **MinHash LSH** on text column values. `threshold=0.7`, `num_perm=512`. Retrieves pre-computed minhash arrays from ES, queries LSH index, creates CONTENT_SIM edges. This is the active text similarity method. |
| 7 | `build_content_sim_relation_num_overlap_distr(network, store)` | ~340 | 🔴 **IMPORTANT** | `network_builder.py::_build_content_sim_numeric()` | **Numeric overlap detection.** Uses median ± IQR overlap (threshold=0.85) as primary check, then inclusion dependency (threshold=0.3), plus DBSCAN (eps=0.1) for single-point clusters. Creates CONTENT_SIM edges for numeric columns. |
| 8 | `build_content_sim_relation_num_overlap_distr_indexed()` | ~430 | ⚪ NOT IMPORTANT | — | Event-sweep variant. Incomplete/unused. |
| 9 | `build_content_sim_relation_num_double_clustering()` | ~480 | ⚪ NOT IMPORTANT | — | Experimental DBSCAN on median AND IQR. Not used. |
| 10 | `build_content_sim_relation_num()` | ~530 | ⚪ NOT IMPORTANT | — | Deprecated DBSCAN on raw features. |
| 11 | `build_pkfk_relation(network, store)` | ~580 | 🔴 **IMPORTANT** | `network_builder.py::_build_pkfk()` | **PK/FK detection.** Cardinality ratio > 0.7, plus neighbor cross-check. Creates PKFK edges. |
| 12 | `index_in_text_engine()` | ~50 | 🟡 MODERATE | — | Indexes TF-IDF vectors into NearPy engine. Internal to schema_sim. |
| 13 | `create_sim_graph_text()` | ~60 | 🟡 MODERATE | — | NearPy-based LSH neighbor search loop. Internal to edge builders. |
| 14 | `lsa_dimensionality_reduction()` | ~70 | ⚪ NOT IMPORTANT | — | TruncatedSVD to 1000 components. Only used in dead LSA paths. |

---

## networkbuildercoordinator.py
*~200 lines — Orchestrates the full network building pipeline*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `main(output_path)` | ~30 | 🔴 **IMPORTANT** | `network_builder.py::NetworkBuilder.build()` | Full pipeline: (1) init_meta_schema, (2) build_schema_sim, (3) entity_sim [commented out], (4) build_content_sim_mh_text, (5) build_content_sim_num_overlap_distr, (6) build_pkfk. Then serializes. The ordering and which functions are called is critical. |
| 2 | `plot_num()` | ~150 | ⚪ NOT IMPORTANT | — | Debug matplotlib visualization. |
| 3 | `test_content_sim_num()` | ~170 | ⚪ NOT IMPORTANT | — | Test harness for numeric similarity. |

---

## modelstore/elasticstore.py
*789 lines — Elasticsearch interface for all profile data*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `StoreHandler.__init__()` | ~30 | 🟡 MODERATE | LakeAgent uses Polars directly, no ES | Connects to ES at `localhost:9200`. |
| 2 | `KWType` enum | ~20 | 🔴 **IMPORTANT** | — | KW_CONTENT, KW_SCHEMA, KW_ENTITIES, KW_TABLE, KW_METADATA. Controls search scope. |
| 3 | `get_all_fields()` | ~80 | 🔴 **IMPORTANT** | `profiler/column_profiler.py` | Scrolls all ES profile docs → `(id, dbName, sourceName, columnName, totalValues, uniqueValues, dataType)`. This populates the graph skeleton. |
| 4 | `search_keywords(kw, kw_type, max_hits)` | ~120 | 🔴 **IMPORTANT** | `algebra.py::search()` | ES match query on text/profile/entities indices. Returns Hits. |
| 5 | `exact_search_keywords()` | ~160 | 🔴 **IMPORTANT** | `algebra.py::exact_search()` | Term query (exact match). |
| 6 | `fuzzy_keyword_match()` | ~190 | 🟡 MODERATE | — | Fuzzy match on text index. |
| 7 | `suggest_schema()` | ~210 | 🟡 MODERATE | — | ES completion suggester on `columnNameSuggest`. |
| 8 | `get_all_fields_text_signatures()` | ~250 | 🔴 **IMPORTANT** | `column_profiler.py` | Retrieves term vectors via ES `mtermvectors` API, filters by frequency (>3) and length (>3 chars). Used by TF-IDF edge builders. |
| 9 | `get_all_mh_text_signatures()` | ~350 | 🔴 **IMPORTANT** | `column_profiler.py` | Retrieves pre-computed minhash arrays for text columns. Used by `build_content_sim_mh_text`. |
| 10 | `get_all_fields_num_signatures()` | ~400 | 🔴 **IMPORTANT** | `column_profiler.py` | Retrieves `(median, iqr, minValue, maxValue)` for numeric columns. Used by numeric overlap builder. |
| 11 | `get_path_of(nid)` | ~70 | 🟡 MODERATE | — | Retrieves filesystem path for a data source given its nid. Used in DoD materialization. |
| 12 | `add_annotation()` / `add_comment()` / `add_tags()` | ~500 | ⚪ NOT IMPORTANT | — | Metadata CRUD. Annotation system. |
| 13 | `search_keywords_md()` / `get_metadata()` / `get_comments()` | ~550 | ⚪ NOT IMPORTANT | — | Metadata search. |
| 14 | `create_metadata_index()` / `delete_metadata_index()` | ~600 | ⚪ NOT IMPORTANT | — | ES index lifecycle management. |

---

## algebra.py
*614 lines — Newer compositional query API*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `Algebra.__init__(network, store_client)` | ~30 | 🔴 **IMPORTANT** | `discovery/algebra.py` | Takes the FieldNetwork + StoreHandler. |
| 2 | `search(kw, kw_type, max_results)` | ~50 | 🔴 **IMPORTANT** | `algebra.py::search()` | ES keyword search → DRS. |
| 3 | `exact_search()` | ~70 | 🔴 **IMPORTANT** | `algebra.py::exact_search()` | ES exact match → DRS. |
| 4 | `search_content()` / `search_attribute()` / `search_exact_attribute()` / `search_table()` | ~80-120 | 🔴 **IMPORTANT** | `algebra.py` | Convenience wrappers that set `KWType`. |
| 5 | `suggest_schema()` | ~130 | 🟡 MODERATE | — | ES completion suggester wrapper. |
| 6 | `__neighbor_search(input, relation)` | ~150 | 🔴 **IMPORTANT** | `algebra.py::neighbor_search()` | Core traversal: converts any input → DRS, iterates hits, gets neighbors by relation from FieldNetwork. Central to all similarity queries. |
| 7 | `content_similar_to()` / `schema_similar_to()` / `pkfk_of()` | ~200 | 🔴 **IMPORTANT** | `algebra.py` | Convenience wrappers for `__neighbor_search` with specific Relation types. |
| 8 | `paths(drs_a, drs_b, relation, max_hops, lean_search)` | ~250 | 🔴 **IMPORTANT** | `algebra.py::paths()` | Path finding between two DRS. Dispatches to `find_path_hit` or `find_path_table` depending on mode. `max_hops` default is 3. |
| 9 | `__traverse(a, primitive, max_hops)` | ~300 | 🟡 MODERATE | — | BFS traversal up to max_hops using a given primitive (e.g., `content_similar_to`). |
| 10 | `intersection()` / `union()` / `difference()` | ~350 | 🔴 **IMPORTANT** | `algebra.py` | Set algebra on DRS. These compose queries. |
| 11 | `make_drs(general_input)` / `_general_to_drs()` | ~400 | 🔴 **IMPORTANT** | `algebra.py` | Converts int/str/tuple/Hit/DRS → DRS. Handles table name lookup, nid lookup, etc. |
| 12 | `_hit_to_drs()` / `drs_from_table_hit()` | ~450 | 🟡 MODERATE | — | Expands a Hit to include all sibling columns from same table. |
| 13 | Metadata API (`annotate`, `add_comments`, `add_tags`, `md_search`) | ~500 | ⚪ NOT IMPORTANT | — | Hidden with `__` prefix. Annotation system. |
| 14 | `Helper` class | ~550 | ⚪ NOT IMPORTANT | — | `reverse_lookup()`, `get_path_nid()`, `help()`. Convenience. |
| 15 | `API(Algebra)` subclass | ~600 | ⚪ NOT IMPORTANT | — | Just a subclass alias. |

---

## ddapi.py
*620 lines — Original/older query API*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `DDAPI.__init__(network)` | ~30 | 🟡 MODERATE | — | Older API that takes only network (no store_client at init). |
| 2 | Seed methods: `drs_from_raw_field()`, `drs_from_hit()`, `drs_from_table()` | ~50-100 | 🟡 MODERATE | — | Create initial DRS from raw inputs. Superseded by `algebra.py::make_drs()`. |
| 3 | `keyword_search()` / `schema_name_search()` / `entity_search()` | ~120-200 | 🟡 MODERATE | — | Search primitives. Duplicate of algebra.py functionality. |
| 4 | `similar_schema_name_to()` / `similar_content_to()` / `pkfk_of()` | ~250-300 | 🟡 MODERATE | — | Neighbor search wrappers. Same as algebra.py. |
| 5 | `paths_between()` / `paths()` / `traverse()` | ~350-400 | 🟡 MODERATE | — | Path finding. Same logic as algebra.py. |
| 6 | `intersection()` / `union()` / `difference()` | ~420-460 | 🟡 MODERATE | — | Set algebra. Same as algebra.py. |
| 7 | `ResultFormatter.format_output_for_webclient()` | ~500 | ⚪ NOT IMPORTANT | — | HTML/JSON formatting for web UI. |
| 8 | `API(DDAPI)` subclass + `init_store()` | ~580 | ⚪ NOT IMPORTANT | — | Creates StoreHandler lazily. Legacy pattern. |

---

## DoD/dod.py
*958 lines — Data-on-Demand: the view search pipeline*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `DoD.__init__(network, store_client, csv_separator)` | ~30 | 🔴 **IMPORTANT** | `discovery/data_on_demand.py` | Creates an internal `API` and configures paths_cache. |
| 2 | `individual_filters(list_attributes, list_samples)` | ~60 | 🔴 **IMPORTANT** | `data_on_demand.py` | Searches for each attr and cell value separately via the algebra API. Returns `{(filter, FilterType): DRS}`. |
| 3 | `joint_filters(list_attributes, list_samples)` | ~100 | 🔴 **IMPORTANT** | `data_on_demand.py` | Combined attr+cell search with intersection on same-column pairs. |
| 4 | `virtual_schema_iterative_search(list_attributes, list_samples)` | ~150 | 🔴 **IMPORTANT** | `data_on_demand.py::search_views()` | **MAIN PIPELINE.** 5 stages: (1) `joint_filters` → search, (2) `eager_candidate_exploration` → greedy set cover, (3) `joinable` → find join paths, (4) `is_join_graph_materializable` → validate, (5) `materialize_join_graphs` → yield DataFrames. **This is the heart of DoD.** |
| 5 | `eager_candidate_exploration()` (inside virtual_schema) | ~200 | 🔴 **IMPORTANT** | `data_on_demand.py` | Greedy set cover: sorts tables by filter coverage, picks tables eagerly, yields candidate groups covering all filters. Uses nested generators. |
| 6 | `joinable(candidate_group)` | ~350 | 🔴 **IMPORTANT** | `data_on_demand.py` | For each pair of tables in group, finds all PKFK paths via `api.paths()`. Uses `itertools.product` to enumerate join graphs. Deduplicates via `compute_join_graph_id()`. |
| 7 | `transform_join_path_to_pair_hop()` | ~450 | 🟡 MODERATE | — | Converts a path list to `[(left, right)]` pairs, removing same-table hops. |
| 8 | `compute_join_graph_id()` | ~470 | 🟡 MODERATE | — | Hash-based deduplication of join graphs. |
| 9 | `format_join_graph_into_nodes_edges()` | ~490 | 🟡 MODERATE | — | Converts join graph to JSON-ready `{nodes, edges}` format for the UI. |
| 10 | `is_join_graph_materializable(join_graph, filters)` | ~520 | 🔴 **IMPORTANT** | `data_on_demand.py` | Validation: applies filters to each table, attempts joins hop-by-hop, verifies cardinality > 0. If any hop produces empty result, the graph is rejected. |
| 11 | `rank_materializable_join_graphs()` | ~700 | 🟡 MODERATE | — | Scores join graphs by key likelihood (uniqueness ratio). |
| 12 | `obtain_table_paths()` | ~750 | 🟡 MODERATE | — | Gets filesystem paths for table sources via `store_client.get_path_of()`. |
| 13 | `test_e2e()` | ~800 | ⚪ NOT IMPORTANT | — | End-to-end test harness. |

---

## DoD/data_processing_utils.py
*736 lines — Join execution engine and data I/O*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `read_relation(path, separator)` / `read_relation_on_copy()` | ~30 | 🔴 **IMPORTANT** | `materialization/join_engine.py` | Cached CSV reading with `pd.read_csv`. `read_relation_on_copy` returns `df.copy()` to avoid mutation. LakeAgent uses Polars instead. |
| 2 | `apply_filter(relation, attribute, cell_value)` | ~80 | 🔴 **IMPORTANT** | `join_engine.py` | Filters rows where `attribute == cell_value` (case-insensitive string comparison). |
| 3 | `find_key_for(relation, attribute, value)` | ~100 | 🟡 MODERATE | — | `SELECT key FROM relation WHERE attribute = value`. |
| 4 | `is_value_in_column(relation, attribute, value)` | ~120 | 🟡 MODERATE | — | Boolean existence check. |
| 5 | `obtain_attributes_to_project(filters)` | ~140 | 🟡 MODERATE | — | Extracts ATTR-type filter names for final column projection. |
| 6 | `project(relation, attrs)` | ~160 | 🟡 MODERATE | — | Column projection (select columns). |
| 7 | `estimate_output_row_size()` | ~180 | 🟡 MODERATE | — | Bytes-per-row estimation for memory planning. |
| 8 | `does_join_fit_in_memory()` | ~200 | 🟡 MODERATE | — | Checks against `memory_limit_join_processing` (60% of RAM via `psutil`). |
| 9 | `join_ab_on_key(a, b, key_a, key_b)` | ~220 | 🔴 **IMPORTANT** | `join_engine.py::join()` | Simple `pd.merge(a, b, left_on=key_a, right_on=key_b, how='inner')`. The basic join primitive. |
| 10 | `join_ab_on_key_optimizer(a, b, key_a, key_b)` | ~250 | 🔴 **IMPORTANT** | `join_engine.py::join()` | **Memory-aware chunked join.** Normalizes keys to lowercase strings, drops NaN/null, shuffles b for uniform sampling, first-chunk memory estimation, **3-minute timeout**, disk-spill fallback. This is the production join. |
| 11 | `join_ab_on_key_spill_disk()` | ~380 | 🟡 MODERATE | — | Always-spill variant. Writes to temp files. |
| 12 | `InTreeNode` class | ~420 | 🟡 MODERATE | — | Tree node for join materialization. Has `relation`, `key`, `children`. |
| 13 | `materialize_join_graph(join_graph, filters)` | ~450 | 🔴 **IMPORTANT** | `join_engine.py::materialize()` | Builds an in-tree from join graph edges, applies filters to leaves, then folds leaves upward via joins. The tree-fold is the core materialization strategy. |
| 14 | `apply_consistent_sample()` | ~550 | 🟡 MODERATE | — | Deterministic sampling by hash-sorting IDs. Used for sampled materialization. |
| 15 | `materialize_join_graph_sample()` | ~580 | 🟡 MODERATE | — | Sampled version of `materialize_join_graph`. |
| 16 | `estimate_join_memory()` | ~650 | 🟡 MODERATE | — | Cartesian product size estimation for memory checking. |

---

## DoD/material_view_analysis.py
*204 lines — Comparing materialized views*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `EQUI` enum | ~10 | 🟡 MODERATE | `materialization/view_analysis.py` | EQUIVALENT, DIF_CARDINALITY, DIF_SCHEMA, DIF_VALUES. |
| 2 | `most_likely_key(view)` | ~20 | 🔴 **IMPORTANT** | `view_analysis.py` | Column with highest `unique/total` ratio. Used for join ranking. |
| 3 | `uniqueness(view)` | ~40 | 🟡 MODERATE | `view_analysis.py` | Per-column uniqueness ratio dictionary. |
| 4 | `curate_view(view)` | ~60 | 🟡 MODERATE | — | Drop NaN, deduplicate, reset+sort indices. Cleanup before comparison. |
| 5 | `equivalent(v1, v2)` | ~80 | 🟡 MODERATE | `view_analysis.py` | Same cardinality + same schema + same values (case-insensitive, sorted). |
| 6 | `contained(v1, v2)` | ~110 | 🟡 MODERATE | `view_analysis.py` | Every value in smaller set exists in larger. |
| 7 | `complementary(v1, v2)` | ~130 | 🟡 MODERATE | `view_analysis.py` | Key sets have non-empty symmetric difference. |
| 8 | `contradictory(v1, v2)` | ~150 | 🟡 MODERATE | `view_analysis.py` | Same key values but different non-key values (groupby comparison). |
| 9 | `inconsistent_value_on_key(view)` | ~180 | 🟡 MODERATE | — | Row-level conflict detection within a single view. |

---

## DoD/utils.py
*7 lines — Tiny enum*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `FilterType` enum | 1-7 | 🔴 **IMPORTANT** | `data_on_demand.py` | CELL=0, ATTR=1. Used throughout DoD to distinguish between content and attribute filters. |

---

## DoD/experimental.py
*~70 lines — Exhaustive search variant*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `virtual_schema_exhaustive_search()` | all | ⚪ NOT IMPORTANT | — | Brute-force set cover using `itertools.combinations`. Not used in production; the iterative search in dod.py is the active algorithm. |

---

## dataanalysis/dataanalysis.py
*~400 lines — Statistical column comparison*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `compute_overlap(a, b)` | ~30 | 🔴 **IMPORTANT** | `profiler/column_profiler.py` | Jaccard-like overlap with early termination. Uses `config.join_overlap_th = 0.4`. |
| 2 | `compute_overlap_of_columns(a, b)` | ~50 | 🔴 **IMPORTANT** | — | Wrapper that reads columns from CSV, builds value dicts, calls `compute_overlap`. |
| 3 | `get_tfidf_docs(corpus)` | ~80 | 🟡 MODERATE | — | Global `TfidfVectorizer(sublinear_tf=True, use_idf=True)`. Used by schema_sim edge builder. |
| 4 | `compare_pair_num_columns()` | ~100 | 🟡 MODERATE | — | KS 2-sample test for numeric column comparison. |
| 5 | `compare_pair_text_columns()` | ~120 | 🟡 MODERATE | — | TF-IDF cosine similarity for text column comparison. |
| 6 | `compare_num_columns_dist_ks()` | ~140 | 🟡 MODERATE | — | `scipy.stats.ks_2samp` wrapper. |
| 7 | `compare_num_columns_dist_odsvm()` | ~160 | ⚪ NOT IMPORTANT | — | One-class SVM prediction. Experimental. |
| 8 | `get_numerical_signature()` | ~180 | 🟡 MODERATE | — | KDE sampling for column signatures. |
| 9 | `get_textual_signature()` | ~200 | 🟡 MODERATE | — | `CountVectorizer` top-5 terms. |
| 10 | `get_sim_matrix_numerical()` / `get_sim_matrix_text()` | ~250 | ⚪ NOT IMPORTANT | — | Full pairwise comparison matrices. Not used in pipeline. |
| 11 | `build_dict_values()` | ~20 | 🟡 MODERATE | — | Value frequency counter for overlap computation. |

---

## dataanalysis/nlp_utils.py
*~55 lines — Text preprocessing utilities*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `camelcase_to_snakecase(term)` | ~30 | 🔴 **IMPORTANT** | `profiler/text_utils.py` | Regex-based CamelCase → snake_case. Used in ALL name tokenization across the codebase. |
| 2 | `tokenize_property(prop)` | ~35 | 🟡 MODERATE | `text_utils.py` | snake_case + split on `_` and `-`. |
| 3 | `curate_tokens(tokens)` | ~40 | 🟡 MODERATE | `text_utils.py` | Lowercase, remove stopwords, remove len≤1, deduplicate. |
| 4 | `curate_string(string)` | ~45 | 🟡 MODERATE | `text_utils.py` | CamelCase→snake, replace `_`/`-` with spaces, lowercase. |
| 5 | `pos_tag_text()` / `get_nouns()` / `get_proper_nouns()` | ~10-25 | ⚪ NOT IMPORTANT | — | NLTK POS tagging. Only used in ontomatch `bow_repr_of`. |

---

## config.py
*~30 lines — Global configuration constants*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `join_overlap_th = 0.4` | ~5 | 🔴 **IMPORTANT** | `config.py` | Threshold for Jaccard overlap in joins. |
| 2 | `k = 512` | ~6 | 🔴 **IMPORTANT** | `config.py` | MinHash permutation count. |
| 3 | `separator = '\|'` | ~7 | 🟡 MODERATE | `config.py` | Default CSV separator. |
| 4 | `join_chunksize = 1000` | ~8 | 🟡 MODERATE | — | Chunk size for chunked joins. |
| 5 | `memory_limit_join_processing = 0.6` | ~9 | 🟡 MODERATE | — | 60% of RAM limit. |
| 6 | Serde paths (graphfile, graphcachedfile, etc.) | ~15 | ⚪ NOT IMPORTANT | — | Legacy file paths. |
| 7 | `db_host = 'localhost'`, `db_port = '9200'` | ~25 | ⚪ NOT IMPORTANT | — | Elasticsearch connection. Not used in LakeAgent. |

---

## ontomatch/ss_api.py
*1956 lines — Semantic Schema Matching API*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `SSAPI.__init__(network, store, schema_sim_idx, content_sim_idx)` | ~25 | ⚪ NOT IMPORTANT | — | Ontology matching system. Not ported to LakeAgent. |
| 2 | `SSAPI.add_krs(kr_name_paths)` | ~45 | ⚪ NOT IMPORTANT | — | Registers ontologies (OWL files) for matching. |
| 3 | `SSAPI.find_matchings()` | ~100 | ⚪ NOT IMPORTANT | — | Multi-level matching pipeline: L1 (class→content), L4 (relation→class syntax), L5 (attr→class syntax), L42 (semantic), L52 (semantic). Most levels are commented out. |
| 4 | `SSAPI.find_links(matchings)` | ~330 | ⚪ NOT IMPORTANT | — | Given matchings, discovers `is_a` links via ontology hierarchy. |
| 5 | `SSAPI.find_coarse_grain_hooks()` | ~480 | ⚪ NOT IMPORTANT | — | Deprecated. LSH-indexed semantic vector matching for tables to ontology classes. |
| 6 | `SSAPI.find_coarse_grain_hooks_n2()` | ~290 | ⚪ NOT IMPORTANT | — | O(n²) variant of coarse grain hooks. |
| 7 | `test_l6()` / module-level test code | ~560+ | ⚪ NOT IMPORTANT | — | Test harnesses. |

---

## ontomatch/ss_utils.py
*589 lines — Semantic similarity computation utilities*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `minhash(str_values)` | ~25 | 🟡 MODERATE | — | Java-compatible MinHash implementation. Uses custom hash matching the Java DDProfiler. Only needed if you must match Java-produced hashes. |
| 2 | `extract_cohesive_groups(table_name, attrs)` | ~70 | ⚪ NOT IMPORTANT | — | Groups semantically similar tokens using GloVe. Experimental. |
| 3 | `generate_table_vectors(path, network)` | ~170 | ⚪ NOT IMPORTANT | — | Creates GloVe-based semantic vectors per table. |
| 4 | `compute_semantic_similarity(sv1, sv2)` | ~260 | ⚪ NOT IMPORTANT | — | Core semantic similarity: pairwise GloVe dot products with penalization and signal strength. Only for ontomatch. |
| 5 | `compute_semantic_similarity_cross_average()` / `max_average()` / `min_average()` / `median()` | ~320-400 | ⚪ NOT IMPORTANT | — | Alternative aggregation strategies. Experimental. |
| 6 | `compute_internal_cohesion(sv)` | ~220 | ⚪ NOT IMPORTANT | — | Mean pairwise semantic distance within a vector set. |
| 7 | `store_signatures()` / `load_signatures()` | ~160 | ⚪ NOT IMPORTANT | — | Pickle serde for semantic vectors. |
| 8 | `read_table_columns(path, network)` | ~150 | ⚪ NOT IMPORTANT | — | Generator yielding `(db, table, [cols])` from FieldNetwork. |

---

## ontomatch/matcher_lib.py
*1660 lines — Matching algorithms library*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `MatchingType` enum | ~15 | ⚪ NOT IMPORTANT | — | L1 through L7 matching types. Only for ontomatch. |
| 2 | `SimpleTrie` class | ~30-130 | ⚪ NOT IMPORTANT | — | Trie for summarizing matchings to ancestor classes. |
| 3 | `Matching` class | ~140-200 | ⚪ NOT IMPORTANT | — | Accumulator for source-level and attr-level matchings. |
| 4 | `summarize_matchings_to_ancestor()` | ~240 | ⚪ NOT IMPORTANT | — | Uses trie to collapse matchings to ontology ancestor nodes. |
| 5 | `combine_matchings()` | ~400 | ⚪ NOT IMPORTANT | — | Merges all matching levels into `Matching` objects keyed by `(db, source)`. |
| 6 | `find_relation_class_name_matchings()` | — | ⚪ NOT IMPORTANT | — | L4: MinHash-based syntax matching between relation names and ontology class names. |
| 7 | `find_relation_class_attr_name_matching()` | — | ⚪ NOT IMPORTANT | — | L5: MinHash-based attr name ↔ class name matching. |
| 8 | `find_relation_class_attr_name_sem_matchings()` | ~550 | ⚪ NOT IMPORTANT | — | L52: GloVe-based semantic matching between attribute names and class names. |
| 9 | `get_ban_indexes()` / `remove_banned_vectors()` | ~510 | ⚪ NOT IMPORTANT | — | Removes shared tokens before semantic comparison to avoid trivial matches. |
| 10 | `double_check_sem_signal_attr_sch_sch()` | ~200 | ⚪ NOT IMPORTANT | — | Re-checks semantic signal between two attributes. |

---

## ontomatch/glove_api.py
*~70 lines — GloVe word embedding loader*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `load_model(path)` / `load_vocab()` | ~35-60 | ⚪ NOT IMPORTANT | LakeAgent uses sentence-transformers | Loads GloVe `.txt` file, normalizes to unit vectors. |
| 2 | `get_embedding_for_word(word)` | ~15 | ⚪ NOT IMPORTANT | — | Lookup in vocab dict. Returns None if not found. |
| 3 | `semantic_distance(v1, v2)` | ~20 | ⚪ NOT IMPORTANT | — | `np.dot(v1, v2.T)` — cosine similarity (vectors are pre-normalized). |
| 4 | `get_lang_model_feature_size()` | ~25 | ⚪ NOT IMPORTANT | — | Returns embedding dimension (e.g., 100 for glove.6B.100d). |

---

## ontomatch/onto_parser.py
*428 lines — OWL ontology parser (ontospy)*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `OntoHandler` class | all | ⚪ NOT IMPORTANT | — | Wraps `ontospy` library. Parses OWL ontologies, caches via pickle. |
| 2 | `parse_ontology()` / `store_ontology()` / `load_ontology()` | ~30-70 | ⚪ NOT IMPORTANT | — | OWL file parsing and caching. |
| 3 | `classes()` / `class_hierarchy_iterator()` | ~80-110 | ⚪ NOT IMPORTANT | — | Iterate ontology class hierarchy. |
| 4 | `ancestors_of_class()` / `parents_of_class()` / `descendants_of_class()` | ~120-160 | ⚪ NOT IMPORTANT | — | Hierarchy traversal. |
| 5 | `compute_classes_signatures()` | ~230 | ⚪ NOT IMPORTANT | — | MinHash signatures of class name groups per hierarchy level. |
| 6 | `bow_repr_of(class_name)` | ~370 | ⚪ NOT IMPORTANT | — | Bag-of-words representation from class description + properties. |

---

## ontomatch/no_matcher.py
*~100 lines — Wikipedia text matching experiment*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `find_matching_to_text()` | all | ⚪ NOT IMPORTANT | — | Matches DB attributes to Wikipedia titles using GloVe semantic similarity. Pure experiment. |

---

## knowledgerepr/lite_graph.py
*~60 lines — Bitarray-based graph (alternative to NetworkX)*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `EdgeType` enum | ~10 | ⚪ NOT IMPORTANT | — | SCHEMA_SIM, CONTENT_SIM, PKFK, SEMANTIC. |
| 2 | `LiteGraph` class | ~20 | ⚪ NOT IMPORTANT | — | Adjacency list using `bitarray` for edge types. Never used in production (NetworkX is used instead). |
| 3 | `add_edge()` / `add_undirected_edge()` / `neighbors()` | ~30-55 | ⚪ NOT IMPORTANT | — | Graph operations on bitarray representation. Prototype code. |

---

## inputoutput/inputoutput.py
*~15 lines — Generic pickle serialization*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `serialize_object(obj, path)` | ~5 | 🟡 MODERATE | — | `pickle.dump`. Used for LSH indexes and network serialization. |
| 2 | `deserialize_object(path)` | ~10 | 🟡 MODERATE | — | `pickle.load`. Used to restore LSH indexes. |

---

## api/annotation.py
*~100 lines — Metadata annotation types*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `MDClass` enum | ~10 | ⚪ NOT IMPORTANT | — | WARNING, INSIGHT, QUESTION. |
| 2 | `MDRelation` enum | ~15 | ⚪ NOT IMPORTANT | — | MEANS_SAME_AS, MEANS_DIFF_FROM, IS_SUBCLASS_OF, etc. |
| 3 | `MDHit` namedtuple | ~25 | ⚪ NOT IMPORTANT | — | Metadata hit: id, author, md_class, text, source, target, relation. |
| 4 | `MDComment` namedtuple | ~50 | ⚪ NOT IMPORTANT | — | Comment: id, author, text, ref_id. |
| 5 | `MRS` class (Metadata Result Set) | ~70 | ⚪ NOT IMPORTANT | — | Iterator over metadata results. |

---

## api/reporting.py
*~80 lines — Network statistics*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `Report.__init__(network)` | ~10 | ⚪ NOT IMPORTANT | — | Computes stats on init. |
| 2 | `compute_all_statistics()` | ~30 | ⚪ NOT IMPORTANT | — | Counts tables, columns, edges by relation type. |
| 3 | `print_content_sim_relations()` / `print_schema_sim_relations()` / `print_pkfk_relations()` | ~50 | ⚪ NOT IMPORTANT | — | Debug printing. |
| 4 | `print_all_indexed_tables()` | ~60 | ⚪ NOT IMPORTANT | — | Lists all table names. |

---

## sugar.py
*~140 lines — REPL convenience shortcuts*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | Short variable aliases (`source`, `field`, `content`, `schema_sim`, etc.) | ~20-35 | ⚪ NOT IMPORTANT | — | Convenience bindings for interactive use. |
| 2 | `search(kws, contexts)` | ~50 | ⚪ NOT IMPORTANT | — | Deprecated. Multi-keyword, multi-context search wrapper. |
| 3 | `neighbors(i_drs, relations)` | ~75 | ⚪ NOT IMPORTANT | — | Deprecated. Multi-relation neighbor search. |
| 4 | `path(drs_a, drs_b, relation)` | ~110 | ⚪ NOT IMPORTANT | — | Deprecated. Path-finding wrapper. |
| 5 | `provenance(i_drs)` | ~130 | ⚪ NOT IMPORTANT | — | Deprecated. Provenance graph edge getter. |

---

## main.py
*~60 lines — System initialization and IPython shell*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `init_system(path, create_reporting)` | ~30 | 🟡 MODERATE | `lakeagent/cli.py` | Deserializes network, creates StoreHandler, returns `(api, reporting)`. This is how you boot the system. |
| 2 | `__init_system()` (old API variant) | ~20 | ⚪ NOT IMPORTANT | — | Uses `ddapi.API` instead of `algebra.API`. Legacy. |
| 3 | `main()` | ~50 | ⚪ NOT IMPORTANT | — | Launches IPython embedded shell. |

---

## run_dod.py
*~20 lines — DoD CLI entry point*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `argparse` CLI | all | ⚪ NOT IMPORTANT | `lakeagent/cli.py` | Parses `--model_path`, `--separator`, `--output_path`, `--list_attributes`, `--list_values`. Calls `dod.main(args)`. |

---

## server_config.py
*~5 lines — Hardcoded server paths*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `path_model` / `separator` | all | ⚪ NOT IMPORTANT | `config.py` | Hardcoded paths. Machine-specific. |

---

## server-api/app.py
*~230 lines — Flask web API for DoD*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `/findvs` POST endpoint | ~60 | 🟡 MODERATE | — | Parses grid payload → `list_attributes` + `list_samples`, calls `dod.virtual_schema_iterative_search()`, returns first view as HTML table + analysis + join graph metadata. |
| 2 | `/next_view` POST endpoint | ~100 | 🟡 MODERATE | — | Calls `next(view_generator)` to get next materialized view. |
| 3 | `/suggest_field` POST endpoint | ~130 | 🟡 MODERATE | — | Calls `dod.aurum_api.suggest_schema(input_text)`. |
| 4 | `/download_view` POST endpoint | ~145 | ⚪ NOT IMPORTANT | — | Saves current view to CSV. Hardcoded path. |
| 5 | `obtain_view_analysis(view)` | ~160 | ⚪ NOT IMPORTANT | — | Per-column `df.describe().to_html()`. |
| 6 | `Ack` / `InvalidUsage` classes | ~180 | ⚪ NOT IMPORTANT | — | Flask error handling boilerplate. |

---

## aurum_cli.py
*283 lines — Fire-based CLI*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | `CSVDataSource` / `DBDataSource` dataclasses | ~30-100 | ⚪ NOT IMPORTANT | — | Data source config generation (YAML). |
| 2 | `AurumWrapper` class | ~120 | ⚪ NOT IMPORTANT | — | Filesystem-based data source and model management. |
| 3 | `AurumCLI.profile()` | ~210 | ⚪ NOT IMPORTANT | — | Invokes Java DDProfiler via `subprocess`. |
| 4 | `AurumCLI.build_model()` | ~220 | ⚪ NOT IMPORTANT | — | Invokes `networkbuildercoordinator.py` via subprocess. |
| 5 | `AurumCLI.export_model()` | ~230 | ⚪ NOT IMPORTANT | — | Exports to Neo4j via `Neo4jExporter`. |
| 6 | `AurumCLI.clear_store()` | ~250 | ⚪ NOT IMPORTANT | — | Deletes ES indices. |
| 7 | `AurumCLI.explore_model()` | ~260 | ⚪ NOT IMPORTANT | — | Opens IPython with loaded model. |

---

## export_network_2_neo4j.py
*~15 lines — Neo4j export entry point*

| # | Feature / Symbol | Lines | Importance | LakeAgent Equivalent | Notes |
|---|---|---|---|---|---|
| 1 | CLI + `serialize_network_to_neo4j()` | all | ⚪ NOT IMPORTANT | — | Thin wrapper to export FieldNetwork to Neo4j. Infrastructure only. |

---

## Quick-Reference: What Matters for Debugging LakeAgent

### 🔴 Must-Know Algorithms (cross-reference these first)

| Aurum File | Feature | LakeAgent File | What to check |
|---|---|---|---|
| `networkbuilder.py` | `build_schema_sim_relation` | `graph/network_builder.py` | TF-IDF + LSH thresholds, edge creation |
| `networkbuilder.py` | `build_content_sim_mh_text` | `graph/network_builder.py` | MinHash threshold=0.7, num_perm=512 |
| `networkbuilder.py` | `build_content_sim_relation_num_overlap_distr` | `graph/network_builder.py` | Median±IQR overlap=0.85, inclusion=0.3, DBSCAN eps=0.1 |
| `networkbuilder.py` | `build_pkfk_relation` | `graph/network_builder.py` | Cardinality ratio > 0.7 |
| `dod.py` | `virtual_schema_iterative_search` | `discovery/data_on_demand.py` | 5-stage pipeline, greedy set cover |
| `dod.py` | `joinable()` | `discovery/data_on_demand.py` | Join graph enumeration via itertools.product |
| `data_processing_utils.py` | `materialize_join_graph` | `materialization/join_engine.py` | Tree-fold join strategy |
| `data_processing_utils.py` | `join_ab_on_key_optimizer` | `materialization/join_engine.py` | Memory-aware chunked join with 3-min timeout |
| `apiutils.py` | `DRS` class | `discovery/result_set.py` | Set operations, mode switching, iteration |
| `apiutils.py` | `Hit` + `compute_field_id` | `discovery/result_set.py` | ID hashing, equality semantics |
| `fieldnetwork.py` | `neighbors_id()` | `graph/field_network.py` | Core graph traversal |
| `fieldnetwork.py` | `find_path_hit()` / `find_path_table()` | `graph/field_network.py` | DFS path finding with provenance |
| `algebra.py` | `__neighbor_search()` | `discovery/algebra.py` | Input→DRS conversion + traversal |
| `config.py` | All thresholds | `lakeagent/config.py` | join_overlap_th=0.4, k=512 |

### ⚪ Safe to Ignore

- **Entire `ontomatch/` directory** — Ontology matching system. Not ported.
- **`sugar.py`** — REPL shortcuts, all deprecated.
- **`api/annotation.py`** — Metadata annotation system.
- **`api/reporting.py`** — Statistics printing.
- **`lite_graph.py`** — Unused bitarray graph prototype.
- **`export_network_2_neo4j.py`** — Neo4j export utility.
- **`aurum_cli.py`** — Fire CLI (calls subprocess for Java profiler).
- **`server-api/app.py`** — Flask endpoints (useful to understand the API contract, but not for algorithm debugging).
