"""
Profiler package — Python replacement for the legacy Java ``ddprofiler``.

Reads data sources (CSV files, databases), computes per-column statistics
(type detection, cardinality, MinHash signatures, numeric distributions,
NER entities), and stores the profiles into Elasticsearch indices.

Modules
-------
column_profiler
    Per-column analysis: type detection, KMinHash, cardinality, range stats, NER.
source_readers
    Pluggable data-source connectors (CSV, PostgreSQL, etc.).
"""
