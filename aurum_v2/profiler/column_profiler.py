"""
Column profiler — Python replacement for the Java ``ddprofiler``.

Reads columns from :mod:`source_readers`, computes per-column statistics
(type, cardinality, MinHash signatures, numeric range, NER entities),
and stores profile + text documents into Elasticsearch.

This is the critical **Stage 0** of the Aurum pipeline.

Elasticsearch indices created (matching legacy Java ``NativeElasticStore``):

* **``profile``** — one document per column:
  ``id, dbName, path, sourceName, sourceNameNA, columnName, columnNameNA,
  dataType, totalValues, uniqueValues, entities, minhash[512],
  minValue, maxValue, avgValue, median, iqr``

* **``text``** — one document per column (raw values for keyword search):
  ``id, dbName, path, sourceName, columnName, columnNameSuggest, text[]``
"""


from __future__ import annotations
from aurum_v2.models.hit import compute_field_id
import re
from datasketch import MinHash, HyperLogLog
from aurum_v2.config import AurumConfig
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from concurrent import futures
import random

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

    from aurum_v2.config import AurumConfig
    from aurum_v2.profiler.source_readers import SourceReader

__all__ = [
    "ColumnProfile",
    "Profiler",
]

logger = logging.getLogger(__name__)
_SPACY_LOADED = False
_SPACY_MODEL = None
# ---------------------------------------------------------------------------
# ColumnProfile — per-column statistics container
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    """Holds all computed statistics for a single column.

    This is the Python equivalent of the Java ``WorkerTaskResult`` class.
    Two constructors in legacy: one for text columns (with minhash, entities)
    and one for numeric columns (with min/max/avg/median/iqr).
    """

    nid: str
    """CRC32 node ID."""

    db_name: str
    source_name: str
    column_name: str
    data_type: str
    """``"T"`` for text, ``"N"`` for numeric."""

    total_values: int = 0
    unique_values: int = 0

    # ── Text column stats ──────────────────────────────────────────────
    entities: str = ""
    """Comma-separated NER entity labels (e.g. ``"PERSON,LOCATION"``)."""

    minhash: list[int] = field(default_factory=list)
    """KMinHash signature array of length K=512. Empty for numeric columns."""

    # ── Numeric column stats ───────────────────────────────────────────
    min_value: float = 0.0
    max_value: float = 0.0
    avg_value: float = 0.0
    median: float = 0.0
    iqr: float = 0.0

    # ── Raw values (for text index) ────────────────────────────────────
    raw_values: list[str] = field(default_factory=list, repr=False)
    """Stored separately in the ES ``text`` index for keyword search."""



# ---------------------------------------------------------------------------
# KMinHash  (replaces Java analysis.modules.KMinHash)
# ---------------------------------------------------------------------------

def compute_kmin_hash(
    values: list[str],
    k: int,
) -> list[int]:
    """Compute a K-MinHash signature for a set of text values.

    USE DATASKETCH INSTEAD OF LEGACY AURUM PROFILER.

    Algorithm (exactly matches ``KMinHash.java``):
    1. Generate *k* random seed pairs ``(a, b)`` from ``Random(seed)``.
    2. Initialise ``minhash[k]`` to ``Long.MAX_VALUE``.
    3. For each value:
       a. Replace ``_`` and ``-`` with spaces, split on spaces.
       b. For each token (lowercased):
          * Compute ``raw_hash`` using the polynomial rolling hash
            ``h = (2^61-1); for c in s: h = 31*h + ord(c)``.
          * For each permutation *i*:
            ``hash = (a[i] * raw_hash + b[i]) % MERSENNE_PRIME``
          * Update ``minhash[i] = min(minhash[i], hash)``.
    4. Return ``minhash`` as a list of *k* longs.
    """
    m = MinHash(num_perm=k)

    # split on spaces
    tokenizer = re.compile(r'[\s_\-]+')
    
    # hash the values
    for val in values:
        # Lowercase and split into tokens
        tokens = tokenizer.split(str(val).lower())
        for token in tokens:
            if token:
                # datasketch requires bytes, so we encode the string
                m.update(token.encode('utf8'))
                
    return m.hashvalues


# ---------------------------------------------------------------------------
# Cardinality  (replaces Java CardinalityAnalyzer / HyperLogLog)
# ---------------------------------------------------------------------------

def compute_cardinality(values: list[str]) -> int:
    """Return the approximate number of unique values.

    Uses a Python set for exact cardinality (adequate for most data-lake
    columns).  For very large columns, consider ``datasketch.HyperLogLog``.
    """
    hll = HyperLogLog(p=14) 
    
    for val in values:
        hll.update(val.encode('utf-8'))
        
    return int(hll.count())


# ---------------------------------------------------------------------------
# Numeric range stats  (replaces Java RangeAnalyzer)
# ---------------------------------------------------------------------------

def compute_numeric_stats(values: list[str]) -> tuple[float, float, float, float, float]:
    """Compute (min, max, avg, median, iqr) safely for numeric strings."""
    # 1. Safely convert to floats (ignoring any dirty strings that snuck in)
    valid_floats = []
    for v in values:
        try:
            valid_floats.append(float(v))
        except ValueError:
            pass

    if not valid_floats:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    # 2. Let numpy handle the math instantly
    arr = np.array(valid_floats)
    q75, q25 = np.percentile(arr, [75, 25])
    
    return (
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.mean(arr)),
        float(np.median(arr)),
        float(q75 - q25) # IQR
    )

# ---------------------------------------------------------------------------
# NER entity detection  (replaces Java EntityAnalyzer / OpenNLP)
# ---------------------------------------------------------------------------
# HINT: THEY DONT EVEN USE NER IN THE ACTUAL AURUM
def detect_entities(
    values: list[str],
    model: str | None = None,
    sample_size: int = 1000
) -> str:
    """Run NER over sampled text values and return a comma-separated entity string."""
    global _SPACY_MODEL, _SPACY_LOADED

    # 1. Graceful Degradation: Load spacy only once, skip if missing
    if not _SPACY_LOADED:
        try:
            import spacy
            target_model = model or "en_core_web_sm"
            
            # Disable components we don't need for a massive speed boost
            _SPACY_MODEL = spacy.load(
                target_model, 
                disable=["tagger", "parser", "attribute_ruler", "lemmatizer"]
            )
        except (ImportError, OSError):
            logger.warning("spaCy or model not found. Skipping NER profiling.")
            _SPACY_MODEL = None
        finally:
            _SPACY_LOADED = True

    if _SPACY_MODEL is None or not values:
        return ""

    # 2. Optimization: Take a random sample so we don't hang on massive columns
    sample = random.sample(values, min(len(values), sample_size))
    found_labels = set()

    # 3. Batch processing via nlp.pipe
    for doc in _SPACY_MODEL.pipe(sample, batch_size=256):
        for ent in doc.ents:
            found_labels.add(ent.label_)

    # Returns something like "PERSON,ORG,DATE"
    return ",".join(sorted(found_labels))


# ---------------------------------------------------------------------------
# Profile a single column
# ---------------------------------------------------------------------------

def profile_column(
    db_name: str,
    table_name:str,
    column_name: str,
    values: list[str],
    aurum_type: str,
    run_ner: bool = False,
) -> ColumnProfile:
    """Profile a single column and return a :class:`ColumnProfile`.

    Steps (mirrors legacy ``Worker.java`` pipeline):
    2. :func:`compute_cardinality` → ``unique_values``.
    3. If text:
       a. :func:`compute_kmin_hash` → ``minhash[512]``.
       b. Optionally :func:`detect_entities` → ``entities``.
    4. If numeric:
       a. :func:`compute_numeric_stats` → min/max/avg/median/iqr.
    5. Wrap everything in a :class:`ColumnProfile`.
    """
    unique_values = compute_cardinality(values)
    kmin_hash = []
    entities = ""
    numeric_stats = (0.0, 0.0, 0.0, 0.0, 0.0)

    if aurum_type == "T":
        kmin_hash = compute_kmin_hash(values=values, k=AurumConfig.minhash_num_perm)
        if run_ner:
            entities = detect_entities(values = values, model = AurumConfig.spacy_model, sample_size=AurumConfig.spacy_size)
    else:
        numeric_stats = compute_numeric_stats(values=values)

    nid = compute_field_id(db_name=db_name, source_name=table_name, field_name= column_name)
    return ColumnProfile(nid=nid, db_name=db_name, source_name=table_name, column_name=column_name,
                         data_type=aurum_type, total_values=len(values),
                         unique_values=unique_values, entities=entities, minhash=kmin_hash, 
                         min_value=numeric_stats[0], max_value=numeric_stats[1], avg_value= numeric_stats[2],
                         median=numeric_stats[3],iqr=numeric_stats[4], raw_values=values)



# ---------------------------------------------------------------------------
# Main Profiler class  (replaces Java Main + Conductor + Worker pipeline)
# ---------------------------------------------------------------------------

class Profiler:
    """Orchestrates profiling of data sources and stores results to ES.

    Parameters
    ----------
    config : AurumConfig
        System configuration (ES host/port, thresholds).

    Usage
    -----
    ::

        profiler = Profiler(config)
        profiler.run(readers)          # readers: list[SourceReader]
        profiler.store_profiles()      # flush to ES
    """

    def __init__(self, config: AurumConfig) -> None:
        self._config = config
        self._profiles: list[ColumnProfile] = []
        self._es_client: Elasticsearch | None = None

    # ------------------------------------------------------------------
    # ES index management  (mirrors NativeElasticStore.initStore)
    # ------------------------------------------------------------------

    def _init_es(self) -> None:
        """Connect to Elasticsearch and create ``profile`` + ``text`` indices
        with the correct mappings if they don't already exist.

        Index mappings match the legacy ``NativeElasticStore.initStore()``
        exactly (see audit_summary §4.3).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Profiling pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        readers: list[SourceReader],
        *,
        run_ner: bool = False,
        max_workers: int = 4,
    ) -> None:
        """Profile all columns from all *readers*.

        Algorithm (mirrors legacy ``Conductor`` + ``Worker``):

        1. For each reader, iterate ``read_columns()``.
        2. For each ``(db, table, column, values)`` call :func:`profile_column`.
        3. Append the resulting :class:`ColumnProfile` to ``self._profiles``.

        *max_workers* controls optional ``concurrent.futures`` parallelism
        (``None`` = sequential, matching legacy default of N=1).
        """
        with futures.ProcessPoolExecutor(max_workers = max_workers) as executor:
            pending_futures = []
            for reader in readers:
                for col_data in reader.read_columns():
                    # (db_name, table_name, column_name, aurum_type, values)
                    future = executor.submit(profile_column, db_name=col_data[0], table_name=col_data[1], 
                                             column_name=col_data[2], aurum_type=col_data[3], values=col_data[4], 
                                             run_ner=run_ner)
                    pending_futures.append(future)

            for future in futures.as_completed(pending_futures):
                try:
                    cprofile = future.result()
                    self._profiles.append(cprofile)
                except Exception as e:
                    logger.error(f"A worker failed to profile a column: {e}")
        
    # ------------------------------------------------------------------
    # Store to ES
    # ------------------------------------------------------------------

    def store_profiles(self) -> None:
        """Bulk-index all collected profiles into Elasticsearch.

        For each :class:`ColumnProfile`:

        * Store a **profile document** in the ``profile`` index (ES bulk API).
          Fields: ``id, dbName, path, sourceName, sourceNameNA, columnName,
          columnNameNA, dataType, totalValues, uniqueValues, entities,
          minhash, minValue, maxValue, avgValue, median, iqr``.

        * Store a **text document** in the ``text`` index.
          Fields: ``id, dbName, path, sourceName, columnName,
          columnNameSuggest, text[]``.

        Uses the ES ``BulkProcessor`` equivalent
        (``elasticsearch.helpers.bulk``).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def profiles(self) -> list[ColumnProfile]:
        """Return all collected profiles (useful for testing / inspection)."""
        return list(self._profiles)
