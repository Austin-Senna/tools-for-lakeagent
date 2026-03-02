"""
Pluggable data-source readers.

Each reader yields ``(db_name, table_name, column_name, values)`` quads
that the :mod:`column_profiler` consumes.  This replaces the legacy Java
``CSVSource``, ``PostgresSource``, ``HiveSource``, and ``SQLServerSource``
implementations in ``ddprofiler/src/main/java/sources/``.

Modern approach: pandas for CSV, SQLAlchemy for databases, boto3 for S3.

The **S3Reader** is the primary reader for the 9.5 TB S3 data-lake use case.
It streams CSVs from an S3 bucket without loading the full file into memory.
"""

from __future__ import annotations
import os
import re
import json

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import boto3
import duckdb
import pandas as pd

__all__ = [
    "SourceConfig",
    "SourceReader",
    "CSVReader",
    "S3Reader",
    "DatabaseReader",
    "discover_sources",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source configuration  (mirrors legacy YAML source configs)
# ---------------------------------------------------------------------------

@dataclass
class SourceConfig:
    """Describes a single data source to be profiled.

    Legacy equivalent: YAML config entries parsed by ``Main.java``.
    """

    name: str
    """Logical name for this data source (e.g. ``"my_database"``)."""

    source_type: str
    """One of ``"csv"``, ``"postgres"``, ``"mysql"``, ``"sqlserver"``, ``"hive"``."""

    config: dict[str, Any] = field(default_factory=dict)
    """Driver-specific configuration.

    * CSV: ``{"path": "/data/folder", "separator": ","}``
    * Database: ``{"connection_string": "postgresql://...", "schema": "public"}``
    """


# ---------------------------------------------------------------------------
# Reader protocol (duck-typed interface for all source types)
# ---------------------------------------------------------------------------

class SourceReader(Protocol):
    """Protocol all source readers must satisfy."""

    def read_columns(self) -> Iterator[tuple[str, str, str, str, list[str], str]]:
        """Yield ``(db_name, table_name, column_name, aurum_type, values, path)`` per column.

        *aurum_type* is 'N' (Numeric) or 'T' (Text).
        *values* is a list of string-coerced cell values for that column.
        *path* is the data-source path (S3 URI, file path, or connection string).
        The profiler uses these to compute statistics and signatures.
        """
        ...


# ---------------------------------------------------------------------------
# CSV reader  (replaces legacy Java CSVSource)
# ---------------------------------------------------------------------------
class CSVReader:
    """Read all CSV files from a directory and yield columns using DuckDB.

    Parameters
    ----------
    db_name : str
        Logical database name assigned to all files in this directory.
    directory : str | Path
        Path to a directory containing ``.csv`` files.
    separator : str
        CSV delimiter.
    limit_values : bool
        If True, applies reservoir sampling to limit the number of rows read.
    max_values : int
        The maximum number of rows to sample per file if limit_values is True (default 10_000).
    """

    def __init__(
        self,
        db_name: str,
        directory: str | Path,
        separator: str = ",",
        limit_values: bool = False,
        max_values: int = 10_000,
    ) -> None:
        self.db_name = db_name
        self.directory = Path(directory)
        self.separator = separator
        self.limit_values = limit_values
        self.max_values = max_values

    def read_columns(self) -> Iterator[tuple[str, str, str, str, list[str], str]]:
        """Yield ``(db_name, table_name, column_name, aurum_type, values, path)`` for every column
        in every CSV file under :attr:`directory`.
        """
        csv_files = sorted(self.directory.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in %s", self.directory)
            return

        con = duckdb.connect()

        for csv_path in csv_files:
            table_name = csv_path.stem
            logger.info("Reading CSV: %s", csv_path)
            
            try:
                if self.limit_values:
                    query = (
                        f"SELECT * FROM read_csv_auto(?, delim=?)"
                        f" USING SAMPLE {self.max_values} ROWS (reservoir)"
                    )
                    df = con.execute(query, [str(csv_path), self.separator]).df()
                else:
                    query = (
                        f"SELECT * FROM read_csv_auto(?, delim=?)"
                    )
                    df = con.execute(query, [str(csv_path), self.separator]).df()
            except Exception:
                logger.exception("Failed to read %s", csv_path)
                continue

            for col_name in df.columns:
                series = df[col_name]
                is_numeric = pd.api.types.is_numeric_dtype(series.dtype)
                aurum_type = "N" if is_numeric else "T"
                values = series.dropna().astype(str).tolist()

                yield (
                    self.db_name,
                    table_name,
                    str(col_name),
                    aurum_type,
                    values,
                    str(csv_path),
                )
                
        con.close()

# ---------------------------------------------------------------------------
# Database reader  (replaces legacy PostgresSource, SQLServerSource, etc.)
# ---------------------------------------------------------------------------

class DatabaseReader:
    """Read tables from a SQL database via DuckDB and yield columns.

    Parameters
    ----------
    db_name : str
        Logical database name.
    connection_string : str
    db_type: str
    schema : str
        Database schema to inspect.
    """

    def __init__(
        self,
        db_name: str,
        db_type: str,
        connection_string: str,
        schema: str = "public",
    ) -> None:
        self.db_name = db_name
        self.connection_string = connection_string
        self.schema = schema
        self.db_type = db_type

    def read_columns(self) -> Iterator[tuple[str, str, str, str, list[str], str]]:
        # 1. Spin up an in-memory DuckDB instance
        con = duckdb.connect()

        # 2. Install and load the specific extension (e.g., postgres)
        con.execute(f"INSTALL {self.db_type};")
        con.execute(f"LOAD {self.db_type};")

        # 3. Attach the remote database directly into DuckDB
        # DuckDB handles the network transfer heavily optimized
        con.execute(f"ATTACH '{self.connection_string}' AS remote_db (TYPE {self.db_type});")

        # 4. Get all table names from the attached database
        tables_query = (
            "SELECT table_name FROM information_schema.tables"
            " WHERE table_schema = 'public'"
        )
        tables = [row[0] for row in con.execute(tables_query).fetchall()]

        for table_name in tables:
            # 5. Read the entire table directly into a Pandas DataFrame
            # (Or you could chunk it using DuckDB's LIMIT/OFFSET if it's massive)
            df = con.execute(f"SELECT * FROM remote_db.{table_name}").df()

            # 6. Apply our exact same N/T logic!
            for col_name in df.columns:
                is_numeric = pd.api.types.is_numeric_dtype(df[col_name].dtype)
                aurum_type = "N" if is_numeric else "T"
                values = df[col_name].dropna().astype(str).tolist()
                yield self.db_name, table_name, str(col_name), aurum_type, values, self.connection_string
                # Note: DatabaseReader doesn't limit text values — config-driven
                # truncation is handled by CSVReader/S3Reader which accept the flag.

        con.close()
        


# ---------------------------------------------------------------------------
# S3 reader  (new — primary reader for the 9.5 TB S3 data-lake)
# ---------------------------------------------------------------------------

class S3Reader:
    """S3 reader using DuckDB's native httpfs + reservoir sampling.

    Parameters
    ----------
    db_name : str
        Logical database name.
    s3_paths : list[str]
        Explicit list of ``s3://`` URIs to profile.
    region : str
        AWS region.
    limit_values : bool
        If True, applies reservoir sampling to limit the number of rows downloaded.
    max_values : int
        The maximum number of rows to sample per file if limit_values is True (default 10_000).
    max_file_size_gb : float
        Files larger than this (in GB) are skipped entirely (default 10.0).
        Prefer setting this via ``AurumConfig.max_file_size_gb``.
    """

    def __init__(
        self,
        db_name: str,
        s3_paths: list[str],
        region: str = "us-east-2",
        limit_values: bool = False,
        max_values: int = 10_000,
        max_file_size_gb: float = 10.0,
    ) -> None:
        self.db_name = db_name
        self.s3_paths = s3_paths
        self.region = region
        self.limit_values = limit_values
        self.max_values = max_values
        self.max_file_size_gb = max_file_size_gb


    def _connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
        con.execute(f"SET s3_region='{self.region}';")

        # Try env vars first, then fall back to boto3 credential chain
        # (covers ~/.aws/credentials, IAM roles, SSO, etc.)
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        if not (access_key and secret_key):
            try:
                import boto3
                session = boto3.Session()
                creds = session.get_credentials()
                if creds:
                    frozen = creds.get_frozen_credentials()
                    access_key = frozen.access_key
                    secret_key = frozen.secret_key
                    token = frozen.token
                    if token:
                        con.execute(f"SET s3_session_token='{token}';")
            except Exception:
                pass

        if access_key and secret_key:
            con.execute(f"SET s3_access_key_id='{access_key}';")
            con.execute(f"SET s3_secret_access_key='{secret_key}';")
        return con

    @staticmethod
    def _table_name_from_path(path: str) -> str:
        """Derive a human-readable table name from an S3 URI.

        S3 layout: ``s3://bucket/datagov/<dataset-slug>/files/<filename>.txt``
        Result:    ``<dataset-slug>/<filename>``

        Falls back to just the filename stem if the path doesn't match.
        """
        parts = path.replace("s3://", "").split("/")
        # Find the 'files' segment and use dataset_slug/filename
        stem = Path(parts[-1]).stem
        try:
            files_idx = parts.index("files")
            if files_idx >= 1:
                dataset = parts[files_idx - 1]
                return f"{dataset}/{stem}"
        except ValueError:
            pass
        # Fallback: use last two path segments
        if len(parts) >= 2:
            return f"{parts[-2]}/{stem}"
        return stem


    # Regex for auto-generated column headers (DuckDB fallback)
    _AUTO_COL_RE = re.compile(r"^column\d+$")

    def _s3_client(self) -> Any:
        """Build a boto3 S3 client using env vars or the default credential chain."""
        return boto3.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
            aws_session_token=os.getenv("AWS_SESSION_TOKEN") or None,
        )

    def _s3_get_meta_and_peek(self, path: str, max_gb: float = 10.0) -> tuple[int, str]:
        """HeadObject for size check + download first 1 KB for type sniffing.

        Raises ValueError if the file exceeds *max_gb*.
        """
        without_scheme = path[len("s3://"):]
        bucket, _, key = without_scheme.partition("/")
        s3 = self._s3_client()

        meta = s3.head_object(Bucket=bucket, Key=key)
        file_size = meta.get("ContentLength", 0)
        max_bytes = max_gb * 1024 ** 3
        if file_size > max_bytes:
            raise ValueError(
                f"File size {file_size / 1024**3:.2f} GB exceeds {max_gb} GB limit"
            )

        response = s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-1023")
        peek_text = response["Body"].read().decode("utf-8", errors="replace")
        return file_size, peek_text

    def _s3_read_full(self, path: str) -> str:
        """Download the entire S3 object as a UTF-8 string."""
        without_scheme = path[len("s3://"):]
        bucket, _, key = without_scheme.partition("/")
        response = self._s3_client().get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8", errors="replace")

    def _df_from_json(self, content: str, con: duckdb.DuckDBPyConnection) -> pd.DataFrame | None:
        """Parse JSON content into a DataFrame using DuckDB where possible.

        Handles three formats:
        * Socrata  – ``{"meta": {"view": {...}}, "data": [...]}``
        * GeoJSON  – ``{"type": "FeatureCollection", "features": [...]}``
        * Plain array – ``[{...}, ...]``

        Respects ``self.limit_values`` / ``self.max_values``.
        Returns None when the content is not valid / recognisable JSON.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None

        limit: int | None = self.max_values if self.limit_values else None

        # --- Socrata ---
        if (
            isinstance(data, dict)
            and "meta" in data
            and isinstance(data["meta"], dict)
            and "view" in data["meta"]
        ):
            all_columns = [col["name"] for col in data["meta"]["view"]["columns"]]
            rows = data.get("data", [])
            if limit is not None:
                rows = rows[:limit]
            df = pd.DataFrame(rows, columns=all_columns)
            hidden = {
                col["name"]
                for col in data["meta"]["view"]["columns"]
                if "hidden" in col.get("flags", [])
            }
            return df.drop(columns=list(hidden), errors="ignore")

        # --- GeoJSON FeatureCollection ---
        if isinstance(data, dict) and data.get("type") == "FeatureCollection":
            features = data.get("features", [])
            if limit is not None:
                features = features[:limit]
            records = [f.get("properties", {}) for f in features]
            return pd.DataFrame(records)

        # --- Plain JSON array — let DuckDB parse it ---
        if isinstance(data, list):
            if limit is not None:
                data = data[:limit]
            try:
                json_str = json.dumps(data)
                return con.execute("SELECT * FROM read_json_auto(?)", [json_str]).df()
            except Exception:
                return pd.DataFrame(data)

        return None

    # Common English stop words to exclude from unstructured text
    _STOP_WORDS = frozenset({
        "about", "above", "after", "again", "also", "another", "because",
        "been", "before", "being", "between", "both", "cannot", "come",
        "could", "does", "doing", "done", "down", "each", "even", "ever",
        "every", "from", "further", "have", "having", "here",
        "hers", "herself", "himself", "his", "into", "itself", "just",
        "like", "make", "many", "more", "most", "much", "myself", "need",
        "never", "next", "none", "nothing", "once", "only", "other",
        "otherwise", "ourselves", "over", "said", "same", "seem", "should",
        "since", "some", "such", "than", "that", "their", "them", "then",
        "there", "these", "they", "this", "those", "through", "together",
        "under", "until", "upon", "very", "want", "were", "what", "when",
        "where", "which", "while", "whom", "will", "with", "would", "your",
        "yourself", "yourselves",
    })

    def _df_from_text(self, content: str, table_name: str) -> pd.DataFrame:
        """Treat raw text as a single-column table of meaningful tokens.

        Extracts unique, non-trivial words (4+ chars, no stop words) from the
        content and yields each as its own row so the profiler sees real values
        rather than one giant string blob.
        """
        col_name = table_name.split("/")[-1]

        # Extract words of 4+ chars, deduplicate, filter stop words
        raw_tokens = re.findall(r'\b[a-zA-Z]{4,}\b', content)
        seen: set[str] = set()
        tokens: list[str] = []
        for tok in raw_tokens:
            lower = tok.lower()
            if lower not in self._STOP_WORDS and lower not in seen:
                seen.add(lower)
                tokens.append(tok)
                if len(tokens) >= self.max_values:
                    break

        return pd.DataFrame({col_name: tokens})

    def read_columns(self) -> Iterator[tuple[str, str, str, str, list[str], str]]:
        con = self._connect()

        for path in self.s3_paths:

            table_name = self._table_name_from_path(path)
            logger.info("Reading %s", path)

            try:
                # Peek at the first 1 KB — also enforces the 10 GB size limit
                _, peek_text = self._s3_get_meta_and_peek(path, max_gb=self.max_file_size_gb)
                first_line = peek_text.split("\n")[0]

                if any(delim in first_line for delim in [",", "\t", "|", ";"]):
                    # CSV: DuckDB streams directly from S3 — never loads into Python memory
                    try:
                        if self.limit_values:
                            df = con.execute(
                                "SELECT * FROM read_csv_auto(?) LIMIT ?",
                                [path, self.max_values],
                            ).df()
                        else:
                            df = con.execute(
                                "SELECT * FROM read_csv_auto(?)", [path]
                            ).df()
                    except Exception:
                        logger.exception("Failed to read CSV %s", path)
                        continue

                elif peek_text.strip().startswith(("{", "[")):
                    # JSON: must download the full file for valid JSON parsing
                    content = self._s3_read_full(path)
                    df = self._df_from_json(content, con)
                    if df is None:
                        df = self._df_from_text(content, table_name)

                else:
                    # Unstructured text: download full file, extract word tokens
                    content = self._s3_read_full(path)
                    df = self._df_from_text(content, table_name)

            except ValueError as ve:
                logger.warning("Skipping %s: %s", path, ve)
                continue
            except Exception:
                logger.exception("Failed to read %s", path)
                continue

            # Skip tiny files (< 2 rows means no real data)
            if len(df) < 2:
                logger.debug("Skipping %s — only %d rows", path, len(df))
                continue

            for col_name in df.columns:
                col_str = str(col_name)
                # Skip auto-generated headers (column0, column1, …)
                if self._AUTO_COL_RE.match(str(col_name)):
                    continue
                
                if len(col_str) > 255:
                    logger.debug(
                        "Skipping absurdly long column name (%d chars) in %s. "
                        "Likely a missing header row.", 
                        len(col_str), path
                    )
                    continue

                series = df[col_name]
                is_numeric = pd.api.types.is_numeric_dtype(series.dtype)
                aurum_type = "N" if is_numeric else "T"
                values = series.dropna().astype(str).tolist()

                yield (
                    self.db_name,
                    table_name,
                    str(col_name),
                    aurum_type,
                    values,
                    path,
                )

        con.close()

# ---------------------------------------------------------------------------
# Source factory
# ---------------------------------------------------------------------------
def discover_sources(configs: list[SourceConfig]) -> list[SourceReader]:
    """Instantiate the appropriate :class:`SourceReader` for each config entry.

    Raises :class:`ValueError` for unsupported ``source_type`` values.
    """
    readers: list[SourceReader] = []
    for cfg in configs:
        if cfg.source_type == "csv":
            readers.append(
                CSVReader(
                    db_name=cfg.name,
                    directory=cfg.config.get("path", "."),
                    separator=cfg.config.get("separator", ","),
                    limit_values=cfg.config.get("limit_values", False),
                    max_values=cfg.config.get("max_values", 10_000),
                )
            )
        elif cfg.source_type == "s3":
            readers.append(
                S3Reader(
                    db_name=cfg.name,
                    s3_paths=cfg.config["s3_paths"],
                    region=cfg.config.get("region", "us-east-2"),
                    limit_values=cfg.config.get("limit_values", False),
                    max_values=cfg.config.get("max_values", 10_000),
                )
            )
        elif cfg.source_type in ("postgres", "mysql", "sqlserver", "hive"):
            readers.append(
                DatabaseReader(
                    db_name=cfg.name,
                    db_type=cfg.source_type,
                    connection_string=cfg.config["connection_string"],
                    schema=cfg.config.get("schema", "public"),
                )
            )
        else:
            raise ValueError(f"Unsupported source type: {cfg.source_type!r}")
    return readers
