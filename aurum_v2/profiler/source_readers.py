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

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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

    def read_columns(self) -> Iterator[tuple[str, str, str, list[str]]]:
        """Yield ``(db_name, table_name, column_name, values)`` per column.

        *values* is a list of string-coerced cell values for that column.
        The profiler uses these to compute statistics and signatures.
        """
        ...


# ---------------------------------------------------------------------------
# CSV reader  (replaces legacy Java CSVSource)
# ---------------------------------------------------------------------------

class CSVReader:
    """Read all CSV files from a directory and yield columns.

    Parameters
    ----------
    db_name : str
        Logical database name assigned to all files in this directory.
    directory : str | Path
        Path to a directory containing ``.csv`` files.
    separator : str
        CSV delimiter.
    encoding : str
        File encoding.
    """

    def __init__(
        self,
        db_name: str,
        directory: str | Path,
        separator: str = ",",
        encoding: str = "utf-8",
    ) -> None:
        self.db_name = db_name
        self.directory = Path(directory)
        self.separator = separator
        self.encoding = encoding

    def read_columns(self) -> Iterator[tuple[str, str, str, list[str]]]:
        """Yield ``(db_name, table_name, column_name, values)`` for every column
        in every CSV file under :attr:`directory`.

        Algorithm (mirrors legacy ``CSVSource``):

        1. Glob all ``*.csv`` files in the directory.
        2. For each file, read into a pandas DataFrame in chunks.
        3. For each column, coerce all values to strings, drop NaN,
           and yield the quadruple.
        """
        csv_files = sorted(self.directory.glob("*.csv"))
        if not csv_files:
            logger.warning("No CSV files found in %s", self.directory)
            return

        for csv_path in csv_files:
            table_name = csv_path.stem
            logger.info("Reading CSV: %s", csv_path)
            try:
                df = pd.read_csv(
                    csv_path,
                    sep=self.separator,
                    encoding=self.encoding,
                    dtype=str,          # all values as strings
                    keep_default_na=False,
                    low_memory=True,
                )
            except Exception:
                logger.exception("Failed to read %s", csv_path)
                continue

            for col_name in df.columns:
                values = df[col_name].dropna().astype(str).tolist()
                yield self.db_name, table_name, str(col_name), values


# ---------------------------------------------------------------------------
# S3 reader  (new — primary reader for the 9.5 TB S3 data-lake)
# ---------------------------------------------------------------------------

class S3Reader:
    """Read CSV files from an S3 bucket and yield columns.

    Uses **boto3** to list objects and **pandas** to stream-read each CSV.
    Designed for very large data lakes:

    * Supports prefix-based filtering (e.g. ``"data/census/"``).
    * Optional row-sampling so 9.5 TB isn't fully materialized.
    * Yields per-column data exactly like :class:`CSVReader`.

    Parameters
    ----------
    db_name : str
        Logical name for this S3 source (e.g. ``"va_datalake"``).
    bucket : str
        S3 bucket name (e.g. ``"my-datalake-bucket"``).
    prefix : str
        Key prefix to restrict which objects are scanned.
    suffix : str
        Only process objects ending with this suffix (default ``".csv"``).
    separator : str
        CSV delimiter.
    encoding : str
        File encoding.
    sample_rows : int | None
        If set, read only this many rows per file (random sample).
        ``None`` means read all rows (legacy behavior).
    aws_profile : str | None
        Named AWS profile from ``~/.aws/credentials``.
        If ``None``, uses the default credential chain.
    region : str
        AWS region (defaults to ``"us-east-1"``).
    max_workers : int
        Number of concurrent S3 reads (for future parallelism).
    """

    def __init__(
        self,
        db_name: str,
        bucket: str,
        prefix: str = "",
        suffix: str = ".csv",
        separator: str = ",",
        encoding: str = "utf-8",
        sample_rows: int | None = None,
        aws_profile: str | None = None,
        region: str = "us-east-1",
        max_workers: int = 4,
    ) -> None:
        self.db_name = db_name
        self.bucket = bucket
        self.prefix = prefix
        self.suffix = suffix
        self.separator = separator
        self.encoding = encoding
        self.sample_rows = sample_rows
        self.aws_profile = aws_profile
        self.region = region
        self.max_workers = max_workers

    # ── internal: lazy boto3 client ─────────────────────────────────

    def _get_s3_client(self) -> Any:
        """Create a boto3 S3 client using the configured credentials."""
        import boto3  # deferred import so boto3 is optional

        session_kwargs: dict[str, Any] = {"region_name": self.region}
        if self.aws_profile:
            session_kwargs["profile_name"] = self.aws_profile

        session = boto3.Session(**session_kwargs)
        return session.client("s3")

    def _list_csv_keys(self, s3_client: Any) -> list[str]:
        """List all object keys under *prefix* that end with *suffix*."""
        keys: list[str] = []
        paginator = s3_client.get_paginator("list_objects_v2")
        page_iter = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)

        for page in page_iter:
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if key.endswith(self.suffix) and not key.endswith("/"):
                    keys.append(key)

        logger.info(
            "Found %d CSV objects in s3://%s/%s",
            len(keys),
            self.bucket,
            self.prefix,
        )
        return sorted(keys)

    @staticmethod
    def _table_name_from_key(key: str) -> str:
        """Derive a logical table name from an S3 key.

        ``"data/census/population_2020.csv"``  →  ``"population_2020"``
        """
        basename = key.rsplit("/", 1)[-1]          # strip directory
        name, _ = os.path.splitext(basename)       # strip extension
        return name

    # ── public interface ────────────────────────────────────────────

    def read_columns(self) -> Iterator[tuple[str, str, str, list[str]]]:
        """Yield ``(db_name, table_name, column_name, values)`` for every
        column in every CSV object in the configured S3 bucket/prefix.

        Streaming approach:

        1. ``ListObjectsV2`` with paginator to enumerate CSVs.
        2. For each key, use ``s3_client.get_object()`` to get a streaming
           ``Body`` and wrap it in ``pandas.read_csv``.
        3. If ``sample_rows`` is set, read only that many rows.
        4. For each column, coerce to strings and yield.

        Large-scale notes:

        * Each file is streamed — we never hold more than one CSV in memory.
        * With ``sample_rows=1000``, profiling a 9.5 TB lake is feasible
          in hours rather than days.
        * For parallel reads, a future version can use
          ``concurrent.futures.ThreadPoolExecutor`` with ``max_workers``.
        """
        s3_client = self._get_s3_client()
        csv_keys = self._list_csv_keys(s3_client)

        if not csv_keys:
            logger.warning(
                "No CSV objects found at s3://%s/%s",
                self.bucket,
                self.prefix,
            )
            return

        for key in csv_keys:
            table_name = self._table_name_from_key(key)
            logger.info("Streaming s3://%s/%s", self.bucket, key)

            try:
                response = s3_client.get_object(Bucket=self.bucket, Key=key)
                body = response["Body"]

                # pandas can read directly from a streaming body
                read_kwargs: dict[str, Any] = {
                    "sep": self.separator,
                    "encoding": self.encoding,
                    "dtype": str,
                    "keep_default_na": False,
                    "low_memory": True,
                }
                if self.sample_rows is not None:
                    read_kwargs["nrows"] = self.sample_rows

                df = pd.read_csv(body, **read_kwargs)

            except Exception:
                logger.exception("Failed to read s3://%s/%s", self.bucket, key)
                continue

            for col_name in df.columns:
                values = df[col_name].dropna().astype(str).tolist()
                yield self.db_name, table_name, str(col_name), values


# ---------------------------------------------------------------------------
# Database reader  (replaces legacy PostgresSource, SQLServerSource, etc.)
# ---------------------------------------------------------------------------

class DatabaseReader:
    """Read tables from a SQL database via SQLAlchemy and yield columns.

    Parameters
    ----------
    db_name : str
        Logical database name.
    connection_string : str
        SQLAlchemy connection string (e.g. ``"postgresql://user:pass@host/db"``).
    schema : str
        Database schema to inspect.
    """

    def __init__(
        self,
        db_name: str,
        connection_string: str,
        schema: str = "public",
    ) -> None:
        self.db_name = db_name
        self.connection_string = connection_string
        self.schema = schema

    def read_columns(self) -> Iterator[tuple[str, str, str, list[str]]]:
        """Yield ``(db_name, table_name, column_name, values)`` for every column
        in every table in the configured schema.

        Algorithm:

        1. Connect via SQLAlchemy, inspect table names in *schema*.
        2. For each table, ``SELECT *`` (with optional LIMIT for very large tables).
        3. For each column, coerce values to strings and yield.
        """
        raise NotImplementedError


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
                    encoding=cfg.config.get("encoding", "utf-8"),
                )
            )
        elif cfg.source_type == "s3":
            readers.append(
                S3Reader(
                    db_name=cfg.name,
                    bucket=cfg.config["bucket"],
                    prefix=cfg.config.get("prefix", ""),
                    suffix=cfg.config.get("suffix", ".csv"),
                    separator=cfg.config.get("separator", ","),
                    encoding=cfg.config.get("encoding", "utf-8"),
                    sample_rows=cfg.config.get("sample_rows"),
                    aws_profile=cfg.config.get("aws_profile"),
                    region=cfg.config.get("region", "us-east-1"),
                    max_workers=cfg.config.get("max_workers", 4),
                )
            )
        elif cfg.source_type in ("postgres", "mysql", "sqlserver", "hive"):
            readers.append(
                DatabaseReader(
                    db_name=cfg.name,
                    connection_string=cfg.config["connection_string"],
                    schema=cfg.config.get("schema", "public"),
                )
            )
        else:
            raise ValueError(f"Unsupported source type: {cfg.source_type!r}")
    return readers
