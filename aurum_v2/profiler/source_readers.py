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

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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
    """

    def __init__(
        self,
        db_name: str,
        s3_paths: list[str],
        region: str = "us-east-2",
        limit_values: bool = False,
        max_values: int = 10_000,
    ) -> None:
        self.db_name = db_name
        self.s3_paths = s3_paths
        self.region = region
        self.limit_values = limit_values
        self.max_values = max_values

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
        return Path(path).stem

    def read_columns(self) -> Iterator[tuple[str, str, str, str, list[str], str]]:
        con = self._connect()

        for path in self.s3_paths:
            table_name = self._table_name_from_path(path)
            logger.info("Reading %s", path)

            try:
                if self.limit_values:
                    query = (
                        f"SELECT * FROM read_csv_auto(?)"
                        f" USING SAMPLE {self.max_values} ROWS (reservoir)"
                    )
                    df = con.execute(query, [path]).df()
                else:
                    query = (
                        "SELECT * FROM read_csv_auto(?)"
                    )
                    df = con.execute(query, [path]).df()
            except Exception:
                logger.exception("Failed to read %s", path)
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