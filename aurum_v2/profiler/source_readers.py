"""
Pluggable data-source readers.

Each reader yields ``(table_name, column_name, values_iterator)`` triples
that the :mod:`column_profiler` consumes.  This replaces the legacy Java
``CSVSource``, ``PostgresSource``, ``HiveSource``, and ``SQLServerSource``
implementations in ``ddprofiler/src/main/java/sources/``.

Modern approach: pandas/polars for CSV, SQLAlchemy for databases.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "SourceConfig",
    "SourceReader",
    "CSVReader",
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
        2. For each file, read into a pandas DataFrame.
        3. For each column, coerce all values to strings, drop NaN,
           and yield the quadruple.
        """
        raise NotImplementedError


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
