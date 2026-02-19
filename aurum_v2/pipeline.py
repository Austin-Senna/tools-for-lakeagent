#!/usr/bin/env python3
"""
Aurum v2 — End-to-end pipeline: Profile S3 tables → Build graph → Serialize.

Usage:
    # Profile + build from s3_list.py search results
    python -m aurum_v2.pipeline --query "veterans disability" --limit 5

    # Profile + build the FIRST 100 tables from a text file of s3:// URIs
    python -m aurum_v2.pipeline --uri-file verified_s3_tables.txt --n-files 100

    # Profile + build from explicit URIs
    python -m aurum_v2.pipeline --uris s3://bucket/path1.csv s3://bucket/path2.csv

    # Skip profiling, just rebuild graph from existing DuckDB
    python -m aurum_v2.pipeline --rebuild

Outputs:
    model/          — graph.pickle, id_info.pickle, table_ids.pickle
    aurum.db        — DuckDB with profile + text_index tables
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure the package root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aurum_v2.config import AurumConfig
from aurum_v2.profiler.column_profiler import Profiler
from aurum_v2.profiler.source_readers import S3Reader
from aurum_v2.store.duck_store import DuckStore
from aurum_v2.builder.coordinator import build_network

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────
# S3 URI resolution (delegates to s3_list.py)
# ─────────────────────────────────────────────────────────────────────

def resolve_uris_from_query(query: str, limit: int = 5) -> list[str]:
    """Use s3_list.py's search_keyword + get_s3_uris_for_tables."""
    from aurum_v2.s3_list import search_keyword, get_s3_uris_for_tables

    logger.info("Searching data.gov for '%s' (limit=%d datasets)...", query, limit)
    results = search_keyword([query], limit=limit)
    dataset_ids = [r["dataset_id"] for r in results.get("results", [])]
    if not dataset_ids:
        logger.warning("No datasets found for '%s'", query)
        return []

    logger.info("Found %d datasets: %s", len(dataset_ids), dataset_ids[:5])
    logger.info("Peeking inside datasets to find actual tables...")
    uris = get_s3_uris_for_tables(dataset_ids)
    logger.info("Resolved %d S3 table URIs", len(uris))
    return uris


def load_uris_from_file(path: str, n_files: int | None = None) -> list[str]:
    """Read s3:// URIs from a text file, optionally limiting to n_files."""
    uris = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("s3://"):
                uris.append(line)
                # Stop reading early if we hit the requested limit
                if n_files and len(uris) >= n_files:
                    break
                    
    logger.info("Loaded %d URIs from %s", len(uris), path)
    return uris


# ─────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────

def stage_profile(
    uris: list[str],
    config: AurumConfig,
    duck: DuckStore,
    *,
    sample_rows: int = 10_000,
    max_workers: int = 4,
    region: str = "us-east-1",
) -> int:
    """Stage 1: Profile S3 tables → DuckDB."""
    t0 = time.time()

    # Pass the standardized S3Reader parameters
    reader = S3Reader(
        db_name="s3",
        s3_paths=uris,
        region=region,
        limit_values=True, # Force sampling so we don't blow up memory
        max_values=sample_rows,
    )

    profiler = Profiler(config)
    profiler.run([reader], max_workers=max_workers)

    result = profiler.store_profiles(duck=duck)
    n = result.get("duckdb", 0)

    logger.info(
        "Profiled %d columns from %d tables in %.1fs",
        n, len(uris), time.time() - t0,
    )
    return n


def stage_build(
    config: AurumConfig,
    duck: DuckStore,
    output_path: str,
) -> None:
    """Stage 2: Build relationship graph from DuckDB profiles."""
    logger.info("Building relationship graph...")
    build_network(config, output_path, duck=duck)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Aurum v2 pipeline: profile S3 tables → build graph",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--query", type=str, help="Search data.gov via s3_list.py")
    src.add_argument("--uri-file", type=str, help="Text file of s3:// URIs (one per line)")
    src.add_argument("--uris", nargs="+", help="Explicit s3:// URIs")
    src.add_argument("--rebuild", action="store_true", help="Rebuild graph from existing DuckDB (skip profiling)")

    parser.add_argument("--n-files", type=int, default=None, help="Max number of S3 URIs to process from the file or list")
    parser.add_argument("--limit", type=int, default=5, help="Max datasets from data.gov search (default 5)")
    parser.add_argument("--sample-rows", type=int, default=10_000, help="Rows to reservoir-sample per S3 file")
    parser.add_argument("--workers", type=int, default=4, help="Profiler worker processes")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--db", type=str, default="aurum.db", help="DuckDB file path")
    parser.add_argument("--model-dir", type=str, default="model", help="Output directory for graph artifacts")
    args = parser.parse_args()

    config = AurumConfig()
    db_path = Path(args.db).resolve()
    model_dir = Path(args.model_dir).resolve()

    duck = DuckStore(config, db_path)
    # Don't recreate the tables if we are just appending more files to an existing DB
    duck.init_tables(recreate=not args.rebuild)

    # ── Resolve S3 URIs ───────────────────────────────────────────────
    if args.rebuild:
        logger.info("Rebuilding graph from existing DuckDB at %s", db_path)
        
    elif args.query:
        uris = resolve_uris_from_query(args.query, limit=args.limit)
        if args.n_files:
            uris = uris[:args.n_files]
            
        if not uris:
            logger.error("No tables found. Exiting.")
            sys.exit(1)
        stage_profile(uris, config, duck, sample_rows=args.sample_rows, max_workers=args.workers, region=args.region)
        
    elif args.uri_file:
        uris = load_uris_from_file(args.uri_file, n_files=args.n_files)
        if not uris:
            logger.error("No URIs in file. Exiting.")
            sys.exit(1)
        stage_profile(uris, config, duck, sample_rows=args.sample_rows, max_workers=args.workers, region=args.region)
        
    elif args.uris:
        uris = args.uris[:args.n_files] if args.n_files else args.uris
        stage_profile(uris, config, duck, sample_rows=args.sample_rows, max_workers=args.workers, region=args.region)
        
    else:
        parser.print_help()
        sys.exit(1)

    # ── Build graph ───────────────────────────────────────────────────
    stage_build(config, duck, str(model_dir))

    duck.close()
    logger.info("Done.  Model at %s, DB at %s", model_dir, db_path)


if __name__ == "__main__":
    main()