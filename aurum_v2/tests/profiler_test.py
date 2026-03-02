#!/usr/bin/env python3
"""
Profile columns for a list of S3 URIs and print the results.

Usage:
    # Explicit URIs
    python profiler_test.py s3://bucket/path/file.csv s3://bucket/other.json

    # From a text file (one URI per line)
    python profiler_test.py --uri-file verified_datasources_for_tests.txt

    # Limit rows sampled per file
    python profiler_test.py --uri-file uris.txt --sample-rows 5000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aurum_v2.config import AurumConfig
from aurum_v2.profiler.column_profiler import Profiler
from aurum_v2.profiler.source_readers import S3Reader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("profiler_test")


def load_uris_from_file(path: str) -> list[str]:
    uris = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("s3://"):
                uris.append(line)
    return uris


def write_profiles(profiler: Profiler, out_path: Path) -> None:
    profiles = profiler.profiles
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Profiled {len(profiles)} columns\n\n")
        for p in profiles:
            f.write(f"nid:          {p.nid}\n")
            f.write(f"db_name:      {p.db_name}\n")
            f.write(f"source_name:  {p.source_name}\n")
            f.write(f"column_name:  {p.column_name}\n")
            f.write(f"data_type:    {p.data_type}\n")
            f.write(f"total_values: {p.total_values}\n")
            f.write(f"unique_values:{p.unique_values}\n")
            f.write(f"path:         {p.path}\n")
            f.write(f"entities:     {p.entities}\n")
            f.write(f"min_value:    {p.min_value}\n")
            f.write(f"max_value:    {p.max_value}\n")
            f.write(f"avg_value:    {p.avg_value}\n")
            f.write(f"median:       {p.median}\n")
            f.write(f"iqr:          {p.iqr}\n")
            f.write(f"minhash:      {p.minhash}\n")
            f.write(f"raw_values:   {p.raw_values[:1000]}\n")
            f.write("\n")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Profile columns from S3 URIs")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("uris", nargs="*", metavar="URI", help="s3:// URIs to profile")
    src.add_argument("--uri-file", type=str, help="Text file of s3:// URIs (one per line)")

    parser.add_argument("--sample-rows", type=int, default=10_000, help="Max rows per file (default 10000)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel profiler workers")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--out", type=str, default="profiler_results.txt", help="Output file path")
    args = parser.parse_args()

    if args.uri_file:
        uris = load_uris_from_file(args.uri_file)
        if not uris:
            logger.error("No s3:// URIs found in %s", args.uri_file)
            sys.exit(1)
        logger.info("Loaded %d URIs from %s", len(uris), args.uri_file)
    else:
        uris = args.uris
        if not uris:
            parser.print_help()
            sys.exit(1)

    reader = S3Reader(
        db_name="s3",
        s3_paths=uris,
        region=args.region,
        limit_values=True,
        max_values=args.sample_rows,
    )

    config = AurumConfig()
    profiler = Profiler(config)
    profiler.run([reader], max_workers=args.workers)

    out_path = Path(args.out)
    write_profiles(profiler, out_path)
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()