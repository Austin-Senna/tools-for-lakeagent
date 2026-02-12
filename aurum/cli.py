"""
aurum CLI — command-line interface for data-lake discovery.

Commands
--------
- ``build-index`` — profile a directory of CSV/Parquet files and build
  the field network (graph + similarity edges).
- ``search`` — keyword search over column names.
- ``discover`` — Data-on-Demand: find virtual schemas from a list of
  desired columns.

Usage::

    aurum build-index --data-dir ./tables --output ./index
    aurum search --index ./index --keyword employee_name
    aurum discover --index ./index --attrs name salary dept
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from aurum.config import aurumConfig

console = Console()


# ── Shared options ───────────────────────────────────────────────────

@click.group()
@click.version_option(package_name="aurum")
def main() -> None:
    """aurum — modern data-lake discovery toolkit."""


# ── build-index ──────────────────────────────────────────────────────

@main.command("build-index")
@click.option("--data-dir", required=True, type=click.Path(exists=True, path_type=Path), help="Directory of CSV/Parquet files.")
@click.option("--output", required=True, type=click.Path(path_type=Path), help="Output directory for the index.")
@click.option("--minhash-perms", default=256, show_default=True, help="Number of MinHash permutations.")
@click.option("--schema-model", default="all-MiniLM-L6-v2", show_default=True, help="Sentence-Transformer model name.")
def build_index(
    data_dir: Path,
    output: Path,
    minhash_perms: int,
    schema_model: str,
) -> None:
    """Profile tables and build the field network index."""
    from aurum.graph.field_network import FieldNetwork
    from aurum.graph.network_builder import build_all
    from aurum.profiler.column_profiler import profile_directory

    cfg = aurumConfig(minhash_perms=minhash_perms, schema_sim_model=schema_model)

    console.print(f"[bold blue]Profiling[/] {data_dir} …")
    profiles = profile_directory(data_dir, cfg)
    console.print(f"  → {len(profiles)} column profiles extracted")

    console.print("[bold blue]Building field network[/] …")
    net = FieldNetwork()
    net.init_from_profiles(profiles)

    console.print("[bold blue]Computing similarity edges[/] …")
    build_all(net, profiles, cfg)

    output.mkdir(parents=True, exist_ok=True)
    net.save(output)
    console.print(f"[bold green]✓[/] Index saved to {output}")


# ── search ───────────────────────────────────────────────────────────

@main.command("search")
@click.option("--index", required=True, type=click.Path(exists=True, path_type=Path), help="Index directory.")
@click.option("--keyword", required=True, help="Keyword to search for in column names.")
@click.option("--max-results", default=25, show_default=True)
def search(index: Path, keyword: str, max_results: int) -> None:
    """Search column names for a keyword."""
    from aurum.discovery.algebra import Algebra
    from aurum.graph.field_network import FieldNetwork

    net = FieldNetwork.load(index)
    api = Algebra(net)

    drs = api.search_attribute(keyword, max_results=max_results)

    table = Table(title=f"Search results for '{keyword}'")
    table.add_column("#", style="dim")
    table.add_column("Table")
    table.add_column("Column")
    table.add_column("Score", justify="right")

    for i, hit in enumerate(drs, 1):
        table.add_row(str(i), hit.source_name, hit.field_name, f"{hit.score:.2f}")

    console.print(table)
    console.print(f"{len(drs)} result(s)")


# ── discover ─────────────────────────────────────────────────────────

@main.command("discover")
@click.option("--index", required=True, type=click.Path(exists=True, path_type=Path), help="Index directory.")
@click.option("--data-dir", required=True, type=click.Path(exists=True, path_type=Path), help="Directory of source tables.")
@click.option("--attrs", required=True, multiple=True, help="Desired column names (repeat --attrs for each).")
@click.option("--max-hops", default=2, show_default=True)
def discover(index: Path, data_dir: Path, attrs: tuple[str, ...], max_hops: int) -> None:
    """Discover virtual schemas covering the requested attributes."""
    from aurum.discovery.algebra import Algebra
    from aurum.discovery.data_on_demand import DataOnDemand
    from aurum.graph.field_network import FieldNetwork

    net = FieldNetwork.load(index)
    api = Algebra(net)
    dod = DataOnDemand(net, api, data_dir=data_dir)

    console.print(f"[bold blue]Discovering virtual schemas[/] for: {', '.join(attrs)}")

    for i, result in enumerate(dod.discover(list(attrs)), 1):
        console.print(f"\n[bold green]View #{i}[/]")
        console.print(f"  Tables: {result.metadata.get('tables', [])}")
        console.print(f"  Joins:  {result.metadata.get('joins', 0)}")
        if result.join_graph:
            for l, r in result.join_graph:
                console.print(f"    {l.source_name}.{l.field_name} ↔ {r.source_name}.{r.field_name}")


if __name__ == "__main__":
    main()
