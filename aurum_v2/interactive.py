#!/usr/bin/env python3
"""
Aurum v2 — Interactive Explorer

Three modes in one script:
  1. **Graph visualization** — renders the relationship graph as an interactive HTML network.
  2. **Query REPL** — run Algebra queries interactively (search, neighbors, paths, set ops).
  3. **Table preview** — read live data from S3 via DuckDB httpfs.

Usage:
    python -m aurum_v2.interactive --db aurum.db --model model

    # Or launch directly into a specific mode:
    python -m aurum_v2.interactive --db aurum.db --model model --viz          # graph only
    python -m aurum_v2.interactive --db aurum.db --model model --query-only   # REPL only
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table as RichTable
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich import print as rprint

from aurum_v2.config import AurumConfig
from aurum_v2.store.duck_store import DuckStore, KWType
from aurum_v2.graph.field_network import FieldNetwork, deserialize_network
from aurum_v2.discovery.api import API, Helper
from aurum_v2.models.relation import Relation

console = Console()

# ═════════════════════════════════════════════════════════════════════
#  1. Graph Visualization
# ═════════════════════════════════════════════════════════════════════

def visualize_graph(network: FieldNetwork, output: str = "graph.html", min_weight: float = 0.0) -> str:
    """Render the FieldNetwork as an interactive pyvis HTML network.

    Nodes = tables (clustered). Edges = Relation types color-coded.
    Edge labels show the max similarity score among underlying column pairs.
    A slider in the HTML lets you hide edges below a weight threshold.

    Parameters
    ----------
    min_weight : float
        Initial minimum edge weight to display (0.0 = show all).
    """
    from pyvis.network import Network

    # Collect unique tables and their columns
    table_columns: dict[str, list[str]] = {}
    id_info = network._get_underlying_repr_id_to_field_info()
    for nid, (db, source, field, dtype) in id_info.items():
        table_columns.setdefault(source, []).append((nid, field, dtype))

    # Color map for relations
    rel_colors = {
        Relation.SCHEMA_SIM: "#3498db",       # blue
        Relation.CONTENT_SIM: "#2ecc71",      # green
        Relation.PKFK: "#e74c3c",             # red
        Relation.INCLUSION_DEPENDENCY: "#f39c12",  # orange
        Relation.ENTITY_SIM: "#9b59b6",       # purple
    }

    # Build pyvis network
    net = Network(
        height="900px", width="100%",
        bgcolor="#1a1a2e", font_color="white",
        directed=False, notebook=False,
    )
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.01,
        damping=0.09,
    )

    # Add table nodes
    import hashlib
    added_tables: set[str] = set()
    nid_to_table: dict[str, str] = {}

    for table, cols in table_columns.items():
        col_list = "\n".join(f"  {'N' if d == 'N' else 'T'} {f}" for _, f, d in cols[:20])
        if len(cols) > 20:
            col_list += f"\n  ... +{len(cols) - 20} more"

        hue = int(hashlib.md5(table.encode()).hexdigest()[:6], 16) % 360
        color = f"hsl({hue}, 70%, 50%)"

        net.add_node(
            table,
            label=table[:40],
            title=f"<b>{table}</b>\n{len(cols)} columns\n{col_list}",
            color=color,
            size=15 + min(len(cols), 30),
            font={"size": 12},
        )
        added_tables.add(table)
        for nid, _, _ in cols:
            nid_to_table[nid] = table

    # Aggregate edges at table level — keep max score per (table_a, table_b, relation)
    edge_scores: dict[tuple[str, str, str], float] = {}
    G = network._get_underlying_repr_graph()

    for u, v, key, data in G.edges(keys=True, data=True):
        if not isinstance(key, Relation):
            continue
        t_u = nid_to_table.get(u)
        t_v = nid_to_table.get(v)
        if not t_u or not t_v or t_u == t_v:
            continue
        edge_key = (min(t_u, t_v), max(t_u, t_v), key.name)
        score = data.get("score", 0.0)
        if edge_key not in edge_scores or score > edge_scores[edge_key]:
            edge_scores[edge_key] = score

    # Add edges with score labels
    for (t_a, t_b, rel_name), score in edge_scores.items():
        if score < min_weight:
            continue
        rel = Relation[rel_name]
        color = rel_colors.get(rel, "#aaaaaa")
        label = f"{score:.2f}" if score > 0 else ""
        net.add_edge(
            t_a, t_b,
            color=color,
            title=f"{rel_name}  score={score:.4f}",
            label=label,
            value=score,    # pyvis uses 'value' to scale width
            width=1 + score * 4,
            font={"size": 10, "color": "#cccccc", "strokeWidth": 0},
        )

    # Legend HTML
    legend_html = "<h3>Aurum v2 — Relationship Graph</h3>"
    legend_html += f"<p>{len(added_tables)} tables, {len(edge_scores)} cross-table edges</p>"
    legend_html += "<ul>"
    for rel, color in rel_colors.items():
        legend_html += f'<li style="color:{color}">■ {rel.name}</li>'
    legend_html += "</ul>"
    net.heading = legend_html

    net.save_graph(output)

    # Inject a weight-threshold slider into the saved HTML
    _inject_weight_slider(output)

    console.print(f"[green]Graph saved to [bold]{output}[/bold][/green]")
    return output


def _inject_weight_slider(html_path: str) -> None:
    """Post-process the pyvis HTML to add a weight filter slider."""
    slider_html = """
<div id="weight-filter" style="position:fixed;top:10px;right:10px;z-index:9999;
  background:rgba(26,26,46,0.95);padding:14px 18px;border-radius:8px;
  color:#fff;font-family:sans-serif;font-size:13px;box-shadow:0 2px 12px rgba(0,0,0,0.5);">
  <label for="weightSlider"><b>Min edge weight:</b></label>
  <input type="range" id="weightSlider" min="0" max="1" step="0.01" value="0"
    style="width:160px;vertical-align:middle;">
  <span id="weightVal" style="margin-left:6px;font-weight:bold;">0.00</span>
</div>
<script>
(function(){
  var slider = document.getElementById('weightSlider');
  var valSpan = document.getElementById('weightVal');
  // stash original edges
  var origEdges = null;
  function waitForNetwork() {
    if (typeof edges === 'undefined' || !edges) {
      setTimeout(waitForNetwork, 200);
      return;
    }
    if (!origEdges) origEdges = edges.get();
    slider.addEventListener('input', function(){
      var th = parseFloat(this.value);
      valSpan.textContent = th.toFixed(2);
      var filtered = origEdges.filter(function(e){ return (e.value || 0) >= th; });
      edges.clear();
      edges.add(filtered);
    });
  }
  waitForNetwork();
})();
</script>
"""
    with open(html_path, "r") as f:
        html = f.read()
    # Insert before closing </body>
    html = html.replace("</body>", slider_html + "\n</body>")
    with open(html_path, "w") as f:
        f.write(html)


# ═════════════════════════════════════════════════════════════════════
#  2. Stats display
# ═════════════════════════════════════════════════════════════════════

def show_stats(network: FieldNetwork, duck: DuckStore):
    """Print summary statistics of the built graph."""
    G = network._get_underlying_repr_graph()
    id_info = network._get_underlying_repr_id_to_field_info()
    source_ids = network._get_underlying_repr_table_to_ids()

    # Count edges by relation type
    rel_counts: dict[str, int] = {}
    for u, v, key in G.edges(keys=True):
        name = key.name if isinstance(key, Relation) else str(key)
        rel_counts[name] = rel_counts.get(name, 0) + 1

    table = RichTable(title="Graph Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")
    table.add_row("Total columns (nodes)", str(len(id_info)))
    table.add_row("Total tables", str(len(source_ids)))
    table.add_row("Total edges", str(G.number_of_edges()))

    for rel_name, count in sorted(rel_counts.items()):
        table.add_row(f"  {rel_name} edges", str(count))

    # Top-degree hubs
    top_k = network.fields_degree(5)
    for i, (nid, deg) in enumerate(top_k):
        db, src, field, _ = id_info.get(nid, ("?", "?", "?", "?"))
        table.add_row(f"  Hub #{i+1}", f"{src}.{field} (degree={deg})")

    console.print(table)


# ═════════════════════════════════════════════════════════════════════
#  3. Query REPL
# ═════════════════════════════════════════════════════════════════════

HELP_TEXT = """
[bold cyan]Available Commands:[/]

[green]search <keyword>[/]            — Content search (find tables/columns by value)
[green]search_attr <keyword>[/]       — Schema search (find columns by name)
[green]search_table <keyword>[/]      — Table name search
[green]neighbors <nid> <relation>[/]  — Find neighbors (PKFK, CONTENT_SIM, SCHEMA_SIM)
[green]path <nid1> <nid2>[/]          — Find join path between two columns
[green]info <nid>[/]                  — Reverse lookup: nid → (db, table, column, path)
[green]tables[/]                      — List all profiled tables
[green]hubs [k][/]                    — Show top-k highest-degree columns
[green]edges <relation>[/]            — List all edges of a relation type
[green]stats[/]                       — Graph summary statistics
[green]viz[/]                         — Open interactive graph visualization
[green]help[/]                        — Show this message
[green]quit[/]                        — Exit
"""

def _print_drs(drs, api: API, max_rows: int = 30):
    """Pretty-print a DRS result as a Rich table."""
    table = RichTable(show_header=True)
    table.add_column("#", style="dim")
    table.add_column("NID", style="cyan")
    table.add_column("Table", style="green")
    table.add_column("Column", style="yellow")
    table.add_column("Score", style="bold white")

    seen = set()
    count = 0
    for hit in drs:
        if hit.nid in seen:
            continue
        seen.add(hit.nid)
        count += 1
        if count > max_rows:
            table.add_row("...", f"+{drs.size() - max_rows} more", "", "", "")
            break
        table.add_row(
            str(count),
            str(hit.nid),
            hit.source_name,
            hit.field_name,
            f"{hit.score:.4f}" if hit.score else "0",
        )

    console.print(table)
    console.print(f"[dim]{drs.size()} total hits[/]")


def _parse_relation(s: str) -> Relation | None:
    """Parse a relation string like 'PKFK' or 'CONTENT_SIM'."""
    s = s.upper().strip()
    try:
        return Relation[s]
    except KeyError:
        console.print(f"[red]Unknown relation: {s}. Options: {', '.join(r.name for r in Relation)}[/]")
        return None


def query_repl(api: API, duck: DuckStore, network: FieldNetwork):
    """Interactive REPL for running Algebra queries."""
    console.print(Panel(HELP_TEXT, title="Aurum v2 Interactive Query Explorer", border_style="cyan"))

    while True:
        try:
            raw = Prompt.ask("\n[bold cyan]aurum[/]")
        except (EOFError, KeyboardInterrupt):
            break

        raw = raw.strip()
        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        try:
            if cmd in ("quit", "exit", "q"):
                break

            elif cmd == "help":
                console.print(HELP_TEXT)

            elif cmd == "search":
                if not arg:
                    console.print("[red]Usage: search <keyword>[/]")
                    continue
                drs = api.search_content(arg, max_results=20)
                _print_drs(drs, api)

            elif cmd == "search_attr":
                if not arg:
                    console.print("[red]Usage: search_attr <keyword>[/]")
                    continue
                drs = api.search_attribute(arg, max_results=20)
                _print_drs(drs, api)

            elif cmd == "search_table":
                if not arg:
                    console.print("[red]Usage: search_table <keyword>[/]")
                    continue
                drs = api.search_table(arg, max_results=20)
                _print_drs(drs, api)

            elif cmd == "neighbors":
                tokens = arg.split()
                if len(tokens) < 2:
                    console.print("[red]Usage: neighbors <nid> <PKFK|CONTENT_SIM|SCHEMA_SIM>[/]")
                    continue
                nid, rel_str = tokens[0], tokens[1]
                rel = _parse_relation(rel_str)
                if rel is None:
                    continue
                drs = api._neighbor_search(nid, rel)
                _print_drs(drs, api)

            elif cmd == "path":
                tokens = arg.split()
                if len(tokens) < 2:
                    console.print("[red]Usage: path <nid1> <nid2>[/]")
                    continue
                nid1, nid2 = tokens[0], tokens[1]
                h1 = api._nid_to_hit(nid1)
                h2 = api._nid_to_hit(nid2)
                drs = network.find_path_hit(h1, h2, Relation.PKFK, max_hops=4)
                if drs.size() == 0:
                    console.print("[yellow]No PKFK path found. Trying CONTENT_SIM...[/]")
                    drs = network.find_path_hit(h1, h2, Relation.CONTENT_SIM, max_hops=4)
                if drs.size() == 0:
                    console.print("[red]No path found within 4 hops.[/]")
                else:
                    _print_drs(drs, api)
                    # Show provenance chain
                    paths = drs.paths()
                    if paths:
                        console.print("\n[bold]Provenance path:[/]")
                        for p in paths[:3]:
                            chain = " → ".join(f"{h.source_name}.{h.field_name}" for h in p)
                            console.print(f"  {chain}")

            elif cmd == "info":
                if not arg:
                    console.print("[red]Usage: info <nid>[/]")
                    continue
                info = network.get_info_for([arg.strip()])
                if info:
                    nid, db, src, field = info[0]
                    s3_path = duck.get_path_of(nid)
                    console.print(f"  NID:    {nid}")
                    console.print(f"  DB:     {db}")
                    console.print(f"  Table:  {src}")
                    console.print(f"  Column: {field}")
                    if s3_path:
                        console.print(f"  Path:   {s3_path}")
                else:
                    console.print(f"[red]NID {arg} not found[/]")

            elif cmd == "tables":
                source_ids = network._get_underlying_repr_table_to_ids()
                table = RichTable(title=f"All Tables ({len(source_ids)})", show_header=True)
                table.add_column("#", style="dim")
                table.add_column("Table Name", style="green")
                table.add_column("Columns", style="cyan")
                for i, (tname, nids) in enumerate(sorted(source_ids.items()), 1):
                    table.add_row(str(i), tname, str(len(nids)))
                console.print(table)

            elif cmd == "hubs":
                k = int(arg) if arg else 10
                top = network.fields_degree(k)
                id_info = network._get_underlying_repr_id_to_field_info()
                table = RichTable(title=f"Top {k} Hub Columns", show_header=True)
                table.add_column("NID", style="cyan")
                table.add_column("Table", style="green")
                table.add_column("Column", style="yellow")
                table.add_column("Degree", style="bold white")
                for nid, deg in top:
                    _, src, field, _ = id_info.get(nid, ("?", "?", "?", "?"))
                    table.add_row(str(nid), src, field, str(deg))
                console.print(table)

            elif cmd == "edges":
                if not arg:
                    console.print("[red]Usage: edges <PKFK|CONTENT_SIM|SCHEMA_SIM>[/]")
                    continue
                rel = _parse_relation(arg)
                if rel is None:
                    continue
                pairs = list(network.enumerate_relation(rel, as_str=True))
                table = RichTable(title=f"{rel.name} edges ({len(pairs)})", show_header=True)
                table.add_column("Source", style="green")
                table.add_column("Target", style="yellow")
                for src, tgt in pairs[:50]:
                    table.add_row(src, tgt)
                if len(pairs) > 50:
                    table.add_row("...", f"+{len(pairs) - 50} more")
                console.print(table)

            elif cmd == "stats":
                show_stats(network, duck)

            elif cmd == "viz":
                out = visualize_graph(network)
                webbrowser.open(f"file://{Path(out).resolve()}")

            else:
                console.print(f"[red]Unknown command: {cmd}. Type 'help' for options.[/]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            import traceback
            traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Aurum v2 Interactive Explorer")
    parser.add_argument("--db", type=str, default="aurum.db", help="DuckDB file path")
    parser.add_argument("--model", type=str, default="model", help="Directory with graph pickles")
    parser.add_argument("--viz", action="store_true", help="Open graph visualization and exit")
    parser.add_argument("--query-only", action="store_true", help="Skip visualization, go straight to REPL")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    model_dir = Path(args.model).resolve()

    if not (model_dir / "graph.pickle").exists():
        console.print(f"[red]No graph found at {model_dir}. Run pipeline.py first.[/]")
        sys.exit(1)

    config = AurumConfig()
    duck = DuckStore(config, db_path)
    network = deserialize_network(str(model_dir))
    api = API(network, duck)

    console.print(Panel(
        f"[bold]Aurum v2 Interactive Explorer[/]\n"
        f"DB: {db_path}\n"
        f"Model: {model_dir}\n"
        f"Tables: {network.get_number_tables()}  |  "
        f"Columns: {network.graph_order()}  |  "
        f"Edges: {network._get_underlying_repr_graph().number_of_edges()}",
        border_style="green",
    ))

    if args.viz:
        # Graph viz only
        out = visualize_graph(network)
        webbrowser.open(f"file://{Path(out).resolve()}")
        return

    if not args.query_only:
        # Show stats + graph on startup
        show_stats(network, duck)
        out = visualize_graph(network)
        console.print(f"[dim]Opening graph in browser...[/]")
        webbrowser.open(f"file://{Path(out).resolve()}")

    # Enter REPL
    query_repl(api, duck, network)

    duck.close()
    console.print("[dim]Goodbye.[/]")


if __name__ == "__main__":
    main()
