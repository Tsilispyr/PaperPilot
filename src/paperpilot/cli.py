"""Command-line interface for PaperPilot."""
from __future__ import annotations

from typing import Literal

import typer
from rich.console import Console

console = Console()

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
ingest_app = typer.Typer(no_args_is_help=True, help="Corpus ingestion pipeline.")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation: golden set, RAGAS, HAIC.")
serve_app = typer.Typer(no_args_is_help=True, help="Serve UI / MCP / API.")
ops_app = typer.Typer(no_args_is_help=True, help="Operational helpers (doctor, stats).")
app.add_typer(ingest_app, name="ingest")
app.add_typer(eval_app, name="eval")
app.add_typer(serve_app, name="serve")
app.add_typer(ops_app, name="ops")



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
@ingest_app.command("fetch")
def ingest_fetch(
    max_papers: int = typer.Option(None, help="Override settings.arxiv_max_papers."),
) -> None:
    """Download ~N ArXiv PDFs into data/raw/."""
    from paperpilot.ingest.arxiv_fetch import fetch_corpus

    n = fetch_corpus(max_papers=max_papers)
    console.print(f"[green]Fetched {n} papers.[/green]")


@ingest_app.command("parse")
def ingest_parse() -> None:
    """Parse data/raw/*.pdf → data/processed/*.md (with section headers)."""
    from paperpilot.ingest.parse import parse_corpus

    n = parse_corpus()
    console.print(f"[green]Parsed {n} papers.[/green]")


@ingest_app.command("index")
def ingest_index(
    version: str = typer.Option("v2", help="v1 (fixed chunks), v2 (section-aware), v3 (table-aware + diversity)."),
    recreate: bool = typer.Option(False, help="Drop & recreate the collection."),
) -> None:
    """Chunk parsed papers, embed, upsert into Qdrant."""
    if version not in ("v1", "v2", "v3"):
        raise typer.BadParameter("version must be v1, v2, or v3")
    from paperpilot.ingest.index import index_corpus

    n = index_corpus(version=version, recreate=recreate)
    console.print(f"[green]Indexed {n} chunks into {version}.[/green]")



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
@eval_app.command("golden-gen")
def eval_golden_gen(n: int = typer.Option(50, help="Candidate questions to generate.")) -> None:
    """LLM-generate candidate golden Qs into data/golden/candidates.jsonl for human review."""
    from paperpilot.eval.golden_gen import generate_candidates

    written = generate_candidates(n_candidates=n)
    console.print(
        f"[green]Wrote {written} candidates.[/green] "
        f"Review them, keep the best 30+, save as data/golden/golden_set.jsonl"
    )


@eval_app.command("ragas")
def eval_ragas(version: str = typer.Option("v2")) -> None:
    """Run RAGAS on the curated golden set against the given pipeline version."""
    if version not in ("v1", "v2", "v3"):
        raise typer.BadParameter("version must be v1, v2, or v3")
    from paperpilot.eval.ragas_eval import run_ragas

    out = run_ragas(version=version)
    console.print(f"[green]Wrote {out}[/green]")


@eval_app.command("tool-call-acc")
def eval_tool_call_acc(version: str = typer.Option("v2")) -> None:
    """Compute Tool Call Accuracy on the curated golden set."""
    if version not in ("v1", "v2", "v3"):
        raise typer.BadParameter("version must be v1, v2, or v3")
    from paperpilot.eval.tool_call_acc import run_tool_call_acc

    out = run_tool_call_acc(version=version)
    console.print(f"[green]Wrote {out}[/green]")


@eval_app.command("haic")
def eval_haic(version: str = typer.Option("v2")) -> None:
    """Run HAIC benchmarking suite on the given version."""
    if version not in ("v1", "v2", "v3"):
        raise typer.BadParameter("version must be v1, v2, or v3")
    from paperpilot.eval.haic_eval import run_haic

    out = run_haic(version=version)
    console.print(f"[green]Wrote {out}[/green]")



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
@serve_app.command("ui")
def serve_ui(port: int = 8000) -> None:
    """Hint: prefer `make ui` (which calls `chainlit run ...`)."""
    console.print(
        "[yellow]Run instead:[/yellow] "
        f"chainlit run src/paperpilot/server/chainlit_app.py --port {port}"
    )


@serve_app.command("mcp")
def serve_mcp() -> None:
    """Launch the MCP server (bonus)."""
    from paperpilot.mcp.server import main as mcp_main

    mcp_main()



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Question to ask the agent."),
    version: str = typer.Option("v2", help="v1, v2, or v3."),
    show_trace: bool = typer.Option(False, "--trace", help="Print intermediate tool calls."),
) -> None:
    """One-shot Q&A from the terminal — handy for smoke-testing without the UI."""
    if version not in ("v1", "v2", "v3"):
        raise typer.BadParameter("version must be v1, v2, or v3")
    from paperpilot.agent.graph import run_agent

    result = run_agent(question, version=version)  # type: ignore[arg-type]

    if show_trace:
        console.rule(f"[cyan]Trace ({version})[/cyan]")
        for tc in result["tool_calls"]:
            console.print(f"  → [magenta]{tc['name']}[/magenta]({tc['args']})")
        console.rule()
    console.print(result["answer"])


@app.command("compare")
def compare(question: str = typer.Argument(..., help="Question to ask both v1 and v2.")) -> None:
    """Run the same question against v1 and v2 — side-by-side for the demo + report."""
    from paperpilot.agent.graph import run_agent

    out: dict[str, dict] = {}
    for v in ("v1", "v2"):
        console.rule(f"[cyan]{v}[/cyan]")
        try:
            r = run_agent(question, version=v)  # type: ignore[arg-type]
            out[v] = r
            console.print(r["answer"])
            console.print(f"\n[dim]tool_calls: {[t['name'] for t in r['tool_calls']]}[/dim]")
        except Exception as exc:
            console.print(f"[red]{v} failed: {exc}[/red]")



        param($m)
        $prefix = $m.Groups[1].Value
        $label = $m.Groups[3].Value.Trim()
        if ($label -ne '') { "$prefix--- $label ---" } else { "${prefix}---" }
    
@ops_app.command("doctor")
def ops_doctor() -> None:
    """Ping every external dependency and report green/red."""
    from paperpilot.config import settings
    import httpx

    def ok(label: str, msg: str = "OK") -> None:
        console.print(f"[green]✓[/green] {label}: {msg}")

    def bad(label: str, msg: str) -> None:
        console.print(f"[red]✗[/red] {label}: {msg}")

    # OpenAI
    if not settings.openai_api_key:
        bad("OpenAI", "OPENAI_API_KEY is empty")
    else:
        try:
            from openai import OpenAI
            OpenAI(api_key=settings.openai_api_key).models.list()
            ok("OpenAI", "auth works")
        except Exception as exc:
            bad("OpenAI", str(exc))

    # Qdrant
    try:
        r = httpx.get(f"{settings.qdrant_url}/", timeout=5.0)
        ok("Qdrant", f"status {r.status_code}")
    except Exception as exc:
        bad("Qdrant", str(exc))

    # Langfuse
    try:
        r = httpx.get(f"{settings.langfuse_host}/api/public/health", timeout=5.0)
        if r.status_code == 200:
            ok("Langfuse", "healthy")
        else:
            bad("Langfuse", f"status {r.status_code}")
    except Exception as exc:
        bad("Langfuse", str(exc))

    # Cache file
    try:
        from paperpilot.cache import get_cache
        get_cache()
        ok("SQLite cache", str(settings.cache_db_full_path))
    except Exception as exc:
        bad("SQLite cache", str(exc))


@ops_app.command("stats")
def ops_stats() -> None:
    """Print corpus + index statistics."""
    from collections import Counter

    from qdrant_client import QdrantClient

    from paperpilot.config import RAW_DIR, PROCESSED_DIR, settings
    from paperpilot.ingest.parse import load_processed

    # Disk
    raw_n = len(list(RAW_DIR.glob("*.pdf")))
    parsed = load_processed()
    console.print(f"[bold]Disk[/bold]: {raw_n} raw PDFs, {len(parsed)} parsed papers")

    # Qdrant
    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=300)
        for v in ("v1", "v2"):
            name = settings.collection_for(v)  # type: ignore[arg-type]
            if not client.collection_exists(name):
                console.print(f"  {name}: [yellow]missing[/yellow]")
                continue
            info = client.get_collection(name)
            console.print(f"  {name}: {info.points_count} points, dim={info.config.params.vectors.size}")
    except Exception as exc:
        console.print(f"[red]Qdrant query failed: {exc}[/red]")

    # Section type distribution from parsed papers
    if parsed:
        from paperpilot.ingest.chunk import SectionAwareChunker
        cnt: Counter = Counter()
        for meta, txt in parsed[: min(20, len(parsed))]:  # sample to keep this fast
            for c in SectionAwareChunker().chunk(meta, txt):
                cnt[c.section_type] += 1
        console.print("[bold]Section distribution[/bold] (sample of 20 papers):")
        for k, v in cnt.most_common():
            console.print(f"  {k:14s} {v}")


if __name__ == "__main__":
    app()
