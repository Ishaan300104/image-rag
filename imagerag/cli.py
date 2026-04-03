import typer
from pathlib import Path
from rich import print as rprint

app = typer.Typer(
    help="ImageRAG — search your local photos and videos by text description.",
    no_args_is_help=True,
)


@app.command()
def index(
    directory: Path = typer.Argument(..., help="Directory to scan for images/videos"),
    frame_interval: int = typer.Option(5, help="Seconds between video keyframes"),
):
    """Index all images and videos in a directory (run once before searching)."""
    from imagerag.indexer import index_directory

    if not directory.exists():
        rprint(f"[red]Directory not found: {directory}[/red]")
        raise typer.Exit(1)

    rprint(f"[bold]Indexing[/bold] {directory} ...")
    index_directory(directory, frame_interval=frame_interval)


@app.command()
def search(
    query: str = typer.Argument(..., help="Text description to search for"),
    top_k: int = typer.Option(10, "--top", "-n", help="Number of results to return"),
):
    """Search indexed photos and videos by text description."""
    from imagerag.searchtool import search as do_search

    try:
        results = do_search(query, top_k=top_k)
    except FileNotFoundError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if not results:
        rprint("[yellow]No results found.[/yellow]")
        return

    for i, r in enumerate(results, 1):
        path = r["path"]
        score = r["score"]
        if r["type"] == "video":
            ts = r.get("timestamp", 0)
            m, s = divmod(int(ts), 60)
            rprint(f"  [cyan]{i}.[/cyan] {path}  [dim]@ {m}:{s:02d}  score={score:.3f}[/dim]")
        else:
            rprint(f"  [cyan]{i}.[/cyan] {path}  [dim]score={score:.3f}[/dim]")


@app.command()
def visualize(
    output: Path = typer.Option(None, "--output", "-o", help="Save HTML to this path instead of a temp file"),
):
    """Visualize all embeddings in an interactive 3D plot (opens in browser)."""
    from imagerag.visualize import visualize as do_visualize

    try:
        rprint("[bold]Reducing embeddings to 3D and building plot...[/bold]")
        do_visualize(output_html=output)
    except FileNotFoundError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command()
def serve(
    port: int = typer.Option(7860, help="Port for the web UI"),
    share: bool = typer.Option(False, help="Create a public Gradio share link"),
):
    """Launch the Gradio web UI."""
    from imagerag.ui import launch

    rprint(f"[bold]Launching web UI on port {port}...[/bold]")
    launch(port=port, share=share)
