"""Olive CLI (replaces scripts/backup.ts, scripts/restore.ts, scripts/download-models.ts)."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import typer

from .main import run_server
from .ml.model_store import ModelStorageError, download_local_models
from .services.backup import BackupService
from .services.census import compute_census, format_census_report, open_readonly_engine
from .settings import get_effective_hf_token, resolve_paths

app = typer.Typer(help="Olive — Self-Hosted AI Meeting Intelligence CLI")
models_app = typer.Typer(help="Manage local ML speech & diarization models")
insights_app = typer.Typer(help="Olive Insights tooling (corpus reports, speaker naming)")
app.add_typer(models_app, name="models")
app.add_typer(insights_app, name="insights")


@app.command("serve")
def serve_cmd() -> None:
    """Start the Olive HTTP and MCP server."""
    run_server()


@app.command("backup")
def backup_cmd(
    output_path: str | None = typer.Argument(
        default=None, help="Optional destination path for the .tar.gz backup archive"
    ),
) -> None:
    """Create a full .tar.gz system backup archive."""

    async def _run() -> None:
        service = BackupService()
        res = await service.create_backup()
        if output_path:
            dest = Path(output_path).resolve()
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(res["path"], dest)
            typer.echo(f"Backup written to {dest} ({res['sizeBytes']} bytes)")
        else:
            typer.echo(f"Backup created at {res['path']} ({res['sizeBytes']} bytes)")

    asyncio.run(_run())


@app.command("restore")
def restore_cmd(
    archive_path: str = typer.Argument(..., help="Path to the .tar.gz backup archive to restore"),
) -> None:
    """Restore an Olive system backup (.tar.gz)."""

    async def _run() -> None:
        service = BackupService()
        res = await service.restore_backup(str(Path(archive_path).resolve()))
        typer.echo(f"Restore complete: {res['stats']}")

    asyncio.run(_run())


@models_app.command("download")
def download_models_cmd() -> None:
    """Pre-download local WeSpeaker, pyannote, MMS aligner, and Cohere INT8 models."""
    token = get_effective_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
    try:
        download_local_models(token)
    except ModelStorageError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@insights_app.command("census")
def insights_census_cmd() -> None:
    """Report read-only corpus stats over stored transcripts (sizing for Insights)."""
    paths = resolve_paths()
    if not Path(paths.database_path).exists():
        typer.echo(f"Error: database not found at {paths.database_path}", err=True)
        raise typer.Exit(code=1)
    engine = open_readonly_engine(paths.database_path)
    try:
        stats = compute_census(engine, paths.meetings_dir)
    finally:
        engine.dispose()
    typer.echo(format_census_report(stats))


@insights_app.command("name")
def insights_name_cmd(
    frm: str | None = typer.Option(None, "--from", help="Only meetings starting on/after this ISO date"),
    to: str | None = typer.Option(None, "--to", help="Only meetings starting on/before this ISO date"),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Maximum meetings to send to the model"),
    force: bool = typer.Option(False, "--force", help="Re-propose meetings that already have proposals"),
) -> None:
    """Propose real names for generic speaker labels across the corpus (pending until confirmed)."""
    from .services.insights.extract import EgressBlockedError
    from .services.insights.naming import SpeakerNamingService

    paths = resolve_paths()
    if not Path(paths.database_path).exists():
        typer.echo(f"Error: database not found at {paths.database_path}", err=True)
        raise typer.Exit(code=1)
    try:
        summary = asyncio.run(SpeakerNamingService().propose_all(frm, to, limit, force))
    except EgressBlockedError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"{summary['meetings']} meetings considered: {summary['processed']} proposed, "
        f"{summary['skipped']} already done, {summary['noGenericLabels']} without generic labels, "
        f"{summary['failed']} failed"
    )
    typer.echo(
        f"Labels: {summary['pending']} pending confirmation, {summary['ungrounded']} rejected "
        f"as ungrounded, {summary['none']} without a name in the transcript"
    )
    for err in summary["errors"]:
        typer.echo(f"  {err['meetingId']}: {err['error']}", err=True)


if __name__ == "__main__":
    app()
