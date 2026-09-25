"""Offline discovery and explicit live replay commands."""

from importlib.resources import files
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console

from jevshift.analyze import analyze_calls
from jevshift.backends.base import BackendConfigurationError
from jevshift.ingest import TraceError, load_calls, parse_calls
from jevshift.models import LLMCall, ScanReport
from jevshift.replay import contains_bundled_examples, replay_calls
from jevshift.replay_models import ReplayOptions
from jevshift.report import print_replay_report, print_report, render_json

app = typer.Typer(
    help="Find bounded LLM decisions offline, then replay them with Jev for historical comparison.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    add_completion=False,
)


def _emit(report: ScanReport, as_json: bool) -> None:
    if as_json:
        typer.echo(render_json(report))
    else:
        print_report(report, Console())


def _load(path: Path) -> list[LLMCall]:
    try:
        return load_calls(path)
    except TraceError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None
    except UnicodeError:
        typer.echo(f"Cannot read '{path}': traces must be UTF-8 text.", err=True)
        raise typer.Exit(code=1) from None
    except OSError as error:
        typer.echo(f"Cannot read '{path}': {error.strerror or error}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def analyze(
    path: Annotated[Path, typer.Argument(help="Normalized JevShift JSONL trace file.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit only a versioned JSON report.")
    ] = False,
) -> None:
    """Analyze one historical decision per JSONL record, fully offline."""
    _emit(analyze_calls(_load(path)), as_json)


@app.command()
def demo(
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit only a versioned JSON report.")
    ] = False,
) -> None:
    """Analyze bundled synthetic traces. No API key, network, or external service."""
    resource = files("jevshift").joinpath("data", "calls.jsonl")
    with resource.open(encoding="utf-8") as stream:
        calls = parse_calls(stream)
    if not as_json:
        typer.echo("Synthetic demo traces; supplied costs and latencies are illustrative.\n")
    _emit(analyze_calls(calls), as_json)


@app.command()
def replay(
    path: Annotated[Path, typer.Argument(help="Normalized JevShift JSONL trace file.")],
    limit: Annotated[
        int | None, typer.Option(min=1, help="Replay at most this many candidates.")
    ] = None,
    noul_threshold: Annotated[
        float,
        typer.Option(min=0, max=1, help="Predict true when the Noul probability >= this value."),
    ] = 0.5,
    model: Annotated[
        str | None, typer.Option(help="Jev model; otherwise use the TypeSafe SDK default.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Write a versioned replay JSON report to stdout.")
    ] = False,
) -> None:
    """Live replay: send prompt/state and criteria to TypeSafe and compare historical answers."""
    try:
        options = ReplayOptions(limit=limit, noul_threshold=noul_threshold, model=model)
    except ValidationError:
        raise typer.BadParameter(
            "Use a positive limit, a finite threshold between 0 and 1, and a nonblank model."
        ) from None
    calls = _load(path)
    synthetic = contains_bundled_examples(calls)
    # Offline commands never initialize the SDK or read its credential.
    from jevshift.backends.jev import JevBackend

    try:
        with JevBackend(model=options.model) as backend:
            typer.echo("Live replay sends prompt/state and criteria to the TypeSafe API.", err=True)
            report = replay_calls(
                calls,
                backend,
                options=options,
                contains_synthetic_examples=synthetic,
            )
    except BackendConfigurationError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None
    if as_json:
        typer.echo(render_json(report))
    else:
        print_replay_report(report, Console())
    if report.summary.failed:
        raise typer.Exit(code=1)
