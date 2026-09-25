"""Human-readable Rich output and deterministic, versioned JSON."""

import json

from rich.console import Console
from rich.table import Table
from rich.text import Text

from jevshift.models import ScanReport
from jevshift.replay_models import ReplayReport, ScoreComparison


def render_json(report: ScanReport | ReplayReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False, allow_nan=False)


def print_report(report: ScanReport, console: Console) -> None:
    summary = report.summary
    console.print("JevShift Analysis", style="bold")
    console.print(f"\n{summary.total_calls} calls analyzed")
    console.print(f"\nPotential Jev candidates: {summary.candidates}\n")
    counts = Table.grid(padding=(0, 2))
    for label, value in (
        ("Choice", summary.choice),
        ("Noul", summary.noul),
        ("Score", summary.score),
        ("Needs review", summary.needs_review),
        ("Not bounded", summary.not_candidate),
    ):
        counts.add_row(label, str(value))
    console.print(counts)
    console.print()

    table = Table(expand=True)
    for heading in ("ID", "Primitive", "Confidence", "Status", "Evidence"):
        table.add_column(heading, overflow="fold")
    for assessment in report.assessments:
        # Trace-derived text is literal, even if it contains Rich markup.
        table.add_row(
            Text(assessment.call_id),
            assessment.primitive.title() if assessment.primitive else "—",
            assessment.confidence.title() if assessment.confidence else "—",
            assessment.status,
            Text("; ".join(assessment.evidence)),
        )
    console.print(table)

    metadata = report.metadata
    if metadata.latency_calls or metadata.cost_calls:
        console.print("\nSupplied historical metadata", style="bold")
        if metadata.mean_latency_ms is not None:
            console.print(
                f"Mean latency: {metadata.mean_latency_ms:g} ms "
                f"({metadata.latency_calls}/{summary.total_calls} calls)"
            )
        if metadata.total_cost_usd is not None:
            console.print(
                f"Total recorded cost: ${metadata.total_cost_usd:g} "
                f"({metadata.cost_calls}/{summary.total_calls} calls)"
            )
    console.print("\nConfidence describes rule evidence, not Jev accuracy.", style="dim")
    console.print(
        "Evaluation candidates require validation before production replacement.", style="dim"
    )


def print_replay_report(report: ReplayReport, console: Console) -> None:
    console.print("JevShift Replay", style="bold")
    for warning in report.warnings:
        console.print(Text(warning, style="yellow"))
    summary = report.summary
    console.print(
        f"\n{summary.candidates_detected} candidates detected\n"
        f"{summary.replayed} replayed\n{summary.succeeded} succeeded\n{summary.failed} failed"
    )
    console.print("\nAgreement with historical decisions", style="bold")
    agreement = Table.grid(padding=(0, 2))
    for name, counts, note in (
        ("Choice", summary.choice, "exact label"),
        ("Noul", summary.noul, f"threshold >= {report.options.noul_threshold:g}"),
        ("Score", summary.score, "nearest-level match (approximation)"),
        ("Overall", summary.overall, "includes nearest-level Score matches"),
    ):
        percentage = f"{counts.rate:.1%}" if counts.rate is not None else "—"
        agreement.add_row(name, f"{counts.matched} / {counts.compared}", percentage, note)
    console.print(agreement)
    console.print("Failed requests are excluded from agreement denominators.", style="dim")
    if summary.score_mean_absolute_level_error is not None:
        console.print(
            f"Score mean absolute level error: {summary.score_mean_absolute_level_error:.3f}"
        )

    console.print("\nLatency", style="bold")
    latency = report.latency
    if latency.jev_median_ms is not None:
        console.print(
            f"Jev median (all successes): {latency.jev_median_ms / 1000:.3f} s "
            f"({latency.successful_calls} successful calls)"
        )
    else:
        console.print("Jev median: unavailable (0 successful calls)")
    if latency.paired_calls:
        console.print(
            f"Historical median (paired): {latency.paired_historical_median_ms / 1000:.3f} s\n"
            f"Jev median (paired):        {latency.paired_jev_median_ms / 1000:.3f} s\n"
            f"Both paired medians use the same {latency.paired_calls} comparable calls."
        )
    else:
        console.print("Paired latency: unavailable (0 comparable calls)")

    table = Table(expand=True)
    for heading in ("ID", "Primitive", "Status", "Comparison / error", "Jev ms"):
        table.add_column(heading, overflow="fold")
    for item in report.items:
        if item.error is not None:
            detail = item.error
        elif isinstance(item.comparison, ScoreComparison):
            comparison = item.comparison
            label = "match" if comparison.exact_level_match else "mismatch"
            detail = (
                f"{comparison.jev_score:g} -> level {comparison.nearest_jev_level}: {label}; "
                f"error {comparison.absolute_level_error:g}"
            )
        elif item.comparison is not None:
            detail = "match" if item.comparison.match else "mismatch"
        else:
            detail = "—"
        table.add_row(
            Text(item.call_id),
            item.primitive.title(),
            item.status,
            Text(detail),
            f"{item.jev_latency_ms:.1f}",
        )
    console.print()
    console.print(table)
    fallback_count = sum(item.state_source == "prompt" for item in report.items)
    if fallback_count:
        console.print(
            f"{fallback_count} replayed calls used their prompt as state (no separate state)."
        )
    console.print("\nHistorical agreement is not a measure of correctness.", style="dim")
