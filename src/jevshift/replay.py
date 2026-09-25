"""Sequential replay, failure isolation, and measured historical comparison."""

from collections.abc import Iterable
from importlib.resources import files
from statistics import mean, median
from time import perf_counter

from pydantic import ValidationError

from jevshift.backends.base import (
    ERROR_MESSAGES,
    BackendError,
    DecisionBackend,
    DecisionResult,
    ReplayCandidate,
)
from jevshift.compare import compare_result, historical_value, is_match
from jevshift.detect import assess
from jevshift.ingest import parse_calls
from jevshift.models import LLMCall, Primitive
from jevshift.replay_models import (
    Agreement,
    ReplayItem,
    ReplayLatencySummary,
    ReplayOptions,
    ReplayReport,
    ReplaySummary,
    ScoreComparison,
)


def contains_bundled_examples(calls: list[LLMCall]) -> bool:
    """Recognize copies and subsets of our samples, even after JSON reformatting."""
    with files("jevshift").joinpath("data", "calls.jsonl").open(encoding="utf-8") as stream:
        examples = {call.id: call for call in parse_calls(stream)}
    return any(examples.get(call.id) == call for call in calls)


def _agreement(items: list[ReplayItem], primitive: Primitive | None = None) -> Agreement:
    comparisons = [
        item.comparison
        for item in items
        if item.status == "success"
        and item.comparison is not None
        and (primitive is None or item.primitive == primitive)
    ]
    matched = sum(is_match(comparison) for comparison in comparisons)
    compared = len(comparisons)
    return Agreement(
        matched=matched, compared=compared, rate=matched / compared if compared else None
    )


def _latency(items: list[ReplayItem]) -> ReplayLatencySummary:
    successful = [item for item in items if item.status == "success"]
    paired = [item for item in successful if item.historical_latency_ms is not None]
    historical = [
        item.historical_latency_ms for item in paired if item.historical_latency_ms is not None
    ]
    return ReplayLatencySummary(
        successful_calls=len(successful),
        jev_median_ms=median(item.jev_latency_ms for item in successful) if successful else None,
        paired_calls=len(paired),
        paired_historical_median_ms=median(historical) if paired else None,
        paired_jev_median_ms=median(item.jev_latency_ms for item in paired) if paired else None,
    )


def replay_calls(
    calls: Iterable[LLMCall],
    backend: DecisionBackend,
    *,
    options: ReplayOptions | None = None,
    contains_synthetic_examples: bool = False,
) -> ReplayReport:
    """Evaluate only detected candidates, preserving order and excluding failures from agreement."""
    options = options or ReplayOptions()
    records = list(calls)
    candidates = []
    for call in records:
        assessment = assess(call)
        if assessment.status == "candidate" and assessment.primitive is not None:
            candidates.append(ReplayCandidate(call=call, primitive=assessment.primitive))

    items = []
    for candidate in candidates[: options.limit]:
        historical = historical_value(candidate)
        result = None
        comparison = None
        error_code = None
        started = perf_counter()
        try:
            raw_result = backend.evaluate(candidate)
        except BackendError as error:
            error_code = error.code
        except Exception:
            # Never copy arbitrary exception messages: they can contain API keys
            # or the submitted prompt/state. KeyboardInterrupt still propagates.
            error_code = "backend_error"
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
        if error_code is None:
            try:
                if not isinstance(raw_result, DecisionResult):
                    raise ValueError("backend must return a DecisionResult")
                # Revalidate even if a backend used model_construct or mutated an instance.
                result = DecisionResult.model_validate(raw_result.model_dump())
                comparison = compare_result(
                    candidate, result, noul_threshold=options.noul_threshold
                )
            except (ValueError, TypeError, ValidationError):
                result = None
                error_code = "invalid_response"
        items.append(
            ReplayItem(
                call_id=candidate.call.id,
                primitive=candidate.primitive,
                status="failed" if error_code else "success",
                historical=historical,
                state_source="prompt" if candidate.call.state is None else "state",
                result=result,
                comparison=comparison,
                jev_latency_ms=elapsed_ms,
                historical_latency_ms=candidate.call.latency_ms,
                error_code=error_code,
                error=ERROR_MESSAGES[error_code] if error_code else None,
            )
        )

    succeeded = sum(item.status == "success" for item in items)
    score_errors = [
        item.comparison.absolute_level_error
        for item in items
        if isinstance(item.comparison, ScoreComparison)
    ]
    warnings = []
    if contains_synthetic_examples:
        warnings.append(
            "Contains synthetic examples: agreement and latency are demonstrations, not benchmarks."
        )
    return ReplayReport(
        options=options,
        contains_synthetic_examples=contains_synthetic_examples,
        warnings=warnings,
        summary=ReplaySummary(
            total_calls=len(records),
            candidates_detected=len(candidates),
            replayed=len(items),
            succeeded=succeeded,
            failed=len(items) - succeeded,
            choice=_agreement(items, "choice"),
            noul=_agreement(items, "noul"),
            score=_agreement(items, "score"),
            overall=_agreement(items),
            score_mean_absolute_level_error=mean(score_errors) if score_errors else None,
        ),
        latency=_latency(items),
        items=items,
    )
