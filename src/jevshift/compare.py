"""Historical comparison policies, with no SDK or network dependency."""

from math import floor

from jevshift.backends.base import DecisionResult, ReplayCandidate
from jevshift.replay_models import (
    ChoiceComparison,
    Comparison,
    NoulComparison,
    ReplayOptions,
    ScoreComparison,
)


def historical_value(candidate: ReplayCandidate) -> str | bool | int:
    output = candidate.call.output
    if candidate.primitive == "score":
        return candidate.call.score_index()
    if candidate.primitive == "noul":
        if type(output) is bool:
            return output
        if isinstance(output, str):
            normalized = output.strip().casefold()
            if normalized in {"yes", "true"}:
                return True
            if normalized in {"no", "false"}:
                return False
        raise ValueError("historical Noul output is not Boolean")
    if not isinstance(output, str):
        raise ValueError("historical Choice output is not a label")
    return output


def compare_result(
    candidate: ReplayCandidate,
    result: DecisionResult,
    *,
    noul_threshold: float = 0.5,
) -> Comparison:
    """Reject invalid responses; nearest Score ties go toward the higher level."""
    threshold = ReplayOptions(noul_threshold=noul_threshold).noul_threshold
    if candidate.primitive != result.primitive:
        raise ValueError("result primitive does not match the candidate")
    historical = historical_value(candidate)
    if result.primitive == "choice":
        options = candidate.call.allowed_outputs
        if options is None or result.value not in options:
            raise ValueError("returned Choice label is outside the declared options")
        if result.probabilities is not None and set(result.probabilities) != set(options):
            raise ValueError("Choice probabilities do not match the declared options")
        return ChoiceComparison(match=historical == result.value)
    if result.primitive == "noul":
        if not isinstance(result.value, float):
            raise ValueError("Noul result must be a probability")
        prediction = result.value >= threshold
        return NoulComparison(
            probability=result.value,
            threshold=threshold,
            prediction=prediction,
            match=historical == prediction,
        )
    levels = candidate.call.score_levels
    if levels is None or not isinstance(result.value, float):
        raise ValueError("Score result requires a rubric and numeric value")
    if not 0 <= result.value <= len(levels) - 1:
        raise ValueError("Score result lies outside the declared rubric")
    if result.probabilities is not None and set(result.probabilities) != {
        str(index) for index in range(len(levels))
    }:
        raise ValueError("Score probabilities do not match the rubric indices")
    # Work from the fractional part rather than round(): ties select the higher
    # level, and values immediately below a half do not round upward by accident.
    lower = floor(result.value)
    nearest = lower + int(result.value - lower >= 0.5)
    index = candidate.call.score_index()
    return ScoreComparison(
        historical_level_index=index,
        jev_score=result.value,
        absolute_level_error=abs(index - result.value),
        nearest_jev_level=nearest,
        exact_level_match=index == nearest,
    )


def is_match(comparison: Comparison) -> bool:
    if isinstance(comparison, ScoreComparison):
        return comparison.exact_level_match
    return comparison.match
