from math import nextafter

import pytest
from pydantic import ValidationError

from jevshift.backends.base import DecisionResult, ReplayCandidate
from jevshift.compare import compare_result, historical_value
from jevshift.models import LLMCall
from jevshift.replay_models import ScoreComparison


def candidate(primitive, output, **kwargs):
    return ReplayCandidate(
        call=LLMCall(id="c", prompt="Decide.", output=output, **kwargs),
        primitive=primitive,
    )


@pytest.mark.parametrize(("answer", "match"), [("billing", True), ("technical", False)])
def test_choice_exact_label_comparison(answer, match):
    item = candidate("choice", "billing", allowed_outputs=["billing", "technical"])
    assert compare_result(item, DecisionResult(primitive="choice", value=answer)).match is match


def test_choice_match_is_case_sensitive():
    item = candidate("choice", "A", allowed_outputs=["A", "a"])
    assert not compare_result(item, DecisionResult(primitive="choice", value="a")).match


@pytest.mark.parametrize(
    ("historical", "normalized"),
    [
        (True, True),
        (False, False),
        ("yes", True),
        (" TRUE ", True),
        ("no", False),
        ("FALSE", False),
    ],
)
def test_normalizes_historical_noul(historical, normalized):
    assert historical_value(candidate("noul", historical)) is normalized


@pytest.mark.parametrize(
    ("probability", "threshold", "prediction"),
    [
        (0.49, 0.5, False),
        (0.5, 0.5, True),
        (0.7, 0.7, True),
        (0.69, 0.7, False),
        (0.0, 0.0, True),
        (0.99, 1.0, False),
        (1.0, 1.0, True),
    ],
)
def test_noul_threshold_boundary_and_raw_probability(probability, threshold, prediction):
    comparison = compare_result(
        candidate("noul", "YES"),
        DecisionResult(primitive="noul", value=probability),
        noul_threshold=threshold,
    )
    assert comparison.prediction is prediction
    assert comparison.match is prediction
    assert comparison.probability == probability
    assert comparison.threshold == threshold


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValidationError):
        compare_result(
            candidate("noul", True),
            DecisionResult(primitive="noul", value=0.5),
            noul_threshold=threshold,
        )


@pytest.mark.parametrize(
    ("value", "nearest", "matches"),
    [
        (0.0, 0, False),
        (0.5, 1, True),
        (1.0, 1, True),
        (1.43, 1, True),
        (1.5, 2, False),
        (2.5, 3, False),
        (3.0, 3, False),
        (nextafter(0.5, 0), 0, False),
    ],
)
def test_fractional_score_uses_half_up_nearest_level(value, nearest, matches):
    item = candidate("score", "Moderate", score_levels=["Low", "Moderate", "High", "Critical"])
    comparison = compare_result(item, DecisionResult(primitive="score", value=value))
    assert isinstance(comparison, ScoreComparison)
    assert comparison.historical_level_index == 1
    assert comparison.jev_score == value
    assert comparison.absolute_level_error == pytest.approx(abs(1 - value))
    assert comparison.nearest_jev_level == nearest
    assert comparison.exact_level_match is matches


def test_rejects_out_of_rubric_score_without_clamping():
    item = candidate("score", 0, score_levels=["Low", "High"])
    with pytest.raises(ValueError, match="outside"):
        compare_result(item, DecisionResult(primitive="score", value=1.00001))


def test_rejects_unrequested_choice():
    item = candidate("choice", "a", allowed_outputs=["a", "b"])
    with pytest.raises(ValueError, match="outside"):
        compare_result(item, DecisionResult(primitive="choice", value="c"))


def test_rejects_mismatched_primitive():
    with pytest.raises(ValueError, match="primitive"):
        compare_result(candidate("noul", True), DecisionResult(primitive="score", value=0.5))


@pytest.mark.parametrize("primitive", ["choice", "score"])
def test_probability_map_must_cover_the_declared_contract(primitive):
    if primitive == "choice":
        item = candidate("choice", "a", allowed_outputs=["a", "b"])
        result = DecisionResult(primitive="choice", value="a", probabilities={"a": 1.0})
    else:
        item = candidate("score", 0, score_levels=["Low", "High"])
        result = DecisionResult(primitive="score", value=0.0, probabilities={"0": 1.0})
    with pytest.raises(ValueError, match="probabilities"):
        compare_result(item, result)
