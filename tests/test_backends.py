import pytest
from pydantic import ValidationError

from jevshift.backends.base import DecisionBackend, DecisionResult, ReplayCandidate
from jevshift.models import LLMCall


class FakeBackend:
    """A fixed result for testing the contract, without using historical answers."""

    def evaluate(self, candidate: ReplayCandidate) -> DecisionResult:
        return DecisionResult(primitive="noul", value=0.85, model="fake")


def test_fake_backend_can_evaluate_a_replay_candidate():
    candidate = ReplayCandidate(
        call=LLMCall(id="b", prompt="Escalate?", state="Needs review.", output=False),
        primitive="noul",
    )
    backend: DecisionBackend = FakeBackend()
    result = backend.evaluate(candidate)
    assert result.value == 0.85
    assert result.confidence is None
    assert candidate.call.output is False


@pytest.mark.parametrize("primitive", ["choice", "score"])
def test_replay_primitive_must_match_candidate(primitive):
    with pytest.raises(ValidationError, match="matching primitive"):
        ReplayCandidate(call=LLMCall(id="b", prompt="Proceed?", output=True), primitive=primitive)


def test_review_record_is_not_replay_ready():
    with pytest.raises(ValidationError, match="matching primitive"):
        ReplayCandidate(call=LLMCall(id="b", prompt="Rate this.", output=3), primitive="score")


@pytest.mark.parametrize(
    "fields",
    [
        {"primitive": "noul", "value": 1.1},
        {"primitive": "noul", "value": True},
        {"primitive": "noul", "value": 0.8, "confidence": 0.9},
        {"primitive": "noul", "value": 0.8, "probabilities": {"yes": 0.8}},
        {"primitive": "choice", "value": 0.5},
        {"primitive": "score", "value": -1.0},
        {"primitive": "score", "value": float("nan")},
    ],
)
def test_result_rejects_invalid_primitive_semantics(fields):
    with pytest.raises(ValidationError):
        DecisionResult(**fields)


def test_replay_score_can_be_fractional():
    result = DecisionResult(primitive="score", value=1.43, confidence=0.35)
    assert result.value == 1.43
