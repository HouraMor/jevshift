import pytest

from jevshift.detect import assess
from jevshift.models import LLMCall


def call(output, **changes):
    return LLMCall(id="test", prompt="Make a decision.", output=output, **changes)


@pytest.mark.parametrize(
    "options",
    [["billing", "technical"], {"billing": "Payments", "technical": "Bugs"}],
)
def test_valid_choice(options):
    result = assess(call("billing", allowed_outputs=options))
    assert (result.status, result.primitive, result.confidence) == ("candidate", "choice", "high")
    assert "2 explicit outputs" in result.evidence


def test_choice_upper_boundary():
    options = [f"option-{i}" for i in range(255)]
    assert assess(call(options[-1], allowed_outputs=options)).primitive == "choice"


def test_explicit_binary_choice_takes_precedence_over_boolean_string():
    assert assess(call("yes", allowed_outputs=["yes", "no"])).primitive == "choice"


@pytest.mark.parametrize("output", [True, False, "yes", "no", "true", "false", " YES ", "False"])
def test_boolean_noul(output):
    result = assess(call(output))
    confidence = "high" if type(output) is bool else "medium"
    assert (result.status, result.primitive, result.confidence) == ("candidate", "noul", confidence)


@pytest.mark.parametrize("output", ["maybe", "likely", "probably", " UNKNOWN "])
def test_ambiguous_pseudo_boolean(output):
    result = assess(call(output))
    assert result.status == "needs_review"
    assert result.primitive is None


@pytest.mark.parametrize("output", ["yes, because it is urgent", "not true", "1", "0", "y", "n"])
def test_does_not_coerce_other_strings_to_booleans(output):
    assert assess(call(output)).status == "not_candidate"


@pytest.mark.parametrize("output", [0, 1, 1.0, "High"])
def test_valid_score(output):
    result = assess(call(output, score_levels=["Low", "High"]))
    assert (result.status, result.primitive, result.confidence) == ("candidate", "score", "high")


def test_score_upper_boundary():
    assert assess(call(9, score_levels=[f"Level {i}" for i in range(10)])).primitive == "score"


@pytest.mark.parametrize("output", [0, 1, 3, 0.7])
def test_number_without_rubric_is_only_for_review(output):
    result = assess(call(output))
    assert (result.status, result.primitive, result.confidence) == ("needs_review", "score", "low")


@pytest.mark.parametrize(
    "prompt",
    ["Write an email.", "Summarize this text.", "Generate code.", "Explain this error."],
)
def test_free_form_generation(prompt):
    result = assess(LLMCall(id="text", prompt=prompt, output="Some generated content."))
    assert result.status == "not_candidate"
    assert result.primitive is None
    assert result.confidence is None
    assert "free-form generation" in result.evidence


def test_bare_routing_label_does_not_establish_closed_set():
    assert assess(call("billing")).status == "not_candidate"


def test_missing_state_is_a_limitation_not_fabricated_context():
    result = assess(call(True))
    assert result.status == "candidate"
    assert any("No separate state" in item for item in result.limitations)
