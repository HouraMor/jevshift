import json
from unittest.mock import Mock

import httpx2
import pytest
from pydantic import ValidationError
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeClient,
    TypeSafeError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    Usage,
)

from jevshift.backends.base import (
    BackendConfigurationError,
    BackendError,
    DecisionResult,
    ReplayCandidate,
)
from jevshift.backends.jev import JevBackend, build_state
from jevshift.models import LLMCall


def candidate(primitive, output, **kwargs):
    return ReplayCandidate(
        primitive=primitive,
        call=LLMCall(
            id="PRIVATE ID", prompt="PRIVATE PROMPT", state="PRIVATE STATE", output=output, **kwargs
        ),
    )


def response(answer):
    return SystemOneResponse(
        model="jev-test",
        usage=Usage(input_tokens=12, output_tokens=3),
        answers={"decision": answer},
    )


@pytest.fixture
def sdk_client(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "unit-test-key")
    client = Mock(spec=TypeSafeClient)
    constructor = Mock(return_value=client)
    monkeypatch.setattr("jevshift.backends.jev.TypeSafeClient", constructor)
    return client, constructor


@pytest.mark.parametrize(
    "options",
    [["a", "b"], {"a": "First route", "b": "Second route"}],
)
def test_choice_uses_real_sdk_question_and_preserves_descriptions(sdk_client, options):
    client, constructor = sdk_client
    client.system_one.return_value = response(
        ChoiceAnswer(choice="b", confidence=0.6, probabilities={"a": 0.2, "b": 0.8})
    )
    with JevBackend(model="jev-pinned") as backend:
        result = backend.evaluate(candidate("choice", "a", allowed_outputs=options))
    request = client.system_one.call_args.kwargs
    assert set(request) == {"state", "questions"}
    assert request["state"] == "PRIVATE STATE"
    question = request["questions"]["decision"]
    assert isinstance(question, Choice)
    assert question.instructions == "PRIVATE PROMPT"
    assert question.criteria == (options if isinstance(options, dict) else {"a": None, "b": None})
    assert result.value == "b"
    assert result.confidence == 0.6
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 3
    assert result.model == "jev-test"
    assert constructor.call_args.kwargs["model"] == "jev-pinned"
    assert constructor.call_args.kwargs["retry"].max_retries == 0
    assert constructor.call_args.kwargs["timeout"] == 30.0
    assert "api_key" not in constructor.call_args.kwargs
    client.close.assert_called_once()


def test_noul_has_probability_and_no_confidence(sdk_client):
    client, _ = sdk_client
    client.system_one.return_value = response(NoulAnswer(noul=0.73))
    with JevBackend() as backend:
        result = backend.evaluate(candidate("noul", False))
    question = client.system_one.call_args.kwargs["questions"]["decision"]
    assert isinstance(question, Noul)
    assert question.model_dump() == {"type": "noul", "instructions": "PRIVATE PROMPT"}
    assert result.value == 0.73
    assert result.confidence is None
    assert result.probabilities is None


def test_score_uses_ordered_criteria_and_normalizes_integer_probability_keys(sdk_client):
    client, _ = sdk_client
    client.system_one.return_value = response(
        ScoreAnswer(
            score=1.43,
            confidence=0.35,
            legend={0: "Low", 1: "Moderate", 2: "High"},
            probabilities={0: 0.0, 1: 0.57, 2: 0.43},
        )
    )
    with JevBackend() as backend:
        result = backend.evaluate(candidate("score", 2, score_levels=["Low", "Moderate", "High"]))
    question = client.system_one.call_args.kwargs["questions"]["decision"]
    assert isinstance(question, Score)
    assert question.criteria == ["Low", "Moderate", "High"]
    assert result.value == 1.43
    assert result.probabilities == {"0": 0.0, "1": 0.57, "2": 0.43}
    assert DecisionResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("primitive", ["choice", "noul", "score"])
def test_changing_historical_answer_does_not_change_any_request_field(sdk_client, primitive):
    client, _ = sdk_client
    if primitive == "choice":
        left, right, fields = "a", "b", {"allowed_outputs": ["a", "b"]}
        answer = ChoiceAnswer(choice="a", confidence=1.0, probabilities={"a": 1.0, "b": 0.0})
    elif primitive == "noul":
        left, right, fields = True, False, {}
        answer = NoulAnswer(noul=0.4)
    else:
        left, right, fields = 0, 1, {"score_levels": ["Low", "High"]}
        answer = ScoreAnswer(
            score=0.4, confidence=0.2, legend={0: "Low", 1: "High"}, probabilities={0: 0.6, 1: 0.4}
        )
    client.system_one.return_value = response(answer)
    with JevBackend() as backend:
        backend.evaluate(candidate(primitive, left, **fields))
        backend.evaluate(candidate(primitive, right, **fields))
    first, second = [call.kwargs for call in client.system_one.call_args_list]
    assert first == second
    assert first["state"] == "PRIVATE STATE"
    assert first["questions"]["decision"].instructions == "PRIVATE PROMPT"
    assert "PRIVATE ID" not in repr(first)


@pytest.mark.parametrize("state", [None, "", " \n", "Context"])
def test_state_policy_uses_only_available_context(state):
    call = LLMCall(
        id="x", prompt="Question with embedded context", state=state, output="HISTORICAL-ONLY"
    )
    assert build_state(call) == (call.prompt if state is None else state)
    assert "HISTORICAL-ONLY" not in build_state(call)


def test_missing_key_fails_before_client_creation(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr("jevshift.backends.jev.TypeSafeClient", constructor)
    with pytest.raises(BackendConfigurationError, match="Set TYPESAFE_API_KEY"):
        JevBackend()
    constructor.assert_not_called()


def test_bad_sdk_configuration_does_not_expose_exception(sdk_client):
    _, constructor = sdk_client
    original = TypeSafeError("SECRET API KEY")
    constructor.side_effect = original
    with pytest.raises(BackendConfigurationError) as error:
        JevBackend()
    assert "SECRET" not in str(error.value)
    assert error.value.__cause__ is original


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (TypeSafeAuthenticationError(401, "SECRET", httpx2.Headers()), "authentication"),
        (TypeSafePermissionDeniedError(403, "SECRET", httpx2.Headers()), "authentication"),
        (TypeSafeAPITimeoutError(30), "timeout"),
        (TypeSafeRateLimitError(429, "SECRET", httpx2.Headers()), "rate_limit"),
        (TypeSafeAPIConnectionError("SECRET"), "connection"),
        (TypeSafeAPIError(500, "SECRET", httpx2.Headers()), "api_error"),
        (
            TypeSafeAPIResponseValidationError(200, "SECRET", httpx2.Headers(), "answers"),
            "invalid_response",
        ),
        (TypeSafeError("SECRET"), "backend_error"),
    ],
)
def test_sdk_errors_have_safe_messages_and_keep_internal_cause(sdk_client, error, code):
    client, _ = sdk_client
    client.system_one.side_effect = error
    with JevBackend() as backend, pytest.raises(BackendError) as raised:
        backend.evaluate(candidate("noul", True))
    assert raised.value.code == code
    assert "SECRET" not in str(raised.value)
    assert raised.value.__cause__ is error


@pytest.mark.parametrize(
    "answer",
    [
        NoulAnswer(noul=float("nan")),
        NoulAnswer(noul=1.01),
        ChoiceAnswer(choice="a", confidence=0.5, probabilities={"a": 1.0}),
    ],
)
def test_invalid_sdk_value_or_wrong_primitive_is_rejected(sdk_client, answer):
    client, _ = sdk_client
    client.system_one.return_value = response(answer)
    with JevBackend() as backend, pytest.raises(BackendError, match="invalid backend response"):
        backend.evaluate(candidate("noul", True))


def test_missing_answer_is_rejected(sdk_client):
    client, _ = sdk_client
    client.system_one.return_value = SystemOneResponse(model="jev-test", usage=Usage(), answers={})
    with JevBackend() as backend, pytest.raises(BackendError, match="invalid backend response"):
        backend.evaluate(candidate("noul", True))


def test_different_score_legend_is_rejected(sdk_client):
    client, _ = sdk_client
    client.system_one.return_value = response(
        ScoreAnswer(
            score=0.2,
            confidence=0.8,
            legend={0: "Wrong", 1: "High"},
            probabilities={0: 0.8, 1: 0.2},
        )
    )
    with JevBackend() as backend, pytest.raises(BackendError, match="invalid backend response"):
        backend.evaluate(candidate("score", 0, score_levels=["Low", "High"]))


@pytest.mark.parametrize("keys", [{0: 1.0}, {"-1": 1.0}, {"00": 1.0}, {"level": 1.0}])
def test_internal_score_probability_keys_are_validated(keys):
    with pytest.raises(ValidationError):
        DecisionResult(primitive="score", value=0.0, probabilities=keys)


def test_real_sdk_encoding_and_decoding_with_mock_transport(monkeypatch):
    """Exercise the installed SDK's current ordered Score wire format without sockets."""
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "model": "jev-test",
                "usage": {"input_tokens": 20, "output_tokens": 4},
                "answers": {
                    "decision": {
                        "type": "score",
                        "score": 0.4,
                        "confidence": 0.2,
                        "legend": {"0": "Low", "1": "High"},
                        "probabilities": {"0": 0.6, "1": 0.4},
                    }
                },
            },
        )

    monkeypatch.setenv("TYPESAFE_API_KEY", "unit-test-key")

    def client_factory(**kwargs):
        return TypeSafeClient(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr("jevshift.backends.jev.TypeSafeClient", client_factory)
    with JevBackend(model="jev-pinned") as backend:
        result = backend.evaluate(candidate("score", 1, score_levels=["Low", "High"]))
    assert requests == [
        {
            "model": "jev-pinned",
            "state": "PRIVATE STATE",
            "questions": {
                "decision": {
                    "type": "score",
                    "instructions": "PRIVATE PROMPT",
                    "criteria": ["Low", "High"],
                }
            },
        }
    ]
    assert result.probabilities == {"0": 0.6, "1": 0.4}
    assert result.usage.input_tokens == 20
