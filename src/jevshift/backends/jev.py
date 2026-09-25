"""Official TypeSafe SDK adapter. Historical answers never enter a request."""

import os
from types import TracebackType

from pydantic import ValidationError
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    RetryPolicy,
    Score,
    ScoreAnswer,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeClient,
    TypeSafeError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
)

from jevshift.backends.base import (
    BackendConfigurationError,
    BackendError,
    DecisionResult,
    ReplayCandidate,
    TokenUsage,
)
from jevshift.models import LLMCall


def build_state(call: LLMCall) -> str:
    """Use state verbatim, or the prompt itself when no separate state was recorded."""
    return call.state if call.state is not None else call.prompt


def build_question(candidate: ReplayCandidate) -> Choice | Noul | Score:
    call = candidate.call
    if candidate.primitive == "choice":
        options = call.allowed_outputs
        if options is None:
            raise ValueError("Choice requires allowed_outputs")
        criteria = options if isinstance(options, dict) else dict.fromkeys(options)
        return Choice(instructions=call.prompt, criteria=criteria)
    if candidate.primitive == "score":
        if call.score_levels is None:
            raise ValueError("Score requires score_levels")
        return Score(instructions=call.prompt, criteria=call.score_levels)
    return Noul(instructions=call.prompt)


class JevBackend:
    """One synchronous SDK request per evaluation, with no automatic retries."""

    def __init__(self, *, model: str | None = None) -> None:
        # The SDK resolves and holds the credential. JevShift neither copies nor saves it.
        if not os.environ.get("TYPESAFE_API_KEY", "").strip():
            raise BackendConfigurationError("Set TYPESAFE_API_KEY before running live replay.")
        if model is not None and not model.strip():
            raise BackendConfigurationError("The Jev model must not be blank.")
        try:
            self._client = TypeSafeClient(
                model=model, timeout=30.0, retry=RetryPolicy(max_retries=0)
            )
        except (TypeSafeError, ValueError, OSError) as error:
            raise BackendConfigurationError(
                "Cannot initialize the TypeSafe SDK; check TYPESAFE_API_KEY and SDK configuration."
            ) from error

    def __enter__(self) -> "JevBackend":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._client.close()

    def evaluate(self, candidate: ReplayCandidate) -> DecisionResult:
        try:
            question = build_question(candidate)
            response = self._client.system_one(
                state=build_state(candidate.call), questions={"decision": question}
            )
            if set(response.answers) != {"decision"}:
                raise ValueError("Expected exactly one decision answer")
            answer = response.answers["decision"]
            usage = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            if candidate.primitive == "choice" and isinstance(answer, ChoiceAnswer):
                return DecisionResult(
                    primitive="choice",
                    value=answer.choice,
                    confidence=answer.confidence,
                    probabilities=answer.probabilities,
                    model=response.model,
                    usage=usage,
                )
            if candidate.primitive == "noul" and isinstance(answer, NoulAnswer):
                return DecisionResult(
                    primitive="noul",
                    value=answer.noul,
                    model=response.model,
                    usage=usage,
                )
            if candidate.primitive == "score" and isinstance(answer, ScoreAnswer):
                levels = candidate.call.score_levels
                if levels is None or answer.legend != dict(enumerate(levels)):
                    raise ValueError("Score legend does not match the requested rubric")
                return DecisionResult(
                    primitive="score",
                    value=answer.score,
                    confidence=answer.confidence,
                    probabilities={
                        str(index): value for index, value in answer.probabilities.items()
                    },
                    model=response.model,
                    usage=usage,
                )
            raise ValueError("Answer primitive does not match the requested question")
        except (TypeSafeAuthenticationError, TypeSafePermissionDeniedError) as error:
            raise BackendError("authentication") from error
        except TypeSafeAPITimeoutError as error:
            raise BackendError("timeout") from error
        except TypeSafeRateLimitError as error:
            raise BackendError("rate_limit") from error
        except TypeSafeAPIConnectionError as error:
            raise BackendError("connection") from error
        except TypeSafeAPIResponseValidationError as error:
            raise BackendError("invalid_response") from error
        except TypeSafeAPIError as error:
            raise BackendError("api_error") from error
        except (ValueError, TypeError, KeyError, AttributeError, ValidationError) as error:
            raise BackendError("invalid_response") from error
        except TypeSafeError as error:
            raise BackendError("backend_error") from error
