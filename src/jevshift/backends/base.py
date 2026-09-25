"""Validated replay contracts and safe errors, independent of any vendor SDK."""

from math import fsum, isclose
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jevshift.detect import assess
from jevshift.models import LLMCall, Primitive

Probability = Annotated[float, Field(ge=0, le=1)]
FailureCode = Literal[
    "authentication",
    "timeout",
    "rate_limit",
    "connection",
    "api_error",
    "invalid_response",
    "backend_error",
]
ERROR_MESSAGES: dict[FailureCode, str] = {
    "authentication": "authentication or permission denied",
    "timeout": "request timed out",
    "rate_limit": "API rate limit exceeded",
    "connection": "could not connect to the API",
    "api_error": "API request failed",
    "invalid_response": "invalid backend response",
    "backend_error": "backend evaluation failed",
}


class BackendConfigurationError(ValueError):
    """Safe configuration error raised before a replay starts."""


class BackendError(Exception):
    """Expose a fixed message, retaining the original exception only as __cause__."""

    def __init__(self, code: FailureCode) -> None:
        self.code = code
        super().__init__(ERROR_MESSAGES[code])


class ReplayCandidate(BaseModel):
    """The validated historical decision and its primitive.

    A backend must use prompt/state/criteria as input, never the historical output.
    The historical output is retained for comparison by the future replay runner.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    call: LLMCall
    primitive: Primitive

    @model_validator(mode="after")
    def require_candidate(self) -> "ReplayCandidate":
        assessment = assess(self.call)
        if assessment.status != "candidate" or assessment.primitive != self.primitive:
            raise ValueError("replay requires a candidate with a matching primitive")
        return self


class TokenUsage(BaseModel):
    """Reported counts only; missing counts are not inferred and no price is applied."""

    model_config = ConfigDict(strict=True, extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class DecisionResult(BaseModel):
    """Choice label, Noul probability, or fractional Score position.

    Choice probability keys are labels. Score keys are canonical zero-based
    integer strings ("0", "1", ...), normalized at the backend boundary for JSON.
    """

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    primitive: Primitive
    value: str | float
    confidence: float | None = Field(default=None, ge=0, le=1)
    probabilities: dict[str, Probability] | None = None
    model: str | None = None
    usage: TokenUsage | None = None

    @model_validator(mode="after")
    def validate_value(self) -> "DecisionResult":
        if self.primitive == "choice":
            if not isinstance(self.value, str) or not self.value.strip():
                raise ValueError("Choice results require a nonempty label")
        elif not isinstance(self.value, float) or self.value < 0:
            raise ValueError("Noul and Score results require a nonnegative number")
        if self.primitive == "noul":
            if self.value > 1:
                raise ValueError("Noul results must be probabilities from 0 to 1")
            if self.confidence is not None or self.probabilities is not None:
                raise ValueError("Noul has no separate confidence or probabilities map")
        if self.probabilities is not None:
            if not self.probabilities or not isclose(
                fsum(self.probabilities.values()), 1.0, abs_tol=0.001, rel_tol=0
            ):
                raise ValueError("probabilities must sum to approximately 1 (tolerance 0.001)")
            if self.primitive == "choice" and self.value not in self.probabilities:
                raise ValueError("Choice result must appear in its probability map")
            if self.primitive == "score" and any(
                not key.isascii() or not key.isdecimal() or str(int(key)) != key
                for key in self.probabilities
            ):
                raise ValueError(
                    "Score probability keys must be canonical nonnegative integer strings"
                )
        return self


class DecisionBackend(Protocol):
    def evaluate(self, candidate: ReplayCandidate) -> DecisionResult:
        """Evaluate a candidate once; the replay runner will own comparisons."""
        ...
