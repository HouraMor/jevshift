"""Versioned replay reports, with no original prompts or contextual state."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jevshift.backends.base import DecisionResult, FailureCode, Probability
from jevshift.models import Primitive

Nonnegative = Annotated[float, Field(ge=0)]
Count = Annotated[int, Field(ge=0)]


class ReplayModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class ReplayOptions(ReplayModel):
    limit: int | None = Field(default=None, gt=0)
    noul_threshold: Probability = 0.5
    model: str | None = None

    @field_validator("model")
    @classmethod
    def nonblank_model(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("model must not be blank")
        return value


class ChoiceComparison(ReplayModel):
    primitive: Literal["choice"] = "choice"
    match: bool


class NoulComparison(ReplayModel):
    primitive: Literal["noul"] = "noul"
    probability: Probability
    threshold: Probability
    prediction: bool
    match: bool


class ScoreComparison(ReplayModel):
    primitive: Literal["score"] = "score"
    historical_level_index: Count
    jev_score: Nonnegative
    absolute_level_error: Nonnegative
    nearest_jev_level: Count
    exact_level_match: bool


Comparison = Annotated[
    ChoiceComparison | NoulComparison | ScoreComparison, Field(discriminator="primitive")
]


class ReplayItem(ReplayModel):
    call_id: str
    primitive: Primitive
    status: Literal["success", "failed"]
    historical: str | bool | int
    state_source: Literal["state", "prompt"]
    result: DecisionResult | None = None
    comparison: Comparison | None = None
    jev_latency_ms: Nonnegative
    historical_latency_ms: Nonnegative | None = None
    error_code: FailureCode | None = None
    error: str | None = None

    @model_validator(mode="after")
    def consistent_item(self) -> "ReplayItem":
        historical_type = {"choice": str, "noul": bool, "score": int}[self.primitive]
        if type(self.historical) is not historical_type:
            raise ValueError("historical value must use its primitive's normalized comparison type")
        if self.primitive == "score" and self.historical < 0:
            raise ValueError("historical Score index cannot be negative")
        if self.status == "success":
            if self.result is None or self.comparison is None:
                raise ValueError("successful items require a result and comparison")
            if (
                self.result.primitive != self.primitive
                or self.comparison.primitive != self.primitive
            ):
                raise ValueError("item, result, and comparison primitives must match")
            if self.error is not None or self.error_code is not None:
                raise ValueError("successful items cannot have errors")
        elif self.error is None or self.error_code is None:
            raise ValueError("failed items require a safe error and error code")
        elif self.result is not None or self.comparison is not None:
            raise ValueError("failed items cannot contribute a result or comparison")
        return self


class Agreement(ReplayModel):
    matched: Count
    compared: Count
    rate: Probability | None

    @model_validator(mode="after")
    def consistent_counts(self) -> "Agreement":
        if self.matched > self.compared:
            raise ValueError("matched count cannot exceed compared count")
        expected = self.matched / self.compared if self.compared else None
        if self.rate != expected:
            raise ValueError("agreement rate must match its denominator, or be null when empty")
        return self


class ReplaySummary(ReplayModel):
    total_calls: Count
    candidates_detected: Count
    replayed: Count
    succeeded: Count
    failed: Count
    choice: Agreement
    noul: Agreement
    score: Agreement
    overall: Agreement
    score_mean_absolute_level_error: Nonnegative | None


class ReplayLatencySummary(ReplayModel):
    successful_calls: Count
    jev_median_ms: Nonnegative | None
    paired_calls: Count
    paired_historical_median_ms: Nonnegative | None
    paired_jev_median_ms: Nonnegative | None


class ReplayReport(ReplayModel):
    report_version: Literal["0.1"] = "0.1"
    report_type: Literal["replay"] = "replay"
    options: ReplayOptions
    contains_synthetic_examples: bool = False
    warnings: list[str] = Field(default_factory=list)
    summary: ReplaySummary
    latency: ReplayLatencySummary
    items: list[ReplayItem]
