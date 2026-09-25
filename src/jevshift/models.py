"""Normalized trace records and the versioned analysis report."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Primitive = Literal["choice", "noul", "score"]
Status = Literal["candidate", "needs_review", "not_candidate"]
Confidence = Literal["high", "medium", "low"]


class LLMCall(BaseModel):
    """One historical decision; explicit contracts must match its historical output."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    id: str
    prompt: str
    output: str | bool | int | float
    state: str | None = None
    allowed_outputs: list[str] | dict[str, str] | None = None
    score_levels: list[str] | None = None
    model: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)

    @field_validator("id", "prompt")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value

    @field_validator("allowed_outputs")
    @classmethod
    def validate_options(cls, value: list[str] | dict[str, str] | None):
        if value is None:
            return value
        if not 2 <= len(value) <= 255:
            raise ValueError("must contain 2–255 Choice options")
        if any(not option.strip() for option in value):
            raise ValueError("Choice options must not be empty or whitespace")
        if len(set(value)) != len(value):
            raise ValueError("duplicate Choice values are not allowed")
        if isinstance(value, dict) and any(
            not description.strip() for description in value.values()
        ):
            raise ValueError("Choice descriptions must not be empty; use a list for bare labels")
        return value

    @field_validator("score_levels")
    @classmethod
    def validate_levels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        if not 2 <= len(value) <= 10:
            raise ValueError("must contain 2–10 ordered Score levels")
        if any(not any(char.isalpha() for char in level) for level in value):
            raise ValueError("Score levels must be descriptive text, not blank or numeric labels")
        if len(set(value)) != len(value):
            raise ValueError("duplicate Score levels make output mapping ambiguous")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "LLMCall":
        if self.allowed_outputs is not None and self.score_levels is not None:
            raise ValueError("declare allowed_outputs or score_levels, not both for one decision")
        if self.allowed_outputs is not None:
            if not isinstance(self.output, str) or self.output not in self.allowed_outputs:
                raise ValueError("field 'output' must exactly match a declared Choice value")
        if self.score_levels is not None:
            self.score_index()
        return self

    def score_index(self) -> int:
        """Map a rubric label or a zero-based integral number to one historical level."""
        if self.score_levels is None:
            raise ValueError("score_levels is required to map a Score output")
        if isinstance(self.output, str) and self.output in self.score_levels:
            return self.score_levels.index(self.output)
        # bool is a subclass of int; it must never become a Score index.
        if type(self.output) in (int, float):
            index = int(self.output)
            if self.output == index and 0 <= index < len(self.score_levels):
                return index
        raise ValueError(
            "field 'output' must be an exact Score level label or a zero-based integer "
            f"from 0 to {len(self.score_levels) - 1}"
        )


class CandidateAssessment(BaseModel):
    call_id: str
    status: Status
    primitive: Primitive | None = None
    confidence: Confidence | None = None
    evidence: list[str]
    limitations: list[str] = Field(default_factory=list)


class AnalysisSummary(BaseModel):
    total_calls: int
    candidates: int
    choice: int
    noul: int
    score: int
    needs_review: int
    not_candidate: int


class MetadataSummary(BaseModel):
    """Observed metadata only, with separate coverage for each measurement."""

    latency_calls: int = 0
    mean_latency_ms: float | None = None
    cost_calls: int = 0
    total_cost_usd: float | None = None


class ScanReport(BaseModel):
    report_version: Literal["0.1"] = "0.1"
    summary: AnalysisSummary
    metadata: MetadataSummary
    assessments: list[CandidateAssessment]
