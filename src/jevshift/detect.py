"""Deterministic rules: no network, model calls, or cross-record inference."""

import re

from jevshift.models import CandidateAssessment, LLMCall

BOOLEAN_STRINGS = {"yes", "no", "true", "false"}
AMBIGUOUS_BOOLEANS = {"maybe", "likely", "probably", "possibly", "uncertain", "unknown"}
GENERATION_PROMPT = re.compile(
    r"^\s*(?:please\s+)?(?:write|draft|compose|summari[sz]e|generate|explain)\b", re.IGNORECASE
)


def assess(call: LLMCall) -> CandidateAssessment:
    """Assess exactly one validated decision. Explicit contracts take precedence."""
    limitations = []
    if call.state is None or not call.state.strip():
        limitations.append(
            "No separate state supplied; replay uses the prompt as the available context."
        )

    if call.allowed_outputs is not None:
        return CandidateAssessment(
            call_id=call.id,
            status="candidate",
            primitive="choice",
            confidence="high",
            evidence=[
                f"{len(call.allowed_outputs)} explicit outputs",
                "output matches declared set",
            ],
            limitations=limitations,
        )
    if call.score_levels is not None:
        return CandidateAssessment(
            call_id=call.id,
            status="candidate",
            primitive="score",
            confidence="high",
            evidence=[
                f"{len(call.score_levels)}-level ordered rubric",
                f"output maps to level {call.score_index()}",
            ],
            limitations=limitations
            + ["Rubric quality and mapping Jev's fractional scores require evaluation."],
        )
    normalized = call.output.strip().casefold() if isinstance(call.output, str) else None
    if type(call.output) is bool or normalized in BOOLEAN_STRINGS:
        evidence = "boolean result" if type(call.output) is bool else "normalized boolean string"
        return CandidateAssessment(
            call_id=call.id,
            status="candidate",
            primitive="noul",
            confidence="high" if type(call.output) is bool else "medium",
            evidence=[evidence],
            limitations=limitations
            + ["Jev returns a probability; a decision threshold needs tuning."],
        )
    if type(call.output) in (int, float):
        return CandidateAssessment(
            call_id=call.id,
            status="needs_review",
            primitive="score",
            confidence="low",
            evidence=["numeric output without an explicit score_levels rubric"],
            limitations=limitations + ["A number alone does not establish a bounded scoring task."],
        )
    if normalized in AMBIGUOUS_BOOLEANS:
        return CandidateAssessment(
            call_id=call.id,
            status="needs_review",
            confidence="low",
            evidence=["ambiguous pseudo-boolean output"],
            limitations=limitations + ["Cannot interpret this output as a Boolean decision."],
        )
    free_form = GENERATION_PROMPT.match(call.prompt) is not None
    return CandidateAssessment(
        call_id=call.id,
        status="not_candidate",
        evidence=["free-form generation" if free_form else "no explicit bounded output contract"],
        limitations=limitations
        + (
            ["Declare allowed_outputs if this is a finite classification task."]
            if not free_form
            else []
        ),
    )
