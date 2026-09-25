"""Combine per-decision assessments and observed metadata."""

from collections import Counter
from collections.abc import Iterable
from math import fsum

from jevshift.detect import assess
from jevshift.models import AnalysisSummary, LLMCall, MetadataSummary, ScanReport


def analyze_calls(calls: Iterable[LLMCall]) -> ScanReport:
    assessments = []
    latencies = []
    costs = []
    for call in calls:
        assessments.append(assess(call))
        if call.latency_ms is not None:
            latencies.append(call.latency_ms)
        if call.cost_usd is not None:
            costs.append(call.cost_usd)
    statuses = Counter(assessment.status for assessment in assessments)
    primitives = Counter(
        assessment.primitive for assessment in assessments if assessment.status == "candidate"
    )
    return ScanReport(
        summary=AnalysisSummary(
            total_calls=len(assessments),
            candidates=statuses["candidate"],
            choice=primitives["choice"],
            noul=primitives["noul"],
            score=primitives["score"],
            needs_review=statuses["needs_review"],
            not_candidate=statuses["not_candidate"],
        ),
        metadata=MetadataSummary(
            latency_calls=len(latencies),
            mean_latency_ms=fsum(value / len(latencies) for value in latencies)
            if latencies
            else None,
            cost_calls=len(costs),
            total_cost_usd=fsum(costs) if costs else None,
        ),
        assessments=assessments,
    )
