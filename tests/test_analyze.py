import json
from importlib.resources import files
from pathlib import Path

import pytest

from jevshift.analyze import analyze_calls
from jevshift.ingest import load_calls
from jevshift.models import LLMCall
from jevshift.report import render_json

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "calls.jsonl"


def test_mixed_report_counts_are_disjoint_and_complete():
    report = analyze_calls(load_calls(EXAMPLES))
    assert report.summary.model_dump() == {
        "total_calls": 20,
        "candidates": 12,
        "choice": 4,
        "noul": 5,
        "score": 3,
        "needs_review": 2,
        "not_candidate": 6,
    }
    assert len(report.assessments) == 20
    assert [item.call_id for item in report.assessments] == [f"call-{i:03}" for i in range(1, 21)]


def test_metadata_coverage_and_zero_values_are_preserved():
    calls = [
        LLMCall(id="zero", prompt="Proceed?", output=True, latency_ms=0, cost_usd=0),
        LLMCall(id="latency", prompt="Proceed?", output=True, latency_ms=100),
        LLMCall(id="cost", prompt="Proceed?", output=True, cost_usd=0.002),
        LLMCall(id="missing", prompt="Proceed?", output=True),
    ]
    metadata = analyze_calls(calls).metadata
    assert metadata.latency_calls == 2
    assert metadata.mean_latency_ms == 50
    assert metadata.cost_calls == 2
    assert metadata.total_cost_usd == pytest.approx(0.002)


def test_empty_report_does_not_invent_metadata():
    report = analyze_calls([])
    assert report.summary.total_calls == 0
    assert report.summary.candidates == 0
    assert report.assessments == []
    assert report.metadata.mean_latency_ms is None
    assert report.metadata.total_cost_usd is None


def test_repeated_outputs_do_not_infer_choice_options():
    calls = [LLMCall(id=str(i), prompt="Choose a route.", output="billing") for i in range(10)]
    report = analyze_calls(calls)
    assert report.summary.candidates == 0
    assert report.summary.not_candidate == 10


def test_json_is_repeatable_versioned_and_omits_original_content():
    calls = [LLMCall(id="safe-id", prompt="Private prompt", state="Private state", output=True)]
    first = render_json(analyze_calls(calls))
    assert first == render_json(analyze_calls(calls))
    data = json.loads(first)
    assert data["report_version"] == "0.1"
    assert set(data) == {"report_version", "summary", "metadata", "assessments"}
    assert set(data["assessments"][0]) == {
        "call_id",
        "status",
        "primitive",
        "confidence",
        "evidence",
        "limitations",
    }
    assert "Private prompt" not in first
    assert "Private state" not in first


def test_packaged_demo_matches_documented_examples():
    assert files("jevshift").joinpath("data", "calls.jsonl").read_bytes() == EXAMPLES.read_bytes()
