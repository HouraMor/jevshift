import json

import pytest
from pydantic import ValidationError

from jevshift.backends.base import BackendError, DecisionResult
from jevshift.models import LLMCall
from jevshift.replay import contains_bundled_examples, replay_calls
from jevshift.replay_models import ReplayOptions
from jevshift.report import render_json


class FakeBackend:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.seen = []

    def evaluate(self, candidate):
        self.seen.append(candidate.call.id)
        result = next(self.outcomes)
        if isinstance(result, Exception):
            raise result
        return result


def boolean(call_id="b", output=True, **kwargs):
    return LLMCall(id=call_id, prompt="Proceed?", output=output, **kwargs)


def noul(value=0.8):
    return DecisionResult(primitive="noul", value=value)


def test_limit_applies_after_detection_and_counts_all_candidates():
    calls = [
        LLMCall(id="text", prompt="Write an email.", output="Hello."),
        LLMCall(id="review", prompt="Rate this.", output=3),
        boolean("first"),
        boolean("second"),
        boolean("third"),
    ]
    backend = FakeBackend([noul(), noul()])
    report = replay_calls(calls, backend, options=ReplayOptions(limit=2))
    assert backend.seen == ["first", "second"]
    assert report.summary.total_calls == 5
    assert report.summary.candidates_detected == 3
    assert report.summary.replayed == 2


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_invalid_limit(limit):
    with pytest.raises(ValidationError):
        ReplayOptions(limit=limit)


def test_partial_failure_continues_and_does_not_count_as_disagreement():
    backend = FakeBackend([noul(), BackendError("timeout"), noul(0.2)])
    report = replay_calls([boolean("a"), boolean("b"), boolean("c")], backend)
    assert backend.seen == ["a", "b", "c"]
    assert [item.status for item in report.items] == ["success", "failed", "success"]
    assert report.items[1].error == "request timed out"
    assert report.items[1].comparison is None
    assert report.items[1].historical is True
    assert report.summary.succeeded == 2
    assert report.summary.failed == 1
    assert report.summary.noul.model_dump() == {"matched": 1, "compared": 2, "rate": 0.5}
    assert report.summary.overall.rate == 0.5


def test_all_failures_have_null_agreement_and_latency():
    report = replay_calls(
        [boolean("a"), boolean("b")],
        FakeBackend([BackendError("api_error"), BackendError("connection")]),
    )
    assert report.summary.failed == 2
    assert report.summary.overall.compared == 0
    assert report.summary.overall.rate is None
    assert report.latency.jev_median_ms is None
    assert report.latency.paired_calls == 0


def test_latency_uses_successful_pairs_only(monkeypatch):
    clock = iter([0, 0.1, 1, 3, 4, 4.9, 5, 5.5])
    monkeypatch.setattr("jevshift.replay.perf_counter", lambda: next(clock))
    calls = [
        boolean("a", latency_ms=1000),
        boolean("b"),
        boolean("c", latency_ms=90000),
        boolean("d", latency_ms=3000),
    ]
    report = replay_calls(calls, FakeBackend([noul(), noul(), BackendError("timeout"), noul()]))
    assert report.latency.successful_calls == 3
    assert report.latency.jev_median_ms == pytest.approx(500)
    assert report.latency.paired_calls == 2
    assert report.latency.paired_historical_median_ms == 2000
    assert report.latency.paired_jev_median_ms == pytest.approx(300)
    assert report.items[2].jev_latency_ms == pytest.approx(900)


def test_missing_historical_latency_is_not_zero():
    report = replay_calls([boolean()], FakeBackend([noul()]))
    assert report.latency.successful_calls == 1
    assert report.latency.paired_calls == 0
    assert report.latency.paired_historical_median_ms is None
    assert report.latency.paired_jev_median_ms is None


def test_recorded_zero_latency_stays_a_valid_pair(monkeypatch):
    monkeypatch.setattr("jevshift.replay.perf_counter", lambda: 1.0)
    report = replay_calls([boolean(latency_ms=0)], FakeBackend([noul()]))
    assert report.latency.paired_calls == 1
    assert report.latency.paired_jev_median_ms == 0
    assert report.latency.paired_historical_median_ms == 0


def test_mixed_primitive_agreement_and_score_error():
    calls = [
        LLMCall(id="c", prompt="Route.", output="a", allowed_outputs=["a", "b"]),
        boolean("n", output="NO"),
        LLMCall(id="s", prompt="Rate.", output=1, score_levels=["Low", "High"]),
    ]
    report = replay_calls(
        calls,
        FakeBackend(
            [
                DecisionResult(primitive="choice", value="b"),
                noul(0.2),
                DecisionResult(primitive="score", value=0.7),
            ]
        ),
    )
    assert report.summary.choice.matched == 0
    assert report.summary.noul.matched == 1
    assert report.summary.score.matched == 1
    assert report.summary.overall.compared == 3
    assert report.summary.overall.rate == pytest.approx(2 / 3)
    assert report.summary.score_mean_absolute_level_error == pytest.approx(0.3)
    assert report.items[1].historical is False


def test_no_candidates_makes_no_backend_calls():
    backend = FakeBackend([])
    report = replay_calls([LLMCall(id="text", prompt="Write an email.", output="Hello.")], backend)
    assert backend.seen == []
    assert report.summary.replayed == 0
    assert report.summary.overall.rate is None


@pytest.mark.parametrize(
    "result",
    [
        None,
        {"primitive": "noul", "value": 0.5},
        DecisionResult(primitive="score", value=0.5),
        DecisionResult.model_construct(primitive="noul", value=float("nan")),
    ],
)
def test_invalid_backend_results_fail_without_aborting(result):
    report = replay_calls([boolean("bad"), boolean("ok")], FakeBackend([result, noul()]))
    assert report.items[0].error_code == "invalid_response"
    assert report.items[1].status == "success"


def test_json_has_no_prompt_state_or_arbitrary_exception_message():
    secret = "test-key-not-for-output"
    call = LLMCall(id="id", prompt="PRIVATE PROMPT", state="PRIVATE STATE", output=True)
    report = replay_calls(
        [call], FakeBackend([RuntimeError(secret + " PRIVATE PROMPT PRIVATE STATE")])
    )
    serialized = render_json(report)
    assert secret not in serialized
    assert "PRIVATE PROMPT" not in serialized
    assert "PRIVATE STATE" not in serialized
    data = json.loads(serialized)
    assert data["report_version"] == "0.1"
    assert data["report_type"] == "replay"
    assert data["items"][0]["historical"] is True
    assert data["items"][0]["error"] == "backend evaluation failed"
    assert "prompt" not in data["items"][0]
    assert "state" not in data["items"][0]


def test_recognizes_copied_subset_of_synthetic_data():
    call = LLMCall(
        id="call-009",
        prompt="Does the message 'Please take your time; next month is fine' express urgency?",
        output=False,
    )
    assert contains_bundled_examples([call])
    assert not contains_bundled_examples([boolean("other")])
    report = replay_calls([call], FakeBackend([noul(0.1)]), contains_synthetic_examples=True)
    assert "not benchmarks" in report.warnings[0]
    assert report.items[0].state_source == "prompt"
