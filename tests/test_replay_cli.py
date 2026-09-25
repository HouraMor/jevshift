import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from jevshift.backends.base import BackendError, DecisionResult
from jevshift.cli import app

runner = CliRunner()


@pytest.fixture
def traces(tmp_path):
    path = tmp_path / "calls.jsonl"
    records = [
        {"id": "text", "prompt": "Write an email.", "output": "Hello."},
        {
            "id": "one",
            "prompt": "PRIVATE PROMPT",
            "state": "PRIVATE STATE",
            "output": True,
            "latency_ms": 100,
        },
        {"id": "two", "prompt": "Proceed?", "output": "no"},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records))
    return path


@pytest.fixture
def backend(monkeypatch):
    fake = Mock()
    fake.__enter__ = Mock(return_value=fake)
    fake.__exit__ = Mock(return_value=False)
    fake.evaluate.return_value = DecisionResult(primitive="noul", value=0.6)
    constructor = Mock(return_value=fake)
    monkeypatch.setattr("jevshift.backends.jev.JevBackend", constructor)
    return fake, constructor


def test_replay_json_limit_threshold_and_model(traces, backend):
    fake, constructor = backend
    result = runner.invoke(
        app,
        [
            "replay",
            str(traces),
            "--limit",
            "1",
            "--noul-threshold",
            "0.7",
            "--model",
            "jev-pinned",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["report_version"] == "0.1"
    assert report["summary"]["candidates_detected"] == 2
    assert report["summary"]["replayed"] == 1
    assert report["summary"]["noul"]["rate"] == 0.0
    assert report["items"][0]["comparison"]["prediction"] is False
    assert "PRIVATE PROMPT" not in result.stdout
    assert "PRIVATE STATE" not in result.stdout
    assert "TypeSafe API" in result.stderr
    constructor.assert_called_once_with(model="jev-pinned")
    fake.evaluate.assert_called_once()
    fake.__exit__.assert_called_once()


def test_missing_key_is_concise_and_no_json_report(traces):
    result = runner.invoke(app, ["replay", str(traces), "--json"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Set TYPESAFE_API_KEY before running live replay.\n"
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "option",
    [
        ["--limit", "0"],
        ["--limit", "-1"],
        ["--noul-threshold", "1.1"],
        ["--noul-threshold", "-0.1"],
        ["--noul-threshold", "nan"],
        ["--noul-threshold", "inf"],
        ["--model", " "],
    ],
)
def test_invalid_options_fail_before_backend_creation(traces, backend, option):
    _, constructor = backend
    result = runner.invoke(app, ["replay", str(traces), *option])
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    constructor.assert_not_called()


def test_partial_failure_emits_complete_json_and_nonzero_exit(traces, backend):
    fake, _ = backend
    fake.evaluate.side_effect = [
        BackendError("timeout"),
        DecisionResult(primitive="noul", value=0.1),
    ]
    result = runner.invoke(app, ["replay", str(traces), "--json"])
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["summary"]["failed"] == 1
    assert report["summary"]["succeeded"] == 1
    assert report["summary"]["overall"] == {"matched": 1, "compared": 1, "rate": 1.0}
    assert report["items"][0]["error"] == "request timed out"
    assert "Traceback" not in result.output


def test_terminal_report_shows_denominators_and_paired_latency(traces, backend):
    result = runner.invoke(app, ["replay", str(traces)], terminal_width=140)
    assert result.exit_code == 0
    assert "JevShift Replay" in result.stdout
    assert "1 / 2" in result.stdout
    assert "50.0%" in result.stdout
    assert "same 1 comparable calls" in result.stdout
    assert "nearest-level match (approximation)" in result.stdout
    assert "cheaper" not in result.stdout
    assert "PRIVATE" not in result.stdout


def test_bad_input_fails_before_backend_creation(tmp_path, backend):
    _, constructor = backend
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"x","prompt":"Proceed?"}')
    result = runner.invoke(app, ["replay", str(path)])
    assert result.exit_code == 1
    assert "Invalid trace at line 1" in result.stderr
    constructor.assert_not_called()


def test_known_synthetic_samples_are_flagged_in_json(tmp_path, backend):
    path = tmp_path / "copied-sample.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "call-009",
                "prompt": (
                    "Does the message 'Please take your time; next month is fine' express urgency?"
                ),
                "output": False,
            }
        )
    )
    result = runner.invoke(app, ["replay", str(path), "--json"])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["contains_synthetic_examples"] is True
    assert "not benchmarks" in report["warnings"][0]
