import json

import pytest
from typer.testing import CliRunner

from jevshift.cli import app

runner = CliRunner()


def test_demo_works_outside_checkout_without_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    result = runner.invoke(app, ["demo"], terminal_width=140)
    assert result.exit_code == 0, result.output
    assert "Synthetic demo traces" in result.output
    assert "20 calls analyzed" in result.output
    assert "Potential Jev candidates: 12" in result.output
    assert "call-001" in result.output
    assert "call-020" in result.output
    assert "Supplied historical metadata" in result.output


def test_json_output_is_pure_and_stable(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text('{"id":"boolean","prompt":"Proceed?","output":true}\n')
    first = runner.invoke(app, ["analyze", str(path), "--json"])
    second = runner.invoke(app, ["analyze", str(path), "--json"])
    assert first.exit_code == 0, first.output
    assert first.stderr == ""
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["report_version"] == "0.1"
    assert report["summary"]["noul"] == 1
    assert report["metadata"]["total_cost_usd"] is None
    assert "\x1b" not in first.stdout


def test_demo_json_has_no_banner():
    result = runner.invoke(app, ["demo", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["summary"]["total_calls"] == 20


def test_terminal_report_without_metadata_does_not_invent_it(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text('{"id":"b","prompt":"Proceed?","output":false}\n')
    result = runner.invoke(app, ["analyze", str(path)], terminal_width=120)
    assert result.exit_code == 0
    assert "JevShift Analysis" in result.stdout
    assert "1 calls analyzed" in result.stdout
    assert "Noul" in result.stdout
    assert "Supplied historical metadata" not in result.stdout
    assert "$" not in result.stdout


@pytest.mark.parametrize("json_flag", [[], ["--json"]])
def test_bad_record_has_line_number_no_traceback_or_partial_report(tmp_path, json_flag):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id":"ok","prompt":"Proceed?","output":true}\n\n{"id":"bad","prompt":"Proceed?"}\n'
    )
    result = runner.invoke(app, ["analyze", str(path), *json_flag])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Invalid trace at line 3:\nfield 'output' is required\n"
    assert "Traceback" not in result.output


def test_missing_file_is_a_human_readable_error(tmp_path):
    result = runner.invoke(app, ["analyze", str(tmp_path / "missing.jsonl")])
    assert result.exit_code == 1
    assert "Cannot read" in result.stderr
    assert "Traceback" not in result.output
    assert result.stdout == ""


def test_invalid_utf8_is_a_human_readable_error(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b"\xff\xfe")
    result = runner.invoke(app, ["analyze", str(path), "--json"])
    assert result.exit_code == 1
    assert "UTF-8" in result.stderr
    assert result.stdout == ""


def test_trace_ids_are_not_interpreted_as_rich_markup(tmp_path):
    path = tmp_path / "literal.jsonl"
    path.write_text('{"id":"[red]x[/red]","prompt":"Proceed?","output":true}\n')
    result = runner.invoke(app, ["analyze", str(path)], terminal_width=160)
    assert result.exit_code == 0
    assert "[red]x[/red]" in result.stdout


def test_help_exposes_only_implemented_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "analyze" in result.output
    assert "demo" in result.output
    assert "replay" in result.output
