import json

import pytest

from jevshift.ingest import TraceError, load_calls, parse_calls


def record(**changes):
    return {"id": "call-1", "prompt": "Choose a route.", "output": "billing", **changes}


def parse_record(**changes):
    return parse_calls([json.dumps(record(**changes))])[0]


@pytest.mark.parametrize("output", [True, False, 1, 1.0, "1", "false"])
def test_preserves_scalar_types(output):
    call = parse_record(output=output)
    assert type(call.output) is type(output)
    assert call.output == output


def test_reads_utf8_and_blank_lines(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text("\n" + json.dumps(record(state="Rückerstattung")) + "\n\n", encoding="utf-8")
    assert load_calls(path)[0].state == "Rückerstattung"


def test_empty_file_is_valid():
    assert parse_calls(["\n", "  \n"]) == []


def test_missing_output_has_physical_line_number():
    lines = ["\n"] * 6 + ['{"id":"bad","prompt":"Choose."}']
    with pytest.raises(TraceError) as error:
        parse_calls(lines)
    assert str(error.value) == "Invalid trace at line 7:\nfield 'output' is required"


@pytest.mark.parametrize("text", ['{"id":', '{"id":"bad",}', "not json"])
def test_malformed_json(text):
    with pytest.raises(TraceError, match="Invalid trace at line 2:\ninvalid JSON at column"):
        parse_calls(["\n", text])


@pytest.mark.parametrize("text", ["[]", '"hello"', "null", "true"])
def test_requires_object_record(text):
    with pytest.raises(TraceError, match="each record must be a JSON object"):
        parse_calls([text])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"prompt": "  "}, "field 'prompt'.*must not be empty"),
        ({"id": ""}, "field 'id'.*must not be empty"),
        ({"allowed_outputs": ["billing"]}, "2–255 Choice options"),
        ({"allowed_outputs": [str(i) for i in range(256)]}, "2–255 Choice options"),
        ({"allowed_outputs": ["billing", "billing"]}, "duplicate Choice values"),
        ({"allowed_outputs": ["billing", " "]}, "Choice options must not be empty"),
        ({"allowed_outputs": {"billing": "", "technical": "Bugs"}}, "Choice descriptions"),
        ({"allowed_outputs": ["technical", "sales"]}, "output.*declared Choice value"),
        ({"output": True, "allowed_outputs": ["true", "false"]}, "declared Choice value"),
        ({"score_levels": ["Low"]}, "2–10 ordered Score levels"),
        ({"score_levels": [f"Level {i}" for i in range(11)]}, "2–10 ordered Score levels"),
        ({"score_levels": ["Low", "Low"]}, "duplicate Score levels"),
        ({"score_levels": ["0", "1"]}, "descriptive text"),
        ({"score_levels": ["Low", ""]}, "descriptive text"),
        ({"output": 3, "score_levels": ["Low", "High"]}, "zero-based integer"),
        ({"output": 0.5, "score_levels": ["Low", "High"]}, "zero-based integer"),
        ({"output": True, "score_levels": ["Low", "High"]}, "zero-based integer"),
        ({"output": "1", "score_levels": ["Low", "High"]}, "zero-based integer"),
        ({"output": -1, "score_levels": ["Low", "High"]}, "zero-based integer"),
        ({"latency_ms": -1}, "latency_ms"),
        ({"cost_usd": -1}, "cost_usd"),
        ({"latency_ms": "100"}, "latency_ms"),
        ({"cost_usd": True}, "cost_usd"),
        ({"output": None}, "output"),
        ({"output": {"answer": True}}, "output"),
        ({"output": ["a", "b"]}, "output"),
        ({"allowed_output": ["a", "b"]}, "Extra inputs are not permitted"),
        (
            {"allowed_outputs": ["billing", "technical"], "score_levels": ["Low", "High"]},
            "not both for one decision",
        ),
    ],
)
def test_invalid_trace_contracts(changes, message):
    with pytest.raises(TraceError, match=message) as error:
        parse_record(**changes)
    assert str(error.value).startswith("Invalid trace at line 1:")


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_numbers_are_invalid(literal):
    with pytest.raises(TraceError):
        parse_calls(['{"id":"bad","prompt":"Rate this.","output":' + literal + "}"])


def test_duplicate_json_choice_keys_are_not_silently_lost():
    line = (
        '{"id":"bad","prompt":"Choose.","output":"a",'
        '"allowed_outputs":{"a":"first","b":"second","a":"third"}}'
    )
    with pytest.raises(TraceError, match="duplicate JSON key 'a'"):
        parse_calls([line])


def test_duplicate_non_choice_key_has_generic_wording():
    with pytest.raises(TraceError) as error:
        parse_calls(['{"id":"first","id":"second","prompt":"Proceed?","output":true}'])
    assert str(error.value) == "Invalid trace at line 1:\nduplicate JSON key 'id'"


def test_duplicate_record_ids_reference_both_lines():
    with pytest.raises(
        TraceError,
        match="line 3:.*",
    ) as error:
        parse_calls([json.dumps(record()), "\n", json.dumps(record())])
    assert "first seen at line 1" in str(error.value)
