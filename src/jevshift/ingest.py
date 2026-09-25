"""Read the normalized JSONL format without provider-specific adapters."""

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from jevshift.models import LLMCall


class TraceError(ValueError):
    """An input problem suitable for display without a traceback."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key '{key}'")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"{value} is not a valid finite JSON number")


def _validation_message(error: ValidationError) -> str:
    messages = []
    for detail in error.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in detail["loc"])
        if detail["type"] == "missing":
            message = f"field '{location}' is required"
        else:
            message = detail["msg"].removeprefix("Value error, ")
            if location:
                message = f"field '{location}': {message}"
        messages.append(message)
    return "\n".join(messages)


def parse_calls(lines: Iterable[str]) -> list[LLMCall]:
    """Fail at the first bad record; blank lines keep their physical line numbers."""
    calls = []
    seen_ids: dict[str, int] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(
                line, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
            if not isinstance(data, dict):
                raise ValueError("each record must be a JSON object")
            call = LLMCall.model_validate(data)
            if call.id in seen_ids:
                raise ValueError(
                    f"duplicate id '{call.id}'; first seen at line {seen_ids[call.id]}"
                )
        except json.JSONDecodeError as error:
            reason = f"invalid JSON at column {error.colno}: {error.msg}"
            raise TraceError(f"Invalid trace at line {line_number}:\n{reason}") from None
        except ValidationError as error:
            raise TraceError(
                f"Invalid trace at line {line_number}:\n{_validation_message(error)}"
            ) from None
        except ValueError as error:
            raise TraceError(f"Invalid trace at line {line_number}:\n{error}") from None
        seen_ids[call.id] = line_number
        calls.append(call)
    return calls


def load_calls(path: Path) -> list[LLMCall]:
    with path.open(encoding="utf-8") as stream:
        return parse_calls(stream)
