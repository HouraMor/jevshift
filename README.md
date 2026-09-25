# JevShift

**Find the LLM calls that should have been Jev calls.**

JevShift scans historical LLM decisions and finds bounded
classification, routing, Boolean, and scoring calls that may be
good candidates for evaluation with Jev.

Find bounded LLM decisions, replay them with Jev, and measure how closely Jev
reproduces the historical decisions.

> JevShift identifies evaluation candidates. It does not automatically recommend that a production LLM call be replaced.

Each input record represents **one historical decision**. Candidate detection is
always deterministic and never calls an LLM.

| Command | Where data goes | API key |
| --- | --- | --- |
| `demo` / `analyze` | **Offline.** Trace data stays local. | Not required |
| `replay` | **Live. Sends prompt, state, and declared criteria to the TypeSafe API.** | `TYPESAFE_API_KEY` |

Historical outputs are used only for comparison and are never added to replay
instructions or contextual state. Replay reports include normalized historical
answers, but omit full prompts and states.

## Quickstart

Python 3.10 or newer is required. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
jevshift demo
jevshift analyze examples/calls.jsonl
jevshift analyze examples/calls.jsonl --json > report.json
```

Then, to make live requests:

```bash
export TYPESAFE_API_KEY="..."
jevshift replay examples/calls.jsonl
jevshift replay examples/calls.jsonl --limit 3 --noul-threshold 0.7 --json > replay.json
```

**The bundled traces and their historical latencies/costs are synthetic. Replaying
them demonstrates the workflow; agreement and latency results are not benchmarks.**
Replay sends data to the API and may incur usage charges. No credentials are
stored by JevShift, and it does not load `.env` files.

On Windows, activate with `.venv\Scripts\activate`. For a regular installation
without development tools, use `python -m pip install .`.

## Demo

`jevshift demo` loads 20 synthetic traces bundled in the installed package. It
works from any directory. `jevshift demo --json` produces the same report format
as `analyze --json`.

The examples cover tool selection, support routing, retry/stop decisions,
escalation, RAG acceptance, severity, email generation, summarization, code
generation, and explanations. They also include missing rubrics, ambiguous
answers, a route without its allowed set, and a Boolean answer mixed with prose.
Costs and latencies in these synthetic records are illustrative, not benchmarks.
The demo produces 12 candidates (4 Choice, 5 Noul, 3 Score), 2 calls needing review,
and 6 non-candidates. Invalid contracts are covered in tests rather than silently
included in a successful demo.

## Trace format

Supply UTF-8 JSONL: one JSON object per nonblank line. IDs must be unique within
the file. These examples are formatted across lines for readability; each must
occupy a single physical line in a trace file.

```json
{
  "id": "route-001",
  "prompt": "Which team should handle this request?",
  "state": "I was charged twice for my subscription.",
  "output": "billing",
  "allowed_outputs": {
    "billing": "Payments, invoices and refunds",
    "technical": "Product bugs and technical problems",
    "sales": "Questions about purchasing"
  },
  "model": "example-model",
  "latency_ms": 1750,
  "cost_usd": 0.0021
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string, required | Nonblank identifier, unique within the input file. |
| `prompt` | string, required | Nonblank historical decision instructions. |
| `output` | string, Boolean, integer, or finite float, required | Historical answer. Objects, arrays, and null are unsupported. |
| `state` | string or null | Context inspected by the historical decision. |
| `allowed_outputs` | list of strings or string-to-description object, or null | Explicit Choice contract. |
| `score_levels` | list of strings or null | Ordered, descriptive rubric, from lowest to highest. |
| `model` | string or null | Historical model identifier. |
| `latency_ms` | nonnegative finite number or null | Supplied historical latency. |
| `cost_usd` | nonnegative finite number or null | Supplied historical cost. |

Unknown fields are rejected to catch misspelled contracts. Types are strict:
`true` remains Boolean, `1` remains numeric, and `"1"` remains a string. Optional
fields can be omitted or null. Choice labels and Score labels match exactly,
including case and whitespace. Empty labels, duplicate values, duplicate JSON
keys, and empty Choice descriptions are rejected; use a list for bare labels.
Declare either `allowed_outputs` or `score_levels`, never both.

A Boolean decision needs no separate declaration:

```json
{"id":"escalate-001","prompt":"Does this require human escalation?","state":"The customer threatens legal action.","output":true}
```

A Score needs its rubric:

```json
{"id":"severity-001","prompt":"How severe is this incident?","state":"Production is unavailable for all users.","output":3,"score_levels":["Low","Moderate","High","Critical"]}
```

Score outputs must exactly name a level or use a **zero-based index**: `3` means
`Critical` above. Integral floats such as `3.0` also map to that index. Fractional
historical values, Boolean indices, numeric strings, and out-of-range indices
are rejected. Numeric-only rubric labels are rejected. Jev's replay results can
be fractional; the comparison policy below preserves the original score and
reports an approximate nearest-level match.

## Detection rules

Rules run independently for each record, in this order:

1. **Choice:** 2–255 explicitly declared, unique options, with the historical
   string output in the declared set. This gives high rule confidence.
2. **Score:** 2–10 explicitly declared, unique descriptive levels, with an
   unambiguous output-to-level mapping. This gives high rule confidence.
3. **Noul:** an actual JSON Boolean gives high rule confidence. One of `yes`,
   `no`, `true`, `false`, ignoring surrounding whitespace and case, gives medium
   confidence: an exact token is weaker contract evidence than a Boolean type.
   Both are candidates. An explicit Choice containing `yes`/`no` remains Choice.
4. **Needs review:** numeric output without a rubric, or exactly `maybe`,
   `likely`, `probably`, `possibly`, `uncertain`, or `unknown` (case/whitespace
   normalized). Numeric cases show Score as a possible primitive with low
   confidence. Pseudo-booleans have no assigned primitive.
5. **Not a candidate:** remaining strings, including free-form email,
   summarization, code, and explanations. A prompt keyword can explain the
   result but does not establish a bounded contract. A bare routing label
   without `allowed_outputs` is also not a candidate under these rules.

Confidence means strength of the deterministic evidence. It is **not** Jev's
model confidence, estimated accuracy, or a probability of successful migration.
Candidate counts exclude all `needs_review` records.

Declared contracts are validated before detection. Invalid contracts (including
an output outside the declared set) stop the scan with exit code 1. Missing
contracts can instead produce `needs_review` or `not_candidate`, as above.

## Reports and errors

The terminal report shows counts, a concise per-call table, and a separate
summary of supplied metadata. Cost and latency coverage are counted separately.
There is no cost extrapolation, estimated savings, or invented latency.

`--json` writes only one JSON object to stdout, with these stable fields:

```json
{
  "report_version": "0.1",
  "summary": {
    "total_calls": 0,
    "candidates": 0,
    "choice": 0,
    "noul": 0,
    "score": 0,
    "needs_review": 0,
    "not_candidate": 0
  },
  "metadata": {
    "latency_calls": 0,
    "mean_latency_ms": null,
    "cost_calls": 0,
    "total_cost_usd": null
  },
  "assessments": []
}
```

Each assessment contains `call_id`, `status`, `primitive`, `confidence`,
`evidence`, and `limitations`. Null denotes unavailable data, while measured
zero stays zero. Assessments preserve input order. There are no timestamps or
random fields: the same input and version produce the same JSON. Consumers
should use the structured fields; evidence and limitation prose may evolve.
Original prompts, states, and outputs are not copied into reports.

Blank files produce an empty report. Blank lines are skipped but count toward
error line numbers. Malformed input fails at the first error, writes a readable
message to stderr, leaves stdout empty, and does not produce a partial report:

```text
Invalid trace at line 7:
field 'output' is required
```

## Limitations

- Only the normalized JevShift format is supported. A historical request with
  multiple decisions must be split by the caller before ingestion.
- A bounded answer does not establish that the task is easy, its historical
  answer is correct, or Jev can perform it reliably. A Boolean result can even
  come from a poorly specified prompt; the scanner does not understand intent.
- Observing a small number of distinct responses never establishes a closed
  set. Options must be declared explicitly, even across repeated calls.
- Rubric validation checks structure and text, not semantic quality. Labels
  such as `Low` and `High` still need clearer descriptions for useful replay.
- Tool selection can be a candidate while generating its free-form arguments
  remains outside the candidate's scope.
- Missing `state` is allowed and flagged as a limitation. Replay uses the prompt
  itself as the available context in that case; it cannot recover missing facts.
- Full traces and assessments are held in memory. Streaming large trace files
  is deferred until there is a concrete scale requirement.
- Logs can contain sensitive data. Analysis stays local, but review local
  traces and exported identifiers before sharing them.

## Live replay

```bash
jevshift replay traces.jsonl
jevshift replay traces.jsonl --limit 5
jevshift replay traces.jsonl --noul-threshold 0.7
jevshift replay traces.jsonl --model MODEL --json
```

Replay runs the existing detector, selects only `candidate` records in input
order, then applies `--limit` to that selection. `--limit` must be positive.
Each selected record receives one synchronous SDK request with one question.
`needs_review` and `not_candidate` records never reach the backend.

`backends/jev.py` implements the small vendor-independent
`DecisionBackend.evaluate(ReplayCandidate) -> DecisionResult` protocol. It sends:

- `prompt` as the question instructions.
- `state` verbatim when present (including an explicitly empty string).
- The prompt itself as state when `state` is absent or null. This fallback is
  isolated in `build_state()` and recorded as `state_source: "prompt"` per item.
- Choice descriptions as supplied; a bare list becomes labels with `None`
  descriptions, which the SDK supports.
- Score levels as an ordered list, preserving their zero-based positions.

It never sends the historical output, call ID, latency, cost, or historical
model as request evidence. A historical Choice label will naturally also occur
among the complete declared options; it is never singled out as the answer.

### Comparison policy

| Primitive | Policy |
| --- | --- |
| Choice | Exact, case-sensitive label equality. |
| Noul | Normalize historical booleans and exact yes/no/true/false tokens; predict true when probability **>= threshold**, default `0.5`. |
| Score | Preserve the fractional expected score and absolute distance from the historical level; compare the nearest Jev level with the historical index. |

The Noul threshold must be finite and between 0 and 1 inclusive. Raw Noul
probabilities and the applied threshold are retained in reports.

Nearest Score levels use **ties toward the higher level**: `0.5 -> 1`,
`1.5 -> 2`, and `2.5 -> 3`. Values below a half go to the lower level; Python's
banker's rounding is not used. Each Score comparison includes
`historical_level_index`, `jev_score`, `absolute_level_error`,
`nearest_jev_level`, and `exact_level_match`. The last field refers to equality
of the rounded level, **not equality of the raw fractional score**.

Scores outside the declared rubric, unexpected Choice labels, and mismatched
probability keys fail validation. They are not clamped or counted as ordinary
disagreements. A supplied distribution must sum to 1 within an absolute
tolerance of `0.001`. Score agreement and overall agreement explicitly include
approximate nearest-level matches. Historical agreement is not correctness:
the historical answers themselves may be wrong.

### Replay reports and failures

`--json` writes a single JSON object to stdout with `report_version: "0.1"` and
`report_type: "replay"`. The live-data notice goes to stderr. Top-level fields
are `options`, `contains_synthetic_examples`, `warnings`, `summary`, `latency`,
and `items`, alongside the report version/type.

Each item contains its ID, primitive, normalized `historical` value, state
source, status, result, comparison, measured Jev latency, supplied historical
latency, and a safe error/code on failure. Full prompts and states are omitted.
Known bundled examples (including copied subsets) are flagged as synthetic;
other synthetic inputs remain the caller's responsibility to identify.

The summary gives detected/replayed/succeeded/failed counts and agreement per
primitive and overall. Every agreement contains `matched`, `compared`, and
`rate` (a fraction in `[0, 1]`). **Only successful comparisons contribute to the
denominator.** When none succeed, the rate is null, not zero. Score mean absolute
level error is also reported when scores succeed.

Authentication, timeout, rate-limit, connection, API, and invalid-response
failures are recorded per item, and the remaining candidates still run. API
error bodies and arbitrary exception messages are not copied into reports.
Missing/invalid local SDK configuration fails before any request, with a concise
message and no traceback. Invalid trace files fail before creating the SDK client.

Exit codes:

- `0`: all selected candidates completed (disagreements are still successful evaluations).
- `1`: input/configuration error, or at least one replay failure. For replay
  failures, a complete report is still emitted, including successful items.
- `2`: invalid CLI options.

### Latency and usage

Jev latency uses a monotonic timer around each backend evaluation, including
request construction, SDK/network time, and response validation, but excluding
historical comparison and report rendering. The client is initialized before
timing. Requests run sequentially with automatic SDK retries disabled and a
30-second timeout for HTTP operations. A failed attempt retains its elapsed
time but is excluded from successful latency statistics.

Reports show the Jev median across all successful calls and a separate pair of
historical/Jev medians over **the same successful records with supplied
historical latency**. Missing historical measurements are not zero; recorded
zero is retained. Different historical timing methods may still limit the
comparison. There is no speedup claim or synthetic benchmark claim.

The SDK's `input_tokens` and `output_tokens` are preserved under `result.usage`
when reported. Missing counts remain null. No token prices, cost estimates,
percentage savings, or extrapolations are applied.

### Verified SDK contract

The installed/latest package retrieved for this milestone was `typesafe-sdk
0.7.1`, verified on 2026-09-25 against its installed source and the
[official Python SDK docs](https://docs.typesafe.ai/sdk/python). The runtime
dependency is constrained to `>=0.7.1,<0.8` because this is a pre-1.0 SDK.

- `TypeSafeClient(model=..., timeout=..., retry=RetryPolicy(max_retries=0))`
  reads `TYPESAFE_API_KEY` normally. JevShift does not keep a separate credential.
- `client.system_one(state=..., questions={"decision": question})` returns
  `SystemOneResponse`; answers are read from `response.answers["decision"]`.
- `Choice` takes a label-to-description mapping, including `None` descriptions.
  `ChoiceAnswer` exposes `choice`, `confidence`, and label-keyed `probabilities`.
- `NoulAnswer.noul` is the probability of true. It has no confidence field.
- `Score.criteria` is an ordered sequence, not a dictionary. `ScoreAnswer`
  exposes `score`, `confidence`, `legend: dict[int, ...]`, and
  `probabilities: dict[int, float]`. The returned legend must match our supplied
  string rubric. JevShift intentionally normalizes probability keys to canonical
  strings such as `"0"` and `"1"`, preserving the indices in JSON.
- `response.model` records the resolved model and `response.usage` supplies
  token counts. `--model` sets the SDK client default; otherwise the SDK honors
  `TYPESAFE_DEFAULT_MODEL` or its `jev-latest` default. The historical trace's
  model is never used to select a Jev model.

Normal SDK configuration, including `TYPESAFE_BASE_URL`, is honored; the default
service is TypeSafe. SDK debug logging can expose request/response bodies, so
leave it disabled when those bodies should stay out of local logs. Model alias
behavior can change; use a supported explicit model version for reproducibility.

## Development

```bash
python -m pytest
ruff check .
ruff format --check .
```

Tests block socket connections and cover validation, rule boundaries, report
counts, metadata coverage, errors, CLI commands, replay comparisons, and SDK
request/response handling through mocks and a mock HTTP transport. Tests remove
real credentials from their environment. No live integration tests run in the
normal suite, and a real API key is not required for development.
The packaged demo mirrors `examples/calls.jsonl`; a test detects drift.

## Roadmap

- Evaluation against independently reviewed labels, beyond historical agreement
- OpenAI importer
- Anthropic importer
- LangChain importer
- LangGraph importer
- HTML report
- Source-code scanning

## License

MIT. See [LICENSE](LICENSE).
