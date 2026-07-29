# Research Agent API - Synchronous Client

A robust Python client for the [Bigdata.com Research Agent API](https://docs.bigdata.com/how-to-guides/agents) that consumes the Server-Sent Events stream and hands back one finished result: the answer, correctly attributed citations, and any charts the agent produced.

> **Note:** The Research Agent API evolves continuously. This wrapper tracks the protocol described in the [concept guides](https://docs.bigdata.com/how-to-guides/agents/concepts/overview) as of July 2026.

## Features

| Feature | Description |
|---------|-------------|
| **Synchronous interface** | One blocking call - no async/await |
| **Complete message coverage** | Every public SSE message type is handled; unknown types are ignored so new API versions do not break the client |
| **Accurate citations** | Document citations and whole-tool attributions are handled separately, so a reference never renders as a blank `N/A` entry |
| **Typed errors** | HTTP status codes and in-stream `ERROR` events map to specific exception classes |
| **Automatic retries** | Exponential backoff with jitter for `429`, `5xx`, and transient network failures |
| **Charts** | `CHART` events are collected as Vega-Lite specs anchored to answer offsets |
| **Conversation continuity** | `chat_id` follow-ups and `checkpoint_id` branching |

---

## Quick start

### 1. Set your API key

```bash
export BIGDATA_API_KEY="your-api-key-here"
```

### 2. Run a query

```python
from research_client import ResearchClient

client = ResearchClient()
result = client.research("How is the Microsoft performing?")

print(result.get_markdown_with_citations())
```

---

## Installation

Copy `research_client.py` into your project. The only third-party dependency is `requests`.

```bash
uv add requests
```

**Requirements:** Python 3.9+, `requests`, and a `BIGDATA_API_KEY`.

---

## API reference

### `ResearchClient`

```python
client = ResearchClient(
    api_key=None,               # or set BIGDATA_API_KEY
    base_url="https://agents.bigdata.com/v1",
    timeout=300,
    stream_timeout=60.0,
    max_retries=3,
    retry_delay=1.0,
    retry_backoff=2.0,
    retry_max_delay=60.0,
    persistence_mode="enabled",
    code_execution=None,
    chart_generation=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | Falls back to the `BIGDATA_API_KEY` environment variable |
| `base_url` | `str` | production | API base URL |
| `timeout` | `int` | `300` | Connection timeout in seconds |
| `stream_timeout` | `float` | `60.0` | Max seconds to wait between SSE chunks before treating the connection as stalled. `None` disables the check |
| `max_retries` | `int` | `3` | Retry attempts for transient failures |
| `retry_delay` | `float` | `1.0` | Initial backoff delay |
| `retry_backoff` | `float` | `2.0` | Exponential backoff multiplier |
| `retry_max_delay` | `float` | `60.0` | Upper bound on the backoff delay |
| `persistence_mode` | `str` | `"enabled"` | Saves conversation history so `chat_id` follow-ups work. The API itself defaults to `"disabled"` |
| `code_execution` | `bool` | `None` | Whether the agent may run sandboxed Python. `None` keeps the server default (on) |
| `chart_generation` | `bool` | `None` | Whether the agent may emit `CHART` events. `None` keeps the server default (off) |

### `client.research()`

```python
result = client.research(
    message: str,
    research_effort: str = "standard",
    chat_id: str | None = None,
    *,
    model_name: str = "base",
    from_checkpoint_id: str | None = None,
    expected_output: str | None = None,
    structured_output_schema: dict | None = None,
    code_execution: bool | None = None,
    chart_generation: bool | None = None,
    on_event: Callable[[str, dict], None] | None = None,
)
```

| Parameter | Description |
|-----------|-------------|
| `message` | The research question. Natural-language time references such as "last 24 hours" are understood |
| `research_effort` | `"lite"` (~10-20s) or `"standard"` (~20-60s, multi-step) |
| `chat_id` | Continue an existing conversation |
| `model_name` | `"base"` for default routing, `"pro"` for the most capable available model |
| `from_checkpoint_id` | Resume or branch from a previous result's `checkpoint_id` |
| `expected_output` | Guidance for the answer's tone, structure, and format |
| `structured_output_schema` | JSON Schema for extraction; the object arrives on `result.structured_output` |
| `code_execution` | Per-request override for sandboxed Python |
| `chart_generation` | Per-request override for chart emission |
| `on_event` | Called as `on_event(msg_type, message)` for every streamed event, for progress display |

### `client.follow_up()`

```python
result2 = client.follow_up("Which sectors are driving that?", previous_result=result1)
```

Reuses the previous result's `chat_id`. Requires `persistence_mode="enabled"`.

---

## Citations

`GROUNDING` events attribute spans of the answer to their sources using `start`/`end` character offsets into the **cumulative** answer text. The client buffers every `ANSWER` chunk verbatim and only resolves those offsets after the stream completes, which is the rule the [grounding guide](https://docs.bigdata.com/how-to-guides/agents/concepts/grounding-and-citations#the-buffering-rule) sets out. Any normalisation between chunks would misalign every citation.

### Two kinds of reference

| `source` | Meaning | Rendered as |
|---|---|---|
| Populated | A document returned by the **search** tool | Headline, publisher, date, and URL |
| Absent | Every **other** tool (market tearsheet, earnings calendar, code execution) grounds the span at the whole-tool level via `audit_id` | The tool's audit title, e.g. *Market Tearsheet* |

An absent `source` is normal and expected, not a broken reference. Treating it as a document is what produced blank `N/A` citations; the client now resolves it against the matching `AUDIT` trace and emits a `ToolCitation` instead.

Pass `include_tool_citations=False` to any citation method for a document-only reference list.

### Source shapes

A populated `source` is one of two shapes, discriminated on `type`:

| | `BIGDATA` | `EXTERNAL` |
|---|---|---|
| Publisher name | `src_name` | `action.name` |
| URL | `url` | `action.url` |
| Headline | `hd` | `hd` |
| Date | `ts` | `ts` |

Reading only the `BIGDATA` field names is why external web results used to appear without a publisher or link. The client normalises both onto the same `Citation` fields.

### Deduplication

Sources are deduplicated by `id`, falling back to `url`, then `hd`. Headlines are the last resort because unrelated documents can share a title. `audit_id` is never used as a key, since one tool call can return many documents.

### Attribution format

Reference lists use the brand-standard `Source name - MMM DD, YYYY`, linked to the canonical URL when one exists.

### Reserved marker slots

The API reserves a space before the closing punctuation of every citable sentence so a marker can drop in without padding, but it only grounds some of them. `get_answer_with_citations()` closes the unused slots, so an ungrounded sentence reads `...excessive capex.` rather than `...excessive capex .`. Pass `tidy_unfilled_slots=False` to keep the answer byte-identical to what the API sent. `get_answer()` is always verbatim.

---

## Working with results

### `ResearchResult` methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_answer()` | `str` | Plain answer text |
| `get_answer_with_citations()` | `str` | Answer with `[1]`, `[2, 3]` markers |
| `get_markdown_with_citations()` | `str` | Annotated answer plus a deduplicated Sources section |
| `get_numbered_citations()` | `list[dict]` | Citations numbered to match the inline markers |
| `get_citations()` | `list[dict]` | Every document returned by search |
| `get_citations_json()` | `str` | The same, as JSON |
| `to_dict()` / `to_json()` | `dict` / `str` | Full result |
| `to_dict_with_inline_citations()` / `to_json_with_inline_citations()` | `dict` / `str` | Full result with the answer annotated |

### `ResearchResult` attributes

| Attribute | Type | Description |
|----------|------|-------------|
| `answer` | `str` | The answer, concatenated verbatim from `ANSWER` chunks |
| `citations` | `list[Citation]` | Documents returned by the search tool |
| `grounding_refs` | `list[GroundingReference]` | Answer spans attributed to a source or a tool |
| `charts` | `list[Chart]` | Vega-Lite charts anchored to answer spans |
| `structured_output` | `Any` | Extracted JSON, when a schema was requested |
| `chat_id` | `str` | Conversation ID for follow-ups |
| `checkpoint_id` | `str` | Checkpoint for resuming or branching |
| `consumption` | `list[dict]` | Per-tier resource usage from `COMPLETE` |
| `audit_traces` | `dict[str, AuditTrace]` | Tool execution traces keyed by `tool_id` |
| `tool_errors` | `dict[str, int]` | `TOOL_ERROR` counts per tool name |
| `plan_steps` | `list[tuple[str, str]]` | Final research plan as `(description, status)` |
| `processing_time_ms` | `int` | Wall-clock time for the run |

---

## Charts

Set `chart_generation=True` to let the agent run Python over Bigdata's structured data and return a [Vega-Lite](https://vega.github.io/vega-lite/) spec. Charts are off by default server-side; opt in only when your client can render them.

```python
result = client.research(
    "Chart the S&P 500 index level over the last 12 months.",
    chart_generation=True,
)

for chart in result.charts:
    # chart.start / chart.end point at the answer span the chart illustrates,
    # using the same offset model as grounding references.
    print(chart.title, chart.chart_type, chart.data_points)
    render(chart.vega_lite_spec)  # any Vega-Lite renderer
```

### Rendering in a notebook

In JupyterLab, display the spec directly as a MIME bundle — no plotting dependency required:

```python
display({"application/vnd.vegalite.v5+json": chart.vega_lite_spec}, raw=True)
```

**GitHub will not render that.** It strips the JavaScript a Vega-Lite renderer needs, so a JSON-only chart output appears blank when the notebook is browsed on GitHub. Add a rasterised copy to the same bundle and each front-end picks the best type it understands:

```bash
uv add vl-convert-python   # optional; renders Vega-Lite without a browser
```

```python
import base64
import vl_convert

png = vl_convert.vegalite_to_png(chart.vega_lite_spec, scale=1.5)

display(
    {
        # JupyterLab: interactive, with tooltips
        "application/vnd.vegalite.v6+json": chart.vega_lite_spec,
        "application/vnd.vegalite.v5+json": chart.vega_lite_spec,
        # GitHub, nbviewer, VS Code, PDF export: static but always visible
        "image/png": base64.b64encode(png).decode("ascii"),
        "text/plain": f"<Vega-Lite {chart.chart_type} chart: {chart.title}>",
    },
    raw=True,
)
```

`scale=1.5` keeps the embedded image sharp without bloating the committed `.ipynb`. `research_client_usage.ipynb` wraps this in a `render_chart()` helper that degrades gracefully when `vl-convert-python` is not installed.

See [Code execution and charts](https://docs.bigdata.com/how-to-guides/agents/concepts/code-execution-and-charts).

---

## Error handling

Failures arrive in two layers, and they need different treatment. All exceptions derive from `ResearchAgentError`.

### HTTP level

Raised before the stream starts. Consuming an SSE stream from a non-2xx response silently yields zero events, so status is always checked first.

| Status | Exception | Retryable |
|-------:|-----------|-----------|
| `400` / `422` | `InvalidRequestError` | No |
| `401` | `AuthenticationError` | No |
| `403` | `EntitlementError` | No |
| `404` | `ResourceNotFoundError` | No |
| `429` | `RateLimitError` | Yes, with backoff |
| `5xx` | `ServerError` | Yes, with backoff |

### Stream level

Typed messages inside a `200` response.

| Message | Handling |
|---------|----------|
| `LLM_RETRY` | Logged. The agent recovers on its own |
| `TOOL_ERROR` | Counted in `result.tool_errors`, never raised. Check it to detect a degraded answer |
| `ERROR` | Raised as `StreamError`. The stream is over |
| No `COMPLETE` | Raised as `TruncatedStreamError`, so a truncated stream never looks like an empty answer |

A stalled connection raises `StreamTimeoutError` and is retried.

```python
from research_client import ResearchAgentError, RateLimitError, StreamError

try:
    result = client.research("Your question")
except RateLimitError:
    print("Rate limited even after backoff")
except StreamError as exc:
    print(f"Agent could not complete the request: {exc}")
except ResearchAgentError as exc:
    print(f"{type(exc).__name__}: {exc}")

if result.tool_errors:
    print("Some sources could not be retrieved; the answer may be incomplete.")
```

See [Error handling](https://docs.bigdata.com/how-to-guides/agents/concepts/error-handling).

---

## Retries

Transient failures are retried with exponential backoff plus **full jitter** (`delay + random(0, delay)`).

Retried: `ConnectionError`, `Timeout`, `ReadTimeout`, `ChunkedEncodingError`, `StreamTimeoutError`, `TruncatedStreamError`, `RateLimitError` (429), `ServerError` (5xx).

Not retried: `400`, `401`, `403`, `404`, `422`, and in-stream `ERROR` events. These reflect client-side or identity problems that will not resolve on retry.

Each attempt starts from a clean result. Grounding offsets index the answer produced by a single response, so accumulating partial answers across attempts would misalign every citation. When a `chat_id` is already known it is carried into the retry, keeping prior turns in context while the answer is regenerated.

---

## Thread safety

`ResearchClient` is thread-safe. Instance attributes are read-only during a request, so a single client may be shared across threads. Call `setup_logging()` once at application startup, not from request handlers.

---

## Logging

```python
import logging
from research_client import ResearchClient, setup_logging

setup_logging(log_file="research.log", level=logging.INFO, console=True, file_mode="w")

client = ResearchClient()
result = client.research("Your query")
```

Records are flushed as they are written, so the log survives a hard failure mid-stream.

```
2026-07-29 15:04:12 - research_client - INFO - Starting research (effort=standard, model=base, chat_id=new)
2026-07-29 15:04:12 - research_client - INFO - Request attempt 1/4
2026-07-29 15:04:18 - research_client - INFO - ACTION: get_market_tearsheet
2026-07-29 15:04:31 - research_client - INFO - ACTION: search
2026-07-29 15:05:01 - research_client - INFO - Research complete: 23 citations, 11 grounding refs, 0 charts, 49411ms
```

---

## Complete example

```python
import logging

from research_client import ResearchClient, ResearchAgentError, setup_logging

setup_logging(log_file="research.log", console=False)

client = ResearchClient(
    stream_timeout=90.0,
    max_retries=3,
    persistence_mode="enabled",
    chart_generation=True,
)

try:
    result = client.research(
        "How is the S&P 500 (SPX) performing?",
        research_effort="standard",
        on_event=lambda kind, msg: print(f"  {kind}") if kind == "ACTION" else None,
    )
except ResearchAgentError as exc:
    raise SystemExit(f"Research failed: {exc}") from exc

print(f"Done in {result.processing_time_ms / 1000:.1f}s, {len(result.citations)} documents")

with open("report.md", "w") as fh:
    fh.write(result.get_markdown_with_citations())

follow_up = client.follow_up("Which sectors are driving that performance?", result)
print(follow_up.get_answer())
```

---

## Files

| File | Description |
|------|-------------|
| `research_client.py` | The client library |
| `research_client_usage.ipynb` | Interactive walkthrough |
| `README.md` | This documentation |

---

## Support

For API issues, contact your Bigdata.com representative or visit [docs.bigdata.com](https://docs.bigdata.com).
