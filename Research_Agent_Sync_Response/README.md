# Research Agent API - Synchronous Client

A robust Python client for the [Bigdata.com Research Agent API](https://docs.bigdata.com/research-agent) that provides synchronous responses with complete citations, automatic retry handling, and network resilience.

> **Note:** The Research Agent API is constantly evolving. This client wrapper implements key parameters as of January 2026.

## Features

| Feature | Description |
|---------|-------------|
| **Synchronous Interface** | Simple blocking API - no async/await complexity |
| **Automatic Retries** | Exponential backoff for connection errors, timeouts, and server errors |
| **Stream Timeout Detection** | Detects stalled connections and automatically triggers retries |
| **Conversation Continuity** | Resumes interrupted conversations using `chat_id` with the original message |
| **Bigdata.com Citations** | Structured citations with source info, timestamps, and text chunks |
| **Inline Citations** | Answer text with `[1]`, `[2]` markers linked to numbered references |

---

## Quick Start

### 1. Set Your API Key

```python
import os
os.environ["BIGDATA_API_KEY"] = "your-api-key-here"
```

Or export in your shell:

```bash
export BIGDATA_API_KEY="your-api-key-here"
```

### 2. Run a Query

```python
from research_client import ResearchClient

client = ResearchClient()
result = client.research("What are the key risks facing NVIDIA?")

# Get the answer
print(result.get_answer())

# Get citations as JSON
print(result.get_citations_json())
```

---

## Installation

Copy `research_client.py` to your project. Only requires the `requests` library.

```bash
pip install requests
```

**Requirements:**
- Python 3.7+
- `requests` library
- `BIGDATA_API_KEY` environment variable

---

## API Reference

### `ResearchClient`

```python
client = ResearchClient(
    api_key=None,           # Or set BIGDATA_API_KEY env var
    base_url=None,          # Defaults to production URL
    timeout=300,            # Connection timeout (seconds)
    stream_timeout=30.0,    # Max wait for streaming data (seconds)
    max_retries=3,          # Retry attempts for transient failures
    retry_delay=1.0,        # Initial retry delay (seconds)
    retry_backoff=2.0,      # Exponential backoff multiplier
    retry_max_delay=60.0    # Maximum retry delay cap (seconds)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | API key. If not provided, reads from `BIGDATA_API_KEY` env var |
| `base_url` | `str` | `None` | API endpoint. Defaults to Bigdata.com production URL |
| `timeout` | `int` | `300` | Connection timeout in seconds |
| `stream_timeout` | `float` | `30.0` | Max seconds to wait for data during streaming. Triggers retry if exceeded |
| `max_retries` | `int` | `3` | Maximum retry attempts for transient failures |
| `retry_delay` | `float` | `1.0` | Initial delay between retries in seconds |
| `retry_backoff` | `float` | `2.0` | Exponential backoff multiplier |
| `retry_max_delay` | `float` | `60.0` | Maximum delay cap between retries |

### `client.research()`

```python
result = client.research(
    message: str,
    research_effort: str = "standard",
    chat_id: str = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | *required* | Your research question. Supports natural language including time references like "last 24 hours" |
| `research_effort` | `str` | `"standard"` | Research depth (see below) |
| `chat_id` | `str` | `None` | Conversation ID from previous response for follow-up questions |

**Research Effort Levels:**

| Value | Speed | Description |
|-------|-------|-------------|
| `"lite"` | ~10-20s | Quick response. Equivalent to former Chat Service. Best for simple queries |
| `"standard"` | ~20-60s | **Recommended.** Deep research with multi-step reasoning. Best for complex analysis |

**Returns:** `ResearchResult` object

**Raises:** `ValueError` if `research_effort` is not `"lite"` or `"standard"`

### `client.follow_up()`

Convenience method for multi-turn conversations:

```python
result2 = client.follow_up(
    message: str,
    previous_result: ResearchResult,
    research_effort: str = "standard"
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Your follow-up question |
| `previous_result` | `ResearchResult` | Result from previous `research()` or `follow_up()` call |
| `research_effort` | `str` | Research depth: `"lite"` or `"standard"` |

---

## Retry Mechanism

The client includes built-in retry logic with exponential backoff for network resilience.

### Retryable Errors (automatic retry)

| Error Type | Description |
|------------|-------------|
| `ConnectionError` | Network unreachable, DNS failures |
| `Timeout` | Connection timeout |
| `ReadTimeout` | No data received within read timeout |
| `StreamTimeoutError` | No data received within `stream_timeout` |
| `ChunkedEncodingError` | Connection broken during streaming |
| HTTP 408 | Request Timeout |
| HTTP 429 | Too Many Requests (rate limiting) |
| HTTP 500, 502, 503, 504 | Server errors |

### Non-Retryable Errors (raised immediately)

| Error Type | Description |
|------------|-------------|
| HTTP 400 | Bad Request |
| HTTP 401 | Unauthorized (invalid API key) |
| HTTP 403 | Forbidden |
| HTTP 404 | Not Found |
| `ValueError` | Invalid parameters |

### Conversation Continuity

When a network interruption occurs mid-stream:
1. The client captures any partial data and the conversation `chat_id`
2. On retry, it sends the original message with the `chat_id` to resume
3. Partial responses are accumulated across retries for a complete answer

### Custom Retry Configuration

```python
# For unstable networks
client = ResearchClient(
    stream_timeout=60.0,    # Wait longer for data
    max_retries=5,          # More retry attempts
    retry_delay=2.0,        # Start with 2s delay
    retry_backoff=2.0,      # Double delay each retry
    retry_max_delay=120.0   # Cap at 2 minutes
)
```

---

## Logging

Enable logging to monitor retry attempts, connection status, and API responses:

```python
from research_client import ResearchClient, setup_logging
import logging

# Configure logging
setup_logging(
    log_file="research_client.log",  # Log file path
    level=logging.INFO,               # Log level
    console=True,                     # Also print to console
    file_mode="w"                     # "w" to overwrite, "a" to append
)

client = ResearchClient()
result = client.research("Your query")
```

**Sample log output:**

```
2026-01-28 16:01:35 - research_client - INFO - Starting research query (effort=standard, chat_id=new)
2026-01-28 16:01:35 - research_client - INFO - Starting request attempt 1/4
2026-01-28 16:01:41 - research_client - INFO - Received chat_id: 17696340...
2026-01-28 16:01:41 - research_client - INFO - Received message type: THINKING
2026-01-28 16:02:47 - research_client - WARNING - Retryable error on attempt 1/4: ReadTimeout: ...
2026-01-28 16:02:47 - research_client - WARNING - Retry attempt 1/3 after 2.0s delay (resuming chat_id=17696340...)
2026-01-28 16:02:49 - research_client - INFO - Starting request attempt 2/4
2026-01-28 16:03:54 - research_client - INFO - Request succeeded after 2 retry attempt(s)
```

---

## Working with Results

### `ResearchResult` Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_answer()` | `str` | Plain answer text |
| `get_citations()` | `list[dict]` | Citations as list of dictionaries |
| `get_citations_json()` | `str` | Citations as formatted JSON string |
| `to_dict()` | `dict` | Full result (answer + citations) |
| `to_json()` | `str` | Full result as JSON string |
| `get_answer_with_citations()` | `str` | Answer with inline `[1]`, `[2]` markers |
| `get_numbered_citations()` | `list[dict]` | Citations with `number` field matching inline markers |
| `to_dict_with_inline_citations()` | `dict` | Answer with inline citations + numbered references |
| `to_json_with_inline_citations()` | `str` | Same as above, as JSON string |

### Result Properties

| Property | Type | Description |
|----------|------|-------------|
| `answer` | `str` | The research answer |
| `citations` | `list[Citation]` | List of Citation objects |
| `chat_id` | `str` | Conversation ID for follow-ups |
| `processing_time_ms` | `int` | API processing time in milliseconds |

---

## Output Formats

### A. Just the Answer

```python
answer = result.get_answer()
```

Returns plain text/markdown answer without citations.

### B. Just Citations (Bigdata.com Format)

```python
citations = result.get_citations()  # List of dicts
citations_json = result.get_citations_json()  # JSON string
```

Each citation follows the Bigdata.com format:

```json
{
  "id": "B7B9DA8A52A784BA285FCCA91F66555F",
  "headline": "Article Title Here",
  "timestamp": "2026-01-04T20:58:59",
  "source": {
    "id": "E5AA62",
    "name": "Yahoo! Finance",
    "rank": "RANK_2"
  },
  "url": "https://...",
  "chunks": [
    {
      "cnum": 6,
      "text": "Relevant excerpt from the article...",
      "relevance": 0.94,
      "sentiment": 0.82
    }
  ]
}
```

**Note:** Only non-null fields are included in the output.

### C. Answer + Citations Together

```python
full_result = result.to_dict()
full_json = result.to_json()
```

### D. Answer with Inline Citation Numbers

```python
# Get answer with [1], [2], [3] markers
answer_with_refs = result.get_answer_with_citations()

# Get numbered citations that match the markers
numbered_refs = result.get_numbered_citations()
```

---

## Complete Example

```python
import os
import logging
from research_client import ResearchClient, setup_logging

# Setup logging
setup_logging(log_file="research.log", console=True)

# Setup client with custom retry config
client = ResearchClient(
    stream_timeout=60.0,  # 60s stream timeout
    max_retries=3         # 3 retry attempts
)

# Execute research
print("Researching...")
result = client.research(
    message="What are the key risks facing NVIDIA?",
    research_effort="standard"
)

print(f"Done in {result.processing_time_ms}ms")
print(f"Found {len(result.citations)} citations\n")

# Option 1: Plain answer
print("=== ANSWER ===")
print(result.get_answer())

# Option 2: Answer with inline citations
print("\n=== ANSWER WITH CITATIONS ===")
print(result.get_answer_with_citations())

# Option 3: Numbered references
print("\n=== REFERENCES ===")
for ref in result.get_numbered_citations():
    print(f"[{ref['number']}] {ref['headline']}")
    if ref.get('url'):
        print(f"    {ref['url']}")

# Option 4: Save to file
with open("research_output.json", "w") as f:
    f.write(result.to_json_with_inline_citations(indent=2))
print("\n✅ Saved to research_output.json")
```

---

## Error Handling

```python
from requests.exceptions import HTTPError, ConnectionError, Timeout

try:
    result = client.research("Your question")
except ValueError as e:
    print(f"Configuration error: {e}")
except HTTPError as e:
    print(f"API error (non-retryable): {e}")
except (ConnectionError, Timeout) as e:
    print(f"Network error (after all retries): {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

Common errors:
- **Missing API key**: Set `BIGDATA_API_KEY` environment variable
- **Invalid API key**: Check your key is correct and active (HTTP 401)
- **Rate limiting**: Request frequency exceeded (HTTP 429)
- **Network issues**: Check connectivity; client will retry automatically

---

## Tips

1. **Research Effort Levels**:
   - `"lite"`: Quick response (~10-20 seconds), best for simple factual queries
   - `"standard"`: Deep research (~20-60 seconds), recommended for complex analysis

2. **Follow-up Questions**: Use `follow_up()` or pass `chat_id` for multi-turn dialogue:
   ```python
   result1 = client.research("What are NVIDIA's main products?")
   result2 = client.follow_up("How do they compare to AMD?", result1)
   result3 = client.follow_up("Compare their valuations", result2)
   ```

3. **Stream Timeout**: Increase `stream_timeout` for slow/unstable networks:
   ```python
   client = ResearchClient(stream_timeout=60.0)  # 60 seconds
   ```

4. **Disable Stream Timeout**: Set to `None` for no timeout checking:
   ```python
   client = ResearchClient(stream_timeout=None)
   ```

5. **Time-Based Queries**: Include time references naturally:
   ```python
   result = client.research("What happened to Tesla stock in the last 24 hours?")
   ```

6. **Citation Deduplication**: The client automatically deduplicates citations from multiple sources.

---

## Files

| File | Description |
|------|-------------|
| `research_client.py` | Main client library |
| `research_client_usage.ipynb` | Interactive examples (Jupyter notebook) |
| `README.md` | This documentation |

---

## Support

For API issues or questions, contact your Bigdata.com representative or visit [docs.bigdata.com](https://docs.bigdata.com).
