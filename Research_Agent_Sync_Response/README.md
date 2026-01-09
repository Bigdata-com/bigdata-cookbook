# Research Agent API Client

A Python wrapper for the Bigdata.com Research Agent API that provides synchronous-style responses with comprehensive citations.

> **Note:** The Research Agent API is constantly evolving. This client wrapper implements few parameters as of January 2026. 

## Features

- **Simple Interface**: Single method call returns complete research results
- **Synchronous Experience**: No need to handle streaming—results returned when ready
- **Bigdata.com Citation Format**: Citations follow the standard Bigdata.com JSON structure
- **Inline Citations**: Get answers with `[1]`, `[2]` markers linked to numbered references
- **Flexible Output**: Access just the answer, just citations, or both

---

## Quick Start

### 1. Set Your API Key

#### Note: Should get from secret manager in Prod env

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

Copy `research_client.py` to your project. No external dependencies beyond the Python standard library.

**Requirements:**
- Python 3.7+
- `BIGDATA_API_KEY` environment variable

---

## API Reference

### `ResearchClient`

```python
client = ResearchClient(api_key=None, base_url=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | API key. If not provided, reads from `BIGDATA_API_KEY` env var |
| `base_url` | `str` | `None` | API endpoint. Defaults to Bigdata.com production URL |

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

Returns:

```json
{
  "answer": "The research answer...",
  "citations": [...],
  "chat_id": "...",
  "processing_time_ms": 12580
}
```

### D. Answer with Inline Citation Numbers

```python
# Get answer with [1], [2], [3] markers
answer_with_refs = result.get_answer_with_citations()

# Get numbered citations that match the markers
numbered_refs = result.get_numbered_citations()
```

**Example output:**

```
NVIDIA faces several key risks [1] [2]:

*   **Intensifying Competition** [3] [4]: The company faces pressure...
*   **Supply Chain Challenges** [5]: NVIDIA has struggled to meet orders...
```

The `numbered_refs` list contains citations with a `number` field:

```json
[
  {
    "number": 1,
    "id": "B7B9DA8A52A784BA285FCCA91F66555F",
    "headline": "Why Nvidia Could Have a Terrible Year",
    "source": {"name": "Yahoo! Finance"},
    "url": "https://...",
    "chunks": [...]
  },
  {
    "number": 2,
    ...
  }
]
```

### Export with Inline Citations

```python
# As dictionary
data = result.to_dict_with_inline_citations()

# As JSON file
with open("result.json", "w") as f:
    f.write(result.to_json_with_inline_citations())
```

---

## Complete Example

```python
import os
import json
from research_client import ResearchClient

# Setup
os.environ["BIGDATA_API_KEY"] = "your-api-key"
client = ResearchClient()

# Execute research
print("Researching...")
result = client.research(
    message="What are the key risks facing NVIDIA?",
    research_effort="lite"
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

## Citation Format Reference

### Source Ranks

| Rank | Description |
|------|-------------|
| `RANK_1` | Premium/Primary sources (e.g., CNBC, Benzinga, Reuters) |
| `RANK_2` | Major sources (e.g., Yahoo Finance, Nasdaq) |
| `RANK_3` | Secondary sources |

### Chunk Fields

| Field | Type | Description |
|-------|------|-------------|
| `cnum` | `int` | Chunk number within the document |
| `text` | `str` | Relevant text excerpt |
| `relevance` | `float` | Relevance score (0.0 - 1.0) |
| `sentiment` | `float` | Sentiment score |

---

## Error Handling

```python
try:
    result = client.research("Your question")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"API error: {e}")
```

Common errors:
- **Missing API key**: Set `BIGDATA_API_KEY` environment variable
- **Invalid API key**: Check your key is correct and active
- **Rate limiting**: Reduce request frequency

---

## Tips

1. **Research Effort Levels** (only two supported):
   - `"lite"`: Quick response, best for simple factual queries (~10-20 seconds)
   - `"standard"`: Deep research with multi-step reasoning, recommended for complex analysis (~20-60 seconds)

2. **Follow-up Questions**: Use `follow_up()` or pass `chat_id` for multi-turn dialogue:
   ```python
   # Option 1: Using follow_up() helper
   result1 = client.research("What are NVIDIA's main products?")
   result2 = client.follow_up("How do they compare to AMD?", result1)
   result3 = client.follow_up("Compare their valuations", result2)
   
   # Option 2: Using chat_id directly
   result1 = client.research("What are NVIDIA's main products?")
   result2 = client.research(
       "How do they compare to AMD?",
       chat_id=result1.chat_id
   )
   ```

3. **Time-Based Queries**: Include time references naturally in your message:
   ```python
   result = client.research("What happened to Tesla stock in the last 24 hours?")
   result = client.research("Summarize Apple's news from this week")
   ```

4. **Citation Deduplication**: The client automatically deduplicates citations from multiple sources.

5. **API Status**: The Research Agent API is currently in **Beta**. Parameters and behavior may evolve.

---

## Files

| File | Description |
|------|-------------|
| `research_client.py` | Main client library |
| `usage.ipynb` | Interactive examples (Jupyter notebook) |
| `README.md` | This documentation |

---

## Support

For API issues or questions, contact your Bigdata.com representative.

