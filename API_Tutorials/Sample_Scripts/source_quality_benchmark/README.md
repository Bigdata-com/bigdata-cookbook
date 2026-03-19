# Source Quality Benchmark

Compare topic coverage across source tiers using [Bigdata.com](https://bigdata.com) source metadata and Search.

## APIs

- `POST /v1/knowledge-graph/sources`
- `POST /v1/search`

## Script

- `fed_source_tier_comparison.py`

## Run

```bash
export BIGDATA_API_KEY=your_api_key_here
uv run python fed_source_tier_comparison.py
```
