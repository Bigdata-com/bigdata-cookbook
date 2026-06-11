# Batch Earnings Sentiment

Run a multi-company earnings sentiment workflow with [Bigdata.com](https://bigdata.com) Batch Search.

## APIs

- `POST /v1/knowledge-graph/companies`
- `POST /v1/search/batch`

## Script

- `earnings_sentiment.py`

## Run

```bash
export BIGDATA_API_KEY=your_api_key_here
uv run python earnings_sentiment.py
```
