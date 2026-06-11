# Daily Briefing Generator

Generate a watchlist morning briefing from the last 24h of coverage using [Bigdata.com](https://bigdata.com).

## APIs

- `POST /v1/knowledge-graph/companies`
- `POST /v1/search/batch`

## Script

- `morning_briefing.py`

## Run

```bash
export BIGDATA_API_KEY=your_api_key_here
uv run python morning_briefing.py
```
