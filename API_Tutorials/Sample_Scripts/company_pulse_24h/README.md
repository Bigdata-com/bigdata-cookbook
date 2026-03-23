# Company Pulse 24h

Generate a one-company 24-hour pulse: media volume, sentiment, and top chunks from [Bigdata.com](https://bigdata.com).

## APIs

- `POST /v1/knowledge-graph/companies`
- `POST /v1/search/volume`
- `POST /v1/search`

## Script

- `company_pulse.py`

## Run

```bash
export BIGDATA_API_KEY=your_api_key_here
uv run python company_pulse.py
```
