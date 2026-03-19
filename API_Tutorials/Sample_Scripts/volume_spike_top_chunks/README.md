# Volume Spike and Top Chunks (Repsol)

Detect the highest-volume day for a query and retrieve the top explanatory chunks from [Bigdata.com](https://bigdata.com).

## APIs

- `POST /v1/knowledge-graph/companies`
- `POST /v1/search/volume`
- `POST /v1/search`

## Script

- `repsol_volume_spike.py`

## Run

```bash
export BIGDATA_API_KEY=your_api_key_here
uv run python repsol_volume_spike.py
```
