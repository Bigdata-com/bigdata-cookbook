# Iran Spike Analysis

Detect unusual Iran coverage spikes and explain them with co-mentioned entities using [Bigdata.com](https://bigdata.com).

## APIs

- `POST /v1/search/volume`
- `POST /v1/search/co-mentions/entities`
- `POST /v1/knowledge-graph/entities/id`

## Scripts

- `iran_volume_spikes.py` (canonical flow)
- `iran_spike_comentions.py` (alternate implementation)

## Run

```bash
export BIGDATA_API_KEY=your_api_key_here
uv run python iran_volume_spikes.py
```
