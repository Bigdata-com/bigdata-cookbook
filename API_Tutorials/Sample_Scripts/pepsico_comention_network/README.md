# PepsiCo Co-mention Network

Build a filtered competitive co-mention network around PepsiCo using [Bigdata.com](https://bigdata.com).

## APIs

- `POST /v1/knowledge-graph/companies`
- `POST /v1/search/co-mentions/entities`
- `POST /v1/knowledge-graph/entities/id`

## Script

- `pepsi_comention_network.py`

## Run

```bash
export BIGDATA_API_KEY=your_api_key_here
uv run python pepsi_comention_network.py
```
