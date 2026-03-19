# Entity Co-mention Map (Apple)

Map top entities co-mentioned with Apple using [Bigdata.com](https://bigdata.com), then fetch supporting chunks per co-mentioned entity.

## APIs

- `POST /v1/knowledge-graph/companies`
- `POST /v1/search/co-mentions/entities`
- `POST /v1/knowledge-graph/entities/id`
- `POST /v1/search`

## Script

- `apple_comentions.py`

## Run

```bash
export BIGDATA_API_KEY=your_api_key_here
uv run python apple_comentions.py
```
