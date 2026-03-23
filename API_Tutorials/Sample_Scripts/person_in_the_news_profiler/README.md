# Person-in-the-News Profiler (Jensen Huang)

Profile companies co-mentioned with a person and fetch evidence chunks via [Bigdata.com](https://bigdata.com).

## APIs

- `POST /v1/search/co-mentions/entities`
- `POST /v1/knowledge-graph/entities/id`
- `POST /v1/search`

## Script

- `jensen_huang_profiler.py`

## Run

```bash
export BIGDATA_API_KEY=your_api_key_here
uv run python jensen_huang_profiler.py
```
