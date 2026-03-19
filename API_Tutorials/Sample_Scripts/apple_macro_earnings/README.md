# Apple Macro Earnings Extraction

Extract macroeconomic commentary from Apple earnings call content via [Bigdata.com](https://bigdata.com) Search.

## APIs

- `POST /v1/knowledge-graph/companies`
- `POST /v1/search`

## Script

- `apple_macro_earnings.py`

## Run

```bash
export BIGDATA_API_KEY=your_api_key_here
uv run python apple_macro_earnings.py
```
