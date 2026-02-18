# Volume API Tutorial

Tutorial for analyzing document/chunk coverage over time with Bigdata.com Volume API.

## What you will learn

- Build volume queries
- Interpret `total` and per-day `volume` output
- Visualize time series and sentiment trends
- Use volume results to plan downstream search jobs

## Setup

```bash
cd Volume_API
uv venv
uv pip install -r requirements.txt
cp .env.example .env
```

Set in `.env`:

```env
BIGDATA_API_KEY=your_api_key_here
```

## Run

```bash
uv run jupyter notebook Volume_API_Tutorial.ipynb
```

## Endpoint used

- `/v1/search/volume`

Docs: https://docs.bigdata.com/api-reference/search/get-volume-data
