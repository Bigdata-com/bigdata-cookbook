# Search API Tutorial

Hands-on tutorial for semantic search on Bigdata.com.

## What you will learn

- Build semantic search queries
- Apply timestamp/entity/source filters
- Tune `max_chunks`, `source_boost`, and `freshness_boost`
- Compare retrieval quality across ranking settings

## Setup

```bash
cd Search_API
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
uv run jupyter notebook Search_API_Tutorial.ipynb
```

## Endpoints used

- `/v1/search`
- `/v1/search/volume`

Docs: https://docs.bigdata.com/api-reference/search/search-documents
