# Co-mentions API Tutorial

Tutorial for identifying entities frequently mentioned with a query.

## What you will learn

- Query co-mentioned entities by topic and date range
- Analyze co-mentions by entity category
- Resolve entity IDs with Knowledge Graph API
- Build relationship views from co-mentions results

## Setup

```bash
cd CoMentions_API
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
uv run jupyter notebook CoMentions_API_Tutorial.ipynb
```

## Endpoints used

- `/v1/search/co-mentions/entities`
- `/v1/knowledge-graph/entities/id`

Docs: https://docs.bigdata.com/api-reference/search/get-co-mentions
