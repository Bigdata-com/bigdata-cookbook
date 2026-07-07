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
uv run jupyter notebook Podcast_most_discussed_companies.ipynb
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `CoMentions_API_Tutorial.ipynb` | General co-mentions API walkthrough |
| `CoMentions_API_Pepsi.ipynb` | PepsiCo competitive landscape via co-mentions |
| `Podcast_most_discussed_companies.ipynb` | Top companies discussed in podcasts (batched US 500 screen) |

## Endpoints used

- `/v1/search/co-mentions/entities`
- `/v1/knowledge-graph/entities/id`

Docs: https://docs.bigdata.com/api-reference/search/get-co-mentions
