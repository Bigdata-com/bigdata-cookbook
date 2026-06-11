# Knowledge Graph API Tutorial

Tutorial for entity/company/source resolution workflows on Bigdata.com.

## What you will learn

- Find company IDs from company names
- Resolve entity IDs to entity metadata
- Discover and filter sources
- Reuse entity IDs in Search and Volume filters

## Setup

```bash
cd Knowledge_Graph_API
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
uv run jupyter notebook Knowledge_Graph_API_Tutorial.ipynb
```

## Endpoints used

- `/v1/knowledge-graph/companies`
- `/v1/knowledge-graph/entities/id`
- `/v1/knowledge-graph/sources`

Docs: https://docs.bigdata.com/api-reference/knowledge-graph
