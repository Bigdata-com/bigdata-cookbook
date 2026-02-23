# Bigdata.com API Tutorials

A focused tutorial bundle for learning core Bigdata.com APIs with five notebooks.

## Included tutorials

| Tutorial | Notebook | Focus |
|---|---|---|
| `Search_API` | `Search_API_Tutorial.ipynb` | Semantic search, filtering, ranking controls |
| `Volume_API` | `Volume_API_Tutorial.ipynb` | Document/chunk volume time series |
| `Knowledge_Graph_API` | `Knowledge_Graph_API_Tutorial.ipynb` | Company/entity/source resolution |
| `CoMentions_API` | `CoMentions_API_Tutorial.ipynb` | Co-mentioned entities and relationship discovery |
| `Workflow_example` | `Workflow_example.ipynb` | End-to-end thematic workflow and rolling signals |

## Authentication

All tutorials use `BIGDATA_API_KEY` from `.env`.

```bash
cp .env.example .env
# edit .env
BIGDATA_API_KEY=your_api_key_here
```

Optional variables:

- `BIGDATA_API_BASE_URL` (defaults to `https://api.bigdata.com`)
- `OPENAI_API_KEY` (only for optional LLM validation in `Workflow_example`)

## Suggested learning path

1. `Search_API`
2. `Volume_API`
3. `Knowledge_Graph_API`
4. `CoMentions_API`
5. `Workflow_example`

Optional advanced follow-up: see [`../Smart_Batching`](../Smart_Batching/).

## Quick start

```bash
cd API_Tutorials/Search_API
uv venv
uv pip install -r requirements.txt
uv run jupyter notebook Search_API_Tutorial.ipynb
```

Repeat in each tutorial folder with its notebook name.
