# Search API Tutorial

Hands-on tutorials for semantic search on Bigdata.com.

## What you will learn

- Build semantic search queries
- Apply timestamp/entity/source filters
- Tune `max_chunks`, `source_boost`, and `freshness_boost`
- Compare retrieval quality across ranking settings
- Post-process search chunks to highlight query-relevant phrases (lexical, LLM, hybrid)

## Notebooks

| Notebook | Focus |
|---|---|
| `Search_API_Tutorial.ipynb` | Semantic search, filters, ranking, volume, reranker |
| `Search_API_Phrase_Highlighting.ipynb` | Client demo: highlight query-relevant phrases in search results |

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
# Optional — only for LLM/hybrid highlighting
# OPENAI_API_KEY=your_openai_api_key
```

## Run

```bash
uv run jupyter notebook Search_API_Tutorial.ipynb
# or
uv run jupyter notebook Search_API_Phrase_Highlighting.ipynb
```

## Endpoints used

- `/v1/search`
- `/v1/search/volume`

Docs: https://docs.bigdata.com/api-reference/search/search-documents
