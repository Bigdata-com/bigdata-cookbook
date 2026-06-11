# Bigdata.com API Examples

A focused set of client-facing API examples on [Bigdata.com](https://bigdata.com), including notebook walkthroughs and script-based use cases.
A focused tutorial bundle for learning core Bigdata.com APIs with six notebooks.

## Included notebooks

| Example | Notebook | Focus |
|---|---|---|
| `Search_API` | `Search_API_Tutorial.ipynb` | Semantic search, filtering, ranking controls |
| `Volume_API` | `Volume_API_Tutorial.ipynb` | Document/chunk volume time series |
| `Knowledge_Graph_API` | `Knowledge_Graph_API_Tutorial.ipynb` | Company/entity/source resolution |
| `CoMentions_API` | `CoMentions_API_Tutorial.ipynb` | Co-mentioned entities and relationship discovery |
| `Document_Download_API` | `document_download.ipynb` | Download full document content (JSON; handles small and large files) |
| `Workflow_example` | `Workflow_example.ipynb` | End-to-end thematic workflow and rolling signals |

## Sample scripts collection

A collection of CLI-style Python scripts demonstrating simple use-cases
- Folder: [`Sample_Scripts`](./Sample_Scripts/)
- One use-case per folder, each with its own README and runnable script.

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

## Suggested exploration path

1. `Search_API`
2. `Volume_API`
3. `Knowledge_Graph_API`
4. `CoMentions_API`
5. `Sample_Scripts` or `Workflow_example`
5. `Document_Download_API` — fetch full document content when you have a document ID
6. `Sample_Scripts` or `Workflow_example`

Optional advanced follow-up: see [`../Smart_Batching`](../Smart_Batching/).

## Quick start (notebooks)

```bash
cd API_Tutorials/Search_API
uv venv
uv pip install -r requirements.txt
uv run jupyter notebook Search_API_Tutorial.ipynb
```

Repeat in each API folder with its notebook name. For script-based examples, use `Sample_Scripts`.
