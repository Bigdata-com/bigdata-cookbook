# Workflow Example Tutorial

End-to-end tutorial that goes from theme definition to rolling sentiment signals.

## What you will learn

- Decompose a theme into sub-themes
- Execute large-scale semantic searches over entities
- Optionally validate relevance/impact with an LLM step
- Build rolling sentiment signals for downstream analysis

## Setup

```bash
cd Workflow_example
uv venv
uv add requests pandas numpy plotly python-dotenv openai
cp .env.example .env
```

Set in `.env`:

```env
BIGDATA_API_KEY=your_api_key_here
# Optional:
# OPENAI_API_KEY=your_openai_api_key
```

## Run

```bash
uv run jupyter notebook Workflow_example.ipynb
```

## Notes

- Uses API key auth only (`BIGDATA_API_KEY`).
- For optional advanced planning patterns, see [`../../Smart_Batching`](../../Smart_Batching/).
