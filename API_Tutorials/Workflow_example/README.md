# Large-Scale Thematic Research Workflow (API Tutorial)

This tutorial shows how to build **auditable signals** from unstructured content across a broad equity universe.

It supports two signal families:

1. **Basic signal** (fast, high recall): direct aggregation of retrieval outputs  
   `score = sum(relevance * sentiment)`, plus `volume`.
2. **Advanced signal** (slower, higher accuracy/precision): LLM-verified relevance/impact before rolling aggregation.

This is an **accuracy/recall trade-off** workflow: use basic signals for broad screening, advanced signals for higher-confidence filtering.

## What this tutorial covers

- Theme decomposition into sub-themes
- Search planning and execution over a large universe
- Entity resolution and deduplication to an investable universe
- Basic daily signal construction
- Optional LLM validation (`is_theme_related`, `impact`)
- Rolling impact signal construction for downstream ranking/backtesting

## Timing integrity (important)

Downstream rolling signals should use `date_nyse_1530` when you want close-to-close-compatible timing.

## Setup (uv)

```bash
cd Workflow_example
uv venv
uv sync
cp .env.example .env
```

If you are not using `pyproject.toml`/lock for this folder, install minimal deps with:

```bash
uv add requests pandas numpy plotly python-dotenv openai jupyter
```

Set credentials in `.env`:

```env
BIGDATA_API_KEY=your_api_key_here
# Optional for LLM validation sections:
# OPENAI_API_KEY=your_openai_api_key
```

## Run

```bash
uv run jupyter notebook Workflow_example.ipynb
```

## Output artifacts (typical)

- `df_basic_signals_daily.csv` (basic daily entity signal)
- `df_rolling_signal.csv` (advanced rolling signal, if LLM section is executed)

## Notes

- Authentication is API-key based (`BIGDATA_API_KEY`) loaded from `.env`.
- LLM validation is optional but recommended when precision matters more than recall.
