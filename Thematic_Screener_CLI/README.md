# Thematic Screener CLI

This Thematic Screener CLI is a Beta environment that presents a Command-line tool for running a four-stage thematic screening pipeline: generate theme labels, build search plans, execute document search, and label sentences with company-level summaries.

It leverages [Bigdata](https://bigdata.com/), [bigdata-smart-batching](https://docs.bigdata.com/use-cases/search-service/smart-batching) library and OpenAI.

Each run is isolated in its own directory under `runs/<run_name>/`, so concurrent or repeated runs never overwrite each other.

## Execution overview example

![Screener Example](assets/screener_flowchart.svg)

## Setup

This project uses [uv](https://docs.astral.sh/uv/) as the package manager. If you use another package manager please refer to it's documentation.

From this directory (`Thematic_Screener_CLI/`):

```bash
uv sync
```

Copy `.env.example` to `.env` and set your API keys:

```bash
cp .env.example .env
```

Required environment variables:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Label generation, sentence labeling, company summaries |
| `BIGDATA_API_KEY` | Search planning and document retrieval |

Run commands from this directory so `.env` is found automatically.

## Claude Desktop MCP

This project also exposes a local MCP server for Claude Desktop:

```bash
uv run thematic-screener-mcp
```

For the full Claude Desktop configuration snippet, human-in-the-loop workflow instructions,
and tool contract details, see `mcp/README.md` and `mcp/claude-skill.md`.

For Cursor, use the project skill at `.cursor/skills/thematic-screener/SKILL.md`.

## Quick start

```bash
# Full pipeline with defaults, uses XNAS_companies.csv
uv run python -m src.cli run-all --run-name demo

# Step by step (same run directory)
uv run python -m src.cli generate-labels --run-name demo
uv run python -m src.cli plans --run-name demo
uv run python -m src.cli search --run-name demo
uv run python -m src.cli label-sentences --run-name demo

# Summarize plans if you want to verify the total chunks to be retrieved
uv run python -m src.cli summarize-plans --run-name demo
```

## Pipeline overview

| Step | Subcommand | Description |
|------|------------|-------------|
| 1 | `generate-labels` | Generate a taxonomy of sub-themes from a main theme and analyst focus |
| 2 | `plans` | Build one Bigdata search plan per theme over a company universe |
| 3 | `search` | Execute all plans and store deduplicated documents |
| 4 | `label-sentences` | Label sentences, summarize companies, export final CSV |
| — | `summarize-plans` | Print per-plan chunk counts and total (no API calls) |

Use `run-all` to execute all four steps in sequence within one isolated run.

### Config persistence

Each step merges its settings into `config.json`. Explicit CLI flags always take precedence over `config.json`. When a flag is omitted, the CLI falls back to the saved config, then to built-in defaults.

## Company universe

Plans use `XNAS_companies.csv` by default. The file must contain:

| Column | Description |
|--------|-------------|
| `RP_COMPANY_ID` | Company identifier passed to Bigdata search |
| `COMPANY_NAME` | Used to resolve company names from search results |

For production runs, pass a larger universe CSV with `--universe`. The cookbook repo includes `Batch_Search_API/global_all_caps.csv` (~10k companies) with the same required columns.

## CLI reference

Entry point:

```bash
python -m src.cli <subcommand> [options]
```

### Common options (all subcommands)

| Option | Default | Description |
|--------|---------|-------------|
| `--run-name` | `run_YYYYMMDD_HHMMSS` | Unique name for this run directory |
| `--runs-root` | `runs` | Parent directory that holds all runs |

### `generate-labels` — generate theme labels

| Option | Default | Description |
|--------|---------|-------------|
| `--main-theme` | see defaults below | Main screening theme |
| `--analyst-focus` | see defaults below | Analyst focus guiding the taxonomy |
| `--labels-model` | `gpt-5.4-nano` | OpenAI model for label generation |

### `plans` — build search plans

| Option | Default | Description |
|--------|---------|-------------|
| `--universe` | `XNAS_companies.csv` | Path to the company universe CSV |
| `--start-date` | `2025-06-01` | Search start date (`YYYY-MM-DD`) |
| `--end-date` | `2026-06-09` | Search end date (`YYYY-MM-DD`) |

### `search` — execute search plans

| Option | Default | Description |
|--------|---------|-------------|
| `--chunk-percentage` | `0.02` | Fraction of expected chunks to retrieve per basket |
| `--requests-per-minute` | `350` | Rate limit for search API requests |

### `label-sentences` — label sentences and summarize

Uses `main_theme` and `universe` from `config.json` (set by earlier pipeline steps). Run `generate-labels` and `plans` first, or use `run-all`.

| Option | Default | Description |
|--------|---------|-------------|
| `--labeling-model` | `gpt-5.4-nano` | OpenAI model for sentence labeling |
| `--summary-model` | `gpt-5.4-nano` | OpenAI model for company summaries |

### `summarize-plans` — summarize search plans

Prints per-plan chunk counts and a total to stdout. Does not call any APIs (no `.env` keys required).

| Option | Default | Description |
|--------|---------|-------------|
| `--plans-dir` | `runs/<run_name>/plans` | Path to a plans folder |

### `run-all` — full pipeline

Runs `generate-labels` → `plans` → `search` → `label-sentences` in one isolated run. Accepts the union of all options above.

## Defaults

| Setting | Value |
|---------|-------|
| Main theme | AI disruption in product development |
| Analyst focus | How companies are including AI in their development cycle |
| Start date | `2025-06-01` |
| End date | `2026-06-09` |
| Labels model | `gpt-5.4-nano` |
| Labeling model | `gpt-5.4-nano` |
| Summary model | `gpt-5.4-nano` |
| Chunk percentage | `0.02` (2%) |
| Search requests/min | `350` |
| Document categories | `news_premium`, `transcripts`, `filings` |

## Examples

```bash
# Custom theme and universe, named run
python -m src.cli run-all \
  --run-name medtech_aesthetics \
  --main-theme "AI disruption in product development" \
  --analyst-focus "How companies are including AI in their development cycle" \
  --universe ../Batch_Search_API/global_all_caps.csv

# Generate labels only
python -m src.cli generate-labels \
  --run-name my_run \
  --main-theme "Ophthalmology medical devices" \
  --analyst-focus "Surgical and diagnostic eye care"

# Resume a run: plans only (reads themes.txt, search_queries.txt, and config from the run dir)
python -m src.cli plans --run-name my_run

# Summarize plans for an existing run (no API keys needed)
python -m src.cli summarize-plans --run-name my_run
```

## Notes

- Steps can be run independently as long as prior outputs exist in the run directory.
- The `label-sentences` step can take several minutes depending on sentence count and OpenAI rate limits.
