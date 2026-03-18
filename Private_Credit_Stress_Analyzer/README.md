# Private Credit Stress Analyzer

Analyze private credit market stress by running thematic signal searches across three entity layers — **Lenders**, **Borrowers**, and **Banks** — using the [Bigdata API](https://bigdata.com). The system scores each entity on a **Terms Power Score** and a **Stress Score**, then outputs a ranked Excel report with an interactive HTML dashboard.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set your API key
cp .env.example .env
# Edit .env and add your BIGDATA_API_KEY

# 3. Run the full pipeline
uv run python main.py
```

## CLI Options

```bash
uv run python main.py                          # Full run (all entities, all layers)
uv run python main.py --skip-search            # Use cached results, regenerate scores + reports
uv run python main.py --clear-cache            # Clear raw cache, then run full pipeline (fresh search)
uv run python main.py --layer lender           # Run only lender layer
uv run python main.py --entity "Blue Owl Capital"  # Single entity
uv run python main.py --max-workers 3          # Control parallelism
```

## Clear cache

Search results are cached as JSON in `output/raw/` (one file per entity × topic). To force a full re-run with fresh API data:

**Option 1 — CLI (recommended)**  
Clear the cache and run the pipeline in one go:

```bash
uv run python main.py --clear-cache
```

**Option 2 — Manual**  
Remove the raw cache directory (or only `*.json` files), then run the pipeline without `--skip-search`:

```bash
rm -rf output/raw/*.json
# or: rm -rf output/raw
uv run python main.py
```

Use `--clear-cache` when you change topics/entities, update the search date range, or want to pick up query reformulation changes.

## Pipeline Stages

| Stage | Command | Description |
|-------|---------|-------------|
| **Search** | `uv run python src/search.py` | Run (entity × topic) searches via Bigdata, cache as JSON |
| **Score** | `uv run python src/scorer.py` | Aggregate mention counts into Terms Power and Stress scores |
| **Report** | `uv run python src/reporter.py` | Generate Excel workbook + standalone HTML dashboard |

## Output Files

| File | Description |
|------|-------------|
| `output/raw/*.json` | One JSON file per (entity, topic) search result |
| `output/scores.csv` | Entity-level scores with per-topic breakdowns |
| `output/private_credit_stress.xlsx` | Multi-sheet Excel: Lender Ranking, Borrower Distress, Bank Contagion, Raw Signal Matrix, Methodology |
| `output/dashboard.html` | Standalone HTML dashboard with Chart.js visualizations |

## Entity Layers

- **Lenders** (12): Blue Owl Capital, Ares Management, Blackstone Credit, KKR Credit, FS KKR Capital, Apollo Global, BlackRock HPS, Cliffwater, Owl Rock Capital, Prospect Capital, Golub Capital BDC, Blue Owl Technology Income
- **Borrowers** (10): Medallia, Peraton, Zendesk, Informatica, Cotiviti, Dun & Bradstreet, Cloudera, Epicor, Solera, First Brands
- **Banks** (5): JPMorgan Chase, Goldman Sachs, Morgan Stanley, Barclays, Wells Fargo

## Scoring

```
terms_power_score = positive_count / (positive_count + negative_count + 1) × 100
stress_score = 100 − terms_power_score
```

- **Lenders** ranked by `terms_power_score` (high = strong)
- **Borrowers** ranked by `stress_score` (high = distressed)
- **Banks** ranked by net position: `market_share_gain − credit_pullback`

**Why stress can be high with “low” heatmap numbers:** The score is a *ratio* of positive vs negative topic counts, not raw volume. An entity with few positive mentions (e.g. revenue growth, refinancing success) and more negative mentions (e.g. AI disruption, maturity wall) gets a high stress score even if the raw counts are modest. The **Distress radar** and **Signal heatmap** both use the same per-topic counts; the radar shows only the negative (distress) topics.

## Project Structure

```
Private_Credit_Stress_Analyzer/
├── main.py                 # Pipeline entry point
├── pyproject.toml           # Dependencies (uv)
├── .env.example             # API key template
├── CLAUDE.md                # Claude Code instructions
├── config/
│   ├── entities.py          # Lenders, Borrowers, Banks universes
│   └── topics.py            # Search topics with polarity
├── src/
│   ├── search.py            # Parallel Bigdata search runner
│   ├── scorer.py            # Score aggregation
│   ├── reporter.py          # Excel + HTML generation
│   └── utils.py             # Helpers, retry, logging
└── output/
    ├── raw/                 # Cached JSON search results
    ├── scores.csv
    ├── private_credit_stress.xlsx
    └── dashboard.html
```
