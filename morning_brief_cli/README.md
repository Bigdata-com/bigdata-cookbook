# Morning Brief CLI

Generate a daily institutional morning brief for an equity portfolio of up to 50 companies, organised by five pre-configured research topics. Powered by [Bigdata.com](https://bigdata.com) smart-batching search and OpenAI summarisation.

## Quick start

```bash
cd morning_brief_cli
uv sync
cp .env.example .env   # fill in BIGDATA_API_KEY and OPENAI_API_KEY

# Generate a brief for the MAG7 (last 30 days, both MD + HTML)
uv run python -m src.cli generate

# Specify a custom portfolio and date window
uv run python -m src.cli generate \
  --portfolio my_portfolio.csv \
  --start-date 2026-06-01 \
  --end-date 2026-06-29 \
  --format html
```

Output lands in `runs/<run_name>/briefs/morning_brief_YYYYMMDD.{md,html}`.

## Research topics

Every company in the portfolio is assessed across five standard sections:

| Topic | Search focus |
|---|---|
| **Earnings & Guidance** | Revenue, EPS, margins, forward guidance |
| **Macro & Policy** | Rate exposure, tariffs, inflation, regulation |
| **Analyst & Sentiment** | Upgrades/downgrades, price targets, positioning |
| **M&A & Corporate** | Deals, buybacks, dividends, restructuring |
| **Supply Chain & Ops** | Constraints, logistics, inventory, efficiency |

## Portfolio CSV format

```csv
TICKER,RP_COMPANY_ID,COMPANY_NAME
AAPL,D8442A,Apple Inc.
MSFT,228D42,Microsoft Corp.
```

`TICKER` is optional but enables cleaner output headings. `RP_COMPANY_ID` and `COMPANY_NAME` are required.

A pre-configured `mag7_companies.csv` (Apple, Microsoft, Alphabet, Amazon, Meta, NVIDIA, Tesla) is included and used by default.

## CLI commands

```bash
# Full pipeline (recommended)
uv run python -m src.cli generate [OPTIONS]

# Individual stages (each reads prior outputs from the run directory)
uv run python -m src.cli plan    --run-name my_run [--portfolio ...] [--start-date ...] [--end-date ...]
uv run python -m src.cli search  --run-name my_run [--chunk-percentage 0.05]
uv run python -m src.cli compile --run-name my_run [--format md|html|both] [--model gpt-4.1-nano]

# Preview expected API volume before executing
uv run python -m src.cli show-plans --run-name my_run
```

### Common options

| Flag | Default | Description |
|---|---|---|
| `--portfolio PATH` | `mag7_companies.csv` | Portfolio CSV |
| `--start-date YYYY-MM-DD` | 30 days ago | Search window start |
| `--end-date YYYY-MM-DD` | today | Search window end |
| `--chunk-percentage FLOAT` | `0.05` | Fraction of index chunks to retrieve |
| `--format md\|html\|both` | `both` | Output format |
| `--model MODEL` | `gpt-4.1-nano` | OpenAI model for summaries |
| `--run-name NAME` | UTC timestamp | Isolates outputs under `runs/<NAME>/` |
| `--runs-root DIR` | `runs` | Parent directory for all run directories |

## Output format

### Markdown

```
# Morning Brief — 2026-06-29
_Portfolio: 7 companies | Topics: 5 | Generated: 2026-06-29 09:00 UTC_

---

## AAPL — Apple Inc.
**Earnings & Guidance:** Revenue grew 8% year-on-year... [1]
**Macro & Policy:** Tariff exposure widened in Q2... [2]
...

---

## Sources
[1] Apple Q2 2026 Earnings Call (2026-04-30) — https://...
```

### HTML

Self-contained single file — inline CSS only, no external dependencies:
- Dark header bar with date and portfolio metadata
- Sticky navigation bar with clickable company chips
- Expandable/collapsible company cards
- Inline citation links (`[N]`) that scroll to the sources appendix
- Sources appendix with hyperlinked headlines

## Run directory layout

```
runs/<run_name>/
├── config.json           — persisted settings (merged across stages)
├── portfolio.csv         — snapshot of the portfolio used
├── plans/
│   ├── earnings.json     — smart-batching plan for Earnings & Guidance
│   ├── macro.json
│   ├── analyst.json
│   ├── ma.json
│   └── supply_chain.json
├── results/
│   ├── earnings.json     — deduplicated search results per topic
│   └── ...
└── briefs/
    ├── morning_brief_20260629.md
    └── morning_brief_20260629.html
```

Runs are fully resumable: re-running a later stage reads prior outputs from the same run directory.

## Environment variables

| Variable | Purpose |
|---|---|
| `BIGDATA_API_KEY` | Bigdata.com API access |
| `OPENAI_API_KEY` | LLM summarisation |

## Tests

```bash
uv run pytest              # all tests (no API calls required)
uv run ruff check src tests
```
