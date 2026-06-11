# Bigdata.com sample scripts

Runnable examples for [Bigdata.com](https://bigdata.com) APIs (Search, Volume, Co-mentions, Knowledge Graph, Batch Search). Each folder is one use case: a primary script, local artifacts (plots, outputs), and a short folder-level README.

These scripts were built with **Claude Code** and the [**Build with Bigdata.com — Claude Plugin**](https://github.com/Bigdata-com/claude-plugin-build-with-bigdata-com).

## Quickstart

```bash
cd API_Tutorials/Sample_Scripts/<use_case_folder>
uv venv
uv pip install -r requirements.txt  # if present
export BIGDATA_API_KEY=your_api_key_here
uv run python <script_name>.py
```

## Catalog

Quick lookup for every folder. The numbered **workflow patterns** below go deeper on the first ten rows (core API combinations); the other rows follow the same layout—open the folder README for specifics.

| # | Use case | Folder | Primary script | APIs |
|---|----------|--------|----------------|------|
| 1 | Entity co-mention map | `entity_comention_map` | `apple_comentions.py` | Knowledge Graph, Co-mentions, Search |
| 2 | Volume spike + top chunks | `volume_spike_top_chunks` | `repsol_volume_spike.py` | Knowledge Graph, Volume, Search |
| 3 | Earnings sentiment (batch) | `earnings_sentiment_batch` | `earnings_sentiment.py` | Knowledge Graph, Batch Search |
| 4 | Competitor narrative comparison | `competitor_narrative_compare` | `nvidia_vs_amd.py` | Knowledge Graph, Search |
| 5 | Macro theme radar | `macro_theme_radar` | `macro_theme_radar.py` | Volume |
| 6 | Person-in-the-news profiler | `person_in_the_news_profiler` | `jensen_huang_profiler.py` | Knowledge Graph, Co-mentions, Search |
| 7 | Daily briefing generator | `daily_briefing_generator` | `morning_briefing.py` | Knowledge Graph, Batch Search |
| 8 | Source quality benchmark | `source_quality_benchmark` | `fed_source_tier_comparison.py` | Knowledge Graph (sources), Search |
| 9 | Geopolitical risk heatmap | `geopolitical_risk_heatmap` | `geopolitical_heatmap.py` | Volume |
| 10 | Thematic ETF universe builder | `thematic_universe_builder` | `quantum_computing_comentions.py` | Co-mentions, Knowledge Graph |
| — | Career network around a person | `career_network` | `people_network_analysis.py` | Co-mentions, Knowledge Graph |
| — | PepsiCo co-mention network | `pepsico_comention_network` | `pepsi_comention_network.py` | Co-mentions, Knowledge Graph |
| — | Iran spike analysis | `iran_spike_analysis` | `iran_volume_spikes.py` | Volume, Co-mentions, Knowledge Graph |
| — | Company pulse (24h) | `company_pulse_24h` | `company_pulse.py` | Volume, Search, Knowledge Graph |

### 1. Entity co-mention map

**What it does:** For a focal company (e.g. Apple), fetch the top N entities co-mentioned today across all categories (companies, people, places, products). For each co-mentioned entity, retrieve the top 2 most relevant chunks.

**APIs used:** Knowledge Graph → Co-mentions → Search

**Example output:**

```
#01  United States [places]   chunks=89
     Chunk 1 — Reuters | 2026-03-11 | relevance=0.94
     Apple expands manufacturing footprint across US states...
```

**File:** `apple_comentions.py`

**Example prompt:**

```
create a script to print top 10 co-mentions of Apple today and use search to bring top 2 chunks for each co-mention
```

### 2. Volume spike + top chunks extraction

**What it does:** Plot daily document volume for a query (e.g. "Repsol oil crisis") over the last month. Identify the day with the highest volume, then extract the top 10 most relevant chunks from that day.

**APIs used:** Knowledge Graph → Volume → Search

**Steps:**

1. Resolve "Repsol" to entity ID via Knowledge Graph.
2. Call Volume with a 30-day window, grouped by day.
3. Find the day with `max(chunk_count)`.
4. Call Search scoped to that single day with `max_chunks=10`.
5. Plot the volume time series (matplotlib) and print the top chunks below the chart.

**Use case:** Detect news spikes around a company or macro theme and immediately surface the most important content driving that spike.

**Example prompt:**

```
create a script that plots the daily volume of documents mentioning Repsol and oil crisis over the last month, then finds the day with the highest volume and prints the top 10 chunks from that day
```

### 3. Earnings surprise sentiment tracker

**What it does:** For a basket of companies reporting earnings this week, search for post-earnings news and score the sentiment. Rank companies from most positive to most negative coverage.

**APIs used:** Knowledge Graph → Batch Search

**Steps:**

1. Resolve each company ticker to entity ID.
2. Submit a Batch Search job with one query per company, filtered to the last 7 days.
3. For each result, average the `relevance`-weighted chunk sentiments.
4. Print a ranked table: `Company | Avg Sentiment | Top Headline`.

**Use case:** Quickly assess post-earnings tone for a portfolio without reading every article.

**Example prompt:**

```
create a script that takes a list of tickers [AAPL, MSFT, GOOGL, AMZN, META], fetches the top 5 post-earnings news chunks for each from the last 7 days using Batch Search, averages the sentiment scores, and prints a table ranking them from most positive to most negative coverage
```

### 4. Competitor narrative comparison

**What it does:** Given two competing companies (e.g. Nvidia vs AMD), run parallel searches for the same theme (e.g. "AI chip demand") and compare how each is covered — chunk count, average relevance, and top headlines.

**APIs used:** Knowledge Graph → Search (two parallel calls)

**Steps:**

1. Resolve both companies to entity IDs.
2. Run two Search calls — same `text`, different `entity.any_of` filter.
3. Print side-by-side: chunk count, average relevance, top 3 headlines per company.

**Use case:** Competitive intelligence — who is getting more favorable AI coverage?

**Example prompt:**

```
create a script that searches for "AI chip demand" news over the last 30 days, runs two parallel searches — one filtered to Nvidia and one to AMD — and prints a side-by-side comparison of chunk count, average relevance, and top 3 headlines for each
```

### 5. Macro theme radar

**What it does:** Track a list of macro themes (e.g. "inflation", "recession", "rate hike", "China trade") and plot their relative document volume over the last 90 days on a single chart.

**APIs used:** Volume (one call per theme)

**Steps:**

1. For each theme string, call Volume with a 90-day window grouped by week.
2. Normalize each series to its own max (0–1 scale).
3. Plot all themes on one chart to visualize which narratives are rising or fading.

**Use case:** Macro research desks monitoring which themes are gaining traction in financial media.

**Example prompt:**

```
create a script that tracks the weekly document volume for the themes "inflation", "recession", "rate hike", and "China trade" over the last 90 days, normalizes each series to 0–1, and plots them all on a single line chart
```

### 6. Person-in-the-news profiler

**What it does:** Given an executive name (e.g. "Jensen Huang"), find all companies co-mentioned with them this week, then pull the top 3 chunks per company to understand what deals, announcements, or controversies are linking them.

**APIs used:** Knowledge Graph → Co-mentions → Search

**Steps:**

1. Search Knowledge Graph for the person entity ID.
2. Call Co-mentions filtered to that entity, `category=companies`, last 7 days.
3. For each co-mentioned company, call Search with `entity.all_of: [person_id, company_id]` and return top 3 chunks.

**Use case:** Investor relations monitoring, executive risk tracking.

**Example prompt:**

```
create a script that looks up Jensen Huang in the Knowledge Graph, finds the top 10 companies co-mentioned with him in the last 7 days, and for each company fetches the top 3 chunks where both Jensen Huang and that company appear together
```

### 7. Daily briefing generator

**What it does:** For a watchlist of tickers, fetch the top 3 chunks published since yesterday for each and render a clean markdown briefing file.

**APIs used:** Knowledge Graph → Batch Search

**Steps:**

1. Resolve all tickers to entity IDs.
2. Submit Batch Search with one query per company, filtered to the last 24 hours, `max_chunks=3`.
3. Write a markdown file: one section per company, with source name, timestamp, and chunk text.

**Use case:** Automated morning briefing for portfolio managers or analysts.

**Example prompt:**

```
create a script that takes the tickers [AAPL, TSLA, NVDA, JPM, AMZN], resolves them to entity IDs, uses Batch Search to fetch the top 3 news chunks for each published in the last 24 hours, and writes the results to a markdown file formatted as a morning briefing
```

### 8. Source quality benchmark

**What it does:** For a query (e.g. "Federal Reserve interest rates"), compare how RANK_1 sources vs RANK_3 sources cover the same topic. Show chunk count, average relevance, and top headlines per tier.

**APIs used:** Knowledge Graph (sources) → Search (two calls with different source filters)

**Steps:**

1. Call `/v1/knowledge-graph/sources` to retrieve source IDs for RANK_1 and RANK_3.
2. Run two Search calls — same query text, different `filters.source`.
3. Print per tier: chunk count, average relevance, and top 5 headlines.

**Use case:** Evaluating signal quality across source tiers before building a production data pipeline.

**Example prompt:**

```
create a script that searches for "Federal Reserve interest rates" in the last 30 days, runs two separate searches — one restricted to RANK_1 sources and one to RANK_3 sources — and compares chunk count, average relevance score, and top 5 headlines for each tier
```

### 9. Geopolitical risk heatmap

**What it does:** For a list of countries (e.g. G7), measure weekly document volume for "geopolitical risk [country]" over the last 90 days. Build a heatmap (country × week) showing where risk narratives are concentrating.

**APIs used:** Volume (one call per country)

**Steps:**

1. For each country, call Volume with `text="geopolitical risk <country>"`, 90-day window, weekly granularity.
2. Build a matrix (country × week) from the results.
3. Render as a heatmap with seaborn, save as PNG.

**Use case:** Macro risk teams monitoring geopolitical exposure across regions.

**Example prompt:**

```
create a script that calls the Volume API for each G7 country with the query "geopolitical risk [country]" over the last 90 days grouped by week, builds a country × week matrix, and renders it as a seaborn heatmap saved to geopolitical_heatmap.png
```

### 10. Thematic ETF universe builder

**What it does:** Given a theme (e.g. "quantum computing"), use Co-mentions to discover which companies are most discussed in that context over the last 30 days. Resolve names and sectors. Output a ranked candidate universe.

**APIs used:** Co-mentions → Knowledge Graph (entities/id)

**Steps:**

1. Call Co-mentions with `text="quantum computing"`, last 30 days.
2. Extract only `category=companies` from the results.
3. Batch-resolve entity IDs to names, sectors, and countries.
4. Sort by `total_chunks_count` and print: `Rank | Company | Sector | Country | Chunk Count`.

**Use case:** Quantitative research teams building thematic factor models or screening for ETF inclusion.

**Example prompt:**

```
create a script that calls the Co-mentions API with the query "quantum computing" over the last 30 days, extracts the top 20 co-mentioned companies, resolves their names and sectors via the Knowledge Graph, and prints a ranked table with columns: rank, company name, sector, country, chunk count
```
