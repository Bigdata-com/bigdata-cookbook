# News Monitor (Edge MRVR)

Entity-scoped **web/public news** pull from RavenPack **Edge**, provider **`MRVR`**.  
Deterministic analytics per company–document row, including relevance, sentiment, novelty, **`rp_document_id`**, and optional **`url`**.

Optional post-processing can use the extracted `rp_document_id` / `url` (URL scrape or Bigdata document fetch) — that step is outside this runner.

---

## Setup

```bash
uv sync
cp .env.example .env
# RAVENPACK_API_KEY  — required for Edge MRVR (scripts/edge_mrvr_stories.py)
# BIGDATA_API_KEY    — required for client-news-monitor (src/client_monitor)
```

Run commands from this directory so `.env` loads automatically.

---

## Quick start

```bash
# Last 15 minutes, first 50 names from us_sml.csv
uv run python scripts/edge_mrvr_stories.py pull \
  --universe us_sml.csv \
  --limit-entities 50 \
  --window-minutes 15 \
  --output-dir runs/edge_pull_smoke

# Skip URL lookups (saves document-URL quota)
uv run python scripts/edge_mrvr_stories.py pull \
  --universe us_sml.csv \
  --limit-entities 50 \
  --window-minutes 15 \
  --skip-urls \
  --output-dir runs/edge_pull_smoke
```

`last15` is an alias for `pull`.

---

## Real-time streaming

Historical `pull` uses the analytics JSON API. Edge also supports a **live NDJSON stream** on a different host:

```text
GET https://feed-edge.ravenpack.com/1.0/json/{dataset_id}?keep_alive=t
```

- Response is **HTTP 200** and stays open; records are JSON objects separated by `\n`.
- ``keep_alive=t`` emits a bare newline after ~30s of silence (reset the connection if silent >60s).
- Streaming uses the **dataset’s baked-in filters only**. Create the dataset with `rp_provider_id=MRVR` and your `rp_entity_id` universe (the sample script does this). Filters passed only on historical query calls are **not** applied to the feed.

### Sample script

```bash
# Stream AAPL + MSFT for 60 seconds → runs/edge_stream_smoke/stream_records.jsonl
uv run python scripts/edge_mrvr_stream.py \
  --tickers AAPL,MSFT \
  --duration-seconds 60 \
  --output-dir runs/edge_stream_smoke

# Larger universe, relevance floor baked into the dataset, stop after 100 records
uv run python scripts/edge_mrvr_stream.py \
  --universe us_sml.csv \
  --limit-entities 50 \
  --min-entity-relevance 90 \
  --max-records 100 \
  --duration-seconds 0 \
  --output-dir runs/edge_stream_us50
```

Equivalent raw curl (after you have a filtered `dataset_id`):

```bash
curl -N -X GET \
  "https://feed-edge.ravenpack.com/1.0/json/${DATASET_ID}?keep_alive=t" \
  -H "api_key: $RAVENPACK_API_KEY" \
  --no-buffer
```

Outputs under `--output-dir`: `dataset_id.txt`, `entity_mapping.csv`, `stream_records.jsonl`, `run_summary.json`.

---

## Universe (CLI)

Pick **one** way to define the company set:

| Flag | Input | Notes |
|------|--------|------|
| `--universe PATH` | CSV with **`RP_ENTITY_ID`** (optional `COMPANY_NAME`, `ticker`) | Preferred for `us_sml.csv` — no ticker mapping calls |
| `--tickers AAPL,MSFT,...` | Comma-separated tickers | Mapped to RP ids via Edge entity-mapping (+ `us_sml.csv` disambiguation when present) |
| `--entity-ids 0157B1,4A6F00,...` | Comma-separated RP ids | Skip mapping entirely |
| `--limit-entities N` | Integer | Cap size after load (`0` = all) |

Examples:

```bash
# Full US small/mid file
uv run python scripts/edge_mrvr_stories.py pull \
  --universe us_sml.csv \
  --window-minutes 15 \
  --skip-urls \
  --output-dir runs/edge_us_sml_15m

# Explicit tickers
uv run python scripts/edge_mrvr_stories.py pull \
  --tickers AAPL,MSFT,NVDA,GOOG \
  --window-minutes 15 \
  --output-dir runs/edge_mega

# Raw RP entity ids
uv run python scripts/edge_mrvr_stories.py pull \
  --entity-ids 0157B1,4A6F00,D6489C \
  --window-minutes 15 \
  --output-dir runs/edge_ids
```

Resolved mappings are written to `{output-dir}/entity_mapping.csv` and **reused** on later runs in the same output directory (avoids remapping rate limits).

---

## Time range (CLI)

All times are **UTC**. Two styles:

### A) Rolling window ending at `--window-end` (default: now)

```bash
# Last 15 minutes ending now
--window-minutes 15

# Last 60 minutes ending at a fixed instant
--window-end 2026-07-29T15:00:00Z --window-minutes 60
```

### B) Explicit interval

```bash
--start 2026-07-28T12:00:00Z --end 2026-07-28T12:15:00Z
```

ISO forms with `Z` or `+00:00` are accepted. `--start` / `--end` must be used together; when set, they override `--window-minutes` / `--window-end`.

Examples:

```bash
# Fixed 15-minute bucket
uv run python scripts/edge_mrvr_stories.py pull \
  --universe us_sml.csv \
  --limit-entities 100 \
  --start 2026-07-29T07:13:00Z \
  --end 2026-07-29T07:28:00Z \
  --skip-urls \
  --output-dir runs/edge_fixed_window

# One hour ending at noon UTC
uv run python scripts/edge_mrvr_stories.py pull \
  --tickers AAPL,AMZN \
  --window-end 2026-07-29T12:00:00Z \
  --window-minutes 60 \
  --output-dir runs/edge_hour
```

`feed` mode always uses rolling buckets of `--interval-minutes` (default 15) ending at “now” for each bucket.

---

## Modes

| Mode | Purpose |
|------|---------|
| `pull` / `last15` | One-shot pull for a universe + time window |
| `feed` | Poll successive buckets (`--interval-minutes`, `--max-buckets`) |

```bash
# Continuous feed: one bucket then exit
uv run python scripts/edge_mrvr_stories.py feed \
  --universe us_sml.csv \
  --limit-entities 50 \
  --interval-minutes 15 \
  --max-buckets 1 \
  --skip-urls \
  --output-dir runs/edge_feed
```

---

## Output columns

`stories_unique.csv`:

| Column | Description |
|--------|-------------|
| `timestamp_utc` | Story timestamp |
| `company_name` | Entity name |
| `title` | Headline |
| `source_name` | Publisher |
| `url` | Article URL (from `get_document_url(rp_document_id)`; empty with `--skip-urls`) |
| `entity_relevance` | How central the company is (≥ 90 ≈ headline / lead) |
| `entity_sentiment` | Sentiment on entity-related text/events |
| `title_similarity_days` | Novelty — days since a similar title (≥ 90 ≈ quarterly-new; &lt; 1 ≈ same-day reprint) |
| `rp_document_id` | Stable document id (dedup + enrichment key) |

Also written: `raw_records.csv`, `entity_mapping.csv`, `dataset_id.txt`, `run_summary.json`.

### Optional filters (post-pull or via `--min-entity-relevance`)

```text
entity_relevance       >= 90
title_similarity_days  >= 90
```

`--min-entity-relevance 90` applies the relevance floor in the Edge query itself.

---

## Post-processing hooks

Every kept row carries:

- **`rp_document_id`** — deterministic key for document-level enrichment  
- **`url`** — optional open/scrape target (when not using `--skip-urls`)

Typical follow-ons: URL scrape, or fetch a full annotated document by id in a downstream system.

---

## CLI reference

| Option | Default | Description |
|--------|---------|-------------|
| `mode` | required | `pull` \| `last15` \| `feed` |
| `--universe` | — | CSV with `RP_ENTITY_ID` |
| `--tickers` | — | Comma-separated tickers |
| `--entity-ids` | — | Comma-separated RP ids |
| `--limit-entities` | `0` | Cap universe size |
| `--start` / `--end` | — | Explicit UTC window |
| `--window-end` | now | End of rolling window |
| `--window-minutes` | `15` | Rolling window length |
| `--interval-minutes` | `15` | Feed bucket length |
| `--max-buckets` | `1` | Feed iterations (`0` = forever) |
| `--min-entity-relevance` | none | Query-time relevance floor |
| `--skip-urls` | off | Skip URL resolution |
| `--output-dir` | `runs/edge_<mode>_<ts>` | Output directory |
| `-v` | off | Verbose logging |

---

## Bigdata.com monitor (`client-news-monitor`)

Also in this repo: a retrieval-only Bigdata.com news monitor over ~3k US names (`us_sml.csv`), with optional topic filters + MAS, or entity-only smart batching (`--skip-mas`). Package: `src/client_monitor/`. Entry point: `uv run client-news-monitor`.

Requires `BIGDATA_API_KEY` in `.env`. Uses [Bigdata.com](https://bigdata.com) smart search and [bigdata-smart-batching](https://docs.bigdata.com/use-cases/search-service/smart-batching). Default document category is **`news`**.

### Quick start

```bash
# Smoke (50 companies, last 15 minutes ending now UTC)
uv run client-news-monitor --limit-entities 50

# Full universe, topic + text search
uv run client-news-monitor \
  --universe us_sml.csv \
  --taxonomy taxonomy.csv \
  --search-mode text+topic \
  --chunk-percentage 0.5 \
  --output-dir runs/client_poc

# Entity-only smart batch (no themes / no MAS)
uv run client-news-monitor \
  --skip-mas \
  --chunk-percentage 0.5 \
  --limit-entities 50 \
  --output-dir runs/skip_mas_50pct

# Fixed window end
uv run client-news-monitor \
  --universe us_sml.csv \
  --window-end 2026-07-29T15:00:00Z \
  --window-minutes 15 \
  --skip-mas \
  --chunk-percentage 0.5 \
  --output-dir runs/skip_mas_fixed
```

### Universe & time

| Flag | Default | Description |
|------|---------|-------------|
| `--universe` | `us_sml.csv` | CSV with `RP_ENTITY_ID`, `COMPANY_NAME` |
| `--limit-entities` | `0` (all) | First N rows of the universe |
| `--window-end` | now (UTC) | ISO end of the monitor window |
| `--window-minutes` | `15` | Window length ending at `--window-end` |

### Search modes & skip-MAS

| Mode / flag | Behavior |
|-------------|----------|
| `--search-mode text` | Document-voice text query; entity in `BODY` |
| `--search-mode topic` | Curated taxonomy topic IDs only |
| `--search-mode text+topic` | Text + topic (default) |
| `--search-mode entity_only` | No taxonomy filter; entity `search_in=ALL` (one `entity_wide` pass) |
| `--compare-modes` | Run `text`, `topic`, and `text+topic` sequentially |
| `--skip-mas` | Skip themes and MAS; `plan_search` + `execute_search` over all companies (including zero-volume densified baskets). `--chunk-percentage` is the retrieval budget |

`--chunk-percentage` is a fraction of the planner’s expected-chunk upper bound (e.g. `0.5` = 50% of that budget), not “half the universe’s news.”

### Monitor topics

When not using `--skip-mas` / `entity_only`, the pipeline loops four topics from `taxonomy.csv` (`TOPIC=business`), defined in `src/client_monitor/topics.py`:

| Topic | Focus |
|-------|--------|
| `earnings` | Earnings, results, analyst ratings |
| `contracts` | Contracts, partnerships, awards |
| `leadership` | Executive / board changes |
| `regulatory` | Regulatory / legal / government |

### CLI reference (`client-news-monitor`)

| Option | Default | Description |
|--------|---------|-------------|
| `--universe` | `us_sml.csv` | Universe CSV |
| `--taxonomy` | `taxonomy.csv` | Bigdata taxonomy CSV |
| `--output-dir` | `runs/client_poc_<timestamp>` | Output directory |
| `--window-end` | now (UTC) | Window end ISO |
| `--window-minutes` | `15` | Window length |
| `--search-mode` | `text+topic` | `text` \| `topic` \| `text+topic` \| `entity_only` |
| `--compare-modes` | off | Run three taxonomy modes |
| `--skip-mas` | off | Entity-only smart batch; no themes/MAS |
| `--category-profile` | `news` | `news` \| `news_premium` |
| `--chunk-percentage` | `0.5` | Fraction of plan expected chunks to retrieve |
| `--limit-entities` | `0` | Cap universe size |
| `--max-chunks-per-basket` | none | Hard cap on `max_chunks` |
| `--seen-headlines-db` | none | SQLite for cross-run headline dedup |
| `--force-baseline-refresh` | off | Refresh 30-day MAS baselines |
| `--requests-per-minute` | `350` | Bigdata search rate limit |
| `-v` | off | Debug logging |

### Outputs (`client-news-monitor`)

```
{output_dir}/
  config.json
  plans/<topic>_<mode>.json
  mas_scores.csv
  mas_baselines.db
  retrieval_chunks.jsonl
  retrieval_digest.json
  alerts_with_stories.csv
  run_summary.json
```

Alerts fire when a company is in the top 1% MAS for a topic or has a primary retrieved chunk. `alerts_with_stories.csv` is the inner join of alerts with primary stories.

### Tests

```bash
uv run python -m pytest tests/test_client_monitor.py -v
uv run ruff check src/client_monitor tests
```

---

## Project layout

```
scripts/edge_mrvr_stories.py   # Edge MRVR runner (pull / feed)
scripts/edge_mrvr_stream.py    # Real-time Edge feed sample (feed-edge NDJSON)
src/client_monitor/            # Bigdata monitor package → client-news-monitor CLI
taxonomy.csv                   # Bigdata topic taxonomy (business rows)
us_sml.csv                     # Default universe (~3k US names, RP_ENTITY_ID)
tests/                         # client_monitor unit tests
runs/                          # Run outputs
.env.example                   # RAVENPACK_API_KEY, BIGDATA_API_KEY
```
