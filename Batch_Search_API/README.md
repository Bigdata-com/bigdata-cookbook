# Batch Search API

**Why it matters.** Scaling search to a full universe (e.g. Global All-Cap with around 10k names) usually means thousands of HTTP calls, rate limits, retries, and timeouts. The Batch Search API removes that. You submit **one file** with all your queries, the service runs them asynchronously and returns **one** result file. No client-side loops, no QPS management, no thousands of round-trips. You get a ranked view of which companies are most affected by a topic (e.g. a policy or theme), plus sector-country heatmaps as bottom-up macro signals, all grounded in document-level evidence, without building or tuning your own large-scale search pipeline.

**What the notebook does.** It walks through the full workflow:
1. **Setup** dependencies and paths
2. **Load universe** from Global All-Cap (or your own) CSV with `RP_ENTITY_ID`, `COMPANY_NAME`, `COUNTRY`, `SECTOR`
3. **Configure** search topic and time window
4. **Build queries** by packing entities into queries (e.g. 10 per query, giving 1,000 queries for 10k companies), with optional entity control (e.g. Trump `22C3AF`)
5. **Submit one batch job** to create job, upload JSONL, poll, and download one result file
6. **Post-process** to deduplicate chunks, assign to query entities only, aggregate **score** = sum(sentiment × relevance) and **volume** per entity, join to sector/country
7. **Results and visuals** top 5 positive/negative by score, top chunks for the most negative company, and a sector-country heatmap (e.g. G12) with optional drill-down

## Features

- **Single batch job**: Submit one JSONL file with all queries; the service processes them asynchronously and returns one result file
- **No client-side rate limiting**: No QPS management, connection pools, or thousands of round-trips
- **Entity-level post-processing**: Deduplicate chunks, assign to query entities only, aggregate score = sum(sentiment × relevance) per entity
- **Sector–country heatmap**: Optional bottom-up macro view by sector and country

## Quick Start

### Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager (or pip)
- Bigdata API access with Batch Search API enabled

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Navigate to the project directory**:
   ```bash
   cd Batch_Search_API
   ```

3. **Create a virtual environment and install dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   uv pip install jupyterlab
   ```

4. **Set your API key**  
   Create a `.env` file in the project directory:
   ```
   BIGDATA_API_KEY=your-api-key-here
   ```
   Or export:
   ```bash
   export BIGDATA_API_KEY="your-api-key-here"
   ```

5. **Run the notebook**  
   From the `Batch_Search_API` directory:
   ```bash
   jupyter lab
   ```
   Open `Batch_Search_API.ipynb` and run all cells.

### Universe data

- **Demo**: If no CSV is provided, the notebook uses a small inline sample (5 companies) so you can run the full workflow.
- **Production**: Place a CSV at `data/universe.csv` with columns: **RP_ENTITY_ID**, **COMPANY_NAME**, **COUNTRY**, **SECTOR** (e.g. Global All-Cap universe). The notebook loads it in section 2.

## Configuration

| Parameter | Description |
|-----------|-------------|
| `TOPIC` | Search topic applied to every query (e.g. policy or theme) |
| `TIME_START` / `TIME_END` | ISO timestamp range for the search window |
| `ENTITIES_PER_QUERY` | Number of entity IDs per query (e.g. 10 → 1,000 queries for 10k companies) |
| `MAX_CHUNKS_PER_QUERY` | Max chunks returned per query |
| `entity.all_of` | Optional control filter (e.g. Trump entity ID `22C3AF`) |
| `reranker.threshold` | Relevance threshold for precision/recall tradeoff |

## Outputs

- **results/batch_{batch_id}_results.jsonl**: Raw API results (one JSON object per line, one line per query).
- **Post-processed metrics**: Entity-level table with `score`, `volume`, `COMPANY_NAME`, `SECTOR`, `COUNTRY` for ranking and heatmaps.

## Project structure

```
Batch_Search_API/
├── Batch_Search_API.ipynb   # Main notebook
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── batch_api_client.py  # Batch API client and polling
├── data/                     # Optional: universe.csv
└── results/                  # Batch result files (created at run time)
```

## API documentation

- [Bigdata.com API Docs](https://docs.bigdata.com)
- [Batch Search API — One job for thousands of queries](https://docs.bigdata.com/use-cases/search-service/one_job_for_thousands_of_queries)

## Support

For API access or Batch Search API permissions, contact your Bigdata.com representative.
