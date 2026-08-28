# SDK → REST migration patterns

Cookbooks in this repo were migrated off the deprecated **`bigdata-client`** SDK and, in most cases, off **`bigdata-research-tools`**. New work should follow the patterns below.

**Reference implementation:** [`Thematic_Screener_CLI/`](Thematic_Screener_CLI/) — full thematic/risk screening pipeline (REST + [`bigdata-smart-batching`](https://docs.bigdata.com/use-cases/search-service/smart-batching) + OpenAI). See also [Thematic Screeners on docs.bigdata.com](https://docs.bigdata.com/use-cases/research-tools/screeners).

---

## Authentication

| Old | New |
|-----|-----|
| `BIGDATA_USERNAME` + `BIGDATA_PASSWORD` | `BIGDATA_API_KEY` (header `X-API-KEY`) |
| `Bigdata(username, password)` | `BigdataRestClient()` from `src/bigdata_rest.py` |

Copy `.env.example` → `.env` in the project directory:

```env
BIGDATA_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

API key setup: [docs.bigdata.com — authentication](https://docs.bigdata.com/api-reference/authentication).

---

## Company universes (replace watchlists)

The SDK used platform **watchlists**. Migrated cookbooks use **CSV files** with at least:

| Column | Aliases accepted |
|--------|------------------|
| Entity ID | `RP_ENTITY_ID`, `RP_COMPANY_ID` |
| Display name | `COMPANY_NAME`, `NAME`, `COMPANY` |

Example universes ship with [`Thematic_Screener_CLI/`](Thematic_Screener_CLI/) (`40_companies.csv`, `XNAS_companies.csv`, `tsx_top150_rp_entities.csv`, etc.). Project-specific CSVs live under each cookbook’s `data/` folder (e.g. sovereign lists for Rising Bond, crypto lists for Screener for Crypto).

```python
from src.bigdata_rest import load_universe, company_ids_from_universe

universe = load_universe("data/my_universe.csv")
company_ids = company_ids_from_universe(universe)
id_to_name = dict(zip(universe["RP_ENTITY_ID"], universe["COMPANY_NAME"]))
```

**Do not** call `bigdata.watchlists` or `bigdata.knowledge_graph` from the old SDK.

---

## Search

### Small / ad hoc queries — REST

Use `BigdataRestClient.search()` with a `/v1/search` query body. See any migrated project’s `src/bigdata_rest.py` (same module copied across cookbooks).

### Multi-company / multi-query workloads — smart batching

Prefer [`bigdata-smart-batching`](https://pypi.org/project/bigdata-smart-batching/) via `plan_search` → `execute_search` → `deduplicate_documents`. Most thematic cookbooks wrap this in `src/search_helper.py`:

```python
from src.search_helper import run_universe_search

df = run_universe_search(
    company_ids,
    queries=["supply chain reshoring nearshoring"],
    start_date="2025-06-01",
    end_date="2026-06-09",
    scope="all",  # news | filings | transcripts | all
    chunk_percentage=0.05,
    id_to_name=id_to_name,
)
```

For the full plan/execute/label pipeline (taxonomy, Excel export, MCP), use **`Thematic_Screener_CLI`** directly instead of reimplementing.

---

## Labeling & mindmaps

| Old (`bigdata-research-tools`) | New |
|--------------------------------|-----|
| `generate_theme_tree` / mindmap helpers | OpenAI structured outputs, or `Thematic_Screener_CLI` `generate_taxonomy()` |
| `ScreenerLabeler` | OpenAI chat completions with JSON schema (see project `src/` or CLI `label_sentences()`) |
| `ExcelManager` | `pandas.ExcelWriter(..., engine="openpyxl")` |
| SDK tracking / `send_trace` | Removed (no-op or deleted) |

`bigdata_research_tools` may still appear in **migration banners** or comments; it is **not** a supported runtime dependency for new runs.

---

## Typical migrated project layout

```
My_Cookbook/
├── .env.example          # BIGDATA_API_KEY + OPENAI_API_KEY
├── requirements.txt      # bigdata-smart-batching, openai, requests, python-dotenv, …
├── data/                 # universe CSV(s)
├── src/
│   ├── bigdata_rest.py   # REST client + load_universe()
│   └── search_helper.py  # smart-batching wrapper (optional)
└── My_Cookbook.ipynb     # migration banner → this file + Thematic_Screener_CLI
```

Smoke tests in migrated projects fail if **`bigdata_client`** is imported in `src/`; `bigdata_research_tools` is allowed where legacy helpers are still referenced.

---

## Features with no drop-in REST replacement

Some SDK-only capabilities were **removed or stubbed** with explicit skip messages in notebooks:

- Platform watchlists (use CSV universes)
- `bigdata_client.knowledge_graph` topic/network graphs that depended on SDK chunk shapes
- SDK document-type enums — use plain string filters or smart-batching `category`
- SDK usage tracking hooks

If a cell prints a “no REST equivalent” skip message, that is intentional; check the notebook for the supported alternative or a reduced visualization.

---

## Per-project notes

These cookbooks have additional migration detail:

- [`Pricing_Power_Analysis/MIGRATION_NOTES.md`](Pricing_Power_Analysis/MIGRATION_NOTES.md)
- [`Risk_Analyzer/MIGRATION_NOTES.md`](Risk_Analyzer/MIGRATION_NOTES.md)
- [`Screener_for_Crypto/MIGRATION_NOTES.md`](Screener_for_Crypto/MIGRATION_NOTES.md)

---

## Docs site

Use-case pages on [docs.bigdata.com](https://docs.bigdata.com) were updated to REST examples and CSV universes (see merged docs PRs **#429** sdk-to-api scrub, **#432** Rising Bond + Crypto alignment).

---

## Quick checklist for a new migration

1. Replace SDK deps in `requirements.txt` with `bigdata-smart-batching`, `requests`, `openai`, `python-dotenv`.
2. Add `src/bigdata_rest.py` (copy from any migrated cookbook).
3. Swap watchlists for a universe CSV; document columns in README.
4. Point search at REST or `search_helper.run_universe_search`.
5. Replace `ScreenerLabeler` / theme tree with OpenAI or delegate to `Thematic_Screener_CLI`.
6. Update `.env.example` and README to `BIGDATA_API_KEY`.
7. Add a smoke test that forbids `bigdata_client` imports in `src/`.
