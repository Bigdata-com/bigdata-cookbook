# Migration Notes for Pricing_Power_Analysis

This project has been migrated from `bigdata-client` + `bigdata-research-tools` to REST API + OpenAI.

**Status: complete.** `Pricing Power.ipynb` has been fully rewritten cell-by-cell onto the
patterns below and executed end-to-end against real `BIGDATA_API_KEY` / `OPENAI_API_KEY`
credentials with no errors (`jupyter nbconvert --execute --inplace`). The former
`raise SystemExit(...)` placeholder cell is gone.

Notable choices made during the rewrite, for anyone re-running or extending this notebook:

- **Company universe**: no CSV ships with this project, so the notebook defines a small
  explicit 5-company `DataFrame` (`RP_ENTITY_ID` + `COMPANY_NAME` + `SECTOR`, one company per
  GICS-like sector) inline instead of pulling from `Thematic_Screener_CLI/40_companies.csv`.
  Swap in `load_universe("path/to/companies.csv")` for a larger run.
- **Search queries**: cost-controlled to a single representative query per theme
  (`"pricing power margin expansion"` / `"lack of pricing power margin compression"`) via
  `src/search_helper.run_universe_search`, rather than iterating the full 27-sentence lists
  defined earlier in the notebook (those lists are kept as illustrative reference / for a
  larger-budget production run).
- **Labeling**: `bigdata_research_tools.labeler.screener_labeler.ScreenerLabeler` has no REST
  replacement anywhere in `src/`, so the notebook now defines a minimal inline OpenAI labeler
  (`label_chunks` in the "Label the Results" section) that classifies each chunk with a single
  `gpt-4o-mini` chat completion (JSON response format) and keeps only relevant rows. It is
  capped at `max_labeling_rows` (10) per theme to bound OpenAI cost. Default model is
  `gpt-5.6-luna` with luna-safe sampling (temperature/top_p omitted for luna models).
- **Schema bridge**: a `to_screener_frame()` helper renames the REST/smart-batching output
  columns (`entity_name`, `entity_id`, `timestamp`, `headline`, `chunk_text`, ...) to the
  `Company` / `Sector` / `Date` / `Headline` / `Motivation` / `Quote` schema that the existing
  plotting helpers in `src/tool.py` expect.
- **Excel export**: `bigdata_research_tools.excel.ExcelManager` is gone; the export cell now
  uses a plain `pandas.ExcelWriter(..., engine="openpyxl")` with one sheet per dataframe.
  `openpyxl` was added to `requirements.txt`.
- With only 5 companies (one per sector), `min_companies=3` in the "Companies Lacking of
  Pricing Power" plot call was lowered to `min_companies=1` — no sector can have 3+ companies
  in this small a universe.

## Changes Made

1. **requirements.txt**: Replaced SDK dependencies with `bigdata-smart-batching`, `requests`, `openai`, `python-dotenv`, `openpyxl`
2. **src/tool.py**: Removed SDK tracking code
3. **src/bigdata_rest.py**: Added REST API helper module
4. **src/search_helper.py**: Added `run_universe_search` (plan/execute/deduplicate wrapper over `bigdata-smart-batching`)

## Notebook Changes Applied

The notebook `Pricing Power.ipynb` has been updated with the following import replacements
(now live in the notebook, not just illustrative):

### Old Imports (lines 270-276):
```python
from bigdata_client import Bigdata
from bigdata_client.models.entities import Company
from bigdata_client.models.search import DocumentType

from bigdata_research_tools.labeler.screener_labeler import ScreenerLabeler
from bigdata_research_tools.search.screener_search import search_by_companies
from bigdata_research_tools.excel import ExcelManager
```

### New Imports:
```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.bigdata_rest import BigdataRestClient, load_universe, company_ids_from_universe

load_dotenv()

# Initialize clients
rest_client = BigdataRestClient()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

## Company Universe Setup

### Old Approach (watchlists):
```python
bigdata = Bigdata()
watchlist = bigdata.watchlists.get("my_watchlist")
companies = bigdata.knowledge_graph.get_entities(watchlist.items)
```

### New Approach (CSV or explicit lists):
```python
# Option 1: Load from CSV
universe = load_universe("path/to/companies.csv")  # Must have RP_ENTITY_ID + COMPANY_NAME columns
company_ids = company_ids_from_universe(universe)

# Option 2: Explicit list
company_ids = ["4F2B", "D8442", "12345"]  # Bigdata entity IDs
company_names = {"4F2B": "Apple Inc.", "D8442": "Microsoft Corp.", ...}
```

## Search Workflow

### Old Approach (research-tools):
```python
df = search_by_companies(
    companies=companies,
    sentences=sentences,
    start_date=start_date,
    end_date=end_date,
    document_type=DocumentType.NEWS,
    ...
)
```

### New Approach (REST + manual iteration):
```python
# Simple REST search example
results = []
for company_id in company_ids:
    for sentence in sentences:
        query = {
            "and": [
                {"type": "similarity", "query": sentence},
                {"type": "entities", "ids": [company_id]},
            ],
            "filter": {
                "type": "date_range",
                "start_date": start_date,
                "end_date": end_date,
            },
            "limit": 50,
        }
        search_results = rest_client.search(query)
        # Process results...
```

## Labeling Workflow

### Old Approach (ScreenerLabeler):
```python
labeler = ScreenerLabeler(
    labels=["Strong Pricing Power", "Weak Pricing Power"],
    llm_model_config="openai::gpt-4o-mini",
)
df = labeler.label(df)
```

### New Approach (OpenAI directly):
```python
def label_text(text: str, entity_name: str, labels: list[str]) -> dict:
    """Label a text chunk using OpenAI."""
    prompt = f"""Classify the following text about {entity_name} into one of these categories: {', '.join(labels)}.
    
Text: {text}

Return JSON: {{"label": "category", "motivation": "explanation"}}
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

# Apply to dataframe
df["label_result"] = df.apply(lambda row: label_text(row["text"], row["entity_name"], labels), axis=1)
df["label"] = df["label_result"].apply(lambda x: x["label"])
df["motivation"] = df["label_result"].apply(lambda x: x["motivation"])
```

## Environment Variables

Update your `.env` file:
```
BIGDATA_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Reference Implementation

See `Thematic_Screener_CLI` for a complete working example of:
- CSV universe loading
- Smart batching for large-scale search
- OpenAI labeling with structured outputs
- Plan + execute search patterns
