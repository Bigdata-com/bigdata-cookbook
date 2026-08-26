# Migration Notes for Risk_Analyzer

This project has been migrated from `bigdata-client` + `bigdata-research-tools` to REST API + OpenAI.

**Status: implemented.** `Risk_Analyzer.ipynb` has been rewritten end-to-end onto this
approach and executes successfully (`uv run jupyter nbconvert --to notebook --execute
--inplace Risk_Analyzer.ipynb`). The old `RiskAnalyzer` SDK class is replaced by three
small functions in `src/labeling.py`:

- `generate_risk_scenarios(main_theme, focus, n, model, client)` — OpenAI call that
  generates the risk sub-scenario taxonomy (replaces `RiskAnalyzer.create_taxonomy`).
- `classify_risk_chunks(df_chunks, risk_scenarios, main_theme, model, max_rows, client)`
  — OpenAI call (`response_format={"type": "json_object"}`) that labels each search-hit
  row with the best-matching sub-scenario (or `"Not Relevant"`) and a 0-3 exposure score
  (replaces `RiskAnalyzer.label_search_results`). `max_rows` caps classification volume
  for cost control.
- `build_company_risk_matrix(df_labeled, risk_scenarios, industry_by_id, sector_by_id)`
  — aggregates labeled chunks into the company x sub-scenario exposure matrix consumed by
  `src/tool.py::display_figures` (replaces `RiskAnalyzer.generate_results`).

Content retrieval uses `src/search_helper.py::run_universe_search` (smart batching) instead
of `RiskAnalyzer.retrieve_results`. The notebook uses a 5-company slice of
`../Thematic_Screener_CLI/40_companies.csv`, a ~20-day date window, `chunk_percentage=0.02`,
and caps OpenAI classification to 10 rows to keep a full run cheap.

## Changes Made

1. **requirements.txt**: Replaced SDK dependencies with `bigdata-smart-batching`, `requests`, `openai`, `python-dotenv`
2. **src/tool.py**: Removed SDK tracking code
3. **src/bigdata_rest.py**: Added REST API helper module

## Notebook Changes Required

The notebook `Risk_Analyzer.ipynb` needs the following import replacements:

### Old Imports (lines 553-558):
```python
from bigdata_client import Bigdata
from bigdata_client.models.entities import Company
from bigdata_client.models.search import DocumentType

from bigdata_research_tools.workflows.risk_analyzer import RiskAnalyzer
```

### New Imports:
```python
import os
import json
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

## Risk Analyzer Workflow

### Old Approach (RiskAnalyzer):
```python
from bigdata_research_tools.workflows.risk_analyzer import RiskAnalyzer

risk_analyzer = RiskAnalyzer(
    llm_model_config="openai::gpt-4o-mini",
    main_theme="Geopolitical Risk",
    focus="Supply chain disruption",
    companies=companies,
    start_date=start_date,
    end_date=end_date,
    document_type=DocumentType.NEWS,
)

result = risk_analyzer.analyze(
    document_limit=20,
    batch_size=10,
    frequency="3M",
)

df_company = result["df_company"]
```

### New Approach (manual implementation):

```python
# 1. Generate risk sub-scenarios using OpenAI
def generate_risk_scenarios(main_theme: str, focus: str = "") -> list[str]:
    """Generate risk scenario taxonomy."""
    prompt = f"""Generate 6-10 specific risk sub-scenarios for: "{main_theme}"
    
{"Focus: " + focus if focus else ""}

Return ONLY a JSON array of risk scenario descriptions:
["Risk scenario 1", "Risk scenario 2", ...]
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    
    text = response.choices[0].message.content.strip()
    if text.startswith("```json"):
        text = text.split("```json")[1].split("```")[0].strip()
    elif text.startswith("```"):
        text = text.split("```")[1].split("```")[0].strip()
    
    return json.loads(text)

risk_scenarios = generate_risk_scenarios("Geopolitical Risk", "Supply chain")

# 2. Search for each company + scenario combination
def search_risk_exposure(company_id: str, scenario: str, start_date: str, end_date: str) -> int:
    """Search for risk exposure mentions."""
    query = {
        "and": [
            {"type": "similarity", "query": scenario},
            {"type": "entities", "ids": [company_id]},
        ],
        "filter": {
            "type": "date_range",
            "start_date": start_date,
            "end_date": end_date,
        },
        "limit": 50,
    }
    
    results = rest_client.search(query)
    return len(results)  # Simple count-based scoring

# 3. Build risk exposure matrix
risk_data = []
for company_id in company_ids:
    company_name = company_names[company_id]
    row = {"company_id": company_id, "company_name": company_name}
    
    for scenario in risk_scenarios:
        exposure_count = search_risk_exposure(company_id, scenario, "2024-01-01", "2024-12-31")
        row[scenario] = exposure_count
    
    row["composite_score"] = sum(row[s] for s in risk_scenarios)
    risk_data.append(row)

df_company = pd.DataFrame(risk_data).sort_values("composite_score", ascending=False)
```

## Alternative: Smart Batching for Large Universes

For large company universes, use `bigdata-smart-batching`:

```python
from bigdata_smart_batching import plan_search, execute_plan

# Plan searches
plan = plan_search(
    sentences=risk_scenarios,
    entity_ids=company_ids,
    start_date="2024-01-01",
    end_date="2024-12-31",
)

print(f"Plan reduces {len(company_ids) * len(risk_scenarios)} queries to {len(plan)} efficient queries")

# Execute plan
results = execute_plan(plan, rest_client)

# Process results into risk matrix
# ... (see Thematic_Screener_CLI for complete pattern)
```

## Visualization

The visualization functions in `src/tool.py` remain unchanged. Use them with the new `df_company` DataFrame:

```python
from src.tool import display_figures

display_figures(df_company, interactive=True, n_companies=10)
```

## Environment Variables

Update your `.env` file:
```
BIGDATA_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Reference Implementation

See `Thematic_Screener_CLI` for complete examples of:
- CSV universe management
- Theme/scenario generation
- Smart batching for scale
- OpenAI-based labeling
- Search + scoring pipeline
