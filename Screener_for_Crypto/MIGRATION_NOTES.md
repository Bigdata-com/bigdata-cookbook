# Migration Notes for Screener_for_Crypto

This project has been migrated from `bigdata-client` + `bigdata-research-tools` to REST API + OpenAI.

## Changes Made

1. **requirements.txt**: Replaced SDK dependencies with `bigdata-smart-batching`, `requests`, `openai`, `python-dotenv`
2. **src/search_entities.py**: Rewritten to use REST API directly (no research-tools dependencies)
3. **src/bigdata_rest.py**: Added REST API helper module

### Fix: `/v1/search` request schema

`search_by_entities` originally built a legacy-shaped payload (`{"and": [...], "filter": {...}, "limit": ...}`)
which the real `/v1/search` endpoint rejects with `400 Bad Request` ("You must limit your query by
adding a query text or any of entity, keyword, ... filters"). The correct REST schema is:

```python
{
    "search_mode": "fast",
    "query": {
        "text": sentence,
        "filters": {
            "entity": {"any_of": [entity_id]},
            "timestamp": {"start": "2025-01-01T00:00:00Z", "end": "2025-01-31T23:59:59Z"},
        },
        "max_chunks": document_limit,
    },
}
```

`src/search_entities.py` has been updated to build this shape; verified against the live API
with real entity IDs (see below) returning non-empty results.

## Notebook Changes Required

The notebook `Screener_for_Crypto.ipynb` needs the following import replacements:

### Old Imports (lines 195-201):
```python
from bigdata_client import Bigdata
from bigdata_client.models.search import DocumentType
from bigdata_research_tools.themes import (
    ThemeTree, generate_theme_tree
)
from bigdata_research_tools.labeler.screener_labeler import ScreenerLabeler
from bigdata_research_tools.workflows.utils import get_scored_df
```

### New Imports:
```python
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from src.bigdata_rest import BigdataRestClient
from src.search_entities import search_by_entities, post_process_dataframe

load_dotenv()

# Initialize clients
rest_client = BigdataRestClient()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

## Entity Universe Setup

### Old Approach (watchlists):
```python
bigdata = Bigdata()
watchlist = bigdata.watchlists.get("Crypto Top 100")
entities = bigdata.knowledge_graph.get_entities(watchlist.items)
```

### New Approach (explicit lists):

Bigdata's knowledge graph covers public companies/entities, not tokens directly, so
resolve crypto-*exposed* public companies via `BigdataRestClient().find_companies(name)`
and hardcode the verified IDs. Example, verified working IDs (as of this migration):

```python
from src.bigdata_rest import BigdataRestClient

client = BigdataRestClient()
client.find_companies("Coinbase")        # -> D69946  Coinbase Global Inc. (COIN)
client.find_companies("Strategy")        # -> C72B8F  Strategy Inc., fka MicroStrategy (MSTR)
client.find_companies("Riot Platforms")  # -> 56AAC4  Riot Platforms Inc. (RIOT)

crypto_ids = ["D69946", "C72B8F", "56AAC4"]
crypto_names = {
    "D69946": "Coinbase Global Inc.",
    "C72B8F": "Strategy Inc. (MicroStrategy)",
    "56AAC4": "Riot Platforms Inc.",
}
```

## Theme Generation

### Old Approach (ThemeTree):
```python
from bigdata_research_tools.themes import generate_theme_tree

theme_tree = generate_theme_tree(
    main_theme="Crypto Cross-Chain Interoperability",
    focus="",
    llm_model_config="openai::gpt-4o-mini",
)
sentences = theme_tree.to_sentences()
```

### New Approach (OpenAI directly):
```python
def generate_themes(main_theme: str, focus: str = "") -> list[str]:
    """Generate theme taxonomy using OpenAI."""
    prompt = f"""Generate 5-8 specific sub-themes for analyzing: "{main_theme}"
    
{"Focus: " + focus if focus else ""}

Return ONLY a JSON array of theme sentences, e.g.:
["Sub-theme 1", "Sub-theme 2", "Sub-theme 3"]
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    
    text = response.choices[0].message.content.strip()
    # Parse JSON (handle code blocks)
    if text.startswith("```json"):
        text = text.split("```json")[1].split("```")[0].strip()
    elif text.startswith("```"):
        text = text.split("```")[1].split("```")[0].strip()
    
    return json.loads(text)

sentences = generate_themes("Crypto Cross-Chain Interoperability", focus="")
```

## Search Workflow

### Old Approach (research-tools):
```python
from src.search_entities import search_by_entities

df = search_by_entities(
    entities=entities,
    sentences=sentences,
    start_date=start_date,
    end_date=end_date,
    scope=DocumentType.NEWS,
    sources=sources,
    ...
)
```

### New Approach (migrated search_entities.py):
```python
from src.search_entities import search_by_entities

df = search_by_entities(
    entity_ids=crypto_ids,
    entity_names=crypto_names,
    sentences=sentences,
    start_date="2025-01-01",
    end_date="2025-09-08",
    rest_client=rest_client,
    document_limit=50,
)
```

## Labeling Workflow

### Old Approach (ScreenerLabeler):
```python
labeler = ScreenerLabeler(
    labels=sentences,  # Use theme sentences as labels
    llm_model_config="openai::gpt-4o-mini",
)
df = labeler.label(df)
```

### New Approach (OpenAI directly):
```python
def label_crypto_text(text: str, entity_name: str, themes: list[str]) -> dict:
    """Label a text chunk for crypto entity against themes."""
    prompt = f"""Analyze this text about {entity_name} and identify which theme it relates to most.

Themes: {', '.join(themes)}

Text: {text}

Return JSON: {{"label": "theme name or 'unclear'", "motivation": "brief explanation"}}
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(response.choices[0].message.content)

# Apply to dataframe
df["label_result"] = df.apply(
    lambda row: label_crypto_text(row["text"], row["entity_name"], sentences), 
    axis=1
)
df["label"] = df["label_result"].apply(lambda x: x["label"])
df["motivation"] = df["label_result"].apply(lambda x: x["motivation"])
```

## Scoring

### Old Approach (get_scored_df):
```python
from bigdata_research_tools.workflows.utils import get_scored_df

df_scored = get_scored_df(df, theme_tree.labels)
```

### New Approach (manual scoring):
```python
def score_entities(df: pd.DataFrame, themes: list[str]) -> pd.DataFrame:
    """Score entities by theme mentions."""
    scores = []
    
    for entity_id in df["entity_id"].unique():
        entity_df = df[df["entity_id"] == entity_id]
        entity_name = entity_df.iloc[0]["entity_name"]
        
        row = {"entity_id": entity_id, "entity_name": entity_name}
        
        for theme in themes:
            theme_count = len(entity_df[entity_df["label"] == theme])
            row[theme] = theme_count
        
        row["composite_score"] = sum(row[t] for t in themes)
        scores.append(row)
    
    return pd.DataFrame(scores).sort_values("composite_score", ascending=False)

df_scored = score_entities(df, sentences)
```

## Environment Variables

Update your `.env` file:
```
BIGDATA_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Reference Implementation

See `Thematic_Screener_CLI` for patterns on:
- CSV universe management
- Label generation with OpenAI
- Structured LLM responses
- Search + label pipeline
