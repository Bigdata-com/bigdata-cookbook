# Career Network (Person to Company)

Build a people-centric company network using [Bigdata.com](https://bigdata.com) Co-mentions data over time.

## Use Case

Given a person name, identify companies frequently co-mentioned with that person and track how the relationship evolves by period.

## APIs

- `POST /v1/search/co-mentions/entities`
- `POST /v1/knowledge-graph/entities/id`

## Script

- `people_network_analysis.py`

## Quickstart

```bash
uv venv
uv pip install -r requirements.txt
export BIGDATA_API_KEY=your_api_key_here
uv run python people_network_analysis.py "Dario Amodei"
```

## CLI Usage

```bash
uv run python people_network_analysis.py [person] [year_start] [year_end] \
  [--frequency yearly|quarterly|monthly] [--threshold FLOAT] [--top N]
```

## Output Artifacts

- `<person_slug>_career.json`
- `<person_slug>_network.png`
- `<person_slug>_heatmap.png`
