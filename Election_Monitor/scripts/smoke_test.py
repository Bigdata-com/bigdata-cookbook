"""Short live smoke test for Election_Monitor REST migration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

load_dotenv(PROJECT_DIR / ".env")

from src.bigdata_rest import BigdataRestClient, company_ids_from_universe, load_universe
from src.labeling import ElectionLabeler
from src.search import search_election_topics
from src.search_helper import run_universe_search
from src.visualization_tools import lookup_sector_information

LLM_MODEL = "gpt-5.6-luna"
START_DATE = "2024-10-01"
END_DATE = "2024-10-07"
CHUNK_PERCENTAGE = 0.01
MAX_LABEL_ROWS = 3
QUERIES = ["U.S. presidential election results"]


def main() -> int:
    api_key = os.getenv("BIGDATA_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("FAIL: BIGDATA_API_KEY not set")
        return 1
    if not openai_key:
        print("FAIL: OPENAI_API_KEY not set")
        return 1

    universe_path = PROJECT_DIR.parent / "Thematic_Screener_CLI" / "40_companies.csv"
    universe = load_universe(universe_path).head(2)
    company_ids = company_ids_from_universe(universe)
    id_to_name = dict(zip(universe["RP_ENTITY_ID"], universe["COMPANY_NAME"], strict=True))

    print(f"Universe: {list(id_to_name.values())}")
    print(f"Window: {START_DATE} -> {END_DATE} | chunk_percentage={CHUNK_PERCENTAGE}")

    search_df = search_election_topics(
        company_ids=company_ids,
        queries=QUERIES,
        start_date=START_DATE,
        end_date=END_DATE,
        id_to_name=id_to_name,
        scope="transcripts",
        chunk_percentage=CHUNK_PERCENTAGE,
        requests_per_minute=200,
    )
    print(f"Search rows: {len(search_df)}")
    if search_df.empty:
        print("WARN: search returned 0 rows (acceptable for narrow smoke window)")

    client = BigdataRestClient(api_key=api_key)
    entity_ids = company_ids[:1]
    entities = client.get_entities_by_id(entity_ids)
    if not entities:
        print(f"FAIL: get_entities_by_id returned empty for {entity_ids}")
        return 1
    sample = next(iter(entities.values()))
    print(f"Entity lookup OK: name={sample.get('name')} sector={sample.get('sector')}")

    if not search_df.empty:
        probe = search_df.head(1).copy()
        probe["rp_entity_id"] = probe["entity_id"]
        enriched = lookup_sector_information(probe, client)
        if "sector" not in enriched.columns:
            print("FAIL: lookup_sector_information did not add sector column")
            return 1
        print("lookup_sector_information OK")

    if search_df.empty:
        print("PASS (search empty; REST + entity lookup verified)")
        return 0

    label_texts = search_df["chunk_text"].dropna().astype(str).head(MAX_LABEL_ROWS).tolist()
    labeler = ElectionLabeler(llm_model=LLM_MODEL, unknown_label="U")
    labels = labeler.get_labels(label_texts, max_workers=1)
    if labels.empty or "label" not in labels.columns:
        print("FAIL: labeler returned no labels")
        return 1
    print(f"Labeler ({LLM_MODEL}) OK: {labels['label'].tolist()}")

    # Static checklist: basket_filtered_entities must be True in smart-batching path
    import inspect

    src = inspect.getsource(run_universe_search)
    if "basket_filtered_entities=True" not in src:
        print("FAIL: basket_filtered_entities=True missing from run_universe_search")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
