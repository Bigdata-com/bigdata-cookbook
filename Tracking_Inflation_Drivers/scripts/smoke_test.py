"""Short smoke test for Tracking_Inflation_Drivers SDK migration.

Uses 5 companies, a 7-day news window, one search query, and up to 3 labeling
calls with ``gpt-5.6-luna`` (luna-safe sampling). Exits 0 on PASS.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.bigdata_rest import company_ids_from_universe, load_universe
from src.labels import (
    create_label_to_parent_mapping,
    deserialize_responses,
    get_labeling_system_prompt,
    get_prompts,
    label_schema,
    process_request_with_schema,
)
from src.mind_map_tools import get_most_granular_elements
from src.search_helper import run_universe_search

SMOKE_COMPANIES = 5
START_DATE = "2026-08-06"
END_DATE = "2026-08-13"
MAIN_THEME = "United States Inflation in 2025"


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def main() -> None:
    if not os.getenv("BIGDATA_API_KEY"):
        _fail("BIGDATA_API_KEY is not set")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        _fail("OPENAI_API_KEY is not set")

    print("=== Step 1: load universe (5 cos) ===")
    universe_path = ROOT.parent / "Thematic_Screener_CLI/40_companies.csv"
    if not universe_path.exists():
        _fail(f"universe CSV not found: {universe_path}")
    universe = load_universe(universe_path).head(SMOKE_COMPANIES)
    company_ids = company_ids_from_universe(universe)
    id_to_name = dict(zip(universe["RP_ENTITY_ID"], universe["COMPANY_NAME"], strict=True))
    print(f"OK: {len(company_ids)} companies")

    print("=== Step 2: minimal taxonomy ===")
    tree_dictionary = {
        "Node": 1,
        "Label": MAIN_THEME,
        "Summary": "Inflation in the United States.",
        "Keywords": ["United States", "inflation"],
        "Children": [
            {
                "Node": 2,
                "Label": "Energy",
                "Summary": "Energy prices drive US inflation.",
                "Children": [
                    {
                        "Node": 3,
                        "Label": "Oil Prices",
                        "Summary": "Rising crude oil prices push US gasoline costs higher.",
                        "Children": [],
                    }
                ],
            }
        ],
    }
    search_queries = get_most_granular_elements(tree_dictionary, "Summary")[:1]
    create_label_to_parent_mapping(tree_dictionary)
    print(f"OK: {len(search_queries)} search query")

    print("=== Step 3: run_universe_search (basket_filtered_entities=True) ===")
    df = run_universe_search(
        company_ids=company_ids,
        queries=search_queries,
        start_date=START_DATE,
        end_date=END_DATE,
        scope="news",
        chunk_percentage=0.02,
        requests_per_minute=350,
        id_to_name=id_to_name,
    )
    n_docs = df["document_id"].nunique() if not df.empty else 0
    print(f"OK: {len(df)} chunk rows, {n_docs} documents")

    print("=== Step 4: label up to 3 chunks (gpt-5.6-luna) ===")
    if not df.empty:
        sample = df.head(3).copy()
        sample["id"] = range(len(sample))
        sample["text"] = sample["chunk_text"].astype(str)
        theme_labels = get_most_granular_elements(tree_dictionary, "Label")
        labeling_prompt = get_labeling_system_prompt(theme_labels)
        prompts = get_prompts(sample, columns=["id", "text"])
        responses = process_request_with_schema(
            prompts, labeling_prompt, label_schema, api_key=openai_key
        )
        if not responses or all(r is None for r in responses):
            _fail("labeling returned no responses")
        df_labels = deserialize_responses(responses)
        merged = sample.merge(df_labels, left_index=True, right_index=True, how="left")
        labeled = merged[merged["label"].notna() & ~merged["label"].isin(["unclear", ""])]
        print(f"OK: labeled {len(labeled)}/{len(sample)} chunks")
    else:
        print("SKIP labeling (no search rows)")

    print("=== Step 5: verify no SDK ===")
    for mod in list(sys.modules):
        if mod and ("bigdata_client" in mod or "bigdata_research_tools" in mod):
            _fail(f"SDK module loaded: {mod}")
    print("OK: no SDK modules")

    print("\nSMOKE_RESULT=PASS")


if __name__ == "__main__":
    main()
