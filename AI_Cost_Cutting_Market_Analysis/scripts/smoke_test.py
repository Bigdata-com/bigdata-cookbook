"""Short smoke test for AI Cost Cutting Market Analysis (REST migration)."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.bigdata_rest import BigdataRestClient, company_ids_from_universe, load_universe
from src.labeling import DEFAULT_LABELING_MODEL, run_ai_cost_cutting_prompt
from src.search import search_by_any

PROVIDER_QUERIES = [
    "Company provides AI solutions to reduce operational costs for clients.",
]
USER_QUERIES = [
    "Company uses AI to cut internal business costs and improve efficiency.",
]


def main() -> int:
    load_dotenv(PROJECT_DIR.parent / ".env")
    load_dotenv(PROJECT_DIR / ".env")

    if not os.getenv("BIGDATA_API_KEY"):
        print("FAIL: BIGDATA_API_KEY is not set")
        return 1
    if not os.getenv("OPENAI_API_KEY"):
        print("FAIL: OPENAI_API_KEY is not set")
        return 1

    universe_path = PROJECT_DIR.parent / "Thematic_Screener_CLI" / "40_companies.csv"
    universe_df = load_universe(universe_path).head(5)
    company_ids = company_ids_from_universe(universe_df)
    id_to_name = dict(zip(universe_df["RP_ENTITY_ID"], universe_df["COMPANY_NAME"], strict=True))

    start_date = "2025-01-01"
    end_date = "2025-01-14"
    model = DEFAULT_LABELING_MODEL

    print(f"Smoke config: {len(company_ids)} companies, {start_date}..{end_date}, model={model}")

    df_provider = search_by_any(
        sentences=PROVIDER_QUERIES,
        start_date=start_date,
        end_date=end_date,
        company_ids=company_ids,
        id_to_name=id_to_name,
        scope="all",
        chunk_percentage=0.02,
        requests_per_minute=350,
    )
    df_user = search_by_any(
        sentences=USER_QUERIES,
        start_date=start_date,
        end_date=end_date,
        company_ids=company_ids,
        id_to_name=id_to_name,
        scope="all",
        chunk_percentage=0.02,
        requests_per_minute=350,
    )

    df_all = pd.concat([df_provider, df_user], ignore_index=True)
    print(f"Search rows: {len(df_all)}")
    if df_all.empty:
        print("WARN: no search hits; continuing with entity metadata check only")

    unique_entity_ids = df_all["entity_id"].dropna().unique().tolist() if not df_all.empty else company_ids[:3]
    client = BigdataRestClient()
    entities_meta = client.get_entities_by_id(unique_entity_ids)
    print(f"Entity metadata rows: {len(entities_meta)}")
    if not entities_meta:
        print("FAIL: get_entities_by_id returned no metadata")
        return 1

    if not df_all.empty:
        df_all = df_all.drop_duplicates(subset=["document_id", "headline", "entity_id", "text"])
        df_all = df_all.reset_index(drop=True)
        df_all["sentence_id"] = df_all.index.astype(str)
        df_all = df_all.rename(
            columns={
                "entity_id": "rp_entity_id",
                "document_id": "rp_document_id",
                "timestamp": "timestamp_utc",
            }
        )
        sample = df_all.head(3)

        def _run_labeling(sentences: pd.DataFrame) -> pd.DataFrame:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    run_ai_cost_cutting_prompt,
                    sentences=sentences,
                    masked_2=True,
                    batch_size=3,
                    model=model,
                    open_ai_credentials=os.environ["OPENAI_API_KEY"],
                )
                return future.result()

        labeled = _run_labeling(sample)
        if "label" not in labeled.columns or "motivation" not in labeled.columns:
            print("FAIL: labeling did not produce label/motivation columns")
            return 1
        print(f"Labeled rows: {len(labeled)}; sample labels: {labeled['label'].tolist()}")

    print("PASS: smoke test completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
