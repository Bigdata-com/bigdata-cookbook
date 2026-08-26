"""Short smoke test for Board Management Monitoring (REST migration)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.bigdata_rest import BigdataRestClient, company_ids_from_universe, load_universe
from src.search_helper import run_universe_search

MANAGEMENT_QUERIES = [
    "Reputation of Management in Media",
]
BOARD_QUERIES = [
    "Board member resignations and director departures",
]


def main() -> int:
    load_dotenv(PROJECT_DIR.parent / ".env")
    load_dotenv(PROJECT_DIR / ".env")

    if not os.getenv("BIGDATA_API_KEY"):
        print("FAIL: BIGDATA_API_KEY is not set")
        return 1

    universe_path = PROJECT_DIR.parent / "Thematic_Screener_CLI" / "40_companies.csv"
    universe_df = load_universe(universe_path).head(5)
    company_ids = company_ids_from_universe(universe_df)
    id_to_name = dict(zip(universe_df["RP_ENTITY_ID"], universe_df["COMPANY_NAME"], strict=True))

    start_date = "2025-01-01"
    end_date = "2025-01-21"
    queries = MANAGEMENT_QUERIES + BOARD_QUERIES

    print(
        f"Smoke config: {len(company_ids)} companies, {start_date}..{end_date}, "
        f"{len(queries)} queries, chunk_percentage=0.02"
    )

    df_raw = run_universe_search(
        company_ids=company_ids,
        queries=queries,
        start_date=start_date,
        end_date=end_date,
        scope="all",
        chunk_percentage=0.02,
        requests_per_minute=350,
        id_to_name=id_to_name,
    )
    print(f"Search rows: {len(df_raw)}")
    if df_raw.empty:
        print("WARN: no search hits; continuing with entity metadata check only")

    unique_entity_ids = (
        df_raw["entity_id"].dropna().unique().tolist() if not df_raw.empty else company_ids[:3]
    )
    client = BigdataRestClient()
    entities_meta = client.get_entities_by_id(unique_entity_ids)
    print(f"Entity metadata rows: {len(entities_meta)}")
    if not entities_meta:
        print("FAIL: get_entities_by_id returned no metadata")
        return 1

    if not df_raw.empty:
        df_processed = df_raw.copy()
        df_processed["Date"] = pd.to_datetime(
            df_processed["timestamp"], errors="coerce", utc=True
        )
        required_cols = {"document_id", "headline", "query", "entity_name", "chunk_text"}
        missing = required_cols - set(df_processed.columns)
        if missing:
            print(f"FAIL: search output missing columns: {sorted(missing)}")
            return 1
        print(
            f"Sample hits: {len(df_processed['document_id'].unique())} documents, "
            f"{df_processed['entity_name'].nunique()} entities"
        )

    print("PASS: smoke test completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
