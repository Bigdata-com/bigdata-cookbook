"""Smart-batching-based search (replaces SDK rate-limited search)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from bigdata_smart_batching import (
    plan_search,
    execute_search,
    deduplicate_documents,
)
from tqdm import tqdm


def run_search(
    company_ids: list[str],
    queries: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    fiscal_year: int | None = None,
    document_type: str = "news",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Execute searches using bigdata-smart-batching.

    Args:
        company_ids: List of RP_ENTITY_ID values
        queries: List of query strings
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        fiscal_year: Fiscal year for filings/transcripts (not used with current API)
        document_type: "news", "filings", or "transcripts"
        limit: Max results per company (not used with current API)

    Returns:
        List of document dicts
    """
    logging.info(f"Planning search for {len(company_ids)} companies, {len(queries)} queries")
    
    # Map document_type to category
    category_map = {
        "news": {"mode": "INCLUDE", "values": ["news", "news_premium"]},
        "filings": {"mode": "INCLUDE", "values": ["filings"]},
        "transcripts": {"mode": "INCLUDE", "values": ["transcripts"]},
    }
    category = category_map.get(document_type, category_map["news"])
    
    all_docs = []
    for query in tqdm(queries, desc="Querying Bigdata..."):
        plan = plan_search(
            universe=company_ids,
            text=query,
            start_date=start_date,
            end_date=end_date,
            volume_query_mode="iterative",
            category=category,
        )
        raw_docs = execute_search(plan)
        deduped = deduplicate_documents(raw_docs)
        all_docs.extend(deduped)
    
    logging.info(f"Retrieved {len(all_docs)} documents")
    return all_docs
