"""Simplified search for Credit Ratings Monitoring (no SDK)."""

from __future__ import annotations

import pandas as pd
from .bigdata_rest import BigdataRestClient
from .search_helper import run_universe_search


def search_credit_ratings(
    company_ids: list[str],
    queries: list[str],
    start_date: str,
    end_date: str,
    id_to_name: dict[str, str] | None = None,
    scope: str = "all",
    **kwargs,
) -> pd.DataFrame:
    """
    Search for credit rating mentions.

    Args:
        company_ids: List of RP_ENTITY_ID values.
        queries: Search query strings.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        id_to_name: Optional entity ID to name mapping.
        scope: Document scope ('news', 'filings', 'transcripts', 'all').

    Returns:
        DataFrame with search results.
    """
    basket_filtered = kwargs.pop("basket_filtered_entities", False)
    return run_universe_search(
        company_ids=company_ids,
        queries=queries,
        start_date=start_date,
        end_date=end_date,
        scope=scope,
        id_to_name=id_to_name,
        basket_filtered_entities=basket_filtered,
        **kwargs,
    )
