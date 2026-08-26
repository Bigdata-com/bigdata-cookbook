"""Search helpers for AI Cost Cutting Market Analysis (no SDK)."""

from __future__ import annotations

import pandas as pd
from typing import Optional

from .bigdata_rest import BigdataRestClient, load_universe, company_ids_from_universe
from .search_helper import run_universe_search


def search_ai_cost_cutting(
    sentences: list[str],
    start_date: str,
    end_date: str,
    company_ids: list[str],
    id_to_name: dict[str, str] | None = None,
    scope: str = "all",
    **kwargs,
) -> pd.DataFrame:
    """
    Screen for documents based on the input sentences.

    Args:
        sentences: The list of sentences to screen for.
        start_date: The start date for the search (YYYY-MM-DD).
        end_date: The end date for the search (YYYY-MM-DD).
        company_ids: List of RP_ENTITY_ID values to search.
        id_to_name: Optional mapping from RP_ENTITY_ID to company name.
        scope: Document type scope ('news', 'filings', 'transcripts', 'all').

    Returns:
        DataFrame: The DataFrame with the screening results.
    """
    return run_universe_search(
        company_ids=company_ids,
        queries=sentences,
        start_date=start_date,
        end_date=end_date,
        scope=scope,
        id_to_name=id_to_name,
        **kwargs,
    )


def search_by_any(
    sentences: list[str],
    start_date: str,
    end_date: str,
    company_ids: list[str],
    id_to_name: dict[str, str] | None = None,
    scope: str = "all",
    **kwargs,
) -> pd.DataFrame:
    """Alias for search_ai_cost_cutting."""
    return search_ai_cost_cutting(
        sentences=sentences,
        start_date=start_date,
        end_date=end_date,
        company_ids=company_ids,
        id_to_name=id_to_name,
        scope=scope,
        **kwargs,
    )
