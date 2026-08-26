"""Simplified search for Tracking Inflation Drivers (no SDK)."""

from __future__ import annotations

import pandas as pd
from .bigdata_rest import BigdataRestClient
from .search_helper import run_universe_search


def search_inflation_topics(
    company_ids: list[str],
    queries: list[str],
    start_date: str,
    end_date: str,
    id_to_name: dict[str, str] | None = None,
    scope: str = "news",
    **kwargs,
) -> pd.DataFrame:
    """Search for inflation-related topics."""
    return run_universe_search(
        company_ids=company_ids,
        queries=queries,
        start_date=start_date,
        end_date=end_date,
        scope=scope,
        id_to_name=id_to_name,
        **kwargs,
    )
