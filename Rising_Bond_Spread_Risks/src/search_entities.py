"""Simplified search for Rising Bond Spread Risks (no SDK).

This module provides basic search functionality. The original SDK-based
search_entities.py had advanced entity sentiment features that would require
additional API endpoint access beyond basic search.
"""

from __future__ import annotations

import pandas as pd
from .bigdata_rest import BigdataRestClient, load_universe, company_ids_from_universe
from .search_helper import run_universe_search


def search_by_entities(
    entities: list[str],  # Now expects list of RP_ENTITY_IDs
    sentences: list[str],
    start_date: str,
    end_date: str,
    id_to_name: dict[str, str] | None = None,
    scope: str = "all",
    **kwargs,
) -> pd.DataFrame:
    """
    Screen for documents based on entities and sentences.

    Args:
        entities: List of RP_ENTITY_ID values to search.
        sentences: The list of sentences to screen for.
        start_date: The start date for the search (YYYY-MM-DD).
        end_date: The end date for the search (YYYY-MM-DD).
        id_to_name: Optional mapping from RP_ENTITY_ID to entity name.
        scope: Document type scope ('news', 'filings', 'transcripts', 'all').

    Returns:
        DataFrame: The DataFrame with the screening results.
    """
    return run_universe_search(
        company_ids=entities,
        queries=sentences,
        start_date=start_date,
        end_date=end_date,
        scope=scope,
        id_to_name=id_to_name,
        **kwargs,
    )


def post_process_dataframe(
    df: pd.DataFrame,
    extra_fields: dict | None = None,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Post-process the labeled DataFrame (simplified version).

    Args:
        df: DataFrame to process.
        extra_fields: Additional field mappings for column renaming.
        extra_columns: Additional columns to include in export.

    Returns:
        Processed DataFrame.
    """
    extra_fields = extra_fields or {}
    extra_columns = extra_columns or []

    # Filter unlabeled sentences
    df = df.loc[df.get("label", "unclear") != "unclear"].copy()
    if df.empty:
        print("Empty dataframe: all rows labelled unclear")
        return df

    # Process timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["Time Period"] = df["timestamp"].dt.strftime("%b %Y")
        df["Date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    # Basic column mappings
    # Note: "chunk_text" is intentionally NOT mapped to "Quote" here — the search
    # helper populates "text", "chunk_text" and "masked_text" with the same
    # string, and renaming two source columns to the same target would produce
    # a duplicate "Quote" column (breaks any code that does df["Quote"]).
    columns_map = {
        "entity_name": "Entity",
        "entity_id": "Entity ID",
        "document_id": "Document ID",
        "headline": "Headline",
        "text": "Quote",
        "motivation": "Motivation",
        "label": "Sub-Scenario",
        "sentiment": "Sentiment",
        "bigdata_sentiment": "Bigdata Sentiment",
    }
    columns_map.update(extra_fields)

    df = df.rename(columns=columns_map)

    # Select available columns
    export_columns = [
        c
        for c in [
            "Time Period",
            "Date",
            "Entity",
            "Entity ID",
            "Document ID",
            "Headline",
            "Quote",
            "Sentiment",
            "Bigdata Sentiment",
            "Motivation",
            "Sub-Scenario",
        ]
        + extra_columns
        if c in df.columns
    ]

    return df[export_columns] if export_columns else df
