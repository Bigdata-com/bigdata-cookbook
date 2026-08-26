"""Simplified entity search using REST API (migrated from research-tools)."""

from __future__ import annotations

from typing import Any
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm


def search_by_entities(
    entity_ids: list[str],
    entity_names: dict[str, str],
    sentences: list[str],
    start_date: str,
    end_date: str,
    rest_client: Any,
    document_limit: int = 50,
) -> DataFrame:
    """
    Screen for documents based on input sentences and entity filters.

    Args:
        entity_ids: List of entity IDs (e.g., ["4F2B", "D8442"])
        entity_names: Dict mapping entity_id -> name
        sentences: List of sentences to screen for
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        rest_client: BigdataRestClient instance
        document_limit: Max documents per query

    Returns:
        DataFrame with columns:
            - timestamp_utc: datetime64
            - document_id: str
            - sentence_id: str
            - headline: str
            - entity_id: str
            - entity_name: str
            - text: str
    """
    rows = []

    start_ts = f"{start_date}T00:00:00Z"
    end_ts = f"{end_date}T23:59:59Z"

    # Build search queries for each entity + sentence combination
    for entity_id in tqdm(entity_ids, desc="Searching entities"):
        entity_name = entity_names.get(entity_id, entity_id)

        for sentence in sentences:
            query = {
                "search_mode": "fast",
                "query": {
                    "text": sentence,
                    "filters": {
                        "entity": {"any_of": [entity_id]},
                        "timestamp": {"start": start_ts, "end": end_ts},
                    },
                    "max_chunks": document_limit,
                },
            }

            try:
                results = rest_client.search(query)

                for doc in results:
                    doc_id = doc.get("id") or doc.get("document_id")
                    timestamp = doc.get("timestamp") or doc.get("timestamp_utc")
                    headline = doc.get("headline") or doc.get("title")
                    chunks = doc.get("chunks") or []

                    for i, chunk in enumerate(chunks):
                        text = chunk.get("text") if isinstance(chunk, dict) else str(chunk)
                        rows.append({
                            "timestamp_utc": pd.to_datetime(timestamp),
                            "document_id": doc_id,
                            "sentence_id": f"{doc_id}-{i}",
                            "headline": headline,
                            "entity_id": entity_id,
                            "entity_name": entity_name,
                            "text": text,
                        })
            except Exception as e:
                print(f"Search failed for entity {entity_name} / sentence '{sentence[:50]}...': {e}")
                continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["timestamp_utc", "document_id", "text", "entity_id"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def post_process_dataframe(df: DataFrame) -> DataFrame:
    """
    Post-process the labeled DataFrame.

    Args:
        df: DataFrame with columns including 'label' and 'motivation'

    Returns:
        Processed DataFrame with formatted columns
    """
    # Filter unlabeled sentences
    df = df.loc[df["label"] != "unclear"].copy()
    if df.empty:
        return df

    # Process timestamps
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
        if df["timestamp_utc"].dt.tz is not None:
            df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize(None)

    # Sort and format
    sort_columns = ["entity_name", "timestamp_utc", "label"]
    df = df.sort_values(by=sort_columns).reset_index(drop=True)

    # Add formatted columns
    if "timestamp_utc" in df.columns:
        df["Time Period"] = df["timestamp_utc"].dt.strftime("%b %Y")
        df["Date"] = df["timestamp_utc"].dt.strftime("%Y-%m-%d")

    df = df.rename(
        columns={
            "document_id": "Document ID",
            "entity_name": "Entity",
            "headline": "Headline",
            "text": "Quote",
            "motivation": "Motivation",
            "label": "Theme",
        }
    )

    # Select and order columns
    export_columns = [
        "Time Period",
        "Date",
        "Entity",
        "Document ID",
        "Headline",
        "Quote",
        "Motivation",
        "Theme",
    ]

    available_columns = [c for c in export_columns if c in df.columns]
    df = df[available_columns].reset_index(drop=True)

    return df
