"""Simplified topic search for Daily Digest (no SDK).

Uses ``POST /v1/search`` directly (fast mode) since a topic digest has no
company/entity universe to plan-batch over. Returns a *chunk-level* frame
(one row per matched text chunk) with the columns the downstream
topic-extraction pipeline (``src/topics_extractor.py``) expects: ``text``,
``headline``, ``timestamp``, ``source_name``.
"""

from __future__ import annotations

import pandas as pd
from .bigdata_rest import BigdataRestClient


def search_topics(
    topics: list[str],
    start_date: str,
    end_date: str,
    sources: list[str] | None = None,
    document_type: str | None = "NEWS",
    document_limit: int = 10,
    **kwargs,
) -> pd.DataFrame:
    """
    Search for topics without entity filter (topic digest mode).

    Args:
        topics: List of topic query strings.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        sources: Optional list of source IDs to filter to.
        document_type: Optional document type filter (e.g. "NEWS").
        document_limit: Max chunks retrieved per topic query (``query.max_chunks``).

    Returns:
        Chunk-level DataFrame with columns: query, document_id, headline,
        timestamp, url, source_name, text, relevance, sentiment.
    """
    client = BigdataRestClient()
    rows = []

    filters: dict = {
        "timestamp": {
            "start": f"{start_date}T00:00:00Z",
            "end": f"{end_date}T23:59:59Z",
        }
    }
    if document_type:
        filters["document_type"] = {"mode": "INCLUDE", "values": [document_type]}
    if sources:
        filters["source"] = {"mode": "INCLUDE", "values": sources}

    for topic_text in topics:
        body = {
            "search_mode": "fast",
            "query": {
                "text": topic_text,
                "filters": filters,
                "max_chunks": kwargs.get("max_chunks", document_limit),
            },
        }

        try:
            data = client.post("/v1/search", body)
            documents = data.get("results") or [] if isinstance(data, dict) else []
            for doc in documents:
                source = doc.get("source") or {}
                for chunk in doc.get("chunks") or []:
                    rows.append(
                        {
                            "query": topic_text,
                            "document_id": doc.get("id", ""),
                            "headline": doc.get("headline", ""),
                            "timestamp": doc.get("timestamp", ""),
                            "url": doc.get("url", ""),
                            "source_name": source.get("name", ""),
                            "text": chunk.get("text", ""),
                            "relevance": chunk.get("relevance"),
                            "sentiment": chunk.get("sentiment"),
                        }
                    )
        except Exception as e:
            print(f"Error searching topic '{topic_text}': {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
