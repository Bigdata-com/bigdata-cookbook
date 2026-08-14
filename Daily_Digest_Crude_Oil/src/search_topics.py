"""Simplified topic search for Daily Digest (no SDK)."""

from __future__ import annotations

import pandas as pd
from .bigdata_rest import BigdataRestClient


def search_topics(
    topics: list[str],
    start_date: str,
    end_date: str,
    sources: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Search for topics without entity filter (topic digest mode).

    Args:
        topics: List of topic query strings.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        sources: Optional source filter list.

    Returns:
        DataFrame with search results.
    """
    client = BigdataRestClient()
    rows = []

    for topic_text in topics:
        query_body = {
            "text": topic_text,
            "start_date": start_date,
            "end_date": end_date,
            "limit": kwargs.get("limit", 50),
        }
        if sources:
            query_body["sources"] = sources

        try:
            results = client.search(query_body)
            for doc in results:
                rows.append(
                    {
                        "query": topic_text,
                        "document_id": doc.get("id", ""),
                        "headline": doc.get("headline", ""),
                        "timestamp": doc.get("timestamp", ""),
                        "url": doc.get("url", ""),
                        "source_name": doc.get("source", {}).get("name", ""),
                    }
                )
        except Exception as e:
            print(f"Error searching topic '{topic_text}': {e}")

    return pd.DataFrame(rows)


_DOC_TYPE_BY_SCOPE = {
    "news": "NEWS",
    "transcripts": "TRANSCRIPT",
    "filings": "FILING",
}


def search_by_keywords(
    keywords: list[str],
    start_date: str,
    end_date: str,
    scope: str = "news",
    freq: str = "D",
    document_limit: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Daily-digest keyword search (replaces the retired bigdata-research-tools helper).

    Splits ``[start_date, end_date]`` into ``freq`` periods and issues one
    ``POST /v1/search`` call per (period, keyword) pair, using the real REST
    schema (``query.filters.timestamp`` / ``query.filters.document_type`` /
    ``query.max_chunks`` — NOT a top-level ``limit``, which the API ignores).

    Returns:
        tuple: (results_df, daily_keyword_count) where results_df has one
        row per unique document with columns 'timestamp', 'text', 'headline',
        'source_name', 'url', and daily_keyword_count maps
        {day_str: {keyword: document_count}}.
    """
    client = BigdataRestClient()
    doc_type = _DOC_TYPE_BY_SCOPE.get(str(scope).lower(), "NEWS")

    periods = pd.date_range(start=start_date, end=end_date, freq=freq)
    if len(periods) == 0:
        periods = pd.DatetimeIndex([pd.Timestamp(start_date)])
    step = periods[1] - periods[0] if len(periods) > 1 else pd.Timedelta(days=1)

    rows: list[dict] = []
    daily_keyword_count: dict[str, dict[str, int]] = {}
    seen_ids: set[str] = set()

    for period_start in periods:
        period_end = period_start + step
        day_str = period_start.strftime("%Y-%m-%d")
        daily_keyword_count[day_str] = {}

        for keyword in keywords:
            payload = {
                "search_mode": "fast",
                "query": {
                    "text": keyword,
                    "filters": {
                        "timestamp": {
                            "start": period_start.strftime("%Y-%m-%dT00:00:00Z"),
                            "end": period_end.strftime("%Y-%m-%dT00:00:00Z"),
                        },
                        "document_type": {"mode": "INCLUDE", "values": [doc_type]},
                    },
                    "max_chunks": document_limit,
                },
            }
            try:
                data = client.post("/v1/search", payload)
            except Exception as e:
                print(f"Error searching '{keyword}' on {day_str}: {e}")
                daily_keyword_count[day_str][keyword] = 0
                continue

            documents = data.get("results", []) if isinstance(data, dict) else []
            daily_keyword_count[day_str][keyword] = len(documents)

            for document in documents:
                doc_id = document.get("id", "")
                if not doc_id or doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                chunks = document.get("chunks") or []
                text = "\n".join(c.get("text", "") for c in chunks).strip()
                if not text:
                    continue
                source = document.get("source") or {}
                rows.append(
                    {
                        "timestamp": document.get("timestamp"),
                        "text": text,
                        "headline": document.get("headline", ""),
                        "source_name": source.get("name", ""),
                        "url": document.get("url", ""),
                        "keyword": keyword,
                    }
                )

    results_df = pd.DataFrame(
        rows, columns=["timestamp", "text", "headline", "source_name", "url", "keyword"]
    )
    if not results_df.empty:
        results_df["timestamp"] = pd.to_datetime(results_df["timestamp"])

    return results_df, daily_keyword_count
