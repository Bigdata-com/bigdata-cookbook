"""Shared smart-batching search helpers used by migrated cookbooks.

Matches Thematic_Screener_CLI / bigdata-smart-batching API:
``plan_search(universe=..., text=..., start_date=..., end_date=..., category=...)``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from bigdata_smart_batching import (
    deduplicate_documents,
    execute_search,
    plan_search,
)

CATEGORY_BY_SCOPE: dict[str, dict[str, Any]] = {
    "news": {"mode": "INCLUDE", "values": ["news", "news_premium"]},
    "news_premium": {"mode": "INCLUDE", "values": ["news_premium"]},
    "filings": {"mode": "INCLUDE", "values": ["filings"]},
    "transcripts": {"mode": "INCLUDE", "values": ["transcripts"]},
    "all": {
        "mode": "INCLUDE",
        "values": ["news_premium", "transcripts", "filings"],
    },
}


def run_universe_search(
    company_ids: list[str],
    queries: list[str],
    *,
    start_date: str,
    end_date: str,
    scope: str = "all",
    chunk_percentage: float = 0.05,
    requests_per_minute: int = 350,
    id_to_name: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Search a company universe for each query text; return chunk-level rows."""
    category = CATEGORY_BY_SCOPE.get(scope, CATEGORY_BY_SCOPE["all"])
    name_map = id_to_name or {}
    rows: list[dict[str, Any]] = []

    for query_text in queries:
        text = (query_text or "").strip()
        if not text:
            continue
        plan = plan_search(
            universe=company_ids,
            start_date=start_date,
            end_date=end_date,
            volume_query_mode="iterative",
            text=text,
            category=category,
        )
        raw = execute_search(
            search_plan=plan,
            chunk_percentage=chunk_percentage,
            requests_per_minute=requests_per_minute,
            basket_filtered_entities=True,
        )
        for document in deduplicate_documents(raw):
            doc_id = document.get("id", "")
            headline = document.get("headline", "")
            timestamp = document.get("timestamp", "")
            url = document.get("url", "")
            source = document.get("source") or {}
            for chunk in document.get("chunks") or []:
                entity_ids = [str(e) for e in (chunk.get("entity_ids") or [])]
                entity_id = entity_ids[0] if entity_ids else ""
                text_chunk = chunk.get("text", "")
                rows.append(
                    {
                        "document_id": doc_id,
                        "headline": headline,
                        "timestamp": timestamp,
                        "url": url,
                        "source_id": source.get("id", ""),
                        "source_name": source.get("name", ""),
                        "chunk_text": text_chunk,
                        "text": text_chunk,
                        "masked_text": text_chunk,
                        "relevance": chunk.get("relevance"),
                        "sentiment": chunk.get("sentiment"),
                        "entity_id": entity_id,
                        "entity_ids": entity_ids,
                        "entity_name": name_map.get(entity_id, entity_id),
                        "query": text,
                        "document_type": scope,
                    }
                )

    return pd.DataFrame(rows)
