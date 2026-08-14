"""Convert smart-batching search results to plain dicts / DataFrames.

No ``bigdata-client`` dependency — results stay as JSON-shaped dicts.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def normalize_documents(raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a shallow-normalized list of document dicts.

    Preserves the smart-batching / REST search shape so callers can use
    ``id``, ``headline``, ``timestamp``, ``url``, ``source``, and ``chunks``
    without SDK ``Document`` models.
    """
    documents: list[dict[str, Any]] = []
    for raw_doc in raw_results:
        raw_source = raw_doc.get("source") or {}
        chunks: list[dict[str, Any]] = []
        for raw_chunk in raw_doc.get("chunks") or []:
            chunks.append(
                {
                    "text": raw_chunk.get("text", ""),
                    "cnum": raw_chunk.get("cnum", raw_chunk.get("chunk_index", 0)),
                    "relevance": raw_chunk.get("relevance"),
                    "sentiment": raw_chunk.get("sentiment"),
                    "entity_ids": list(raw_chunk.get("entity_ids") or []),
                    "detections": list(raw_chunk.get("detections") or []),
                }
            )
        documents.append(
            {
                "id": raw_doc.get("id", ""),
                "headline": raw_doc.get("headline", ""),
                "timestamp": raw_doc.get("timestamp", ""),
                "url": raw_doc.get("url"),
                "source": {
                    "id": raw_source.get("id", ""),
                    "name": raw_source.get("name", "Unknown"),
                    "rank": raw_source.get("rank"),
                },
                "chunks": chunks,
                "reporting_entities": raw_doc.get("reporting_entities"),
            }
        )
    return documents


# Backwards-compatible alias (previously returned SDK Document objects).
convert_smart_batching_to_documents = normalize_documents


def convert_to_dataframe(raw_results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert smart-batching output to a DataFrame exploded by chunk."""
    rows: list[dict[str, Any]] = []

    for raw_doc in raw_results:
        doc_id = raw_doc.get("id", "")
        headline = raw_doc.get("headline", "")
        timestamp = raw_doc.get("timestamp", "")
        url = raw_doc.get("url", "")
        reporting_entities = raw_doc.get("reporting_entities", [])

        raw_source = raw_doc.get("source") or {}
        source_id = raw_source.get("id", "")
        source_name = raw_source.get("name", "")
        source_rank = raw_source.get("rank", "")

        chunks = raw_doc.get("chunks") or []

        if not chunks:
            rows.append(
                {
                    "date": timestamp,
                    "doc_id": doc_id,
                    "headline": headline,
                    "source_id": source_id,
                    "source_name": source_name,
                    "source_rank": source_rank,
                    "chunk_index": None,
                    "chunk_text": "",
                    "chunk_relevance": None,
                    "chunk_sentiment": None,
                    "entity_ids": [],
                    "url": url,
                    "reporting_entities": reporting_entities,
                }
            )
            continue

        for chunk in chunks:
            rows.append(
                {
                    "date": timestamp,
                    "doc_id": doc_id,
                    "headline": headline,
                    "source_id": source_id,
                    "source_name": source_name,
                    "source_rank": source_rank,
                    "chunk_index": chunk.get("cnum"),
                    "chunk_text": chunk.get("text", ""),
                    "chunk_relevance": chunk.get("relevance"),
                    "chunk_sentiment": chunk.get("sentiment"),
                    "entity_ids": chunk.get("entity_ids", []),
                    "url": url,
                    "reporting_entities": reporting_entities,
                }
            )

    df = pd.DataFrame(rows)
    if "date" in df.columns and not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    return df
