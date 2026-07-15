"""Execute search plans and flatten results to JSONL rows."""

from __future__ import annotations

import logging
from typing import Any

from bigdata_smart_batching import deduplicate_documents, execute_search

from src.client_monitor.query import QuerySpec

logger = logging.getLogger(__name__)


def plan_entity_ids(plan: dict[str, Any]) -> set[str]:
    entity_ids: set[str] = set()
    for basket in plan.get("baskets") or []:
        for entity_id in basket.get("companies") or []:
            if entity_id:
                entity_ids.add(str(entity_id))
    return entity_ids


def run_plan_search(
    plan: dict[str, Any],
    *,
    chunk_percentage: float,
    requests_per_minute: int,
) -> list[dict[str, Any]]:
    """Execute one search plan and return deduplicated documents."""
    if not plan.get("baskets"):
        return []
    documents = execute_search(
        search_plan=plan,
        chunk_percentage=chunk_percentage,
        requests_per_minute=requests_per_minute,
        basket_filtered_entities=True,
    )
    return deduplicate_documents(documents)


def flatten_documents(
    documents: list[dict[str, Any]],
    *,
    spec: QuerySpec,
    plan_entity_ids: set[str],
    id_to_name: dict[str, str],
) -> list[dict[str, Any]]:
    """Flatten documents to one row per chunk × matched universe entity."""
    rows: list[dict[str, Any]] = []
    for document in documents:
        document_id = str(document.get("id") or "")
        headline = document.get("headline")
        timestamp = document.get("timestamp")
        url = document.get("url")
        source_category = document.get("source_category") or document.get("category")
        source_obj = document.get("source")
        source_name = ""
        if isinstance(source_obj, dict):
            source_name = str(source_obj.get("name") or "")

        for chunk in document.get("chunks") or []:
            if not isinstance(chunk, dict):
                continue
            entity_ids = [str(value) for value in (chunk.get("entity_ids") or [])]
            matched = [entity_id for entity_id in entity_ids if entity_id in plan_entity_ids]
            if not matched and entity_ids:
                matched = [entity_id for entity_id in entity_ids if entity_id in id_to_name]
            if not matched:
                matched = [None]

            for entity_id in matched:
                rows.append(
                    {
                        "monitor_topic": spec.monitor_topic,
                        "search_mode": spec.search_mode.value,
                        "document_id": document_id,
                        "headline": headline,
                        "timestamp": timestamp,
                        "url": url,
                        "source_category": source_category,
                        "source_name": source_name,
                        "entity_id": entity_id,
                        "company_name": id_to_name.get(str(entity_id), "") if entity_id else "",
                        "cnum": chunk.get("cnum"),
                        "chunk_text": str(chunk.get("text") or ""),
                        "search_relevance": float(chunk.get("relevance") or 0.0),
                        "sentiment": float(chunk.get("sentiment") or 0.0),
                        "window_start": spec.window.start_iso,
                        "window_end": spec.window.end_iso,
                    }
                )
    return rows


def dedupe_rows_by_document(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cross-topic dedupe: keep first row per (document_id, entity_id, monitor_topic)."""
    seen: set[tuple[str | None, str | None, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (
            row.get("document_id"),
            row.get("entity_id"),
            str(row.get("monitor_topic")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped
