"""Deterministic digest and run summary for client monitor PoC."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALERTS_WITH_STORIES_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "company_name",
    "monitor_topic",
    "search_mode",
    "MAS",
    "PCT_RANK",
    "V_NOW",
    "headline",
    "timestamp",
    "url",
    "document_id",
    "search_relevance",
    "sentiment",
    "chunk_text",
    "window_start",
    "window_end",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def build_retrieval_digest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize retrieval rows by topic, mode, and company."""
    primary_rows = [row for row in rows if row.get("is_primary_story", True)]
    topic_counts: Counter[str] = Counter()
    company_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()

    for row in primary_rows:
        topic_counts[str(row.get("monitor_topic"))] += 1
        mode_counts[str(row.get("search_mode"))] += 1
        company = str(row.get("company_name") or row.get("entity_id") or "")
        if company:
            company_counts[company] += 1

    top_companies = [
        {"company_name": name, "primary_chunk_count": count}
        for name, count in company_counts.most_common(20)
    ]
    representative = sorted(
        primary_rows,
        key=lambda row: float(row.get("search_relevance") or 0.0),
        reverse=True,
    )[:30]

    return {
        "created_at": _utc_now(),
        "retrieval_stats": {
            "total_rows": len(rows),
            "primary_rows": len(primary_rows),
            "syndicated_rows": len(rows) - len(primary_rows),
            "monitor_topics": len(topic_counts),
        },
        "by_monitor_topic": dict(topic_counts),
        "by_search_mode": dict(mode_counts),
        "top_companies": top_companies,
        "representative_chunks": [
            {
                "monitor_topic": row.get("monitor_topic"),
                "search_mode": row.get("search_mode"),
                "company_name": row.get("company_name"),
                "headline": row.get("headline"),
                "search_relevance": row.get("search_relevance"),
                "document_id": row.get("document_id"),
            }
            for row in representative
        ],
    }


def _compute_alerts(
    chunk_rows: list[dict[str, Any]],
    mas_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build alert list from MAS scores and primary retrieval rows."""
    primary_rows = [row for row in chunk_rows if row.get("is_primary_story", True)]
    alerts: list[dict[str, Any]] = []
    by_topic: dict[str, list[dict[str, Any]]] = {}
    for row in mas_rows:
        by_topic.setdefault(str(row["monitor_topic"]), []).append(row)

    primary_companies_by_topic: dict[str, set[str]] = {}
    for row in primary_rows:
        topic = str(row.get("monitor_topic"))
        entity_id = str(row.get("entity_id") or "")
        if entity_id:
            primary_companies_by_topic.setdefault(topic, set()).add(entity_id)

    for topic, scores in by_topic.items():
        if not scores:
            continue
        threshold_index = max(int(len(scores) * 0.99) - 1, 0)
        pct_threshold = scores[threshold_index]["PCT_RANK"]
        for score in scores:
            entity_id = str(score["RP_ENTITY_ID"])
            high_mas = score["PCT_RANK"] >= pct_threshold and score["MAS"] > 0
            has_primary = entity_id in primary_companies_by_topic.get(topic, set())
            if high_mas or has_primary:
                alerts.append(
                    {
                        "monitor_topic": topic,
                        "entity_id": entity_id,
                        "company_name": score.get("COMPANY_NAME"),
                        "MAS": score.get("MAS"),
                        "PCT_RANK": score.get("PCT_RANK"),
                        "has_primary_chunk": has_primary,
                    }
                )
    return alerts


def build_alerts_with_stories(
    *,
    chunk_rows: list[dict[str, Any]],
    mas_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Inner join alerts with primary retrieval chunks (one row per story)."""
    alerts = _compute_alerts(chunk_rows, mas_rows)
    alert_keys = {
        (str(alert["entity_id"]), str(alert["monitor_topic"]))
        for alert in alerts
        if alert.get("has_primary_chunk")
    }
    mas_by_key = {
        (str(row["RP_ENTITY_ID"]), str(row["monitor_topic"])): row for row in mas_rows
    }

    joined: list[dict[str, Any]] = []
    for row in chunk_rows:
        if not row.get("is_primary_story", True):
            continue
        entity_id = str(row.get("entity_id") or "")
        monitor_topic = str(row.get("monitor_topic") or "")
        key = (entity_id, monitor_topic)
        if key not in alert_keys:
            continue
        mas = mas_by_key.get(key, {})
        joined.append(
            {
                "entity_id": entity_id,
                "company_name": row.get("company_name") or mas.get("COMPANY_NAME", ""),
                "monitor_topic": monitor_topic,
                "search_mode": row.get("search_mode") or mas.get("search_mode", ""),
                "MAS": mas.get("MAS"),
                "PCT_RANK": mas.get("PCT_RANK"),
                "V_NOW": mas.get("V_NOW"),
                "headline": row.get("headline"),
                "timestamp": row.get("timestamp"),
                "url": row.get("url"),
                "document_id": row.get("document_id"),
                "search_relevance": row.get("search_relevance"),
                "sentiment": row.get("sentiment"),
                "chunk_text": row.get("chunk_text"),
                "window_start": row.get("window_start"),
                "window_end": row.get("window_end"),
            }
        )

    joined.sort(
        key=lambda row: (
            -float(row.get("MAS") or 0.0),
            -float(row.get("search_relevance") or 0.0),
        )
    )
    return joined


def write_alerts_with_stories_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write inner-join alert rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ALERTS_WITH_STORIES_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in ALERTS_WITH_STORIES_COLUMNS})


def build_run_summary(
    *,
    config: dict[str, Any],
    chunk_rows: list[dict[str, Any]],
    mas_rows: list[dict[str, Any]],
    alerts_with_stories: list[dict[str, Any]] | None = None,
    timings: dict[str, float],
    mode_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build top-level run summary JSON."""
    primary_rows = [row for row in chunk_rows if row.get("is_primary_story", True)]
    syndicated_rows = len(chunk_rows) - len(primary_rows)
    alerts = _compute_alerts(chunk_rows, mas_rows)
    if alerts_with_stories is None:
        alerts_with_stories = build_alerts_with_stories(
            chunk_rows=chunk_rows,
            mas_rows=mas_rows,
        )
    alert_keys_with_story = {
        (str(row["entity_id"]), str(row["monitor_topic"])) for row in alerts_with_stories
    }

    estimated_chunks = len(chunk_rows)
    estimated_cost_usd = (estimated_chunks / 10.0) * 0.015

    summary: dict[str, Any] = {
        "created_at": _utc_now(),
        "config": config,
        "timings_seconds": timings,
        "retrieval": {
            "chunk_rows": len(chunk_rows),
            "primary_stories": len(primary_rows),
            "syndicated_rows": syndicated_rows,
        },
        "syndication_stats": {
            "primary_stories": len(primary_rows),
            "syndicated_rows": syndicated_rows,
        },
        "mas": {
            "scored_pairs": len(mas_rows),
            "alert_count": len(alerts),
            "alerts_with_stories_count": len(alert_keys_with_story),
            "story_row_count": len(alerts_with_stories),
        },
        "alerts": alerts[:100],
        "estimated_cost_usd": round(estimated_cost_usd, 4),
    }
    if mode_stats is not None:
        summary["mode_comparison"] = mode_stats
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
