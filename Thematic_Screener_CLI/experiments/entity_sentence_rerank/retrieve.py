"""Fresh per-plan retrieval with search-query provenance preserved."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from bigdata_smart_batching import execute_search, load_plan


def _plan_search_query(plan: dict[str, Any]) -> str:
    baskets = plan.get("baskets") or []
    if not baskets:
        return ""
    query = baskets[0].get("query") or {}
    return str(query.get("text") or "")


def _plan_entity_ids(plan: dict[str, Any]) -> set[str]:
    """Entity IDs targeted by this search plan (basket companies)."""
    entity_ids: set[str] = set()
    for basket in plan.get("baskets") or []:
        for entity_id in basket.get("companies") or []:
            if entity_id:
                entity_ids.add(str(entity_id))
    return entity_ids


def _leaf_label_from_plan_file(plan_file: str) -> str:
    stem = Path(plan_file).stem
    return stem.replace("_", " ")


def _flatten_plan_documents(
    documents: list[dict[str, Any]],
    *,
    plan_file: str,
    search_query: str,
    leaf_label: str,
    plan_entity_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        document_id = str(document.get("id") or "")
        headline = document.get("headline")
        timestamp = document.get("timestamp")
        url = document.get("url")
        for chunk in document.get("chunks") or []:
            rows.append(
                {
                    "plan_file": plan_file,
                    "search_query": search_query,
                    "leaf_label": leaf_label,
                    "plan_entity_ids": sorted(plan_entity_ids),
                    "document_id": document_id,
                    "headline": headline,
                    "timestamp": timestamp,
                    "url": url,
                    "cnum": chunk.get("cnum"),
                    "chunk_text": str(chunk.get("text") or ""),
                    "search_relevance": float(chunk.get("relevance") or 0.0),
                    "sentiment": float(chunk.get("sentiment") or 0.0),
                    "primary_entity_id": chunk.get("primary_entity_id"),
                    "entity_ids": [str(eid) for eid in (chunk.get("entity_ids") or [])],
                    "detections": chunk.get("detections") or [],
                }
            )
    return rows


def retrieve_tagged_chunks(
    *,
    plans_dir: Path,
    plan_files: list[str],
    chunk_percentage: float,
    requests_per_minute: int,
    sample_size: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Execute selected plans and return query-tagged chunk rows."""
    all_rows: list[dict[str, Any]] = []
    for plan_file in plan_files:
        plan_path = plans_dir / plan_file
        if not plan_path.exists():
            msg = f"Plan file not found: {plan_path}"
            raise FileNotFoundError(msg)
        plan = load_plan(str(plan_path))
        search_query = _plan_search_query(plan)
        plan_entity_ids = _plan_entity_ids(plan)
        leaf_label = _leaf_label_from_plan_file(plan_file)
        documents = execute_search(
            search_plan=plan,
            chunk_percentage=chunk_percentage,
            requests_per_minute=requests_per_minute,
            basket_filtered_entities=True,
        )
        all_rows.extend(
            _flatten_plan_documents(
                documents,
                plan_file=plan_file,
                search_query=search_query,
                leaf_label=leaf_label,
                plan_entity_ids=plan_entity_ids,
            )
        )

    if len(all_rows) <= sample_size:
        return all_rows

    rng = random.Random(seed)
    by_plan: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_plan.setdefault(row["plan_file"], []).append(row)

    plan_keys = sorted(by_plan)
    per_plan = sample_size // len(plan_keys)
    remainder = sample_size % len(plan_keys)
    sampled: list[dict[str, Any]] = []
    for index, plan_key in enumerate(plan_keys):
        quota = per_plan + (1 if index < remainder else 0)
        pool = by_plan[plan_key]
        sampled.extend(rng.sample(pool, min(quota, len(pool))))

    if len(sampled) < sample_size:
        chosen_ids = {id(row) for row in sampled}
        leftovers = [row for row in all_rows if id(row) not in chosen_ids]
        sampled.extend(rng.sample(leftovers, sample_size - len(sampled)))

    return sampled[:sample_size]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
