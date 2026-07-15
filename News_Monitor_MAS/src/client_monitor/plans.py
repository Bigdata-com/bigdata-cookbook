"""Fixed entity baskets and search plans for client monitor retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bigdata_smart_batching.search_function import MAX_ENTITIES_IN_ANY_OF

from src.client_monitor.query import QuerySpec
from src.client_monitor.volumes import batch_size_for_spec

DEFAULT_BASKET_SIZE = MAX_ENTITIES_IN_ANY_OF


def chunk_entity_ids(
    entity_ids: list[str],
    batch_size: int = DEFAULT_BASKET_SIZE,
) -> list[list[str]]:
    """Split entity IDs into sequential baskets."""
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)
    return [
        entity_ids[index : index + batch_size]
        for index in range(0, len(entity_ids), batch_size)
    ]


def basket_expected_chunks(entities: list[str], volumes: dict[str, int]) -> int:
    """Sum ``V_now`` for entities in one basket."""
    return sum(int(volumes.get(entity_id, 0)) for entity_id in entities)


def build_search_plan(
    spec: QuerySpec,
    entity_ids: list[str],
    volumes: dict[str, int],
    *,
    max_chunks_per_basket: int | None = None,
) -> dict[str, Any]:
    """Build an ``execute_search`` plan with expected chunks from volume pass."""
    baskets: list[dict[str, Any]] = []
    total_expected = 0

    for index, companies in enumerate(chunk_entity_ids(entity_ids, batch_size_for_spec(spec))):
        active = [entity_id for entity_id in companies if int(volumes.get(entity_id, 0)) > 0]
        expected = basket_expected_chunks(active, volumes)
        if expected <= 0:
            continue

        proportional_cap = expected
        if max_chunks_per_basket is not None:
            proportional_cap = min(proportional_cap, max_chunks_per_basket)

        query = spec.to_search_query(active, max_chunks=max(proportional_cap, 1))
        basket = {
            "basket_id": f"basket_{index}_{spec.monitor_topic}",
            "companies": active,
            "expected_chunks": expected,
            "period_start": spec.window.start_iso,
            "period_end": spec.window.end_iso,
            "query": query,
        }
        baskets.append(basket)
        total_expected += expected

    return {
        "chunk_upper_bound_estimate": total_expected,
        "baskets": baskets,
        "planning_metadata": {
            "monitor_topic": spec.monitor_topic,
            "search_mode": spec.search_mode.value,
            "uses_fixed_baskets": True,
        },
    }


def save_search_plan(plan: dict[str, Any], path: Path) -> None:
    """Persist plan JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
