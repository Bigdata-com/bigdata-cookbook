"""Co-mention volume fetching using shared QuerySpec payloads."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from src.client_monitor.query import QuerySpec

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.bigdata.com"
CO_MENTION_PATH = "/v1/search/co-mentions/entities"
BATCH_SIZE = 500
MIN_BATCH_SIZE = 25
MAX_WORKERS = 8
MAX_ROUNDS = 4


def batch_size_for_spec(spec: QuerySpec, default: int = BATCH_SIZE) -> int:
    """Shrink entity batches when topic/text filters increase query complexity."""
    size = default
    topic_filter = spec.topic_filter or {}
    topic_ids = topic_filter.get("any_of") if isinstance(topic_filter, dict) else None
    if isinstance(topic_ids, list) and topic_ids:
        size = min(size, max(MIN_BATCH_SIZE, 25_000 // len(topic_ids)))
    if spec.text:
        size = min(size, 200)
    return max(MIN_BATCH_SIZE, size)


def _api_base_url() -> str:
    return os.getenv("BIGDATA_API_BASE_URL", DEFAULT_API_BASE).rstrip("/")


def _headers(api_key: str | None) -> dict[str, str]:
    key = api_key or os.getenv("BIGDATA_API_KEY")
    if not key:
        msg = "BIGDATA_API_KEY is not set"
        raise ValueError(msg)
    return {"X-API-KEY": key, "Content-Type": "application/json"}


def post_with_retry(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None = None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """POST JSON with simple 429 backoff."""
    headers = _headers(api_key)
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 429:
            wait = 2.0**attempt
            logger.warning("Rate limited; waiting %.1fs", wait)
            time.sleep(wait)
            continue
        if response.status_code >= 400:
            snippet = (response.text or "")[:300]
            last_error = requests.HTTPError(
                f"{response.status_code} {response.reason}: {snippet}",
                response=response,
            )
            raise last_error
        data = response.json()
        if not isinstance(data, dict):
            msg = f"unexpected co-mention response type: {type(data).__name__}"
            raise TypeError(msg)
        return data
    msg = f"Failed after {max_attempts} attempts (rate limited)"
    raise RuntimeError(msg)


def _parse_company_volumes(response: dict[str, Any], entity_ids: list[str]) -> dict[str, int]:
    allowed = set(entity_ids)
    results = response.get("results", response)
    companies = results.get("companies", []) if isinstance(results, dict) else []
    volumes: dict[str, int] = {}
    for item in companies:
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("id") or item.get("entity_id") or "")
        if entity_id not in allowed:
            continue
        if "total_chunks_count" not in item:
            continue
        chunks = int(item.get("total_chunks_count") or 0)
        if chunks > 0:
            volumes[entity_id] = chunks
    return volumes


def fetch_batch_volumes(
    spec: QuerySpec,
    entity_ids: list[str],
    *,
    api_key: str | None = None,
) -> dict[str, int]:
    """Fetch co-mention volumes for one entity batch."""
    if not entity_ids:
        return {}
    url = f"{_api_base_url()}{CO_MENTION_PATH}"
    payload = spec.to_comention_payload(entity_ids)
    try:
        response = post_with_retry(url, payload, api_key=api_key)
    except requests.HTTPError as error:
        message = str(error).lower()
        if "too complex" in message and len(entity_ids) > 1:
            midpoint = len(entity_ids) // 2
            left = fetch_batch_volumes(spec, entity_ids[:midpoint], api_key=api_key)
            right = fetch_batch_volumes(spec, entity_ids[midpoint:], api_key=api_key)
            return {**left, **right}
        raise
    return _parse_company_volumes(response, entity_ids)


def fetch_entity_volumes(
    spec: QuerySpec,
    entity_ids: list[str],
    *,
    api_key: str | None = None,
    batch_size: int | None = None,
    max_workers: int = MAX_WORKERS,
    max_rounds: int = MAX_ROUNDS,
) -> dict[str, int]:
    """Iteratively fetch per-entity chunk volumes for the query spec window."""
    effective_batch_size = batch_size if batch_size is not None else batch_size_for_spec(spec)
    universe = list(entity_ids)
    universe_set = set(universe)
    found: dict[str, int] = {}

    for round_num in range(max_rounds):
        remaining = [entity_id for entity_id in universe if entity_id not in found]
        if not remaining:
            break

        batches = [
            remaining[index : index + effective_batch_size]
            for index in range(0, len(remaining), effective_batch_size)
        ]
        round_found = 0

        def _task(batch: list[str]) -> dict[str, int]:
            return fetch_batch_volumes(spec, batch, api_key=api_key)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_task, batch) for batch in batches]
            for future in as_completed(futures):
                batch_volumes = future.result()
                for entity_id, chunks in batch_volumes.items():
                    if entity_id in universe_set and entity_id not in found and chunks > 0:
                        found[entity_id] = chunks
                        round_found += 1

        logger.info(
            "Volume round %d for %s: +%d entities (%d total with volume)",
            round_num + 1,
            spec.monitor_topic,
            round_found,
            len(found),
        )
        if round_found == 0:
            break

    return found
