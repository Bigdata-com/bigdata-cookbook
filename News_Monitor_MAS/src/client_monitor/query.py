"""Shared query specification for retrieval and co-mention volume/MAS."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from src.client_monitor.config import DEFAULT_SEARCH_CATEGORY
from src.client_monitor.topics import ENTITY_WIDE_MONITOR_TOPIC, SearchMode
from src.client_monitor.window import TimeWindow

CO_MENTION_ENTITY_LIMIT = 500


@dataclass(frozen=True)
class QuerySpec:
    """One monitor-topic query over a time window and search mode."""

    monitor_topic: str
    search_mode: SearchMode
    text: str | None
    topic_filter: dict[str, Any] | None
    category: dict[str, Any]
    window: TimeWindow

    def query_hash(self) -> str:
        """Stable hash for baseline cache keys."""
        payload = {
            "monitor_topic": self.monitor_topic,
            "search_mode": self.search_mode.value,
            "text": self.text,
            "topic_filter": self.topic_filter,
            "category": self.category,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def _entity_search_in(self) -> str:
        """Entity-wide mode uses ALL; taxonomy modes use BODY co-mentions."""
        return "ALL" if self.search_mode.is_entity_wide() else "BODY"

    def _base_filters(self, entities: list[str]) -> dict[str, Any]:
        filters: dict[str, Any] = {
            "timestamp": {"start": self.window.start_iso, "end": self.window.end_iso},
            "entity": {"any_of": entities, "search_in": self._entity_search_in()},
            "category": self.category,
        }
        if self.search_mode.includes_topic_filter() and self.topic_filter is not None:
            filters["topic"] = self.topic_filter
        return filters

    def to_search_query(self, entities: list[str], max_chunks: int) -> dict[str, Any]:
        """Build inner ``query`` dict for ``/v1/search``."""
        query: dict[str, Any] = {
            "auto_enrich_filters": False,
            "filters": self._base_filters(entities),
            "ranking_params": {
                "source_boost": 1,
                "freshness_boost": 0,
                "reranker": {"enabled": False},
                "content_diversification": {"enabled": True},
            },
            "max_chunks": max_chunks,
        }
        if self.search_mode.includes_text() and self.text:
            query["text"] = self.text
        return query

    def to_comention_payload(
        self,
        entities: list[str],
        limit: int = CO_MENTION_ENTITY_LIMIT,
    ) -> dict[str, Any]:
        """Build payload for ``/v1/search/co-mentions/entities``."""
        filters = self._base_filters(entities)
        filters["entity"] = {
            "all_of": [],
            "any_of": entities,
            "none_of": [],
            "search_in": self._entity_search_in(),
        }
        query: dict[str, Any] = {
            "auto_enrich_filters": False,
            "entity_categories": ["companies"],
            "filters": filters,
            "limit": limit,
        }
        if self.search_mode.includes_text() and self.text:
            query["text"] = self.text
        return {"query": query}


def build_query_spec(
    *,
    monitor_topic: str,
    search_mode: SearchMode,
    document_voice_text: str,
    topic_filter: dict[str, Any] | None,
    window: TimeWindow,
    category: dict[str, Any] | None = None,
) -> QuerySpec:
    """Construct a ``QuerySpec`` respecting search mode inclusions."""
    if search_mode.is_entity_wide():
        msg = "use build_entity_wide_spec() for entity_only search mode"
        raise ValueError(msg)
    text: str | None = document_voice_text if search_mode.includes_text() else None
    topic: dict[str, Any] | None = topic_filter if search_mode.includes_topic_filter() else None
    return QuerySpec(
        monitor_topic=monitor_topic,
        search_mode=search_mode,
        text=text,
        topic_filter=topic,
        category=category if category is not None else DEFAULT_SEARCH_CATEGORY,
        window=window,
    )


def build_entity_wide_spec(
    *,
    window: TimeWindow,
    category: dict[str, Any] | None = None,
) -> QuerySpec:
    """Entity + timestamp + category only — no taxonomy topic filter."""
    return QuerySpec(
        monitor_topic=ENTITY_WIDE_MONITOR_TOPIC,
        search_mode=SearchMode.ENTITY_ONLY,
        text=None,
        topic_filter=None,
        category=category if category is not None else DEFAULT_SEARCH_CATEGORY,
        window=window,
    )
