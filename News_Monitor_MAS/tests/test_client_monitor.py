"""Unit tests for client news monitor PoC."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.client_monitor.config import DEFAULT_SEARCH_CATEGORY, resolve_search_category
from src.client_monitor.digest import build_alerts_with_stories
from src.client_monitor.mas import compute_mas, scale_lambda_from_total
from src.client_monitor.novelty import headline_hash, mark_syndication, normalize_headline
from src.client_monitor.plans import basket_expected_chunks, build_search_plan
from src.client_monitor.query import QuerySpec, build_entity_wide_spec, build_query_spec
from src.client_monitor.taxonomy import build_taxonomy_index, load_taxonomy
from src.client_monitor.topics import MONITOR_TOPICS, SearchMode
from src.client_monitor.volumes import batch_size_for_spec
from src.client_monitor.window import (
    TimeWindow,
    build_time_window,
    build_time_window_from_range,
    parse_iso_datetime,
    parse_window_end,
)


@pytest.fixture
def taxonomy_path() -> Path:
    return Path(__file__).resolve().parents[1] / "taxonomy.csv"


@pytest.fixture
def sample_window() -> TimeWindow:
    end = datetime(2026, 7, 14, 14, 30, tzinfo=UTC)
    return build_time_window(window_end=end, window_minutes=15)


def test_build_time_window_15_minutes() -> None:
    end = datetime(2026, 7, 14, 14, 30, tzinfo=UTC)
    window = build_time_window(window_end=end, window_minutes=15)
    assert window.start == end - timedelta(minutes=15)
    assert window.end == end
    assert window.start_iso == "2026-07-14T14:15:00Z"
    assert window.end_iso == "2026-07-14T14:30:00Z"


def test_parse_window_end_defaults_to_utc() -> None:
    parsed = parse_window_end("2026-07-14T14:30:00Z")
    assert parsed.tzinfo is not None
    assert parsed.isoformat().endswith("+00:00")


def test_taxonomy_topic_counts(taxonomy_path: Path) -> None:
    taxonomy = load_taxonomy(taxonomy_path)
    index = build_taxonomy_index(taxonomy)
    assert len(index["earnings"]) >= 200
    assert 15 <= len(index["contracts"]) <= 25
    assert 30 <= len(index["leadership"]) <= 45
    assert 15 <= len(index["regulatory"]) <= 30


def test_query_spec_search_modes(sample_window: TimeWindow) -> None:
    topic_filter = {"search_in": "ALL", "any_of": ["business,earnings,earnings-guidance,,"]}

    text_spec = build_query_spec(
        monitor_topic="earnings",
        search_mode=SearchMode.TEXT,
        document_voice_text="Earnings text",
        topic_filter=topic_filter,
        window=sample_window,
    )
    text_query = text_spec.to_search_query(["E1"], max_chunks=5)
    assert text_query["text"] == "Earnings text"
    assert "topic" not in text_query["filters"]

    topic_spec = build_query_spec(
        monitor_topic="earnings",
        search_mode=SearchMode.TOPIC,
        document_voice_text="Earnings text",
        topic_filter=topic_filter,
        window=sample_window,
    )
    topic_query = topic_spec.to_search_query(["E1"], max_chunks=5)
    assert "text" not in topic_query
    assert topic_query["filters"]["topic"] == topic_filter

    both_spec = build_query_spec(
        monitor_topic="earnings",
        search_mode=SearchMode.TEXT_AND_TOPIC,
        document_voice_text="Earnings text",
        topic_filter=topic_filter,
        window=sample_window,
    )
    both_query = both_spec.to_search_query(["E1"], max_chunks=5)
    assert both_query["text"] == "Earnings text"
    assert both_query["filters"]["topic"] == topic_filter
    assert both_query["auto_enrich_filters"] is False


def test_comention_payload_includes_topic(sample_window: TimeWindow) -> None:
    topic_filter = {"search_in": "ALL", "any_of": ["business,regulatory,,"]}
    spec = build_query_spec(
        monitor_topic="regulatory",
        search_mode=SearchMode.TEXT_AND_TOPIC,
        document_voice_text="Regulatory news",
        topic_filter=topic_filter,
        window=sample_window,
    )
    payload = spec.to_comention_payload(["E1", "E2"])
    query = payload["query"]
    assert query["filters"]["topic"] == topic_filter
    assert query["text"] == "Regulatory news"


def test_default_search_category_is_news_premium_only() -> None:
    assert DEFAULT_SEARCH_CATEGORY == {"mode": "INCLUDE", "values": ["news_premium"]}


def test_resolve_search_category() -> None:
    profile, category = resolve_search_category("news_premium")
    assert profile == "news_premium"
    assert category == {"mode": "INCLUDE", "values": ["news_premium"]}

    profile, category = resolve_search_category("NEWS")
    assert profile == "news"
    assert category == {"mode": "INCLUDE", "values": ["news"]}

    with pytest.raises(ValueError):
        resolve_search_category("filings")


def test_build_search_plan_skips_zero_volume_baskets(sample_window: TimeWindow) -> None:
    spec = QuerySpec(
        monitor_topic="earnings",
        search_mode=SearchMode.TEXT,
        text="Earnings",
        topic_filter=None,
        category={"mode": "INCLUDE", "values": ["news_premium"]},
        window=sample_window,
    )
    entity_ids = ["E1", "E2", "E3"]
    volumes = {"E1": 3, "E2": 0, "E3": 0}
    plan = build_search_plan(spec, entity_ids, volumes)
    assert len(plan["baskets"]) == 1
    assert plan["baskets"][0]["expected_chunks"] == 3
    assert basket_expected_chunks(["E1", "E2"], volumes) == 3


def test_compute_mas_zero_volume() -> None:
    metrics = compute_mas(0, lambda_bucket=10.0)
    assert metrics["mas"] == 0.0
    assert metrics["mar"] == pytest.approx(0.09, rel=0.01)


def test_compute_mas_spike() -> None:
    metrics = compute_mas(50, lambda_bucket=2.0)
    assert metrics["mas"] > 50.0
    assert metrics["z_score"] > 0


def test_scale_lambda_from_total() -> None:
    total = 30 * 24 * 60
    scaled = scale_lambda_from_total(total, window_minutes=15, span_days=30)
    assert scaled == pytest.approx(15.0, rel=0.01)


def test_headline_syndication_within_run() -> None:
    rows = [
        {"headline": "Acme wins contract - Reuters", "document_id": "d1"},
        {"headline": "Acme wins contract", "document_id": "d2"},
        {"headline": "Other story", "document_id": "d3"},
    ]
    marked = mark_syndication(rows)
    assert marked[0]["is_primary_story"] is True
    assert marked[1]["is_syndicated_repeat"] is True
    assert marked[2]["is_primary_story"] is True


def test_normalize_headline_strips_source_suffix() -> None:
    normalized = normalize_headline("Big Deal Announced - Reuters")
    assert normalized == "big deal announced"
    assert headline_hash("Big Deal Announced - Reuters") == headline_hash("big deal announced")


def test_search_mode_parse() -> None:
    assert SearchMode.parse("text+topic") == SearchMode.TEXT_AND_TOPIC
    assert SearchMode.parse("TEXT") == SearchMode.TEXT
    assert SearchMode.parse("entity_only") == SearchMode.ENTITY_ONLY
    assert SearchMode.parse("entity-only") == SearchMode.ENTITY_ONLY
    with pytest.raises(ValueError):
        SearchMode.parse("invalid")


def test_entity_wide_spec_has_no_text_or_topic(sample_window: TimeWindow) -> None:
    spec = build_entity_wide_spec(window=sample_window)
    query = spec.to_search_query(["E1"], max_chunks=50)
    assert spec.monitor_topic == "entity_wide"
    assert spec.search_mode == SearchMode.ENTITY_ONLY
    assert "text" not in query
    assert "topic" not in query["filters"]
    assert query["filters"]["entity"]["search_in"] == "ALL"

    comention = spec.to_comention_payload(["E1"])
    assert "text" not in comention["query"]
    assert "topic" not in comention["query"]["filters"]
    assert comention["query"]["filters"]["entity"]["search_in"] == "ALL"


def test_build_query_spec_rejects_entity_only(sample_window: TimeWindow) -> None:
    with pytest.raises(ValueError, match="build_entity_wide_spec"):
        build_query_spec(
            monitor_topic="earnings",
            search_mode=SearchMode.ENTITY_ONLY,
            document_voice_text="ignored",
            topic_filter=None,
            window=sample_window,
        )


def test_batch_size_shrinks_for_large_topic_filters(sample_window: TimeWindow) -> None:
    large_filter = {"search_in": "ALL", "any_of": [f"id{i}," for i in range(211)]}
    spec = build_query_spec(
        monitor_topic="earnings",
        search_mode=SearchMode.TEXT_AND_TOPIC,
        document_voice_text="Earnings text",
        topic_filter=large_filter,
        window=sample_window,
    )
    assert batch_size_for_spec(spec) <= 200


def test_monitor_topics_count() -> None:
    assert len(MONITOR_TOPICS) == 4
    keys = {topic.key for topic in MONITOR_TOPICS}
    assert keys == {"earnings", "contracts", "leadership", "regulatory"}


def test_build_alerts_with_stories_inner_join() -> None:
    mas_rows = [
        {
            "RP_ENTITY_ID": "E1",
            "COMPANY_NAME": "Acme Corp.",
            "monitor_topic": "earnings",
            "search_mode": "topic",
            "V_NOW": 5,
            "MAS": 95.0,
            "PCT_RANK": 100.0,
        },
        {
            "RP_ENTITY_ID": "E2",
            "COMPANY_NAME": "Beta Inc.",
            "monitor_topic": "earnings",
            "search_mode": "topic",
            "V_NOW": 4,
            "MAS": 90.0,
            "PCT_RANK": 99.9,
        },
        {
            "RP_ENTITY_ID": "E3",
            "COMPANY_NAME": "Gamma LLC",
            "monitor_topic": "earnings",
            "search_mode": "topic",
            "V_NOW": 1,
            "MAS": 15.0,
            "PCT_RANK": 99.0,
        },
    ]
    chunk_rows = [
        {
            "entity_id": "E1",
            "company_name": "Acme Corp.",
            "monitor_topic": "earnings",
            "search_mode": "topic",
            "is_primary_story": True,
            "headline": "Acme beats estimates",
            "document_id": "d1",
            "search_relevance": 0.9,
            "chunk_text": "Acme reported strong earnings.",
            "window_start": "2026-07-14T11:32:13Z",
            "window_end": "2026-07-14T11:47:13Z",
        },
        {
            "entity_id": "E2",
            "company_name": "Beta Inc.",
            "monitor_topic": "earnings",
            "search_mode": "topic",
            "is_primary_story": False,
            "headline": "Acme beats estimates",
            "document_id": "d1",
            "search_relevance": 0.8,
            "chunk_text": "Acme reported strong earnings.",
            "window_start": "2026-07-14T11:32:13Z",
            "window_end": "2026-07-14T11:47:13Z",
        },
    ]

    joined = build_alerts_with_stories(chunk_rows=chunk_rows, mas_rows=mas_rows)

    assert len(joined) == 1
    assert joined[0]["entity_id"] == "E1"
    assert joined[0]["headline"] == "Acme beats estimates"
    assert joined[0]["MAS"] == 95.0
    assert "E2" not in {row["entity_id"] for row in joined}
    assert "E3" not in {row["entity_id"] for row in joined}


def test_build_time_window_from_range() -> None:
    start = parse_iso_datetime("2026-06-16T05:00:00Z")
    end = parse_iso_datetime("2026-06-17T04:59:59Z")
    window = build_time_window_from_range(window_start=start, window_end=end)
    assert window.start_iso == "2026-06-16T05:00:00Z"
    assert window.end_iso == "2026-06-17T04:59:59Z"
    assert window.minutes == 1439
