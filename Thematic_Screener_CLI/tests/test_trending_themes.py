"""Tests for current macro-theme discovery helpers."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from typing import Any

from src.trending_themes import (
    MacroEvent,
    ThemeSeed,
    consolidate_macro_events,
    demo_lookback_window,
    fetch_macro_events,
    ground_theme_seeds,
)


class _Cursor:
    description = [
        ("IMPORTANCE",),
        ("ENTITY_NAME",),
        ("RP_ENTITY_ID",),
        ("SENT",),
        ("CATEGORY",),
        ("TITLES",),
        ("SAMPLE_TITLE",),
    ]

    def __init__(self) -> None:
        self.executed_sql = ""
        self.closed = False

    def execute(self, sql: str) -> None:
        self.executed_sql = sql

    def fetchall(self) -> list[tuple[object, ...]]:
        return [(2.5, "Federal Reserve", "ABC123", -0.4, "rates", "A /n B", "A")]

    def close(self) -> None:
        self.closed = True


class _Connection:
    def __init__(self) -> None:
        self.active_cursor = _Cursor()
        self.closed = False

    def cursor(self) -> _Cursor:
        return self.active_cursor

    def close(self) -> None:
        self.closed = True


class _Completions:
    def __init__(self, content: dict[str, object]) -> None:
        self.content = content

    def create(self, **_: Any) -> SimpleNamespace:
        message = SimpleNamespace(content=json.dumps(self.content))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client(content: dict[str, object]) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions(content)))


def _event() -> MacroEvent:
    return MacroEvent(
        importance=2.5,
        entity_name="Federal Reserve",
        rp_entity_id="ABC123",
        sent=-0.4,
        category="rates",
        titles="A /n B",
        sample_title="A",
    )


def test_fetch_macro_events_closes_snowflake_resources() -> None:
    connection = _Connection()

    events = fetch_macro_events(lambda: connection)

    assert events == [_event()]
    assert "DATEADD(day, -1" in connection.active_cursor.executed_sql
    assert connection.active_cursor.closed is True
    assert connection.closed is True


def test_consolidate_macro_events_parses_structured_themes() -> None:
    client = _client(
        {
            "themes": [
                {
                    "main_theme": "Rate-cut repricing",
                    "rationale": "Policy expectations shifted.",
                    "search_queries": ["Companies discuss lower rates and refinancing."],
                }
            ]
        }
    )

    seeds = consolidate_macro_events([_event()], client=client)

    assert seeds[0].main_theme == "Rate-cut repricing"
    assert len(seeds[0].search_queries) == 1


def test_ground_theme_seeds_preserves_source_attribution() -> None:
    seed = ThemeSeed(
        main_theme="Rate-cut repricing",
        rationale="Policy expectations shifted.",
        search_queries=["Companies discuss lower rates and refinancing."],
    )

    grounded = ground_theme_seeds(
        [seed],
        start_date=date(2026, 8, 1),
        end_date=date(2026, 9, 1),
        retriever=lambda *_: [
            {
                "text": "Borrowing costs declined.",
                "source_name": "Example News",
                "timestamp": "2026-09-01T08:00:00Z",
                "url": "https://example.test/story",
                "headline": "Rates decline",
                "relevance": 0.9,
            }
        ],
    )

    assert grounded[0].sources[0].source_name == "Example News"
    assert grounded[0].sources[0].url == "https://example.test/story"


def test_demo_lookback_window_is_one_calendar_year() -> None:
    start, end = demo_lookback_window(end_date=date(2026, 9, 1))

    assert start == date(2025, 9, 1)
    assert end == date(2026, 9, 1)


def test_demo_lookback_window_handles_leap_day() -> None:
    start, end = demo_lookback_window(end_date=date(2024, 2, 29))

    assert start == date(2023, 2, 28)
    assert end == date(2024, 2, 29)
