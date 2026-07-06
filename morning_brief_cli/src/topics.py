"""Pre-configured research topics for the morning brief pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    """A research topic with its display label and Bigdata search query."""

    id: str
    label: str
    search_query: str


TOPICS: list[Topic] = [
    Topic(
        id="earnings",
        label="Earnings & Guidance",
        search_query=(
            "The company reports quarterly earnings results, revenue growth, profit margins, "
            "EPS, and provides forward guidance on financial outlook and performance targets."
        ),
    ),
    Topic(
        id="macro",
        label="Macro & Policy",
        search_query=(
            "The company discusses macroeconomic headwinds or tailwinds, interest rate exposure, "
            "inflation impact, regulatory changes, trade tariffs, or government policy effects."
        ),
    ),
    Topic(
        id="analyst",
        label="Analyst & Sentiment",
        search_query=(
            "Analyst rating upgrades or downgrades, price target revisions, broker recommendation"
            " changes, and institutional investor sentiment or positioning shifts on the company."
        ),
    ),
    Topic(
        id="ma",
        label="M&A & Corporate",
        search_query=(
            "The company announces a merger, acquisition, divestiture, strategic partnership, "
            "joint venture, buyback program, dividend change, or material corporate restructuring."
        ),
    ),
    Topic(
        id="supply_chain",
        label="Supply Chain & Ops",
        search_query=(
            "The company addresses supply chain disruptions, manufacturing capacity constraints, "
            "logistics challenges, inventory build or drawdown, or operational efficiency programs."
        ),
    ),
]

TOPIC_BY_ID: dict[str, Topic] = {t.id: t for t in TOPICS}
