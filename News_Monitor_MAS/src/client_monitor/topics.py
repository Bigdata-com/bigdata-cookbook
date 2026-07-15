"""Fixed monitor topics and curated taxonomy rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

ENTITY_WIDE_MONITOR_TOPIC = "entity_wide"


class SearchMode(StrEnum):
    """Bigdata query composition mode."""

    TEXT = "text"
    TOPIC = "topic"
    TEXT_AND_TOPIC = "text+topic"
    ENTITY_ONLY = "entity_only"

    @classmethod
    def all_modes(cls) -> tuple[SearchMode, ...]:
        return (cls.TEXT, cls.TOPIC, cls.TEXT_AND_TOPIC)

    @classmethod
    def taxonomy_modes(cls) -> tuple[SearchMode, ...]:
        """Modes that scope retrieval to curated monitor-topic taxonomy."""
        return cls.all_modes()

    @classmethod
    def parse(cls, value: str) -> SearchMode:
        normalized = value.strip().lower()
        if normalized in {"text+topic", "text_and_topic"}:
            return cls.TEXT_AND_TOPIC
        if normalized in {"entity_only", "entity-only", "entityonly"}:
            return cls.ENTITY_ONLY
        for mode in cls:
            if mode.value == normalized:
                return mode
        msg = (
            f"unknown search mode: {value!r} "
            "(expected text, topic, text+topic, or entity_only)"
        )
        raise ValueError(msg)

    def includes_text(self) -> bool:
        return self in (SearchMode.TEXT, SearchMode.TEXT_AND_TOPIC)

    def includes_topic_filter(self) -> bool:
        return self in (SearchMode.TOPIC, SearchMode.TEXT_AND_TOPIC)

    def is_entity_wide(self) -> bool:
        return self is SearchMode.ENTITY_ONLY


@dataclass(frozen=True)
class TaxonomyRule:
    """Curated subset of taxonomy.csv rows for one monitor topic."""

    include_groups: frozenset[str] = frozenset()
    include_types_by_group: frozenset[tuple[str, str]] = frozenset()
    type_prefixes_by_group: frozenset[tuple[str, str]] = frozenset()
    exact_types_by_group: frozenset[tuple[str, str]] = frozenset()


@dataclass(frozen=True)
class MonitorTopic:
    """One client monitor topic (label, semantic text, taxonomy rule)."""

    key: str
    document_voice_text: str
    taxonomy_rule: TaxonomyRule


LEADERSHIP_EXACT_TYPES: frozenset[tuple[str, str]] = frozenset(
    {("labor-issues", "board-diversity")}
)
LEADERSHIP_TYPE_PREFIXES: frozenset[tuple[str, str]] = frozenset(
    {
        ("labor-issues", "executive-"),
        ("labor-issues", "board-member-"),
    }
)

CONTRACT_TYPES: frozenset[tuple[str, str]] = frozenset(
    {
        ("products-services", "business-contract"),
        ("products-services", "government-contract"),
        ("products-services", "award"),
    }
)

MONITOR_TOPICS: tuple[MonitorTopic, ...] = (
    MonitorTopic(
        key="earnings",
        document_voice_text=(
            "The company reports earnings, financial results, or analyst ratings."
        ),
        taxonomy_rule=TaxonomyRule(
            include_groups=frozenset({"earnings", "revenues", "dividends", "analyst-ratings"}),
        ),
    ),
    MonitorTopic(
        key="contracts",
        document_voice_text=(
            "The company announces major contracts, partnerships, or strategic developments."
        ),
        taxonomy_rule=TaxonomyRule(
            include_groups=frozenset({"partnerships"}),
            exact_types_by_group=CONTRACT_TYPES,
        ),
    ),
    MonitorTopic(
        key="leadership",
        document_voice_text=(
            "The company reports executive changes, leadership transitions, or management news."
        ),
        taxonomy_rule=TaxonomyRule(
            exact_types_by_group=LEADERSHIP_EXACT_TYPES,
            type_prefixes_by_group=LEADERSHIP_TYPE_PREFIXES,
        ),
    ),
    MonitorTopic(
        key="regulatory",
        document_voice_text=(
            "Regulatory actions, legal developments, or government news affect the company."
        ),
        taxonomy_rule=TaxonomyRule(include_groups=frozenset({"regulatory"})),
    ),
)


def get_monitor_topic(key: str) -> MonitorTopic:
    for topic in MONITOR_TOPICS:
        if topic.key == key:
            return topic
    msg = f"unknown monitor topic: {key!r}"
    raise KeyError(msg)
