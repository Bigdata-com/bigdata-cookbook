"""Load taxonomy.csv and build curated topic filter IDs per monitor topic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.client_monitor.topics import MONITOR_TOPICS, MonitorTopic, TaxonomyRule

TOPIC_COLUMN = "TOPIC"
GROUP_COLUMN = "GROUP"
TYPE_COLUMN = "TYPE"
SUB_TYPE_COLUMN = "SUB_TYPE"
DEFAULT_TOP_TOPIC = "business"


def load_taxonomy(path: Path) -> pd.DataFrame:
    """Load taxonomy CSV and keep ``TOPIC=business`` rows only."""
    frame = pd.read_csv(path)
    required = {TOPIC_COLUMN, GROUP_COLUMN, TYPE_COLUMN, SUB_TYPE_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        msg = f"taxonomy file {path} missing columns: {sorted(missing)}"
        raise ValueError(msg)
    filtered = frame[frame[TOPIC_COLUMN].astype(str) == DEFAULT_TOP_TOPIC].copy()
    return filtered.reset_index(drop=True)


def topic_id_from_row(row: pd.Series) -> str:
    """Build Bigdata topic ID ``TOPIC,GROUP,TYPE,SUB_TYPE,``."""

    def _cell(column: str) -> str:
        raw = row.get(column)
        if pd.isna(raw):
            return ""
        text = str(raw).strip()
        return text

    return (
        f"{_cell(TOPIC_COLUMN)},{_cell(GROUP_COLUMN)},{_cell(TYPE_COLUMN)},"
        f"{_cell(SUB_TYPE_COLUMN)},"
    )


def _row_matches_rule(row: pd.Series, rule: TaxonomyRule) -> bool:
    group = str(row[GROUP_COLUMN]).strip()
    type_name = str(row[TYPE_COLUMN]).strip()

    if group in rule.include_groups:
        return True

    pair = (group, type_name)
    if pair in rule.exact_types_by_group:
        return True

    for rule_group, prefix in rule.type_prefixes_by_group:
        if group == rule_group and type_name.startswith(prefix):
            return True

    return False


def topic_ids_for_rule(taxonomy: pd.DataFrame, rule: TaxonomyRule) -> list[str]:
    """Return sorted unique topic IDs matching a taxonomy rule."""
    matched = taxonomy[taxonomy.apply(lambda row: _row_matches_rule(row, rule), axis=1)]
    ids = sorted({topic_id_from_row(row) for _, row in matched.iterrows()})
    return ids


def topic_ids_for_monitor_topic(taxonomy: pd.DataFrame, topic: MonitorTopic) -> list[str]:
    """Return topic filter IDs for one monitor topic."""
    return topic_ids_for_rule(taxonomy, topic.taxonomy_rule)


def build_topic_filter(topic_ids: list[str]) -> dict[str, str | list[str]]:
    """Build ``filters.topic`` dict for Bigdata search/co-mention payloads."""
    return {"search_in": "ALL", "any_of": topic_ids}


def build_taxonomy_index(taxonomy: pd.DataFrame) -> dict[str, list[str]]:
    """Map monitor topic key → topic ID list."""
    return {
        topic.key: topic_ids_for_monitor_topic(taxonomy, topic) for topic in MONITOR_TOPICS
    }
