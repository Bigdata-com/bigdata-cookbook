"""Tests for universe entity-type vocabulary and wiring."""

from __future__ import annotations

import pandas as pd

from src.entity_types import EntityType, get_entity_config, resolve_entity_type
from src.prompts import RISK_SUMMARY_TEMPLATE, SYSTEM_MESSAGE_RISK, SYSTEM_PROMPT_RISK_LABELING
from src.screener import (
    _entity_metadata_lookup,
    _labeling_payload,
    build_report_json,
)
from src.search_query import normalize_summary_to_search_query


def test_resolve_entity_type_prefers_explicit_value() -> None:
    assert resolve_entity_type("country", {"entity_type": "company"}) is EntityType.COUNTRY


def test_country_prompt_vocabulary() -> None:
    config = get_entity_config(EntityType.COUNTRY)
    assert config.target_entity == "Target Country"
    assert config.payload_name_field == "country_name"

    risk_labels = SYSTEM_MESSAGE_RISK.format(
        main_theme="Energy shock",
        analyst_focus="G20 spillovers",
        entity_noun_plural="countries",
        evidence_sources="news and macro commentary",
    )
    assert "exposure of countries to the risk Energy shock" in risk_labels

    labeling = SYSTEM_PROMPT_RISK_LABELING.format(
        main_theme="Energy shock",
        labels=["Oil import cost spike"],
        target_entity=config.target_entity,
        entity_noun=config.entity_noun,
    )
    assert "Target Country" in labeling
    assert "country name" in labeling

    summary = RISK_SUMMARY_TEMPLATE.format(
        main_theme="Energy shock",
        entity_noun_plural="countries",
        entity_noun="country",
        entity_level="country-level",
    )
    assert "countries" in summary
    assert "country-level" in summary


def test_labeling_payload_uses_entity_name_field() -> None:
    sentence = {
        "sentence_id": "1",
        "text": "Germany faces higher energy import costs.",
        "company_name": "Germany",
    }
    payload = _labeling_payload(sentence, EntityType.COUNTRY)
    assert payload["country_name"] == "Germany"
    assert "company_name" not in payload


def test_normalize_summary_to_search_query_country_prefix() -> None:
    rewritten = normalize_summary_to_search_query(
        "Countries facing higher LNG import costs.",
        entity_type=EntityType.COUNTRY,
    )
    assert rewritten.startswith("The country")


def test_entity_metadata_lookup_defaults_country_for_country_universe() -> None:
    universe_df = pd.DataFrame(
        {
            "RP_COMPANY_ID": ["DE001"],
            "COMPANY_NAME": ["Germany"],
        }
    )
    metadata = _entity_metadata_lookup(universe_df, EntityType.COUNTRY)
    assert metadata["Germany"]["country"] == "Germany"


def test_build_report_json_includes_entity_type() -> None:
    screener_df = pd.DataFrame(
        columns=[
            "company_name",
            "label",
            "text",
            "motivation",
            "timestamp",
            "document_id",
            "headline",
        ]
    )
    universe_df = pd.DataFrame({"RP_COMPANY_ID": ["DE001"], "COMPANY_NAME": ["Germany"]})
    from src.screener import Node

    root = Node.model_validate(
        {
            "node": 1,
            "label": "Root",
            "summary": "Root",
            "search_query": "",
            "children": [],
        }
    )
    report = build_report_json(
        screener_df,
        root,
        universe_df,
        "risk-analyzer",
        entity_type=EntityType.COUNTRY,
    )
    assert report["entity_type"] == "country"
    assert "risk_scoring" in report
