"""Universe entity-type vocabulary for prompts, labeling, and exports.

The pipeline stores universe names in ``COMPANY_NAME`` for historical CSV
compatibility, but prompts and summaries should refer to countries, companies,
currencies, etc. depending on ``--entity-type``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EntityType(str, Enum):
    """Kind of entities in the screening universe."""

    COMPANY = "company"
    COUNTRY = "country"
    CURRENCY = "currency"
    ORGANIZATION = "organization"


DEFAULT_ENTITY_TYPE = EntityType.COMPANY


@dataclass(frozen=True)
class EntityTypeConfig:
    """Prompt and export vocabulary for one universe entity type."""

    entity_type: EntityType
    target_entity: str
    entity_noun: str
    entity_noun_plural: str
    entity_level: str
    taxonomy_subject: str
    evidence_sources: str
    search_query_prefix: str
    payload_name_field: str
    summary_prompt_label: str


_ENTITY_CONFIGS: dict[EntityType, EntityTypeConfig] = {
    EntityType.COMPANY: EntityTypeConfig(
        entity_type=EntityType.COMPANY,
        target_entity="Target Company",
        entity_noun="company",
        entity_noun_plural="companies",
        entity_level="company-level",
        taxonomy_subject="companies",
        evidence_sources="company news, filings, and transcripts",
        search_query_prefix="The company",
        payload_name_field="company_name",
        summary_prompt_label="Company",
    ),
    EntityType.COUNTRY: EntityTypeConfig(
        entity_type=EntityType.COUNTRY,
        target_entity="Target Country",
        entity_noun="country",
        entity_noun_plural="countries",
        entity_level="country-level",
        taxonomy_subject="sovereign countries and economies",
        evidence_sources="news, macro commentary, and policy reports about countries",
        search_query_prefix="The country",
        payload_name_field="country_name",
        summary_prompt_label="Country",
    ),
    EntityType.CURRENCY: EntityTypeConfig(
        entity_type=EntityType.CURRENCY,
        target_entity="Target Currency",
        entity_noun="currency",
        entity_noun_plural="currencies",
        entity_level="currency-level",
        taxonomy_subject="currencies and monetary areas",
        evidence_sources="news, FX commentary, and central-bank communications",
        search_query_prefix="The currency",
        payload_name_field="currency_name",
        summary_prompt_label="Currency",
    ),
    EntityType.ORGANIZATION: EntityTypeConfig(
        entity_type=EntityType.ORGANIZATION,
        target_entity="Target Organization",
        entity_noun="organization",
        entity_noun_plural="organizations",
        entity_level="organization-level",
        taxonomy_subject="organizations",
        evidence_sources="news, filings, and transcripts about organizations",
        search_query_prefix="The organization",
        payload_name_field="organization_name",
        summary_prompt_label="Organization",
    ),
}


def get_entity_config(entity_type: EntityType | str | None) -> EntityTypeConfig:
    """Return vocabulary config for ``entity_type`` (defaults to company)."""
    if entity_type is None:
        return _ENTITY_CONFIGS[DEFAULT_ENTITY_TYPE]
    resolved = EntityType(entity_type) if not isinstance(entity_type, EntityType) else entity_type
    return _ENTITY_CONFIGS[resolved]


def prompt_context(
    entity_type: EntityType | str | None,
    *,
    main_theme: str = "",
    analyst_focus: str = "",
) -> dict[str, str]:
    """Build placeholder dict for ``str.format`` on prompt templates."""
    config = get_entity_config(entity_type)
    return {
        "main_theme": main_theme,
        "analyst_focus": analyst_focus,
        "target_entity": config.target_entity,
        "entity_noun": config.entity_noun,
        "entity_noun_plural": config.entity_noun_plural,
        "entity_level": config.entity_level,
        "taxonomy_subject": config.taxonomy_subject,
        "evidence_sources": config.evidence_sources,
        "search_query_prefix": config.search_query_prefix,
    }


def format_prompt(
    template: str,
    entity_type: EntityType | str | None,
    *,
    main_theme: str = "",
    analyst_focus: str = "",
    **extra: str,
) -> str:
    """Format a prompt template with entity-type and theme placeholders."""
    context = prompt_context(
        entity_type,
        main_theme=main_theme,
        analyst_focus=analyst_focus,
    )
    context.update(extra)
    return template.format(**context)


def resolve_entity_type(
    explicit: str | None = None,
    config: dict[str, Any] | None = None,
) -> EntityType:
    """Resolve entity type from an explicit CLI value, then ``config.json``."""
    if explicit is not None:
        return EntityType(explicit)
    if config and config.get("entity_type") is not None:
        return EntityType(str(config["entity_type"]))
    return DEFAULT_ENTITY_TYPE
