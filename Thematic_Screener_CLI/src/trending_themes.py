"""Discover current macro themes and ground them with Bigdata.com evidence."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any, TypeAlias

import snowflake.connector
from openai import OpenAI
from pydantic import BaseModel, Field

from src.derivative_grounding import retrieve_theme_chunks_http
from src.openai_parallel import sampling_params_for_model

DEFAULT_DISCOVERY_MODEL = "gpt-5.6-luna"
DEFAULT_GROUNDING_DAYS = 30
DEFAULT_GROUNDING_CHUNKS = 6
MAX_SEARCH_QUERIES = 3

SQL_MACRO = """
with granular as (
select timestamp_utc, event_sentiment, category, title, event_text, CASE
WHEN event_similarity_days < 90 THEN event_similarity_days
ELSE 90
END AS event_similarity_days, event_similarity_key,
rp_source_id, source_rank, rp_entity_id, entity_name, original_language,
case
when fact_level = 'fact' then 1
when fact_level = 'mention' then 0.8
when fact_level = 'opinion' then 0.6
else 0
end as fact_score from RAVENPACK.NEWS.NEWS_ANALYTICS_FULL_1_0
where event_relevance >= 90
and event_sentiment != 0
and rp_provider_id in ('MRVR','FLY','BZG','DJ','DJTP','MT','AN','RP')
and timestamp_utc >= DATEADD(day, -1, CURRENT_TIMESTAMP)
and entity_type != 'COMP'
and source_rank <= 4
and unit is null
and rp_source_id not in ('B5297D','44B3BE','C98333','CAF988','EE1FC6','57B51C',
'A0BAFE','834599','FAD0C0','E5F58B','91217B','EF40FE')
),
aggregate as (select count(distinct rp_source_id) nsour,
count(distinct event_similarity_key) nesk, max(event_similarity_days) sim,
LISTAGG(distinct title, ' /n ') titles, ANY_VALUE(title) sample_title,
count(distinct original_language) nlan, avg(fact_score) FF,
max(abs(event_sentiment))*sign(avg(event_sentiment)) sent, category,
min(timestamp_utc), rp_entity_id, entity_name
from granular
group by rp_entity_id, entity_name, category
)
select log(10,1+nsour+nesk)*(exp(sim/90)-1)*abs(sent)*log(10,1+nlan)*FF importance,
entity_name, rp_entity_id, sent, category, titles, sample_title from aggregate
order by importance DESC
limit 10
"""

ConnectionFactory: TypeAlias = Callable[[], Any]
ChunkRetriever: TypeAlias = Callable[[str, str, str, int], list[dict[str, Any]]]


class MacroEvent(BaseModel):
    """A ranked macro event returned by the RavenPack Snowflake query."""

    importance: float
    entity_name: str
    rp_entity_id: str
    sent: float
    category: str
    titles: str
    sample_title: str


class ThemeSeed(BaseModel):
    """An LLM-consolidated current theme before external grounding."""

    main_theme: str
    rationale: str
    search_queries: list[str] = Field(min_length=1, max_length=MAX_SEARCH_QUERIES)


class ThemeSeedList(BaseModel):
    """Structured response containing current theme seeds."""

    themes: list[ThemeSeed] = Field(min_length=1)


class GroundingSource(BaseModel):
    """A source-attributed Bigdata.com passage used to assess a theme."""

    query: str
    headline: str
    source_name: str
    timestamp: str
    url: str
    text: str
    relevance: float | None = None


class GroundedTheme(BaseModel):
    """A theme seed and its retrieved Bigdata.com evidence."""

    seed: ThemeSeed
    sources: list[GroundingSource]


class ThemeCandidate(BaseModel):
    """A ranked theme ready to drive the thematic screener."""

    main_theme: str
    analyst_focus: str
    rationale: str
    search_queries: list[str] = Field(min_length=1, max_length=MAX_SEARCH_QUERIES)


class ThemeCandidateList(BaseModel):
    """Structured ranked output from the grounded-theme analyst."""

    candidates: list[ThemeCandidate] = Field(min_length=1)


def get_snowflake_connection() -> Any:
    """Create a Snowflake connection from environment variables."""
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    )


def fetch_macro_events(
    connection_factory: ConnectionFactory = get_snowflake_connection,
) -> list[MacroEvent]:
    """Fetch the top macro events from the rolling last-24-hours query."""
    connection = connection_factory()
    cursor = connection.cursor()
    try:
        cursor.execute(SQL_MACRO)
        columns = [str(description[0]).lower() for description in cursor.description]
        rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    finally:
        cursor.close()
        connection.close()
    return [MacroEvent.model_validate(row) for row in rows]


def _chat_json(
    *,
    system_prompt: str,
    user_payload: object,
    model: str,
    client: OpenAI,
) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, default=str)},
        ],
        response_format={"type": "json_object"},
        **sampling_params_for_model(model, temperature=0.0, seed=42),
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("The LLM returned an empty response")
    return content


def consolidate_macro_events(
    events: Sequence[MacroEvent],
    *,
    model: str = DEFAULT_DISCOVERY_MODEL,
    client: OpenAI | None = None,
) -> list[ThemeSeed]:
    """Consolidate overlapping events into financially relevant theme seeds."""
    if not events:
        raise ValueError("At least one macro event is required")
    active_client = client or OpenAI()
    prompt = """
You are a macro equity strategist. Consolidate overlapping event rows into 3-5 distinct,
investable themes relevant to US mid- and large-cap equities. Drop human-interest stories
without a credible financial transmission channel. For each theme, provide a concise
main_theme, a factual rationale tied only to the supplied rows, and 1-3 natural-language
search_queries suitable for grounded financial-document search. Return JSON only:
{"themes": [{"main_theme": "...", "rationale": "...", "search_queries": ["..."]}]}.
"""
    content = _chat_json(
        system_prompt=prompt,
        user_payload=[event.model_dump(mode="json") for event in events],
        model=model,
        client=active_client,
    )
    return ThemeSeedList.model_validate_json(content).themes


def ground_theme_seeds(
    seeds: Sequence[ThemeSeed],
    *,
    start_date: date,
    end_date: date,
    max_chunks_per_query: int = DEFAULT_GROUNDING_CHUNKS,
    retriever: ChunkRetriever = retrieve_theme_chunks_http,
) -> list[GroundedTheme]:
    """Retrieve source-attributed Bigdata.com evidence for each theme seed."""
    grounded: list[GroundedTheme] = []
    for seed in seeds:
        sources: list[GroundingSource] = []
        for query in seed.search_queries:
            chunks = retriever(
                query,
                start_date.isoformat(),
                end_date.isoformat(),
                max_chunks_per_query,
            )
            sources.extend(
                GroundingSource(
                    query=query,
                    headline=str(chunk.get("headline") or ""),
                    source_name=str(chunk.get("source_name") or "Unknown source"),
                    timestamp=str(chunk.get("timestamp") or ""),
                    url=str(chunk.get("url") or ""),
                    text=str(chunk.get("text") or ""),
                    relevance=chunk.get("relevance")
                    if isinstance(chunk.get("relevance"), int | float)
                    else None,
                )
                for chunk in chunks
                if str(chunk.get("text") or "").strip()
            )
        grounded.append(GroundedTheme(seed=seed, sources=sources))
    return grounded


def one_year_before(value: date) -> date:
    """Return the calendar date one year before ``value``, handling leap days."""
    try:
        return value.replace(year=value.year - 1)
    except ValueError:
        return value.replace(year=value.year - 1, day=28)


def demo_lookback_window(*, end_date: date | None = None) -> tuple[date, date]:
    """Return a one-year demo lookback ending on ``end_date`` (UTC today by default).

    This is a reporting window, not an estimated onset date for the theme.
    """
    resolved_end = end_date or datetime.now(UTC).date()
    resolved_start = one_year_before(resolved_end)
    if resolved_start > resolved_end:
        raise ValueError(f"Lookback start {resolved_start} is after end date {resolved_end}")
    return resolved_start, resolved_end


def rank_grounded_candidates(
    grounded_themes: Sequence[GroundedTheme],
    *,
    end_date: date,
    model: str = DEFAULT_DISCOVERY_MODEL,
    client: OpenAI | None = None,
) -> list[ThemeCandidate]:
    """Rank grounded themes and define screener-ready parameters."""
    if not grounded_themes:
        raise ValueError("At least one grounded theme is required")
    active_client = client or OpenAI()
    prompt = """
You are an equity thematic analyst. Rank the supplied current themes for a US mid/large-cap
exposure screen. Favor themes with strong current evidence, broad company transmission
channels, and differentiated direct and indirect impacts. Do not invent facts beyond the
sources. Return JSON only with:
{"candidates": [{"main_theme": "...", "analyst_focus": "...", "rationale": "...",
"search_queries": ["..."]}]}.
Keep 1-3 document-voice search queries per candidate. Do not propose a start date; the
screen uses a separately chosen lookback window.
"""
    payload = [
        theme.model_dump(mode="json", exclude={"sources": {"__all__": {"text"}}})
        | {
            "source_excerpts": [
                {
                    "query": source.query,
                    "headline": source.headline,
                    "source_name": source.source_name,
                    "timestamp": source.timestamp,
                    "url": source.url,
                    "excerpt": source.text[:500],
                    "relevance": source.relevance,
                }
                for source in theme.sources
            ]
        }
        for theme in grounded_themes
    ]
    content = _chat_json(
        system_prompt=prompt,
        user_payload=payload,
        model=model,
        client=active_client,
    )
    return ThemeCandidateList.model_validate_json(content).candidates


def discover_theme_candidates(
    *,
    as_of_date: date | None = None,
    model: str = DEFAULT_DISCOVERY_MODEL,
    client: OpenAI | None = None,
    connection_factory: ConnectionFactory = get_snowflake_connection,
    retriever: ChunkRetriever = retrieve_theme_chunks_http,
) -> tuple[list[MacroEvent], list[GroundedTheme], list[ThemeCandidate]]:
    """Run macro discovery, Bigdata.com grounding, and candidate ranking."""
    resolved_end = as_of_date or datetime.now(UTC).date()
    events = fetch_macro_events(connection_factory)
    seeds = consolidate_macro_events(events, model=model, client=client)
    grounded = ground_theme_seeds(
        seeds,
        start_date=resolved_end - timedelta(days=DEFAULT_GROUNDING_DAYS),
        end_date=resolved_end,
        retriever=retriever,
    )
    candidates = rank_grounded_candidates(
        grounded,
        end_date=resolved_end,
        model=model,
        client=client,
    )
    return events, grounded, candidates
