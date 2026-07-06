"""Core pipeline functions for the morning brief.

Pure building blocks with no CLI parsing or run-directory logic.
Orchestration and persistence are handled by :mod:`src.cli` together with
:class:`src.run_context.RunContext`.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from bigdata_smart_batching import (
    deduplicate_documents,
    execute_search,
    load_plan,
    plan_search,
    save_plan,
)
from pydantic import BaseModel

from src.openai_parallel import (
    ChatRequest,
    RateLimitConfig,
    run_chat_requests_parallel,
)
from src.topics import Topic

logger = logging.getLogger(__name__)

PORTFOLIO_ID_COLUMN = "RP_COMPANY_ID"
PORTFOLIO_NAME_COLUMN = "COMPANY_NAME"
PORTFOLIO_TICKER_COLUMN = "TICKER"

DEFAULT_START_DATE = "2026-05-30"
DEFAULT_END_DATE = "2026-06-29"
DEFAULT_CHUNK_PERCENTAGE = 0.05
DEFAULT_REQUESTS_PER_MINUTE = 350
DEFAULT_SUMMARY_MODEL = "gpt-4.1-nano"
MAX_CHUNKS_PER_SECTION = 12
MAX_CHUNK_TEXT_CHARS = 500
MAX_PORTFOLIO_SIZE = 50

DEFAULT_SEARCH_CATEGORY: dict[str, Any] = {
    "mode": "INCLUDE",
    "values": ["news_premium", "transcripts", "filings"],
}


# ──────────────────────────────────────────────── data models ─────────────────


@dataclass(frozen=True)
class PortfolioCompany:
    company_id: str
    name: str
    ticker: str


@dataclass(frozen=True)
class Source:
    index: int
    url: str
    headline: str
    source_name: str
    timestamp: str


@dataclass
class CompanySection:
    topic_id: str
    topic_label: str
    summary: str
    cited_indices: list[int] = field(default_factory=list)


@dataclass
class CompanyBrief:
    company: PortfolioCompany
    sections: dict[str, CompanySection]


@dataclass
class BriefData:
    brief_date: str
    generated_at: str
    companies: list[CompanyBrief]
    sources: list[Source]
    topics: list[Topic]


class _TopicSummaryResponse(BaseModel):
    summary: str
    cited_indices: list[int]


# ──────────────────────────────────────────────── portfolio ───────────────────


def load_portfolio(portfolio_path: str | Path) -> list[PortfolioCompany]:
    """Load companies from a CSV with RP_COMPANY_ID and COMPANY_NAME columns.

    An optional TICKER column is used for display if present.
    """
    df = pd.read_csv(portfolio_path)
    missing = {PORTFOLIO_ID_COLUMN, PORTFOLIO_NAME_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"portfolio CSV is missing required columns: {sorted(missing)}")

    companies: list[PortfolioCompany] = []
    has_ticker = PORTFOLIO_TICKER_COLUMN in df.columns
    for row in df.itertuples(index=False):
        ticker = ""
        if has_ticker:
            raw = getattr(row, PORTFOLIO_TICKER_COLUMN, None)
            if raw is not None and not (isinstance(raw, float) and math.isnan(raw)):
                ticker = str(raw).strip()
        companies.append(PortfolioCompany(
            company_id=str(getattr(row, PORTFOLIO_ID_COLUMN)).strip(),
            name=str(getattr(row, PORTFOLIO_NAME_COLUMN)).strip(),
            ticker=ticker,
        ))

    if len(companies) > MAX_PORTFOLIO_SIZE:
        logger.warning(
            "Portfolio has %d companies; only the first %d will be used.",
            len(companies),
            MAX_PORTFOLIO_SIZE,
        )
        companies = companies[:MAX_PORTFOLIO_SIZE]

    return companies


# ──────────────────────────────────────────────── plan ────────────────────────


def build_plans(
    topics: list[Topic],
    company_ids: list[str],
    plans_dir: Path,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> list[Path]:
    """Create and persist one smart-batching search plan per topic."""
    plans_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for topic in topics:
        plan = plan_search(
            universe=company_ids,
            start_date=start_date,
            end_date=end_date,
            volume_query_mode="iterative",
            text=topic.search_query,
            category=DEFAULT_SEARCH_CATEGORY,
        )
        plan_path = plans_dir / f"{topic.id}.json"
        save_plan(plan, str(plan_path))
        saved.append(plan_path)
        baskets = plan.get("baskets", [])
        chunks = sum(int(b.get("expected_chunks", 0) or 0) for b in baskets)
        logger.info(
            "Plan '%s': %d baskets, ~%d expected chunks -> %s",
            topic.label,
            len(baskets),
            chunks,
            plan_path.name,
        )
    return saved


def summarize_plans(topics: list[Topic], plans_dir: Path) -> pd.DataFrame:
    """Return a DataFrame of per-topic chunk estimates from saved plan files."""
    rows: list[dict[str, Any]] = []
    for topic in topics:
        plan_path = plans_dir / f"{topic.id}.json"
        if not plan_path.exists():
            continue
        plan = load_plan(str(plan_path))
        baskets = plan.get("baskets", [])
        chunks = sum(int(b.get("expected_chunks", 0) or 0) for b in baskets)
        rows.append({"topic_id": topic.id, "topic": topic.label, "chunks": chunks})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("chunks", ascending=False).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────── search ─────────────────────


def run_search_per_topic(
    topics: list[Topic],
    plans_dir: Path,
    results_dir: Path,
    chunk_percentage: float = DEFAULT_CHUNK_PERCENTAGE,
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
) -> dict[str, list[dict[str, Any]]]:
    """Execute one search plan per topic; return and persist per-topic results."""
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict[str, Any]]] = {}

    for topic in topics:
        plan_path = plans_dir / f"{topic.id}.json"
        if not plan_path.exists():
            raise FileNotFoundError(
                f"plan not found for topic '{topic.id}' at {plan_path}; run 'plan' step first"
            )
        plan = load_plan(str(plan_path))
        raw = execute_search(
            search_plan=plan,
            chunk_percentage=chunk_percentage,
            requests_per_minute=requests_per_minute,
            basket_filtered_entities=True,
        )
        deduped = deduplicate_documents(raw)
        all_results[topic.id] = deduped

        out_path = results_dir / f"{topic.id}.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(deduped, handle, default=str)
        logger.info(
            "Topic '%s': %d documents stored -> %s",
            topic.label,
            len(deduped),
            out_path.name,
        )

    return all_results


def load_topic_results(
    topics: list[Topic], results_dir: Path
) -> dict[str, list[dict[str, Any]]]:
    """Load per-topic results from disk."""
    all_results: dict[str, list[dict[str, Any]]] = {}
    for topic in topics:
        path = results_dir / f"{topic.id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"results not found for topic '{topic.id}' at {path}; run 'search' step first"
            )
        with path.open(encoding="utf-8") as handle:
            all_results[topic.id] = json.load(handle)
    return all_results


# ──────────────────────────────────────────────── compile ────────────────────


def _extract_company_chunks(
    documents: list[dict[str, Any]],
    company_id: str,
) -> list[dict[str, Any]]:
    """Return chunks where company_id appears in entity_ids, sorted by relevance desc."""
    chunks: list[dict[str, Any]] = []
    for doc in documents:
        doc_id = doc.get("id", "")
        for chunk in doc.get("chunks", []):
            if company_id in set(chunk.get("entity_ids", [])):
                chunks.append({
                    "text": chunk.get("text", ""),
                    "relevance": float(chunk.get("relevance", 0.0)),
                    "doc_id": doc_id,
                })
    return sorted(chunks, key=lambda c: c["relevance"], reverse=True)


def _build_source_index(
    topic_results: dict[str, list[dict[str, Any]]],
) -> tuple[list[Source], dict[str, int]]:
    """Build a deduplicated source list and a doc_id → source_index map."""
    url_to_index: dict[str, int] = {}
    doc_id_to_index: dict[str, int] = {}
    sources: list[Source] = []

    for documents in topic_results.values():
        for doc in documents:
            url = (doc.get("url") or "").strip()
            doc_id = (doc.get("id") or "").strip()
            if not url:
                continue
            if url not in url_to_index:
                idx = len(sources) + 1
                url_to_index[url] = idx
                raw_source = doc.get("source") or {}
                sources.append(Source(
                    index=idx,
                    url=url,
                    headline=(doc.get("headline") or "").strip(),
                    source_name=(raw_source.get("name") or "").strip(),
                    timestamp=(doc.get("timestamp") or "").strip(),
                ))
            if doc_id:
                doc_id_to_index[doc_id] = url_to_index[url]

    return sources, doc_id_to_index


def _build_section_prompt(
    company: PortfolioCompany,
    topic: Topic,
    chunks: list[dict[str, Any]],
    sources: list[Source],
    doc_id_to_index: dict[str, int],
) -> str:
    top_chunks = chunks[:MAX_CHUNKS_PER_SECTION]

    source_map: dict[int, Source] = {}
    for chunk in top_chunks:
        idx = doc_id_to_index.get(chunk["doc_id"])
        if idx:
            src = next((s for s in sources if s.index == idx), None)
            if src:
                source_map[idx] = src

    source_lines = []
    for s in sorted(source_map.values(), key=lambda x: x.index):
        ts = s.timestamp[:10] if s.timestamp else ""
        prefix = f"{s.source_name}, " if s.source_name else ""
        source_lines.append(f"[{s.index}] {s.headline or s.url} ({prefix}{ts})")
    source_context = "\n".join(source_lines) if source_lines else "No source metadata available."

    excerpts = "\n\n".join(
        f"[{doc_id_to_index.get(c['doc_id'], '?')}] {c['text'][:MAX_CHUNK_TEXT_CHARS]}"
        for c in top_chunks
    )

    ticker_str = f" ({company.ticker})" if company.ticker else ""
    no_info_rule = (
        f"- If the excerpts contain no material information relevant to {topic.label},"
        " return an empty summary string"
    )
    json_schema = (
        '{"summary": "<prose with [N] inline citations>",'
        ' "cited_indices": [<list of integer N values cited>]}'
    )
    return (
        f"You are a professional equity research analyst writing an institutional morning brief.\n"
        f"\nCompany: {company.name}{ticker_str}"
        f"\nSection: {topic.label}"
        f"\n\nAvailable sources — cite inline with [N] markers for specific, verifiable claims:"
        f"\n{source_context}"
        f"\n\nRelevant excerpts (prefixed by source number):"
        f"\n{excerpts}"
        f"\n\nWrite a concise 2–3 sentence summary for the **{topic.label}** section. Rules:"
        f"\n- Use [N] inline citation markers tied to the source list above"
        f"\n- Prioritise the most material, recent developments"
        f"\n- Write in flowing professional prose; no bullet points"
        f"\n- Do not open with the company name or ticker"
        f"\n{no_info_rule}"
        f"\n\nReturn valid JSON only — no markdown, no commentary:"
        f"\n{json_schema}"
    )


def compile_brief(
    topics: list[Topic],
    portfolio: list[PortfolioCompany],
    topic_results: dict[str, list[dict[str, Any]]],
    brief_date: str,
    generated_at: str,
    model: str = DEFAULT_SUMMARY_MODEL,
    requests_per_minute: int = 10000,
    max_concurrent_requests: int = 30,
) -> BriefData:
    """Generate per-company, per-topic summaries and assemble the final BriefData."""
    sources, doc_id_to_index = _build_source_index(topic_results)

    requests: list[ChatRequest] = []
    for company in portfolio:
        for topic in topics:
            chunks = _extract_company_chunks(topic_results.get(topic.id, []), company.company_id)
            if not chunks:
                continue
            prompt = _build_section_prompt(company, topic, chunks, sources, doc_id_to_index)
            requests.append(ChatRequest(
                request_id=f"{company.company_id}|{topic.id}",
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.0,
                seed=42,
                response_format={"type": "json_object"},
            ))

    logger.info(
        "Dispatching %d LLM requests (%d companies × %d topics, skipping empty)",
        len(requests),
        len(portfolio),
        len(topics),
    )
    responses = run_chat_requests_parallel(
        requests,
        rate_limit=RateLimitConfig(
            requests_per_minute=requests_per_minute,
            max_concurrent_requests=max_concurrent_requests,
        ),
        max_retries=5,
        progress_desc="Summarising sections",
    )

    sections_map: dict[str, dict[str, CompanySection]] = {}
    for response in responses:
        if not response.succeeded:
            logger.warning("Section '%s' failed: %s", response.request_id, response.error)
            continue
        try:
            payload = json.loads(response.content)
            parsed = _TopicSummaryResponse.model_validate(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Parse error for '%s': %s", response.request_id, exc)
            continue
        if not parsed.summary.strip():
            continue
        company_id, topic_id = response.request_id.split("|", 1)
        topic_obj = next((t for t in topics if t.id == topic_id), None)
        if topic_obj is None:
            continue
        sections_map.setdefault(company_id, {})[topic_id] = CompanySection(
            topic_id=topic_id,
            topic_label=topic_obj.label,
            summary=parsed.summary,
            cited_indices=parsed.cited_indices,
        )

    company_briefs = [
        CompanyBrief(company=c, sections=sections_map.get(c.company_id, {}))
        for c in portfolio
    ]
    return BriefData(
        brief_date=brief_date,
        generated_at=generated_at,
        companies=company_briefs,
        sources=sources,
        topics=topics,
    )
