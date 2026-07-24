"""Central-bank sentiment feed for the rate-differential driver.

Adapted from the `Daily_Digest_Central_Banks` pattern, but parameterized on the two
central banks of the pair and generating the monetary-policy lexicon dynamically
(instead of a hardcoded Fed/ECB term set). The evidence gathered here feeds the
rate-differential driver in `scoring.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from src.bigdata_mcp_client import BigdataMCPClient, EvidenceDoc, SearchSpec
from src.data_layer import ForecastParams
from src.llm import LLMClient


class SearchPhrases(BaseModel):
    """Structured LLM output: natural-language search phrases for one central bank."""

    phrases: list[str] = Field(default_factory=list)


@dataclass
class CentralBankFeed:
    """Result of the central-bank evidence gathering step."""

    base_terms: list[str]
    quote_terms: list[str]
    evidence: list[EvidenceDoc] = field(default_factory=list)


_LEXICON_SYSTEM = (
    "You are an FX macro strategist. You craft concise, natural-language search "
    "queries for a financial news and research search engine. Each query is a full "
    "phrase (not a keyword bag) that retrieves the latest monetary-policy signals."
)


def generate_lexicon(
    llm: LLMClient, central_bank: str, currency: str, country: str, n: int = 5
) -> list[str]:
    """Generate `n` search phrases capturing a central bank's latest policy signals."""
    user = (
        f"Generate {n} distinct search queries to retrieve the most recent monetary "
        f"policy signals from the {central_bank} ({country}) that matter for the "
        f"{currency}. Across the {n} queries, cover: (1) the latest policy-rate "
        f"decision and near-term rate expectations, (2) forward guidance and tone "
        f"(hawkish vs dovish), (3) the inflation trajectory versus target, and "
        f"(4) the next policy meeting or officials' recent commentary. Each query "
        f"must be a complete natural-language phrase suitable for semantic news search."
    )
    result = llm.complete_structured(_LEXICON_SYSTEM, user, SearchPhrases)
    phrases = [p.strip() for p in result.phrases if p.strip()]
    return phrases[:n] or [f"{central_bank} latest interest rate decision and guidance"]


def _dedupe(docs: list[EvidenceDoc], max_docs: int) -> list[EvidenceDoc]:
    """Drop duplicate documents (by id) and keep the most recent `max_docs`."""
    seen: set[str] = set()
    unique: list[EvidenceDoc] = []
    for doc in docs:
        key = doc.doc_id or f"{doc.source}|{doc.headline}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    unique.sort(key=lambda d: d.timestamp, reverse=True)
    return unique[:max_docs]


async def gather_central_bank_evidence(
    client: BigdataMCPClient,
    llm: LLMClient,
    params: ForecastParams,
    per_bank: int = 5,
    max_docs: int = 18,
    max_chunks: int = 8,
) -> CentralBankFeed:
    """Build lexicons for both central banks, search in parallel, and collect evidence."""
    base_terms = generate_lexicon(
        llm, params.central_bank_base, params.base_ccy, params.base_name, n=per_bank
    )
    quote_terms = generate_lexicon(
        llm, params.central_bank_quote, params.quote_ccy, params.quote_name, n=per_bank
    )

    specs = [
        SearchSpec(text=term, label="rate_differential", max_chunks=max_chunks)
        for term in (*base_terms, *quote_terms)
    ]
    grouped = await client.search_many(specs)
    flat = [doc for group in grouped for doc in group]

    return CentralBankFeed(
        base_terms=base_terms,
        quote_terms=quote_terms,
        evidence=_dedupe(flat, max_docs),
    )
