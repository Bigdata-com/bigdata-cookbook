"""Per-driver LLM scoring and weighted aggregation into an overall directional call.

Adapted from the `Rising_Bond_Spread_Risks` sub-scenario scoring pattern: each driver
gets a directional lean, a confidence score, a one-line rationale, and attributed
sources; the drivers are then combined with configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from config.drivers import active_drivers
from src.bigdata_mcp_client import EvidenceDoc
from src.data_layer import ForecastParams, select_sections
from src.llm import LLMClient

Lean = Literal["base_up", "base_down", "neutral"]

_LEAN_SIGN: dict[str, int] = {"base_up": 1, "base_down": -1, "neutral": 0}

# Tearsheet sections most relevant to each driver (title keyword match).
_DRIVER_SECTIONS: dict[str, tuple[str, ...]] = {
    "rate_differential": ("Treasury Yields", "Country Comparison", "Central Bank"),
    "trade_capital_flows": ("Macroeconomic Overview", "Country Comparison"),
    "intervention_risk": ("Currency",),
    "risk_sentiment_carry": ("Currency", "Market Indices"),
    "geopolitical": (),
    "technical_positioning": ("Currency", "Market Indices"),
}

_MAX_EVIDENCE_PER_DRIVER = 10
_EVIDENCE_SNIPPET_CHARS = 600
_SECTION_CONTEXT_CHARS = 4000


class DriverScoreLLM(BaseModel):
    """Structured LLM output for a single driver."""

    lean: Lean = Field(description="Directional lean for the base currency vs the quote.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence 0-1 from evidence volume, recency, agreement."
    )
    rationale: str = Field(description="One-sentence rationale with the key driver logic.")
    evidence_indices: list[int] = Field(
        default_factory=list,
        description="Indices of the numbered evidence items relied upon (may be empty).",
    )


@dataclass
class DriverResult:
    """A scored driver with its attributed sources and weight."""

    key: str
    label: str
    lean: str
    confidence: float
    rationale: str
    weight: float
    sources: list[EvidenceDoc] = field(default_factory=list)
    evidence_count: int = 0

    @property
    def signed_score(self) -> float:
        return _LEAN_SIGN.get(self.lean, 0) * self.confidence

    @property
    def contribution(self) -> float:
        return self.signed_score * self.weight


@dataclass
class Aggregate:
    """Overall directional call for the horizon."""

    net_score: float
    overall_lean: str
    direction_text: str
    conviction_label: str
    conviction_score: float


_SCORE_SYSTEM = (
    "You are a senior FX strategist producing a short-horizon directional read for a "
    "currency pair. You reason strictly from the structured tearsheet context and the "
    "numbered news/research evidence provided. Convention: 'base_up' means the BASE "
    "currency APPRECIATES against the quote (the pair {pair} RISES); 'base_down' means "
    "the base currency depreciates (the pair FALLS); 'neutral' means no clear lean. "
    "Set confidence from the amount, recency, and agreement of the evidence: little or "
    "conflicting evidence => low confidence."
)


def _format_evidence(docs: list[EvidenceDoc]) -> str:
    lines: list[str] = []
    for i, doc in enumerate(docs):
        snippet = doc.text[:_EVIDENCE_SNIPPET_CHARS].replace("\n", " ").strip()
        date = doc.timestamp[:10]
        lines.append(f"[{i}] ({date}) {doc.source} — {doc.headline}\n    {snippet}")
    return "\n".join(lines) if lines else "(no news/research evidence retrieved)"


def _driver_tearsheet_context(
    driver_key: str, params: ForecastParams, base_md: str | None, quote_md: str | None
) -> str:
    keywords = _DRIVER_SECTIONS.get(driver_key, ())
    if not keywords:
        return ""
    parts: list[str] = []
    base_ctx = select_sections(base_md, keywords)
    quote_ctx = select_sections(quote_md, keywords)
    if base_ctx:
        parts.append(
            f"### Base — {params.base_name} tearsheet\n{base_ctx[:_SECTION_CONTEXT_CHARS]}"
        )
    if quote_ctx:
        parts.append(
            f"### Quote — {params.quote_name} tearsheet\n{quote_ctx[:_SECTION_CONTEXT_CHARS]}"
        )
    return "\n\n".join(parts)


def score_driver(
    llm: LLMClient,
    params: ForecastParams,
    driver: dict[str, object],
    base_md: str | None,
    quote_md: str | None,
    evidence: list[EvidenceDoc],
    weight: float,
) -> DriverResult:
    """Score one driver from tearsheet context + retrieved evidence."""
    docs = evidence[:_MAX_EVIDENCE_PER_DRIVER]
    tearsheet_ctx = _driver_tearsheet_context(
        str(driver["key"]), params, base_md, quote_md
    )

    user = (
        f"Currency pair: {params.pair} (base {params.base_ccy} / quote {params.quote_ccy}).\n"
        f"Forecast horizon: {params.horizon_days} calendar days.\n\n"
        f"DRIVER: {driver['label']}\n{driver['description']}\n\n"
        f"STRUCTURED TEARSHEET CONTEXT:\n{tearsheet_ctx or '(none for this driver)'}\n\n"
        f"NUMBERED EVIDENCE:\n{_format_evidence(docs)}\n\n"
        f"Assess how this driver alone leans {params.pair} over the horizon. Return the "
        f"lean, a confidence 0-1, a one-sentence rationale, and the indices of the "
        f"evidence items you used."
    )
    system = _SCORE_SYSTEM.format(pair=params.pair)
    raw = llm.complete_structured(system, user, DriverScoreLLM)

    used = [docs[i] for i in raw.evidence_indices if 0 <= i < len(docs)]
    return DriverResult(
        key=str(driver["key"]),
        label=str(driver["label"]),
        lean=raw.lean,
        confidence=max(0.0, min(1.0, raw.confidence)),
        rationale=raw.rationale.strip(),
        weight=weight,
        sources=used or docs[:3],
        evidence_count=len(docs),
    )


def group_evidence_by_label(
    grouped_results: list[list[EvidenceDoc]],
) -> dict[str, list[EvidenceDoc]]:
    """Flatten per-query search results into a {driver_key: [docs]} map by label."""
    by_label: dict[str, list[EvidenceDoc]] = {}
    seen: dict[str, set[str]] = {}
    for group in grouped_results:
        for doc in group:
            label = doc.label or "unlabeled"
            by_label.setdefault(label, [])
            seen.setdefault(label, set())
            key = doc.doc_id or f"{doc.source}|{doc.headline}"
            if key in seen[label]:
                continue
            seen[label].add(key)
            by_label[label].append(doc)
    for docs in by_label.values():
        docs.sort(key=lambda d: d.timestamp, reverse=True)
    return by_label


def score_all_drivers(
    llm: LLMClient,
    params: ForecastParams,
    base_md: str | None,
    quote_md: str | None,
    evidence_by_driver: dict[str, list[EvidenceDoc]],
) -> list[DriverResult]:
    """Score every active driver for the pair."""
    weights = params.weights()
    results: list[DriverResult] = []
    for driver in active_drivers(params.intervention_history):
        key = str(driver["key"])
        results.append(
            score_driver(
                llm,
                params,
                driver,
                base_md,
                quote_md,
                evidence_by_driver.get(key, []),
                weights.get(key, 0.0),
            )
        )
    return results


def aggregate(params: ForecastParams, results: list[DriverResult]) -> Aggregate:
    """Combine driver results into an overall directional call and conviction."""
    net = sum(r.contribution for r in results)

    if net > 0.15:
        overall = "base_up"
        direction = (
            f"{params.base_ccy} appreciates vs {params.quote_ccy} "
            f"({params.pair} rises)"
        )
    elif net < -0.15:
        overall = "base_down"
        direction = (
            f"{params.base_ccy} depreciates vs {params.quote_ccy} "
            f"({params.pair} falls)"
        )
    else:
        overall = "neutral"
        direction = f"{params.pair} range-bound / no clear directional edge"

    magnitude = abs(net)
    if magnitude >= 0.5:
        conviction = "High"
    elif magnitude >= 0.25:
        conviction = "Medium"
    else:
        conviction = "Low"

    return Aggregate(
        net_score=round(net, 4),
        overall_lean=overall,
        direction_text=direction,
        conviction_label=conviction,
        conviction_score=round(magnitude, 4),
    )
