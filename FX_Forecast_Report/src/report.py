"""Assemble the FX forecast markdown report.

Mirrors the `Rising_Bond_Spread_Risks` output pattern: executive summary, driver table
(lean / confidence / rationale / sources), risk flags (intervention, event risk landing
inside the horizon, geopolitical tail), and a source appendix for auditability.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from src.bigdata_mcp_client import EvidenceDoc
from src.central_bank_feed import CentralBankFeed
from src.data_layer import ForecastParams, TearsheetBundle
from src.llm import LLMClient
from src.scoring import Aggregate, DriverResult

_LEAN_DISPLAY: dict[str, str] = {
    "base_up": "▲ base up",
    "base_down": "▼ base down",
    "neutral": "→ neutral",
}

_EXEC_SYSTEM = (
    "You are an FX strategist writing the executive summary of a short-horizon forecast "
    "report. Write a single tight paragraph (4-6 sentences). State the directional call "
    "and conviction for the pair, name the two or three drivers doing the most work, and "
    "flag the main risk. Use only the information provided; do not invent specific "
    "numbers. Always refer to the pair by name."
)


def _lean_for_pair(lean: str, pair: str) -> str:
    label = _LEAN_DISPLAY.get(lean, lean)
    if lean == "base_up":
        return f"{label} ({pair} ↑)"
    if lean == "base_down":
        return f"{label} ({pair} ↓)"
    return f"{label} ({pair} ~)"


def _executive_summary(
    llm: LLMClient,
    params: ForecastParams,
    aggregate: Aggregate,
    results: list[DriverResult],
) -> str:
    driver_lines = "\n".join(
        f"- {r.label}: lean={r.lean}, confidence={r.confidence:.2f}, "
        f"weight={r.weight:.2f} — {r.rationale}"
        for r in results
    )
    user = (
        f"Pair: {params.pair} (base {params.base_ccy}, quote {params.quote_ccy}).\n"
        f"Horizon: {params.horizon_days} days.\n"
        f"Overall call: {aggregate.direction_text}.\n"
        f"Conviction: {aggregate.conviction_label} "
        f"(net score {aggregate.net_score:+.2f}).\n\n"
        f"Driver reads:\n{driver_lines}\n\n"
        f"Write the executive summary paragraph."
    )
    try:
        return llm.complete_text(_EXEC_SYSTEM, user, temperature=0.3)
    except Exception:  # noqa: BLE001 - narrative is best-effort
        return (
            f"Over the next {params.horizon_days} days the base case for {params.pair} is "
            f"{aggregate.direction_text}, with {aggregate.conviction_label.lower()} "
            f"conviction (net score {aggregate.net_score:+.2f})."
        )


def _driver_table(params: ForecastParams, results: list[DriverResult]) -> str:
    header = (
        "| Driver | Lean | Confidence | Weight | Rationale | Sources |\n"
        "|---|---|---|---|---|---|"
    )
    rows = []
    for r in results:
        rationale = r.rationale.replace("|", "/").replace("\n", " ")
        src = f"{len(r.sources)} doc(s)" if r.sources else "—"
        rows.append(
            f"| {r.label} | {_lean_for_pair(r.lean, params.pair)} | "
            f"{r.confidence:.2f} | {r.weight:.0%} | {rationale} | {src} |"
        )
    return "\n".join([header, *rows])


def _event_risk_table(bundle: TearsheetBundle) -> str:
    material = [e for e in bundle.events if e.impact.upper() in {"HIGH", "MEDIUM"}]
    if not material:
        return "_No high- or medium-impact economic releases land inside the horizon._"
    header = (
        "| Date (UTC) | Country | Impact | Event | Consensus |\n"
        "|---|---|---|---|---|"
    )
    rows = [
        f"| {e.date.strftime('%Y-%m-%d %H:%M')} | {e.country} | {e.impact.upper()} | "
        f"{e.event.replace('|', '/')} | {e.consensus or '—'} |"
        for e in material
    ]
    return "\n".join([header, *rows])


def _risk_flags(
    params: ForecastParams,
    bundle: TearsheetBundle,
    results: list[DriverResult],
) -> str:
    by_key = {r.key: r for r in results}
    parts: list[str] = []

    if params.intervention_history:
        interv = by_key.get("intervention_risk")
        if interv:
            parts.append(
                f"- **Intervention risk** ({params.central_bank_quote}): "
                f"{interv.rationale}"
            )
    else:
        parts.append(
            "- **Intervention risk**: not flagged for this pair "
            "(`intervention_history=False`)."
        )

    geo = by_key.get("geopolitical")
    if geo:
        parts.append(f"- **Geopolitical tail risk**: {geo.rationale}")

    parts.append("- **Event risk (releases inside the horizon):**")
    parts.append(_event_risk_table(bundle))
    return "\n".join(parts)


def _collect_sources(
    results: list[DriverResult], cb_feed: CentralBankFeed
) -> list[EvidenceDoc]:
    all_docs = [d for r in results for d in r.sources] + list(cb_feed.evidence)
    seen: set[str] = set()
    unique: list[EvidenceDoc] = []
    for doc in all_docs:
        key = doc.doc_id or doc.url or f"{doc.source}|{doc.headline}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    unique.sort(key=lambda d: d.timestamp, reverse=True)
    return unique


def _appendix(results: list[DriverResult], cb_feed: CentralBankFeed) -> str:
    docs = _collect_sources(results, cb_feed)
    if not docs:
        return "_No sources retrieved._"
    lines = []
    for i, d in enumerate(docs, start=1):
        date = d.timestamp[:10] or "n/a"
        if d.url:
            lines.append(f"{i}. [{d.headline}]({d.url}) — {d.source} ({date})")
        else:
            lines.append(f"{i}. {d.headline} — {d.source} ({date})")
    return "\n".join(lines)


def assemble_report(
    params: ForecastParams,
    bundle: TearsheetBundle,
    cb_feed: CentralBankFeed,
    results: list[DriverResult],
    aggregate: Aggregate,
    llm: LLMClient,
    now: datetime | None = None,
) -> str:
    """Build the full markdown report."""
    now = now or datetime.now(timezone.utc)
    summary = _executive_summary(llm, params, aggregate, results)

    coverage = ""
    if bundle.notes:
        coverage = "\n\n> **Data coverage:** " + " ".join(bundle.notes)

    sector = (
        f" Sector focus: {', '.join(params.sector_driver_terms)}."
        if params.sector_driver_terms
        else ""
    )

    return f"""# {params.pair} — {params.horizon_days}-Day FX Forecast

*Generated {now.strftime('%Y-%m-%d %H:%M UTC')} · Base: {params.base_name} \
({params.central_bank_base}) · Quote: {params.quote_name} ({params.central_bank_quote})*{sector}

## Executive Summary

{summary}

**Overall call:** {aggregate.direction_text}
**Conviction:** {aggregate.conviction_label} (net score {aggregate.net_score:+.2f}){coverage}

## Driver Table

{_driver_table(params, results)}

_Lean is expressed for the base currency: "base up" means {params.base_ccy} strengthens \
against {params.quote_ccy} (i.e. {params.pair} rises)._

## Risk Flags

{_risk_flags(params, bundle, results)}

## Appendix — Sources

{_appendix(results, cb_feed)}

---
*Data: Bigdata.com (country tearsheets + news/research search via the remote MCP \
server). Synthesis: OpenAI. This report is a research aid, not investment advice.*
"""


def save_report(markdown: str, params: ForecastParams, output_dir: str | Path) -> Path:
    """Write the report to output/<PAIR>_<date>_fx_forecast.md and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_pair = re.sub(r"[^A-Za-z0-9]+", "", params.pair)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = output_dir / f"{safe_pair}_{stamp}_fx_forecast.md"
    path.write_text(markdown, encoding="utf-8")
    return path
