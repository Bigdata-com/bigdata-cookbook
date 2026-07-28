"""Data layer for the FX Forecast Report.

Responsibilities:
    - Hold the run parameters (`ForecastParams`).
    - Pull the base/quote country tearsheets (skipping unsupported countries).
    - Parse the tearsheet economic-calendar markdown into events landing inside the
      forecast horizon (deterministic, for auditable event-risk flags).
    - Build the parameterized `bigdata_search` query set (per driver) from the inputs.

All data comes through `BigdataMCPClient` (Bigdata.com MCP).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from config.drivers import country_name, is_tearsheet_supported, resolve_weights
from src.bigdata_mcp_client import BigdataMCPClient, SearchSpec

_IMPACT_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0, "": 0}


@dataclass
class ForecastParams:
    """All inputs for a single FX forecast run (set once at the top of the notebook)."""

    base_country: str
    quote_country: str
    pair: str
    central_bank_base: str
    central_bank_quote: str
    horizon_days: int = 5
    sector_driver_terms: list[str] = field(default_factory=list)
    intervention_history: bool = False
    weight_overrides: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_country = self.base_country.upper()
        self.quote_country = self.quote_country.upper()
        self.pair = self.pair.upper()

    @property
    def base_ccy(self) -> str:
        return self.pair.split("/")[0]

    @property
    def quote_ccy(self) -> str:
        return self.pair.split("/")[1]

    @property
    def base_name(self) -> str:
        return country_name(self.base_country)

    @property
    def quote_name(self) -> str:
        return country_name(self.quote_country)

    def weights(self) -> dict[str, float]:
        """Normalized active-driver weights, applying any ad-hoc overrides."""
        merged = resolve_weights(self.pair, self.intervention_history)
        if self.weight_overrides:
            merged.update(
                {k: v for k, v in self.weight_overrides.items() if k in merged}
            )
            total = sum(merged.values())
            if total > 0:
                merged = {k: v / total for k, v in merged.items()}
        return merged


@dataclass
class CalendarEvent:
    """An economic-calendar release landing inside the forecast horizon."""

    date: datetime
    event: str
    impact: str
    consensus: str
    country: str

    @property
    def impact_rank(self) -> int:
        return _IMPACT_RANK.get(self.impact.upper(), 0)


@dataclass
class TearsheetBundle:
    """Structured output of the data-pull step."""

    base_markdown: str | None
    quote_markdown: str | None
    events: list[CalendarEvent]
    notes: list[str]

    @property
    def base_available(self) -> bool:
        return bool(self.base_markdown)

    @property
    def quote_available(self) -> bool:
        return bool(self.quote_markdown)


# ---------------------------------------------------------------------------
# Tearsheet pull
# ---------------------------------------------------------------------------
async def pull_tearsheets(
    client: BigdataMCPClient, params: ForecastParams, now: datetime | None = None
) -> TearsheetBundle:
    """Pull base/quote tearsheets and parse the in-horizon economic calendar."""
    now = now or datetime.now(timezone.utc)
    horizon_end = now + timedelta(days=params.horizon_days)

    base_md = await client.country_tearsheet(params.base_country)
    quote_md = await client.country_tearsheet(params.quote_country)

    notes: list[str] = []
    if not is_tearsheet_supported(params.base_country):
        notes.append(
            f"Base country {params.base_name} ({params.base_country}) has no country "
            f"tearsheet; base side relies on search only."
        )
    if not is_tearsheet_supported(params.quote_country):
        notes.append(
            f"Quote country {params.quote_name} ({params.quote_country}) has no country "
            f"tearsheet; quote side relies on search only."
        )

    events: list[CalendarEvent] = []
    if base_md:
        events += _parse_calendar(base_md, params.base_country, now, horizon_end)
    if quote_md:
        events += _parse_calendar(quote_md, params.quote_country, now, horizon_end)
    events.sort(key=lambda e: (e.date, -e.impact_rank))

    return TearsheetBundle(
        base_markdown=base_md, quote_markdown=quote_md, events=events, notes=notes
    )


# ---------------------------------------------------------------------------
# Calendar parsing
# ---------------------------------------------------------------------------
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})")


def _parse_calendar(
    markdown: str, country: str, now: datetime, horizon_end: datetime
) -> list[CalendarEvent]:
    """Extract upcoming-events rows from a tearsheet that fall inside the horizon."""
    section = _extract_section(markdown, ("Upcoming Events", "Economic Calendar"))
    if not section:
        return []

    events: list[CalendarEvent] = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        # Skip header and separator rows.
        if cells[0].lower() in {"date"} or set(cells[0]) <= {"-", ":", " "}:
            continue
        match = _DATE_RE.search(cells[0])
        if not match:
            continue
        try:
            when = datetime.strptime(
                f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if not (now - timedelta(hours=12) <= when <= horizon_end):
            continue
        # Columns: Date | Event | Frequency | Impact | Consensus
        event_name = cells[1]
        impact = cells[3] if len(cells) > 3 else ""
        consensus = cells[4] if len(cells) > 4 else ""
        consensus = "" if consensus in {"–", "-", "—"} else consensus
        events.append(
            CalendarEvent(
                date=when,
                event=event_name,
                impact=impact,
                consensus=consensus,
                country=country,
            )
        )
    return events


def _extract_section(markdown: str, title_keywords: tuple[str, ...]) -> str:
    """Return the body of the first `##` section whose title matches a keyword."""
    lines = markdown.splitlines()
    capture = False
    out: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if capture:
                break
            title = line[3:].strip().lower()
            capture = any(kw.lower() in title for kw in title_keywords)
            continue
        if capture:
            out.append(line)
    return "\n".join(out).strip()


def extract_sections(markdown: str | None) -> dict[str, str]:
    """Split a tearsheet into a {section_title: body} map (top-level `##` headers)."""
    if not markdown:
        return {}
    sections: dict[str, str] = {}
    current = "Preamble"
    buffer: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if buffer:
                sections[current] = "\n".join(buffer).strip()
            current = line[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    if buffer:
        sections[current] = "\n".join(buffer).strip()
    return sections


def select_sections(markdown: str | None, keywords: tuple[str, ...]) -> str:
    """Concatenate tearsheet sections whose titles match any keyword (for LLM context)."""
    sections = extract_sections(markdown)
    picked = [
        f"## {title}\n{body}"
        for title, body in sections.items()
        if any(kw.lower() in title.lower() for kw in keywords)
    ]
    return "\n\n".join(picked).strip()


# ---------------------------------------------------------------------------
# Search query construction (per driver)
# ---------------------------------------------------------------------------
def build_driver_queries(params: ForecastParams) -> list[SearchSpec]:
    """Build the parameterized search set for the non-rate drivers.

    The rate-differential driver is fed separately by `central_bank_feed`, so it is not
    included here.
    """
    quote = params.quote_name
    base = params.base_name
    pair = params.pair
    base_ccy = params.base_ccy
    quote_ccy = params.quote_ccy

    specs: list[SearchSpec] = []

    # Trade & capital flows
    specs.append(
        SearchSpec(
            text=f"{quote} trade balance, current account, and export performance outlook",
            label="trade_capital_flows",
        )
    )
    if params.sector_driver_terms:
        terms = ", ".join(params.sector_driver_terms)
        specs.append(
            SearchSpec(
                text=f"{quote} {terms} exports and external demand affecting the {quote_ccy}",
                label="trade_capital_flows",
            )
        )
    specs.append(
        SearchSpec(
            text=f"portfolio and foreign capital flows into {quote} assets and the {quote_ccy}",
            label="trade_capital_flows",
        )
    )

    # Intervention risk (only when applicable)
    if params.intervention_history:
        specs.append(
            SearchSpec(
                text=(
                    f"{params.central_bank_quote} foreign exchange intervention, "
                    f"reserves, and verbal signaling on the {quote_ccy}"
                ),
                label="intervention_risk",
            )
        )
        specs.append(
            SearchSpec(
                text=f"{quote} authorities smoothing operations to defend the {quote_ccy}",
                label="intervention_risk",
            )
        )

    # Risk sentiment / carry
    specs.append(
        SearchSpec(
            text=f"broad {base_ccy} strength and dollar index direction",
            label="risk_sentiment_carry",
        )
    )
    specs.append(
        SearchSpec(
            text=f"{pair} carry trade dynamics and global risk-on risk-off sentiment",
            label="risk_sentiment_carry",
        )
    )

    # Geopolitical
    specs.append(
        SearchSpec(
            text=f"geopolitical and trade-policy tensions between {base} and {quote}",
            label="geopolitical",
        )
    )
    specs.append(
        SearchSpec(
            text=f"{quote} political risk and policy uncertainty affecting the {quote_ccy}",
            label="geopolitical",
        )
    )

    # Technical / positioning
    specs.append(
        SearchSpec(
            text=f"{pair} technical analysis, realized volatility, and market positioning",
            label="technical_positioning",
        )
    )

    return specs
