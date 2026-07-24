"""Driver taxonomy, country coverage, and scoring weights for the FX Forecast Report.

Everything here is generic across currency pairs. The *content* under each driver is
populated per-pair at runtime from the notebook parameters (base/quote country,
central-bank names, sector driver terms, intervention flag). Only the taxonomy and the
default weights live here.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Country tearsheet coverage
# ---------------------------------------------------------------------------
# `bigdata_country_tearsheet` accepts ONLY these 2-letter codes. A country outside
# this set (e.g. Taiwan / "TW") has no structured tearsheet, so its side of the pair
# is built from `bigdata_search` alone (graceful degradation).
SUPPORTED_TEARSHEET_COUNTRIES: Final[frozenset[str]] = frozenset(
    {
        "AE", "AR", "AT", "AU", "BE", "BR", "CA", "CH", "CL", "CN", "CO", "CZ",
        "DE", "DK", "EG", "EMU", "ES", "FI", "FR", "GR", "HK", "HU", "ID", "IE",
        "IL", "IN", "IS", "IT", "JP", "KR", "KW", "MX", "NL", "NO", "NZ", "PL",
        "PT", "QA", "RO", "RU", "SA", "SE", "SG", "SK", "TH", "TR", "UK", "US",
        "ZA",
    }
)

# Human-readable names used in prompts and the report. Codes not listed fall back to
# the raw code. Only the codes likely to appear in FX pairs are enumerated.
COUNTRY_NAMES: Final[dict[str, str]] = {
    "US": "United States",
    "JP": "Japan",
    "UK": "United Kingdom",
    "EMU": "Eurozone",
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "CA": "Canada",
    "AU": "Australia",
    "NZ": "New Zealand",
    "CH": "Switzerland",
    "CN": "China",
    "KR": "South Korea",
    "HK": "Hong Kong",
    "SG": "Singapore",
    "IN": "India",
    "ID": "Indonesia",
    "TH": "Thailand",
    "BR": "Brazil",
    "MX": "Mexico",
    "ZA": "South Africa",
    "TR": "Turkey",
    "NO": "Norway",
    "SE": "Sweden",
    "PL": "Poland",
    "RU": "Russia",
    "TW": "Taiwan",  # not tearsheet-supported; search-only
}


def country_name(code: str) -> str:
    """Return the display name for a country code (falls back to the code itself)."""
    return COUNTRY_NAMES.get(code.upper(), code.upper())


def is_tearsheet_supported(code: str) -> bool:
    """True when `bigdata_country_tearsheet` covers this country code."""
    return code.upper() in SUPPORTED_TEARSHEET_COUNTRIES


# ---------------------------------------------------------------------------
# Driver taxonomy
# ---------------------------------------------------------------------------
# `key`          — stable identifier used across modules and the weights table
# `label`        — display name in the report
# `description`  — what the driver captures (also fed to the scoring LLM)
# `intervention` — driver only activated when intervention_history is True
DRIVER_CATEGORIES: Final[list[dict[str, object]]] = [
    {
        "key": "rate_differential",
        "label": "Rate Differential",
        "description": (
            "Relative monetary-policy stance and forward guidance of the two central "
            "banks, the real and nominal yield gap, and rate-decision expectations. A "
            "wider rate advantage for the base currency is base-supportive."
        ),
        "intervention": False,
    },
    {
        "key": "trade_capital_flows",
        "label": "Trade & Capital Flows",
        "description": (
            "Trade balance, export performance (including any pair-specific sector "
            "export terms), current account, and portfolio/FDI flows into or out of the "
            "quote country's assets."
        ),
        "intervention": False,
    },
    {
        "key": "intervention_risk",
        "label": "Intervention Risk",
        "description": (
            "Central-bank FX smoothing operations, verbal intervention, and FX-reserve "
            "dynamics. Only relevant when a central bank in the pair has a history of "
            "intervening to manage its currency."
        ),
        "intervention": True,
    },
    {
        "key": "risk_sentiment_carry",
        "label": "Risk Sentiment / Carry",
        "description": (
            "Broad base-currency strength (e.g. the US dollar index), global and "
            "regional risk-on/risk-off, and carry-trade dynamics relevant to the pair."
        ),
        "intervention": False,
    },
    {
        "key": "geopolitical",
        "label": "Geopolitical",
        "description": (
            "Bilateral or regional political and trade-policy tensions specific to the "
            "base and quote countries that could move the pair."
        ),
        "intervention": False,
    },
    {
        "key": "technical_positioning",
        "label": "Technical / Positioning",
        "description": (
            "Recent realized volatility, trend and momentum (moving-average signals, "
            "distance to 52-week extremes), and positioning extremes where available."
        ),
        "intervention": False,
    },
]

DRIVER_KEYS: Final[tuple[str, ...]] = tuple(str(d["key"]) for d in DRIVER_CATEGORIES)


# ---------------------------------------------------------------------------
# Aggregation weights
# ---------------------------------------------------------------------------
# Default relative importance of each driver. Weights are normalized at runtime over
# the *active* drivers (intervention_risk is dropped when intervention_history is
# False), so they need not sum to 1 here.
DEFAULT_WEIGHTS: Final[dict[str, float]] = {
    "rate_differential": 0.30,
    "trade_capital_flows": 0.20,
    "intervention_risk": 0.10,
    "risk_sentiment_carry": 0.20,
    "geopolitical": 0.10,
    "technical_positioning": 0.10,
}

# Per-pair weight overrides. Keys are "BASE/QUOTE" tickers. Only the drivers you want
# to change need to be listed; unlisted drivers keep their DEFAULT_WEIGHTS value.
# - Export-driven currencies (e.g. TWD, KRW): rate differential + trade flows dominate.
# - Intervention-prone / carry pairs (e.g. JPY): rate differential + intervention +
#   carry matter more.
PAIR_WEIGHT_OVERRIDES: Final[dict[str, dict[str, float]]] = {
    "USD/JPY": {
        "rate_differential": 0.35,
        "intervention_risk": 0.20,
        "risk_sentiment_carry": 0.20,
        "trade_capital_flows": 0.10,
        "geopolitical": 0.05,
        "technical_positioning": 0.10,
    },
    "USD/TWD": {
        "rate_differential": 0.25,
        "trade_capital_flows": 0.30,
        "intervention_risk": 0.15,
        "risk_sentiment_carry": 0.15,
        "geopolitical": 0.10,
        "technical_positioning": 0.05,
    },
}


def resolve_weights(pair: str, intervention_history: bool) -> dict[str, float]:
    """Return normalized weights for the active drivers of a pair.

    Merges DEFAULT_WEIGHTS with any per-pair override, drops the intervention driver
    when it is not applicable, and normalizes the remainder to sum to 1.0.
    """
    merged = dict(DEFAULT_WEIGHTS)
    merged.update(PAIR_WEIGHT_OVERRIDES.get(pair.upper(), {}))

    active = {
        key: weight
        for key, weight in merged.items()
        if intervention_history or key != "intervention_risk"
    }

    total = sum(active.values())
    if total <= 0:
        # Degenerate config: fall back to equal weights.
        equal = 1.0 / len(active)
        return {key: equal for key in active}
    return {key: weight / total for key, weight in active.items()}


def active_drivers(intervention_history: bool) -> list[dict[str, object]]:
    """Return the driver definitions active for a run."""
    return [
        driver
        for driver in DRIVER_CATEGORIES
        if intervention_history or not driver["intervention"]
    ]
