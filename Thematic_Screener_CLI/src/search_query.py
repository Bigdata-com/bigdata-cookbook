"""Heuristics for document-voice Bigdata search queries."""

from __future__ import annotations

import re

EXPOSURE_META_PATTERN = re.compile(
    r"\b(exposed to|exposure|ipo-?driven|capex scaling|beneficiar(?:y|ies)|spillover|"
    r"earning returns|profiting from|gain from|benefiting from)\b",
    re.IGNORECASE,
)

_NOMINAL_PREFIX_PATTERN = re.compile(
    r"^(companies|suppliers|vendors|operators|contractors|lenders|prime contractors)\b",
    re.IGNORECASE,
)


def normalize_summary_to_search_query(text: str) -> str:
    """Best-effort rewrite of taxonomy phrasing into document-voice retrieval text."""
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    rewritten = _NOMINAL_PREFIX_PATTERN.sub("The company", cleaned)
    rewritten = EXPOSURE_META_PATTERN.sub("", rewritten)
    rewritten = re.sub(r"\s{2,}", " ", rewritten).strip(" ,.;")
    if rewritten and not rewritten.endswith("."):
        rewritten = f"{rewritten}."
    return rewritten


def has_exposure_meta_language(text: str) -> bool:
    """Return True when query text still uses analyst exposure framing."""
    return bool(EXPOSURE_META_PATTERN.search(text))
