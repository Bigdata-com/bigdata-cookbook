"""Theme-level Bigdata.com search used to ground derivative mindmaps."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.prompts import DERIVATIVE_BRANCH_LABELS

DEFAULT_START_DATE = "2025-06-01"
DEFAULT_END_DATE = "2026-06-09"
DEFAULT_SEARCH_CATEGORY: dict[str, Any] = {
    "mode": "INCLUDE",
    "values": ["news_premium", "transcripts", "filings"],
}

logger = logging.getLogger(__name__)

BIGDATA_SEARCH_URL = os.environ.get(
    "BIGDATA_API_BASE_URL",
    "https://api.bigdata.com",
).rstrip("/") + "/v1/search"

ChunkRetriever = Callable[[str, str, str, int], list[dict[str, Any]]]

_HOP_QUERY_TEMPLATES: dict[str, str] = {
    DERIVATIVE_BRANCH_LABELS[0]: (
        "How {theme} directly changes company costs, revenues, fuel, commodity, "
        "or operating expenses in filings, earnings calls, and news."
    ),
    DERIVATIVE_BRANCH_LABELS[1]: (
        "How companies respond to {theme} with capacity cuts, price increases, "
        "hedging, or adjacent platform pricing in filings, transcripts, and news."
    ),
    DERIVATIVE_BRANCH_LABELS[2]: (
        "Knock-on consumer mix-shift, discount retail, and discretionary demand "
        "effects after {theme}, including sectors not obviously tied to the theme."
    ),
}


def build_grounding_queries(main_theme: str) -> list[dict[str, str]]:
    """Return one natural-language query per derivative hop."""
    return [
        {
            "branch": branch,
            "query": template.format(theme=main_theme),
        }
        for branch, template in _HOP_QUERY_TEMPLATES.items()
    ]


def _chunk_text(chunk: dict[str, Any]) -> str:
    for key in ("text", "chunk", "content", "snippet"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _source_name(chunk: dict[str, Any]) -> str:
    source = chunk.get("source")
    if isinstance(source, dict):
        for key in ("name", "source_name", "title"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("source_name", "document_title", "headline"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Unknown source"


def _timestamp(chunk: dict[str, Any]) -> str:
    for key in ("timestamp", "document_timestamp", "date", "published_at"):
        value = chunk.get(key)
        if value:
            return str(value)
    return ""


def _url(chunk: dict[str, Any]) -> str:
    for key in ("url", "document_url", "source_url"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    source = chunk.get("source")
    if isinstance(source, dict):
        url = source.get("url")
        if isinstance(url, str):
            return url
    return ""


def _document_chunks(document: dict[str, Any], max_chunks: int) -> list[dict[str, Any]]:
    """Flatten one search result into chunk dicts carrying document metadata."""
    source_name = _source_name(document)
    timestamp = _timestamp(document)
    url = _url(document)
    headline = str(document.get("headline") or "").strip()

    nested = document.get("chunks")
    if not isinstance(nested, list):
        nested = [document]

    flattened: list[dict[str, Any]] = []
    for chunk in nested:
        if not isinstance(chunk, dict):
            continue
        text = _chunk_text(chunk)
        if not text:
            continue
        flattened.append(
            {
                "text": text,
                "source_name": source_name,
                "timestamp": timestamp,
                "url": url,
                "headline": headline,
                "relevance": chunk.get("relevance"),
            }
        )
        if len(flattened) >= max_chunks:
            break
    return flattened


def flatten_search_results(parsed: Any, max_chunks: int) -> list[dict[str, Any]]:
    """Normalize a /v1/search response into flat, citable chunk dicts.

    The API returns ``results`` as documents whose text lives in a nested
    ``chunks`` array, while source, timestamp, and url stay on the document.
    """
    if isinstance(parsed, list):
        documents = parsed
    elif isinstance(parsed, dict):
        documents = next(
            (
                value
                for key in ("results", "documents", "data", "chunks")
                if isinstance(value := parsed.get(key), list) and value
            ),
            [],
        )
    else:
        return []

    flattened: list[dict[str, Any]] = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        flattened.extend(_document_chunks(document, max_chunks))

    ranked = sorted(
        flattened,
        key=lambda chunk: (
            chunk.get("relevance") if isinstance(chunk.get("relevance"), int | float) else 0.0
        ),
        reverse=True,
    )
    return ranked[:max_chunks]


def _iso_bound(date_value: str, *, end: bool) -> str:
    """Expand a YYYY-MM-DD date to an ANSI UTC timestamp."""
    text = date_value.strip()
    if "T" in text:
        return text
    suffix = "T23:59:59Z" if end else "T00:00:00Z"
    return f"{text}{suffix}"


def retrieve_theme_chunks_http(
    query: str,
    start_date: str,
    end_date: str,
    max_chunks: int = 8,
) -> list[dict[str, Any]]:
    """Search Bigdata.com without a company universe (theme-level documents)."""
    api_key = os.environ.get("BIGDATA_API_KEY")
    if not api_key:
        raise RuntimeError("BIGDATA_API_KEY is not set")

    payload = {
        "search_mode": "fast",
        "query": {
            "text": query,
            "max_chunks": max_chunks,
            "filters": {
                "timestamp": {
                    "start": _iso_bound(start_date, end=False),
                    "end": _iso_bound(end_date, end=True),
                },
                "category": {
                    "mode": DEFAULT_SEARCH_CATEGORY.get("mode", "INCLUDE"),
                    "values": list(DEFAULT_SEARCH_CATEGORY.get("values") or []),
                },
            },
        },
    }
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        BIGDATA_SEARCH_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-API-KEY": api_key,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Bigdata.com theme search failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Bigdata.com theme search connection error: {exc}") from exc

    return flatten_search_results(parsed, max_chunks)


def collect_grounding(
    main_theme: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    max_chunks_per_hop: int = 8,
    retriever: ChunkRetriever | None = None,
) -> dict[str, Any]:
    """Run three theme-level searches and return a serializable grounding payload."""
    retrieve = retriever or retrieve_theme_chunks_http
    hops: list[dict[str, Any]] = []
    for item in build_grounding_queries(main_theme):
        raw_chunks = retrieve(item["query"], start_date, end_date, max_chunks_per_hop)
        chunks: list[dict[str, str]] = []
        for raw in raw_chunks:
            text = _chunk_text(raw)
            if not text:
                continue
            chunks.append(
                {
                    "text": text,
                    "source_name": _source_name(raw),
                    "timestamp": _timestamp(raw),
                    "url": _url(raw),
                }
            )
        hops.append({"branch": item["branch"], "query": item["query"], "chunks": chunks})
        logger.info(
            "Grounding hop %s retrieved %d chunks",
            item["branch"],
            len(chunks),
        )
    return {
        "main_theme": main_theme,
        "start_date": start_date,
        "end_date": end_date,
        "hops": hops,
    }


def format_grounding_brief(payload: dict[str, Any], max_bullets_per_hop: int = 4) -> str:
    """Build a cited briefing string for taxonomy generation."""
    lines: list[str] = []
    hops = payload.get("hops") or []
    if not isinstance(hops, list) or not hops:
        return (
            "No Bigdata.com briefing was provided. "
            "Infer derivative pathways from economic reasoning."
        )

    for hop in hops:
        branch = str(hop.get("branch") or "")
        chunks = hop.get("chunks") or []
        lines.append(f"{branch}:")
        if not chunks:
            lines.append("- No retrieved chunks for this hop.")
            continue
        for chunk in chunks[:max_bullets_per_hop]:
            source = chunk.get("source_name") or "Unknown source"
            timestamp = chunk.get("timestamp") or "undated"
            url = chunk.get("url") or ""
            text = " ".join(str(chunk.get("text") or "").split())
            excerpt = text[:280]
            citation = f"{source} ({timestamp})"
            if url:
                citation = f"{citation} {url}"
            lines.append(f"- {excerpt} [{citation}]")
    return "\n".join(lines)


def write_grounding(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
