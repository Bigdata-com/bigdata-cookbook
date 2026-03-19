# Requirements: pip install requests
"""
Company 24-hour Pulse
----------------------
Given a ticker (e.g. AAPL), produces:

  1. Media attention  — total chunk volume over the last 24 hours.
                        The Volume API aggregates by calendar date_utc, so the
                        24h window almost always spans two days. We fetch both
                        and sum their chunk counts.

  2. Avg sentiment    — chunk-weighted average of the Volume API's per-day
                        sentiment field across the two calendar days:
                        Σ(sentiment_day × chunks_day) / Σ(chunks_day).

  3. Top 10 chunks    — most impactful passages from Search, ranked by
                        relevance × |sentiment|  (signal strength).

Usage:
    python company_pulse.py          # defaults to AAPL
    python company_pulse.py TSLA
    python company_pulse.py NVDA
"""

import os
import sys
from datetime import datetime, timezone, timedelta

import requests

API_KEY = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

TICKER = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

now      = datetime.now(timezone.utc)
since    = now - timedelta(hours=24)
START_TS = since.strftime("%Y-%m-%dT%H:%M:%SZ")
END_TS   = now.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── 1. Resolve ticker → entity ID ────────────────────────────────────────────

def resolve_ticker(ticker: str) -> tuple[str, str]:
    """Returns (entity_id, canonical_name)."""
    resp = requests.post(
        f"{BASE_URL}/v1/knowledge-graph/companies",
        headers=HEADERS,
        json={"query": ticker, "types": ["PUBLIC"]},
    )
    resp.raise_for_status()
    companies = resp.json().get("results", [])
    if not companies:
        raise ValueError(f"Could not resolve ticker '{ticker}'")
    top = companies[0]
    return top["id"], top.get("name", ticker)


# ── 2. Volume: chunk counts + chunk-weighted sentiment ───────────────────────

def fetch_volume(entity_id: str) -> dict:
    """
    Returns:
        total_chunks   — sum of chunks across both calendar days in the window.
        avg_sentiment  — chunk-weighted average of the per-day sentiment values:
                         Σ(sentiment_day × chunks_day) / Σ(chunks_day).
        days           — raw daily entries for the breakdown display.
    """
    body = {
        "query": {
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {"start": START_TS, "end": END_TS},
                "entity": {"any_of": [entity_id], "all_of": [], "none_of": []},
            },
        }
    }
    resp = requests.post(f"{BASE_URL}/v1/search/volume", headers=HEADERS, json=body)
    resp.raise_for_status()
    volume = resp.json().get("results", {}).get("volume", [])

    # Normalise date/day field inconsistency
    for entry in volume:
        if "day" in entry and "date" not in entry:
            entry["date"] = entry["day"]

    total_chunks       = 0
    weighted_sentiment = 0.0
    for entry in volume:
        chunks    = entry.get("chunks", 0)
        sentiment = entry.get("sentiment", 0.0) or 0.0
        total_chunks       += chunks
        weighted_sentiment += sentiment * chunks

    avg_sentiment = weighted_sentiment / total_chunks if total_chunks else 0.0

    return {"total_chunks": total_chunks, "avg_sentiment": avg_sentiment, "days": volume}


# ── 3. Search: top 10 chunks ranked by relevance × |sentiment| ───────────────

def fetch_top_chunks(entity_id: str, max_chunks: int = 100) -> list[dict]:
    """
    Fetches up to max_chunks from Search and re-ranks by relevance × |sentiment|
    to surface the most impactful (strong signal + strong relevance) passages.
    """
    body = {
        "query": {
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {"start": START_TS, "end": END_TS},
                "entity": {"any_of": [entity_id], "all_of": [], "none_of": []},
            },
            "ranking_params": {"freshness_boost": 0, "source_boost": 10},
            "max_chunks": max_chunks,
        }
    }
    resp = requests.post(f"{BASE_URL}/v1/search", headers=HEADERS, json=body)
    resp.raise_for_status()

    scored = []
    for doc in resp.json().get("results", []):
        headline    = doc.get("headline", "")
        source_name = doc.get("source", {}).get("name", "")
        timestamp   = doc.get("timestamp", "")[:16].replace("T", " ")
        url         = doc.get("url", "")
        for ch in doc.get("chunks", []):
            relevance = ch.get("relevance", 0.0)
            sentiment = ch.get("sentiment", 0.0) or 0.0
            scored.append({
                "score":     relevance * abs(sentiment),
                "relevance": relevance,
                "sentiment": sentiment,
                "text":      ch.get("text", ""),
                "headline":  headline,
                "source":    source_name,
                "timestamp": timestamp,
                "url":       url,
            })

    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:10]


# ── 4. Print report ───────────────────────────────────────────────────────────

def print_report(
    ticker: str,
    name: str,
    entity_id: str,
    vol: dict,
    top_chunks: list[dict],
) -> None:
    sent  = vol["avg_sentiment"]
    label = "positive" if sent > 0.05 else "negative" if sent < -0.05 else "neutral"
    bar   = "▲" if sent > 0.05 else "▼" if sent < -0.05 else "─"

    print(f"\n{'═' * 66}")
    print(f"  {ticker}  ·  {name}  [{entity_id}]")
    print(f"  Window: {START_TS[:16]} → {END_TS[:16]} UTC  (last 24 h)")
    print(f"{'═' * 66}")

    print(f"\n  {'Media attention':<28}  {vol['total_chunks']:>8,}  chunks")
    print(f"  {'Avg sentiment (weighted)':<28}  {sent:>+8.4f}  {bar} {label}")

    print(f"\n  Daily breakdown:")
    for d in vol["days"]:
        date_str = d.get("date", "?")
        ch       = d.get("chunks", 0)
        docs     = d.get("documents", 0)
        s        = d.get("sentiment", 0.0) or 0.0
        print(f"    {date_str}   chunks: {ch:>6,}   docs: {docs:>5,}   sentiment: {s:>+.4f}")

    print(f"\n  Top 10 chunks  (ranked by relevance × |sentiment|)")
    print(f"  {'─' * 62}")
    for i, ch in enumerate(top_chunks, start=1):
        sent_ch = ch["sentiment"]
        sign    = "▲" if sent_ch > 0 else "▼" if sent_ch < 0 else "─"
        print(
            f"\n  [{i:>2}]  score={ch['score']:.4f}  "
            f"rel={ch['relevance']:.3f}  "
            f"sent={sent_ch:>+.3f} {sign}"
        )
        print(f"        {ch['source']}  ·  {ch['timestamp']}")
        print(f"        {ch['headline'][:72]}")
        words, line, lines = ch["text"].split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > 72:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        for j, ln in enumerate(lines[:3]):
            prefix = "        > " if j == 0 else "          "
            print(f"{prefix}{ln}")
        if len(lines) > 3:
            print(f"          [+{len(lines)-3} more lines]")

    print(f"\n{'═' * 66}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Resolving {TICKER}...")
    entity_id, name = resolve_ticker(TICKER)
    print(f"  → {name}  [{entity_id}]")

    print(f"Fetching volume  ({START_TS} → {END_TS})...")
    vol = fetch_volume(entity_id)
    print(f"  → {vol['total_chunks']:,} chunks | avg sentiment: {vol['avg_sentiment']:+.4f}")

    print(f"Fetching top chunks...")
    top_chunks = fetch_top_chunks(entity_id)
    print(f"  → {len(top_chunks)} chunks scored and ranked")

    print_report(TICKER, name, entity_id, vol, top_chunks)


if __name__ == "__main__":
    main()
