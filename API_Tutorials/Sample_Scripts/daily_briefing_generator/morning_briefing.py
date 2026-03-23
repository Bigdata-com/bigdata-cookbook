# Requirements: pip install requests
"""
Morning Briefing Generator
--------------------------
1. Resolves tickers to entity IDs via Knowledge Graph.
2. Builds a Batch Search .jsonl — one query per ticker, filtered to the last 24 hours.
3. Polls until the job completes, then downloads results.
4. Writes a morning_briefing.md with top 3 chunks per ticker.
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta

import requests

API_KEY = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

TICKERS = ["AAPL", "TSLA", "NVDA", "JPM", "AMZN"]
CHUNKS_PER_TICKER = 3
POLL_INTERVAL = 15  # seconds

now = datetime.now(timezone.utc)
START_TS = (now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
END_TS = now.strftime("%Y-%m-%dT%H:%M:%SZ")


# ── 1. Resolve tickers → entity IDs ──────────────────────────────────────────

def resolve_tickers(tickers: list[str]) -> dict[str, dict]:
    """Returns {ticker: {"id": ..., "name": ...}} for each ticker found."""
    resolved = {}
    for ticker in tickers:
        resp = requests.post(
            f"{BASE_URL}/v1/knowledge-graph/companies",
            headers=HEADERS,
            json={"query": ticker, "types": ["PUBLIC"]},
        )
        resp.raise_for_status()
        companies = resp.json().get("results", [])
        if companies:
            top = companies[0]
            resolved[ticker] = {"id": top["id"], "name": top.get("name", ticker)}
        else:
            print(f"  WARNING: could not resolve '{ticker}', skipping.")
    return resolved


# ── 2. Build batch input .jsonl ───────────────────────────────────────────────

def build_jsonl(ticker_entities: dict[str, dict]) -> tuple[str, list[str]]:
    """Writes batch_input.jsonl and returns (path, ordered list of tickers)."""
    ordered_tickers = list(ticker_entities.keys())
    path = "batch_input.jsonl"
    with open(path, "w") as f:
        for ticker in ordered_tickers:
            entity_id = ticker_entities[ticker]["id"]
            query = {
                "query": {
                    "text": f"{ticker} {ticker_entities[ticker]['name']} latest news",
                    "auto_enrich_filters": False,
                    "filters": {
                        "timestamp": {"start": START_TS, "end": END_TS},
                        "entity": {"any_of": [entity_id], "all_of": [], "none_of": []},
                    },
                    "ranking_params": {"freshness_boost": 2, "source_boost": 1},
                    "max_chunks": CHUNKS_PER_TICKER,
                }
            }
            f.write(json.dumps(query) + "\n")
    return path, ordered_tickers


# ── 3. Submit + poll batch job ────────────────────────────────────────────────

def run_batch(jsonl_path: str) -> list[dict]:
    # Create job
    resp = requests.post(f"{BASE_URL}/v1/search/batches", headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    batch_id = data["batch_id"]
    presigned_url = data["presigned_url"]
    print(f"  Batch job created: {batch_id}")

    # Upload input file
    with open(jsonl_path, "rb") as f:
        upload_resp = requests.put(
            presigned_url,
            headers={"Content-Type": "application/jsonl"},
            data=f,
        )
        upload_resp.raise_for_status()
    print(f"  Input uploaded. Polling every {POLL_INTERVAL}s...")

    # Poll until completed
    while True:
        status_resp = requests.get(
            f"{BASE_URL}/v1/search/batches/{batch_id}", headers=HEADERS
        )
        status_resp.raise_for_status()
        info = status_resp.json()
        status = info["status"]
        print(f"    status: {status}")

        if status == "completed":
            output_url = info["output_file_url"]
            break
        elif status in ("failed", "cancelled"):
            raise RuntimeError(f"Batch job {status}: {info}")

        time.sleep(POLL_INTERVAL)

    # Download results
    results_resp = requests.get(output_url)
    results_resp.raise_for_status()
    return [json.loads(line) for line in results_resp.text.strip().splitlines()]


# ── 4. Parse results → {ticker: [chunks]} ────────────────────────────────────

def parse_results(raw_results: list[dict], ordered_tickers: list[str]) -> dict[str, list[dict]]:
    briefing: dict[str, list[dict]] = {t: [] for t in ordered_tickers}

    for result in raw_results:
        line_num = result.get("line_number", 1)  # 1-indexed
        idx = line_num - 1
        if not (0 <= idx < len(ordered_tickers)):
            continue
        ticker = ordered_tickers[idx]

        if result.get("status") != "success":
            print(f"  WARNING: {ticker} result status={result.get('status')}")
            continue

        chunks = []
        for doc in result.get("response", {}).get("results", []):
            for ch in doc.get("chunks", []):
                chunks.append({
                    "headline": doc.get("headline", ""),
                    "source":   doc.get("source", {}).get("name", ""),
                    "url":      doc.get("url", ""),
                    "timestamp": doc.get("timestamp", "")[:16].replace("T", " "),
                    "text":     ch.get("text", ""),
                    "relevance": ch.get("relevance", 0.0),
                    "sentiment": ch.get("sentiment", 0.0),
                })
                if len(chunks) >= CHUNKS_PER_TICKER:
                    break
            if len(chunks) >= CHUNKS_PER_TICKER:
                break
        briefing[ticker] = chunks

    return briefing


# ── 5. Write markdown briefing ────────────────────────────────────────────────

def write_markdown(briefing: dict[str, list[dict]], ticker_entities: dict[str, dict]) -> str:
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%H:%M UTC")
    lines = [
        f"# Morning Briefing — {date_str}",
        f"*Generated at {time_str} · Last 24 hours · Top {CHUNKS_PER_TICKER} chunks per ticker*",
        "",
        "---",
        "",
    ]

    for ticker, chunks in briefing.items():
        name = ticker_entities.get(ticker, {}).get("name", ticker)
        lines.append(f"## {ticker} — {name}")
        lines.append("")

        if not chunks:
            lines.append("*No news found in the last 24 hours.*")
            lines.append("")
            continue

        for i, ch in enumerate(chunks, start=1):
            sentiment_label = (
                "positive" if ch["sentiment"] > 0.1
                else "negative" if ch["sentiment"] < -0.1
                else "neutral"
            )
            lines.append(f"### {i}. {ch['headline'] or '(no headline)'}")
            lines.append(
                f"**Source:** {ch['source']} &nbsp;|&nbsp; "
                f"**Published:** {ch['timestamp']} &nbsp;|&nbsp; "
                f"**Sentiment:** {sentiment_label} ({ch['sentiment']:+.2f}) &nbsp;|&nbsp; "
                f"**Relevance:** {ch['relevance']:.2f}"
            )
            if ch["url"]:
                lines.append(f"**URL:** <{ch['url']}>")
            lines.append("")
            lines.append(f"> {ch['text'][:300].strip()}")
            lines.append("")

        lines.append("---")
        lines.append("")

    lines.append(f"*End of briefing — {len(TICKERS)} tickers · {date_str}*")

    out_path = "morning_briefing.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Morning Briefing  |  Window: {START_TS} → {END_TS}\n")

    print("Resolving tickers...")
    ticker_entities = resolve_tickers(TICKERS)
    for ticker, info in ticker_entities.items():
        print(f"  {ticker:5s}  →  {info['name']}  [{info['id']}]")

    print("\nBuilding batch input...")
    jsonl_path, ordered_tickers = build_jsonl(ticker_entities)
    print(f"  {len(ordered_tickers)} queries written to {jsonl_path}")

    print("\nSubmitting batch job...")
    raw_results = run_batch(jsonl_path)
    print(f"  {len(raw_results)} results received")

    print("\nParsing results...")
    briefing = parse_results(raw_results, ordered_tickers)
    for ticker, chunks in briefing.items():
        print(f"  {ticker:5s}  {len(chunks)} chunks")

    print("\nWriting markdown...")
    out_path = write_markdown(briefing, ticker_entities)
    print(f"\nBriefing saved → {out_path}")


if __name__ == "__main__":
    main()
