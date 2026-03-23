# Requirements: pip install requests
"""
Earnings Sentiment Tracker — Batch Search
------------------------------------------
1. Resolves [AAPL, MSFT, GOOGL, AMZN, META] to entity IDs via Knowledge Graph.
2. Builds one "post-earnings news" query per ticker (last 7 days).
3. Submits all queries as a single Batch Search job.
4. Polls until complete, downloads results.
5. Averages chunk sentiment per company and prints a ranked table.
"""

import json
import os
import time
from datetime import date, timedelta

import requests

API_KEY = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
MAX_CHUNKS = 5
DAYS_BACK = 7

END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=DAYS_BACK)
START_TS = f"{START_DATE.isoformat()}T00:00:00Z"
END_TS = f"{END_DATE.isoformat()}T23:59:59Z"


# ── 1. Resolve tickers to entity IDs ─────────────────────────────────────────

def resolve_tickers(tickers: list[str]) -> dict[str, dict]:
    """Returns {ticker: {"id": ..., "name": ...}}"""
    resolved: dict[str, dict] = {}
    for ticker in tickers:
        resp = requests.post(
            f"{BASE_URL}/v1/knowledge-graph/companies",
            headers=HEADERS,
            json={"query": ticker, "types": ["PUBLIC"]},
        )
        resp.raise_for_status()
        companies = resp.json().get("results", [])
        if not companies:
            print(f"  WARNING: no match for {ticker}, skipping.")
            continue
        resolved[ticker] = {"id": companies[0]["id"], "name": companies[0]["name"]}
        print(f"  {ticker} → {companies[0]['name']} (id={companies[0]['id']})")
    return resolved


# ── 2. Build .jsonl batch input ───────────────────────────────────────────────

def build_batch_lines(ticker_map: dict[str, dict]) -> list[dict]:
    lines = []
    for ticker, info in ticker_map.items():
        lines.append({
            "query": {
                "text": f"{info['name']} earnings results revenue profit",
                "auto_enrich_filters": False,
                "filters": {
                    "timestamp": {"start": START_TS, "end": END_TS},
                    "entity": {"any_of": [info["id"]], "all_of": [], "none_of": []},
                },
                "ranking_params": {"freshness_boost": 1, "source_boost": 0},
                "max_chunks": MAX_CHUNKS,
            }
        })
    return lines


# ── 3. Submit batch job ───────────────────────────────────────────────────────

def submit_batch(lines: list[dict]) -> tuple[str, str]:
    # Create job → get batch_id + presigned_url
    resp = requests.post(f"{BASE_URL}/v1/search/batches", headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    batch_id = data["batch_id"]
    presigned_url = data["presigned_url"]
    print(f"\nBatch job created: {batch_id}")

    # Write .jsonl and upload
    input_path = "earnings_batch_input.jsonl"
    with open(input_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    with open(input_path, "rb") as f:
        upload_resp = requests.put(
            presigned_url,
            headers={"Content-Type": "application/jsonl"},
            data=f,
        )
        upload_resp.raise_for_status()
    print(f"Input uploaded ({len(lines)} queries).")
    return batch_id, presigned_url


# ── 4. Poll until completed ───────────────────────────────────────────────────

def poll_batch(batch_id: str) -> str:
    print("Polling", end="", flush=True)
    while True:
        resp = requests.get(f"{BASE_URL}/v1/search/batches/{batch_id}", headers=HEADERS)
        resp.raise_for_status()
        info = resp.json()
        status = info["status"]

        if status == "completed":
            print(" done.")
            return info["output_file_url"]
        elif status in ("failed", "cancelled"):
            raise RuntimeError(f"Batch job {status}: {info}")

        print(".", end="", flush=True)
        time.sleep(10)


# ── 5. Download and parse results ─────────────────────────────────────────────

def download_results(output_url: str) -> list[dict]:
    resp = requests.get(output_url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.strip().splitlines()]


# ── 6. Compute per-ticker sentiment stats ─────────────────────────────────────

def compute_sentiment(
    results: list[dict],
    ticker_map: dict[str, dict],
) -> list[dict]:
    tickers = list(ticker_map.keys())
    rows = []

    for result in results:
        line_num = result.get("line_number", 1)  # line_number is 1-indexed
        ticker = tickers[line_num - 1] if 0 <= line_num - 1 < len(tickers) else "?"
        name = ticker_map.get(ticker, {}).get("name", ticker)

        if result.get("status") != "success":
            print(f"  WARNING: {ticker} query failed — {result.get('error', '?')}")
            rows.append({
                "ticker": ticker,
                "name": name,
                "avg_sentiment": None,
                "chunk_count": 0,
                "top_headline": "(no results)",
            })
            continue

        docs = result.get("response", {}).get("results", [])
        sentiments: list[float] = []
        top_headline = ""

        for doc in docs:
            if not top_headline:
                top_headline = doc.get("headline", "")
            for ch in doc.get("chunks", []):
                s = ch.get("sentiment")
                if s is not None:
                    sentiments.append(float(s))

        avg = sum(sentiments) / len(sentiments) if sentiments else None
        rows.append({
            "ticker": ticker,
            "name": name,
            "avg_sentiment": avg,
            "chunk_count": len(sentiments),
            "top_headline": top_headline or "(no results)",
        })

    # Sort: highest sentiment first; None goes to bottom
    rows.sort(key=lambda r: r["avg_sentiment"] if r["avg_sentiment"] is not None else -99, reverse=True)
    return rows


# ── 7. Print ranked table ─────────────────────────────────────────────────────

def print_table(rows: list[dict]) -> None:
    print(f"\n{'─' * 80}")
    print(f"{'Rank':<5} {'Ticker':<7} {'Avg Sentiment':>14} {'Chunks':>7}  Top Headline")
    print(f"{'─' * 80}")
    for rank, row in enumerate(rows, start=1):
        sentiment_str = f"{row['avg_sentiment']:+.3f}" if row["avg_sentiment"] is not None else "   N/A"
        headline = row["top_headline"][:55] + "…" if len(row["top_headline"]) > 55 else row["top_headline"]
        print(f"  {rank:<4} {row['ticker']:<7} {sentiment_str:>14} {row['chunk_count']:>7}  {headline}")
    print(f"{'─' * 80}")
    print(f"\nSentiment range: -1.0 (most negative) → 0.0 (neutral) → +1.0 (most positive)")
    print(f"Window: last {DAYS_BACK} days ({START_DATE} → {END_DATE})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Resolving {len(TICKERS)} tickers...")
    ticker_map = resolve_tickers(TICKERS)
    if not ticker_map:
        print("No tickers resolved. Exiting.")
        return

    lines = build_batch_lines(ticker_map)
    batch_id, _ = submit_batch(lines)
    output_url = poll_batch(batch_id)
    results = download_results(output_url)
    rows = compute_sentiment(results, ticker_map)
    print_table(rows)


if __name__ == "__main__":
    main()
