# Requirements: pip install requests
"""
Federal Reserve Source Tier Comparison
---------------------------------------
1. Fetches source IDs for RANK_1 and RANK_3 via Knowledge Graph sources endpoint.
2. Runs two parallel searches for "Federal Reserve interest rates" (last 30 days),
   each restricted to one tier's source IDs.
3. Prints a side-by-side comparison: chunk count, average relevance, top 5 headlines.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import requests

API_KEY = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

QUERY_TEXT = (
    "Federal Reserve interest rates monetary policy rate decision"
)
TIERS = ["RANK_1", "RANK_3"]
MAX_CHUNKS = 50        # fetch enough to compute meaningful stats
TOP_HEADLINES = 5

END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=30)
START_TS = f"{START_DATE.isoformat()}T00:00:00Z"
END_TS = f"{END_DATE.isoformat()}T23:59:59Z"


# ── 1. Fetch source IDs for a given rank ─────────────────────────────────────

def fetch_source_ids(rank: str) -> list[str]:
    """Returns all source IDs for the given rank tier (e.g. 'RANK_1')."""
    source_ids: list[str] = []
    resp = requests.post(
        f"{BASE_URL}/v1/knowledge-graph/sources",
        headers=HEADERS,
        json={"ranks": [rank]},
    )
    resp.raise_for_status()
    for source in resp.json().get("results", []):
        sid = source.get("id")
        if sid:
            source_ids.append(sid)
    return source_ids


# ── 2. Search restricted to a set of source IDs ───────────────────────────────

def search_tier(rank: str, source_ids: list[str]) -> dict:
    """Runs a search filtered to the given source IDs; returns parsed result dict."""
    body = {
        "query": {
            "text": QUERY_TEXT,
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {"start": START_TS, "end": END_TS},
                "source": {"mode": "INCLUDE", "values": source_ids},
            },
            "ranking_params": {"freshness_boost": 0, "source_boost": 0},
            "max_chunks": MAX_CHUNKS,
        }
    }
    resp = requests.post(f"{BASE_URL}/v1/search", headers=HEADERS, json=body)
    resp.raise_for_status()

    chunks = []
    headlines = []   # (relevance, headline, source_name, timestamp)

    for doc in resp.json().get("results", []):
        headline = doc.get("headline", "")
        source_name = doc.get("source", {}).get("name", "")
        timestamp = doc.get("timestamp", "")[:10]
        for ch in doc.get("chunks", []):
            rel = ch.get("relevance", 0.0)
            chunks.append(rel)
            headlines.append((rel, headline, source_name, timestamp))

    headlines.sort(key=lambda x: x[0], reverse=True)

    avg_relevance = sum(chunks) / len(chunks) if chunks else 0.0
    return {
        "rank": rank,
        "chunk_count": len(chunks),
        "avg_relevance": avg_relevance,
        "top_headlines": headlines[:TOP_HEADLINES],
        "source_pool_size": len(source_ids),
    }


# ── 3. Print comparison ───────────────────────────────────────────────────────

def print_comparison(results: dict[str, dict]) -> None:
    tier1 = results.get("RANK_1", {})
    tier3 = results.get("RANK_3", {})

    print("\n" + "═" * 70)
    print(f'  Query: "{QUERY_TEXT}"')
    print(f"  Window: {START_DATE} → {END_DATE}  (last 30 days)")
    print("═" * 70)

    # Summary table
    print(f"\n{'Metric':<28} {'RANK_1':>14} {'RANK_3':>14}")
    print("─" * 58)
    print(f"  {'Source pool size':<26} {tier1.get('source_pool_size', 0):>14,} {tier3.get('source_pool_size', 0):>14,}")
    print(f"  {'Chunks returned':<26} {tier1.get('chunk_count', 0):>14,} {tier3.get('chunk_count', 0):>14,}")
    print(f"  {'Avg relevance score':<26} {tier1.get('avg_relevance', 0):>14.4f} {tier3.get('avg_relevance', 0):>14.4f}")
    print()

    # Top headlines per tier
    for rank in TIERS:
        res = results.get(rank, {})
        print(f"  Top {TOP_HEADLINES} headlines — {rank}")
        print("  " + "─" * 66)
        top = res.get("top_headlines", [])
        if not top:
            print("    (no results)")
        for i, (rel, headline, source, ts) in enumerate(top, start=1):
            print(f"    [{i}] rel={rel:.3f}  {source}  {ts}")
            # Wrap long headlines at 60 chars
            words = headline.split()
            line, wrapped = "", []
            for word in words:
                if len(line) + len(word) + 1 > 60:
                    wrapped.append(line)
                    line = word
                else:
                    line = (line + " " + word).strip()
            if line:
                wrapped.append(line)
            for j, wline in enumerate(wrapped):
                prefix = "         " if j > 0 else "         "
                print(f"{prefix}{wline}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Fetching source IDs for {TIERS}...")
    tier_sources: dict[str, list[str]] = {}
    for rank in TIERS:
        ids = fetch_source_ids(rank)
        tier_sources[rank] = ids
        print(f"  {rank}: {len(ids)} sources")

    print("\nRunning parallel searches...")
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(search_tier, rank, tier_sources[rank]): rank
            for rank in TIERS
        }
        for future in as_completed(futures):
            rank = futures[future]
            res = future.result()
            results[rank] = res
            print(f"  {rank}: {res['chunk_count']} chunks, avg_relevance={res['avg_relevance']:.4f}")

    print_comparison(results)


if __name__ == "__main__":
    main()
