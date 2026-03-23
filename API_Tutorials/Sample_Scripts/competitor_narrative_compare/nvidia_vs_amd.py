# Requirements: pip install requests
"""
Nvidia vs AMD — Competitor Narrative Comparison
------------------------------------------------
1. Resolves Nvidia and AMD entity IDs via Knowledge Graph.
2. Runs two Search calls in parallel — same query "AI chip demand",
   each filtered to one company, last 30 days.
3. Prints a side-by-side comparison: chunk count, average relevance,
   top 3 headlines per company.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import requests

API_KEY = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

QUERY_TEXT = "AI chip demand"
MAX_CHUNKS = 20  # fetch more so we can surface 3 distinct headlines
DAYS_BACK = 30

END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=DAYS_BACK)
START_TS = f"{START_DATE.isoformat()}T00:00:00Z"
END_TS = f"{END_DATE.isoformat()}T23:59:59Z"

COMPANIES = ["Nvidia", "AMD"]


# ── 1. Resolve company name to entity ID ─────────────────────────────────────

def resolve_company(name: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/v1/knowledge-graph/companies",
        headers=HEADERS,
        json={"query": name, "types": ["PUBLIC"]},
    )
    resp.raise_for_status()
    companies = resp.json().get("results", [])
    if not companies:
        raise ValueError(f"No match found for '{name}'")
    c = companies[0]
    return {"name": c["name"], "id": c["id"]}


# ── 2. Search filtered to one entity ─────────────────────────────────────────

def search_for_company(entity: dict) -> dict:
    body = {
        "query": {
            "text": QUERY_TEXT,
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {"start": START_TS, "end": END_TS},
                "entity": {"any_of": [entity["id"]], "all_of": [], "none_of": []},
            },
            "ranking_params": {"freshness_boost": 0, "source_boost": 0},
            "max_chunks": MAX_CHUNKS,
        }
    }
    resp = requests.post(f"{BASE_URL}/v1/search", headers=HEADERS, json=body)
    resp.raise_for_status()
    docs = resp.json().get("results", [])

    chunks: list[dict] = []
    headlines: list[str] = []

    for doc in docs:
        headline = doc.get("headline", "")
        source = doc.get("source", {}).get("name", "")
        ts = doc.get("timestamp", "")[:10]
        if headline and headline not in headlines:
            headlines.append(f"{headline}  [{source}, {ts}]")
        for ch in doc.get("chunks", []):
            chunks.append(ch.get("relevance", 0))

    avg_relevance = sum(chunks) / len(chunks) if chunks else 0.0

    return {
        "name": entity["name"],
        "chunk_count": len(chunks),
        "avg_relevance": avg_relevance,
        "top_headlines": headlines[:3],
    }


# ── 3. Print side-by-side comparison ─────────────────────────────────────────

def print_comparison(left: dict, right: dict) -> None:
    col = 46  # width of each company column

    def pad(s: str) -> str:
        return s[:col].ljust(col)

    sep = "  │  "
    divider = "─" * col + "──┼──" + "─" * col

    print(f"\n{'Query:':<10} \"{QUERY_TEXT}\"   |   Last {DAYS_BACK} days ({START_DATE} → {END_DATE})\n")
    print(pad(left["name"].upper()) + sep + right["name"].upper())
    print(divider)

    # Chunk count
    print(pad(f"Chunks:       {left['chunk_count']}") + sep + f"Chunks:       {right['chunk_count']}")

    # Avg relevance
    print(pad(f"Avg relevance: {left['avg_relevance']:.3f}") + sep + f"Avg relevance: {right['avg_relevance']:.3f}")

    print(divider)
    print(pad("Top 3 Headlines:") + sep + "Top 3 Headlines:")

    max_h = max(len(left["top_headlines"]), len(right["top_headlines"]))
    for i in range(max_h):
        l_line = left["top_headlines"][i] if i < len(left["top_headlines"]) else ""
        r_line = right["top_headlines"][i] if i < len(right["top_headlines"]) else ""

        # Word-wrap each headline at col chars
        l_lines = [l_line[j:j+col] for j in range(0, max(len(l_line), 1), col)] if l_line else [""]
        r_lines = [r_line[j:j+col] for j in range(0, max(len(r_line), 1), col)] if r_line else [""]

        for k in range(max(len(l_lines), len(r_lines))):
            lp = l_lines[k] if k < len(l_lines) else ""
            rp = r_lines[k] if k < len(r_lines) else ""
            print(pad(lp) + sep + rp)

        if i < max_h - 1:
            print(pad("") + sep + "")

    print(divider)

    # Winner summary
    winner_chunks = left["name"] if left["chunk_count"] >= right["chunk_count"] else right["name"]
    winner_rel = left["name"] if left["avg_relevance"] >= right["avg_relevance"] else right["name"]
    print(f"\n  More coverage:       {winner_chunks}")
    print(f"  Higher relevance:    {winner_rel}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Resolving {COMPANIES}...")
    entities = {}
    for name in COMPANIES:
        e = resolve_company(name)
        entities[name] = e
        print(f"  {name} → {e['name']} (id={e['id']})")

    print(f"\nRunning parallel searches for \"{QUERY_TEXT}\"...")
    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(search_for_company, entities[name]): name
            for name in COMPANIES
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            print(f"  {name} done — {results[name]['chunk_count']} chunks")

    print_comparison(results[COMPANIES[0]], results[COMPANIES[1]])


if __name__ == "__main__":
    main()
