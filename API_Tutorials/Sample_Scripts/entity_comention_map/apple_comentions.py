# Requirements: pip install requests
"""
Apple Co-mentions + Search Script
----------------------------------
1. Resolves Apple's entity ID via Knowledge Graph.
2. Fetches the top 10 co-mentioned entities today via Co-mentions API.
3. Resolves co-mention entity IDs to names.
4. For each co-mention, runs a Search query (Apple + co-mentioned entity) and
   prints the top 2 chunks.
"""

import os
from datetime import date

import requests

API_KEY = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

TODAY = date.today().isoformat()
DAY_START = f"{TODAY}T00:00:00Z"
DAY_END = f"{TODAY}T23:59:59Z"


# ── 1. Resolve Apple entity ID ────────────────────────────────────────────────

def resolve_apple_id() -> str:
    resp = requests.post(
        f"{BASE_URL}/v1/knowledge-graph/companies",
        headers=HEADERS,
        json={"query": "Apple", "types": ["PUBLIC"]},
    )
    resp.raise_for_status()
    companies = resp.json().get("results", [])
    # First result is Apple Inc (AAPL)
    apple_id = companies[0]["id"]
    apple_name = companies[0]["name"]
    print(f"Resolved Apple → {apple_name} (id={apple_id})\n")
    return apple_id


# ── 2. Top 10 co-mentions of Apple today ─────────────────────────────────────

def get_top_comentions(apple_id: str) -> list[dict]:
    body = {
        "query": {
            "text": "Apple",
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {"start": DAY_START, "end": DAY_END},
                "entity": {"any_of": [apple_id], "all_of": [], "none_of": []},
            },
        },
        "limit": 10,
    }
    resp = requests.post(
        f"{BASE_URL}/v1/search/co-mentions/entities",
        headers=HEADERS,
        json=body,
    )
    resp.raise_for_status()
    results = resp.json().get("results", {})

    # Collect all co-mentioned entities across categories, ranked by chunk count
    all_entities: list[dict] = []
    for category, entities in results.items():
        for ent in entities:
            all_entities.append({**ent, "category": category})

    all_entities.sort(key=lambda e: e.get("total_chunks_count", 0), reverse=True)
    return all_entities[:10]


# ── 3. Resolve entity IDs to names ───────────────────────────────────────────

def resolve_entity_names(entity_ids: list[str]) -> dict[str, str]:
    resp = requests.post(
        f"{BASE_URL}/v1/knowledge-graph/entities/id",
        headers=HEADERS,
        json={"values": entity_ids},
    )
    resp.raise_for_status()
    raw = resp.json().get("results", {})
    return {eid: info.get("name", eid) for eid, info in raw.items()}


# ── 4. Search: top 2 chunks for Apple + co-mentioned entity ──────────────────

def search_chunks(apple_id: str, comention_id: str, comention_name: str) -> list[dict]:
    body = {
        "query": {
            "text": f"Apple {comention_name}",
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {"start": DAY_START, "end": DAY_END},
                "entity": {
                    "any_of": [apple_id, comention_id],
                    "all_of": [],
                    "none_of": [],
                },
            },
            "ranking_params": {"freshness_boost": 1, "source_boost": 0},
            "max_chunks": 2,
        }
    }
    resp = requests.post(f"{BASE_URL}/v1/search", headers=HEADERS, json=body)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    chunks: list[dict] = []
    for doc in results:
        for chunk in doc.get("chunks", []):
            chunks.append(
                {
                    "headline": doc.get("headline", ""),
                    "source": doc.get("source", {}).get("name", ""),
                    "timestamp": doc.get("timestamp", ""),
                    "text": chunk.get("text", ""),
                    "relevance": chunk.get("relevance", 0),
                }
            )
            if len(chunks) >= 2:
                return chunks
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Date range: {DAY_START} → {DAY_END}\n{'=' * 70}\n")

    apple_id = resolve_apple_id()
    comentions = get_top_comentions(apple_id)

    if not comentions:
        print("No co-mentions found for Apple today.")
        return

    ids = [e["id"] for e in comentions]
    names = resolve_entity_names(ids)

    print(f"Top {len(comentions)} co-mentioned entities with Apple today:\n")

    for rank, entity in enumerate(comentions, start=1):
        eid = entity["id"]
        name = names.get(eid, eid)
        category = entity.get("category", "?")
        chunks_count = entity.get("total_chunks_count", 0)
        headlines_count = entity.get("total_headlines_count", 0)

        print(f"{'─' * 70}")
        print(f"#{rank:02d}  {name}  [{category}]")
        print(f"     chunks={chunks_count}  headlines={headlines_count}  id={eid}")

        chunks = search_chunks(apple_id, eid, name)
        if not chunks:
            print("     (no search results found)\n")
            continue

        for i, ch in enumerate(chunks, start=1):
            print(f"\n     Chunk {i} — {ch['source']} | {ch['timestamp'][:10]} | relevance={ch['relevance']:.3f}")
            print(f"     Headline: {ch['headline']}")
            # Wrap text at 80 chars for readability
            text = ch["text"]
            for j in range(0, len(text), 80):
                prefix = "     " if j == 0 else "             "
                print(f"     {text[j:j+80]}")
        print()


if __name__ == "__main__":
    main()
