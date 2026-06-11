# Requirements: pip install requests
"""
Person-in-the-News Profiler — Jensen Huang
-------------------------------------------
1. Calls Co-mentions with text="Jensen Huang" (auto_enrich_filters=True) to
   let the API resolve him as an entity automatically.
2. Resolves the top person result → confirms it is Jensen Huang, extracts his ID.
3. Takes the top 10 co-mentioned companies from the same response.
4. Resolves company entity IDs → names.
5. For each company, fetches the top 3 chunks where BOTH Jensen Huang AND
   that company appear together (entity.all_of filter).
"""

import os
from datetime import date, timedelta

import requests

API_KEY = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

PERSON_QUERY = "Jensen Huang"
TOP_N_COMPANIES = 10
CHUNKS_PER_COMPANY = 3
DAYS_BACK = 7

END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=DAYS_BACK)
START_TS = f"{START_DATE.isoformat()}T00:00:00Z"
END_TS = f"{END_DATE.isoformat()}T23:59:59Z"


# ── 1. Co-mentions: discover person entity ID + top companies ─────────────────

def get_comentions() -> tuple[list[dict], list[dict]]:
    """Returns (people_entities, company_entities) from co-mentions."""
    body = {
        "query": {
            "text": PERSON_QUERY,
            "auto_enrich_filters": True,   # API resolves the person entity automatically
            "filters": {
                "timestamp": {"start": START_TS, "end": END_TS},
            },
        },
        "limit": TOP_N_COMPANIES,
    }
    resp = requests.post(
        f"{BASE_URL}/v1/search/co-mentions/entities", headers=HEADERS, json=body
    )
    resp.raise_for_status()
    results = resp.json().get("results", {})
    return results.get("people", []), results.get("companies", [])


# ── 2. Resolve entity IDs → names ────────────────────────────────────────────

def resolve_entities(entity_ids: list[str]) -> dict[str, str]:
    if not entity_ids:
        return {}
    resp = requests.post(
        f"{BASE_URL}/v1/knowledge-graph/entities/id",
        headers=HEADERS,
        json={"values": entity_ids[:100]},
    )
    resp.raise_for_status()
    raw = resp.json().get("results", {})
    return {eid: info.get("name", eid) for eid, info in raw.items()}


# ── 3. Search: top N chunks where both person AND company appear ──────────────

def fetch_chunks(person_id: str, company_id: str, company_name: str) -> list[dict]:
    body = {
        "query": {
            "text": f"{PERSON_QUERY} {company_name}",
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {"start": START_TS, "end": END_TS},
                "entity": {
                    "any_of": [],
                    "all_of": [person_id, company_id],  # BOTH must be present
                    "none_of": [],
                },
            },
            "ranking_params": {"freshness_boost": 1, "source_boost": 0},
            "max_chunks": CHUNKS_PER_COMPANY,
        }
    }
    resp = requests.post(f"{BASE_URL}/v1/search", headers=HEADERS, json=body)
    resp.raise_for_status()

    chunks: list[dict] = []
    for doc in resp.json().get("results", []):
        for ch in doc.get("chunks", []):
            chunks.append({
                "headline": doc.get("headline", ""),
                "source": doc.get("source", {}).get("name", ""),
                "timestamp": doc.get("timestamp", "")[:10],
                "url": doc.get("url", ""),
                "text": ch.get("text", ""),
                "relevance": ch.get("relevance", 0),
            })
            if len(chunks) >= CHUNKS_PER_COMPANY:
                return chunks
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Profile: \"{PERSON_QUERY}\"  |  Last {DAYS_BACK} days ({START_DATE} → {END_DATE})")
    print("=" * 70)

    people_entities, company_entities = get_comentions()

    if not people_entities:
        print("No person entity found in co-mentions. Try a longer date range.")
        return

    # Resolve the top person to confirm identity
    person_id = people_entities[0]["id"]
    person_names = resolve_entities([person_id])
    confirmed_name = person_names.get(person_id, person_id)
    person_chunks = people_entities[0].get("total_chunks_count", 0)
    print(f"\nFocal entity: {confirmed_name} (id={person_id})  |  {person_chunks:,} chunks this week\n")

    if not company_entities:
        print("No companies found in co-mentions.")
        return

    # Resolve all company IDs in one batch call
    company_ids = [e["id"] for e in company_entities]
    company_names = resolve_entities(company_ids)

    print(f"Top {len(company_entities)} companies co-mentioned with {confirmed_name}:\n")

    for rank, entity in enumerate(company_entities, start=1):
        cid = entity["id"]
        cname = company_names.get(cid, cid)
        co_chunks = entity.get("total_chunks_count", 0)
        co_headlines = entity.get("total_headlines_count", 0)

        print(f"{'─' * 70}")
        print(f"#{rank:02d}  {cname}  [id={cid}]")
        print(f"     co-mention chunks={co_chunks}  headlines={co_headlines}")

        chunks = fetch_chunks(person_id, cid, cname)

        if not chunks:
            # Fallback: search may find nothing with all_of — try any_of + text
            print(f"     (no results with both entities — trying text-only fallback)")
            body_fallback = {
                "query": {
                    "text": f"{PERSON_QUERY} {cname}",
                    "auto_enrich_filters": False,
                    "filters": {
                        "timestamp": {"start": START_TS, "end": END_TS},
                        "entity": {"any_of": [cid], "all_of": [], "none_of": []},
                    },
                    "ranking_params": {"freshness_boost": 1, "source_boost": 0},
                    "max_chunks": CHUNKS_PER_COMPANY,
                }
            }
            resp = requests.post(f"{BASE_URL}/v1/search", headers=HEADERS, json=body_fallback)
            resp.raise_for_status()
            for doc in resp.json().get("results", []):
                for ch in doc.get("chunks", []):
                    chunks.append({
                        "headline": doc.get("headline", ""),
                        "source": doc.get("source", {}).get("name", ""),
                        "timestamp": doc.get("timestamp", "")[:10],
                        "url": doc.get("url", ""),
                        "text": ch.get("text", ""),
                        "relevance": ch.get("relevance", 0),
                    })
                    if len(chunks) >= CHUNKS_PER_COMPANY:
                        break
                if len(chunks) >= CHUNKS_PER_COMPANY:
                    break

        if not chunks:
            print("     (no search results found)\n")
            continue

        for i, ch in enumerate(chunks, start=1):
            print(f"\n     [{i}] relevance={ch['relevance']:.3f}  |  {ch['source']}  |  {ch['timestamp']}")
            print(f"         {ch['headline']}")
            print(f"         {ch['text'][:220].strip()}")
        print()


if __name__ == "__main__":
    main()
