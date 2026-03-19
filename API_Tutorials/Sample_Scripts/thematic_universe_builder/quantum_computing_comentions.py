# Requirements: pip install requests
"""
Quantum Computing Co-mention Landscape
---------------------------------------
The co-mentions endpoint returns entities across ALL types (companies, places,
people, organizations, products, concepts) up to `limit` total. Setting a small
limit (e.g. 20) may yield very few companies since the budget is shared across
all categories.

Strategy: use a large limit (FETCH_N = 500) to pull a wide candidate pool,
filter to companies client-side, then sort by total_chunks_count (depth of
coverage) and display the top TOP_N.
"""

import os
from datetime import date, timedelta

import requests

API_KEY = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

# Natural-language sentence — semantically meaningful so chunk-level embeddings
# match documents about companies involved in quantum computing.
QUERY_TEXT = (
    "The company is investing in quantum computing technology"
)

TOP_N   = 20   # companies to display after filtering and ranking
FETCH_N = 1000  # limit sent to API — large pool ensures enough companies survive
               # the cross-category split; max allowed is 1000

RANK_BY = "headlines"  # "headlines" (breadth) or "chunks" (depth of coverage)
                       # headline counts are reliably present; chunk counts may be
                       # absent for entities detected only at the headline level

END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=30)
START_TS = f"{START_DATE.isoformat()}T00:00:00Z"
END_TS = f"{END_DATE.isoformat()}T23:59:59Z"


# ── 1. Co-mentions: fetch large pool, filter to companies, rank ───────────────

def fetch_comentions() -> tuple[list[dict], dict]:
    """Returns (top companies, raw_results) so caller can inspect full breakdown."""
    body = {
        "query": {
            "text": QUERY_TEXT,
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {"start": START_TS, "end": END_TS},
            },
        },
        "limit": FETCH_N,
    }
    resp = requests.post(
        f"{BASE_URL}/v1/search/co-mentions/entities", headers=HEADERS, json=body
    )
    resp.raise_for_status()
    raw = resp.json().get("results", {})
    companies = raw.get("companies", [])

    if RANK_BY == "chunks":
        companies.sort(
            key=lambda e: (e.get("total_chunks_count", 0), e.get("total_headlines_count", 0)),
            reverse=True,
        )
    else:
        companies.sort(
            key=lambda e: (e.get("total_headlines_count", 0), e.get("total_chunks_count", 0)),
            reverse=True,
        )
    return companies[:TOP_N], raw


def print_raw_breakdown(raw: dict) -> None:
    """Show how many entities were returned per category and chunk distribution."""
    print("\n  Raw co-mention breakdown (all entity types):")
    for category, entities in raw.items():
        with_chunks = [e for e in entities if e.get("total_chunks_count")]
        print(
            f"    {category:<16} {len(entities):>4} entities  |  "
            f"{len(with_chunks)} with chunk count"
            + (
                f"  (top: {max(e['total_chunks_count'] for e in with_chunks):,})"
                if with_chunks else ""
            )
        )


# ── 2. Resolve entity IDs → name, sector, country ────────────────────────────

def resolve_entities(entity_ids: list[str]) -> dict[str, dict]:
    resolved: dict[str, dict] = {}
    for i in range(0, len(entity_ids), 100):
        batch = entity_ids[i : i + 100]
        resp = requests.post(
            f"{BASE_URL}/v1/knowledge-graph/entities/id",
            headers=HEADERS,
            json={"values": batch},
        )
        resp.raise_for_status()
        for eid, info in resp.json().get("results", {}).items():
            resolved[eid] = {
                "name":    info.get("name", eid),
                "sector":  info.get("sector") or "—",
                "country": info.get("country") or "—",
            }
    return resolved


# ── 3. Print ranked table ─────────────────────────────────────────────────────

def print_table(companies: list[dict], entity_info: dict[str, dict]) -> None:
    col_rank      = 5
    col_name      = 32
    col_sector    = 28
    col_country   = 10
    col_chunks    = 10
    col_headlines = 10

    header = (
        f"{'Rank':>{col_rank}}  "
        f"{'Company':<{col_name}}  "
        f"{'Sector':<{col_sector}}  "
        f"{'Country':<{col_country}}  "
        f"{'Chunks':>{col_chunks}}  "
        f"{'Headlines':>{col_headlines}}"
    )
    divider = "─" * len(header)

    rank_label = "chunk volume" if RANK_BY == "chunks" else "headline volume"
    secondary  = "headline volume" if RANK_BY == "chunks" else "chunk volume"
    print(f"\nQuery  : \"{QUERY_TEXT[:75]}...\"")
    print(f"Window : {START_DATE} → {END_DATE}  (last 30 days)")
    print(f"Fetch  : limit={FETCH_N} → filter to companies → rank by {rank_label} (tiebreak: {secondary}) → top {TOP_N}")
    print(f"\n{header}")
    print(divider)

    for rank, entity in enumerate(companies, start=1):
        eid   = entity["id"]
        info  = entity_info.get(eid, {})
        name  = info.get("name", eid)[:col_name]
        sec   = info.get("sector", "—")[:col_sector]
        cty   = info.get("country", "—")

        chunks    = entity.get("total_chunks_count", 0)
        headlines = entity.get("total_headlines_count", 0)
        ch_disp   = f"{chunks:,}"    if chunks else "—"
        hl_disp   = f"{headlines:,}" if headlines else "—"

        print(
            f"{rank:>{col_rank}}  "
            f"{name:<{col_name}}  "
            f"{sec:<{col_sector}}  "
            f"{cty:<{col_country}}  "
            f"{ch_disp:>{col_chunks}}  "
            f"{hl_disp:>{col_headlines}}"
        )

    print(divider)
    total_ch = sum(e["total_chunks_count"]    for e in companies if "total_chunks_count"    in e)
    total_hl = sum(e["total_headlines_count"] for e in companies if "total_headlines_count" in e)
    print(
        f"{'Total':>{col_rank}}  {'':>{col_name}}  {'':>{col_sector}}  {'':>{col_country}}  "
        f"{total_ch:>{col_chunks},}  {total_hl:>{col_headlines},}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Fetching co-mentions  ({START_DATE} → {END_DATE})...")
    companies, raw = fetch_comentions()
    print_raw_breakdown(raw)
    print(f"\n  {len(companies)} companies selected (ranked by {RANK_BY})")

    if not companies:
        print("No companies found. Try increasing FETCH_N or extending the date range.")
        return

    print("Resolving entity metadata...")
    entity_ids = [e["id"] for e in companies]
    entity_info = resolve_entities(entity_ids)
    print(f"  {len(entity_info)} entities resolved")

    print_table(companies, entity_info)


if __name__ == "__main__":
    main()
