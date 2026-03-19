"""
Apple Q2 FY2025 Earnings Call — Macro Economics
-------------------------------------------------
Finds what Apple said about macroeconomic conditions in the earnings call
4 quarters ago (Q2 FY2025, reported ~May 1 2025).

API calls made:
  1. POST /v1/knowledge-graph/companies  → resolve AAPL entity ID
  2. POST /v1/search                     → earnings_call chunks about macro
"""

import os
import requests

API_KEY  = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS  = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}


# ── 1. Resolve AAPL ───────────────────────────────────────────────────────────

resp = requests.post(
    f"{BASE_URL}/v1/knowledge-graph/companies",
    headers=HEADERS,
    json={"query": "AAPL", "types": ["PUBLIC"]},
)
resp.raise_for_status()
company   = resp.json().get("results", [])[0]
entity_id = company["id"]
print(f"Entity : {company['name']}  [{entity_id}]")


# ── 2. Search earnings call for macro content ─────────────────────────────────

body = {
    "query": {
        "text": (
            "The company discussed macroeconomic conditions, consumer spending "
            "trends, and the global economic outlook."
        ),
        "auto_enrich_filters": False,
        "filters": {
            "timestamp":     {"start": "2025-04-15T00:00:00Z", "end": "2025-05-15T23:59:59Z"},
            "entity":        {"any_of": [entity_id], "all_of": [], "none_of": []},
            "document_type": ["TRANSCRIPT"],
        },
        "ranking_params": {"freshness_boost": 0, "source_boost": 0},
        "max_chunks": 10,
    }
}

resp2 = requests.post(f"{BASE_URL}/v1/search", headers=HEADERS, json=body)
if not resp2.ok:
    print(f"HTTP {resp2.status_code}: {resp2.text}")
    raise SystemExit(1)
results = resp2.json().get("results", [])
print(f"Documents: {len(results)}\n")

for doc in results:
    print(f"{'═' * 70}")
    print(f"  {doc.get('headline', '')}")
    print(f"  {doc.get('timestamp', '')[:10]}  |  {doc.get('source', {}).get('name', '')}  |  {doc.get('url', '')}")
    print()
    for ch in doc.get("chunks", []):
        sent = ch.get("sentiment", 0) or 0
        sign = "▲" if sent > 0 else "▼" if sent < 0 else "─"
        print(f"  rel={ch.get('relevance', 0):.3f}  sent={sent:+.3f} {sign}")
        print(f"  > {ch.get('text', '')}")
        print()
