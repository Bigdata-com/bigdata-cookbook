#!/usr/bin/env python3
# Requirements: pip install requests networkx matplotlib pandas
"""
Pepsi Co-mention Network Graph
--------------------------------
Builds an entity co-mention network graph for PepsiCo using the Bigdata.com API.

Methodology:
  1. Resolve PepsiCo entity ID via Knowledge Graph.
  2. Fetch top co-mentioned COMPANY entities (large pool, 90-day lookback).
  3. Resolve entity metadata (name, sector, country) in batches.
  4. Filter out noise sectors: Media, Financial, Government, and other
     non-peer categories per standard co-mention best practices.
  5. Rank remaining companies by headline volume.
  6. Build and render a weighted network graph (node size ∝ co-mentions).
  7. Export underlying data to CSV.
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY  = os.environ.get("BIGDATA_API_KEY", "")
if not API_KEY:
    sys.exit("Error: BIGDATA_API_KEY environment variable is not set.")

BASE_URL = "https://api.bigdata.com"
HEADERS  = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

# Lookback window — 90 days is the standard for co-mention peer analysis
END_DATE   = date.today()
START_DATE = END_DATE - timedelta(days=90)
START_TS   = f"{START_DATE.isoformat()}T00:00:00Z"
END_TS     = f"{END_DATE.isoformat()}T23:59:59Z"

# Fetch a large pool so enough companies survive the cross-category split
FETCH_N  = 1000
# Final number of peer nodes displayed in the graph
TOP_N    = 25

# ── Sector exclusion list ─────────────────────────────────────────────────────
# These sectors are filtered out because they co-mention Pepsi due to
# coverage/distribution relationships, not competitive or peer relationships.
# Standard best-practice filters for CPG / Beverage co-mention analysis.

EXCLUDED_SECTOR_KEYWORDS = [
    # Media & Entertainment — co-mention Pepsi as advertiser / content partner
    "media", "broadcasting", "publishing", "entertainment", "newspapers",
    "television", "radio", "streaming", "advertising", "marketing",
    "public relations", "communications",
    # Financial companies — appear because they cover/own Pepsi as an asset
    "bank", "banking", "financial", "finance", "insurance", "investment",
    "asset management", "hedge fund", "private equity", "venture capital",
    "brokerage", "securities", "credit", "mortgage", "fintech",
    "accounting", "audit",
    # Government & Regulatory — not commercial peers
    "government", "public administration", "regulatory",
    # Retail / Distribution — channels, not peers
    "supermarket", "grocery", "convenience store", "wholesale",
    # Technology infrastructure — appear due to digital transformation stories
    "cloud", "software", "semiconductor",
]

OUTPUT_DIR   = Path(__file__).parent
GRAPH_FILE   = OUTPUT_DIR / "pepsi_comention_network.png"
DATA_CSV     = OUTPUT_DIR / "pepsi_comention_data.csv"


# ── 1. Resolve PepsiCo entity ID ──────────────────────────────────────────────

def resolve_pepsi_id() -> tuple[str, str]:
    resp = requests.post(
        f"{BASE_URL}/v1/knowledge-graph/companies",
        headers=HEADERS,
        json={"query": "PepsiCo", "types": ["PUBLIC"]},
    )
    resp.raise_for_status()
    companies = resp.json().get("results", [])
    if not companies:
        sys.exit("Could not resolve PepsiCo entity. Check API key or query.")
    # First result is PepsiCo Inc (PEP)
    entity = companies[0]
    print(f"Resolved: {entity['name']} (id={entity['id']})")
    return entity["id"], entity["name"]


# ── 2. Fetch co-mentioned companies ──────────────────────────────────────────

def fetch_comentions(pepsi_id: str) -> list[dict]:
    """Fetch large pool of co-mentioned entities anchored to PepsiCo."""
    body = {
        "query": {
            # Semantic query anchored to PepsiCo; the entity filter ensures
            # every document in the pool mentions PepsiCo
            "text": "PepsiCo beverage food consumer goods",
            "auto_enrich_filters": False,
            "filters": {
                "timestamp": {"start": START_TS, "end": END_TS},
                "entity": {
                    "any_of": [pepsi_id],
                    "all_of": [],
                    "none_of": [],
                },
            },
        },
        "limit": FETCH_N,
    }
    resp = requests.post(
        f"{BASE_URL}/v1/search/co-mentions/entities",
        headers=HEADERS,
        json=body,
    )
    resp.raise_for_status()
    raw = resp.json().get("results", {})

    # Print breakdown across all entity types
    print("\nRaw co-mention breakdown (all types):")
    for cat, entities in raw.items():
        print(f"  {cat:<18} {len(entities):>4} entities")

    # Return only companies; sort by headline volume (breadth) then chunks (depth)
    companies = raw.get("companies", [])
    companies.sort(
        key=lambda e: (e.get("total_headlines_count", 0), e.get("total_chunks_count", 0)),
        reverse=True,
    )
    return companies


# ── 3. Resolve entity metadata ────────────────────────────────────────────────

def resolve_entities(entity_ids: list[str]) -> dict[str, dict]:
    """Batch-resolve entity IDs to name + sector + country (max 100 per call)."""
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
                "sector":  (info.get("sector") or "").strip(),
                "country": (info.get("country") or "").strip(),
            }
    return resolved


# ── 4. Filter out noise sectors ───────────────────────────────────────────────

def is_excluded(sector: str) -> bool:
    sector_lower = sector.lower()
    return any(kw in sector_lower for kw in EXCLUDED_SECTOR_KEYWORDS)


def apply_filters(
    companies: list[dict], entity_info: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    """Split companies into (kept, filtered_out)."""
    kept, dropped = [], []
    for ent in companies:
        info = entity_info.get(ent["id"], {})
        sector = info.get("sector", "")
        if is_excluded(sector):
            dropped.append({**ent, **info})
        else:
            kept.append({**ent, **info})
    return kept, dropped


# ── 5. Export underlying data to CSV ─────────────────────────────────────────

def export_csv(companies: list[dict], filtered: list[dict], pepsi_name: str) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed — skipping CSV export (pip install pandas).")
        return

    rows = []
    for rank, ent in enumerate(companies, start=1):
        rows.append({
            "rank":             rank,
            "entity_id":        ent.get("id", ""),
            "name":             ent.get("name", ""),
            "sector":           ent.get("sector", ""),
            "country":          ent.get("country", ""),
            "total_headlines":  ent.get("total_headlines_count", 0),
            "total_chunks":     ent.get("total_chunks_count", 0),
            "included":         True,
        })
    for ent in filtered:
        rows.append({
            "rank":             None,
            "entity_id":        ent.get("id", ""),
            "name":             ent.get("name", ""),
            "sector":           ent.get("sector", ""),
            "country":          ent.get("country", ""),
            "total_headlines":  ent.get("total_headlines_count", 0),
            "total_chunks":     ent.get("total_chunks_count", 0),
            "included":         False,
        })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_CSV, index=False)
    print(f"  Data saved → {DATA_CSV}")


# ── 6. Build and render network graph ────────────────────────────────────────

# Sector → color mapping for node coloring
SECTOR_COLORS = {
    "beverages":           "#E63946",   # red       — direct competitors
    "food":                "#F4A261",   # orange    — food peers
    "consumer":            "#2A9D8F",   # teal      — CPG peers
    "tobacco":             "#8338EC",   # purple
    "restaurant":          "#FB8500",   # amber
    "retail":              "#457B9D",   # blue-grey
    "pharmaceutical":      "#06D6A0",   # mint
    "agriculture":         "#8CB369",   # green
    "chemicals":           "#A8DADC",   # light blue
    "":                    "#ADB5BD",   # grey      — unknown / other
}

def sector_color(sector: str) -> str:
    sector_l = sector.lower()
    for key, color in SECTOR_COLORS.items():
        if key and key in sector_l:
            return color
    return SECTOR_COLORS[""]


def build_graph(companies: list[dict], pepsi_name: str) -> None:
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("networkx or matplotlib not installed — skipping graph (pip install networkx matplotlib).")
        return

    G = nx.Graph()

    # Central node — PepsiCo
    G.add_node(pepsi_name, node_type="focal", sector="Focal", headlines=0)

    max_hl = max((e.get("total_headlines_count", 1) for e in companies), default=1)

    for ent in companies:
        name = ent.get("name", ent.get("id"))
        hl   = ent.get("total_headlines_count", 0)
        G.add_node(name, node_type="peer", sector=ent.get("sector", ""), headlines=hl)
        G.add_edge(pepsi_name, name, weight=hl)

    # Layout — spring with Pepsi fixed at center
    pos = nx.spring_layout(G, seed=42, k=2.2, weight="weight")
    pos[pepsi_name] = (0.0, 0.0)

    # Node sizes
    node_sizes = []
    for node in G.nodes():
        if node == pepsi_name:
            node_sizes.append(4200)
        else:
            hl = G.nodes[node].get("headlines", 0)
            node_sizes.append(400 + 2800 * (hl / max_hl))

    # Node colors
    node_colors = []
    for node in G.nodes():
        if node == pepsi_name:
            node_colors.append("#1D3557")   # dark navy — focal
        else:
            node_colors.append(sector_color(G.nodes[node].get("sector", "")))

    # Edge widths ∝ co-mention weight
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 4.0 * (w / max_w) for w in edge_weights]

    fig, ax = plt.subplots(figsize=(18, 14))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    # Edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color="#CCCCCC",
        alpha=0.6,
    )

    # Nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.92,
        linewidths=1.2,
        edgecolors="#FFFFFF",
    )

    # Labels — PepsiCo always visible, peers if space allows
    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=7.5,
        font_color="#1A1A2E",
        font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
    )

    # Legend — sectors present
    seen_sectors: dict[str, str] = {}
    for node in G.nodes():
        if node == pepsi_name:
            seen_sectors["Focal (PepsiCo)"] = "#1D3557"
        else:
            sec = G.nodes[node].get("sector", "")
            color = sector_color(sec)
            label = sec if sec else "Other / Unknown"
            seen_sectors[label] = color
    # Deduplicate
    legend_handles = [
        mpatches.Patch(color=c, label=s)
        for s, c in sorted(seen_sectors.items(), key=lambda x: x[0])
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=8,
        title="Sector",
        title_fontsize=9,
        framealpha=0.85,
    )

    ax.set_title(
        f"PepsiCo Co-mention Network  ·  {START_DATE} → {END_DATE}  (90-day lookback)\n"
        f"Top {len(companies)} companies  |  filtered: media, fincos, gov't, retail/distribution  "
        f"|  edge weight ∝ co-mention headline volume  |  data: Bigdata.com",
        fontsize=11,
        pad=14,
        color="#1A1A2E",
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Graph saved → {GRAPH_FILE}")


# ── 7. Print ranked table ─────────────────────────────────────────────────────

def print_table(companies: list[dict], dropped: list[dict]) -> None:
    W_RANK, W_NAME, W_SECTOR, W_CTY, W_HL, W_CH = 5, 34, 30, 12, 11, 11
    header = (
        f"{'#':>{W_RANK}}  {'Company':<{W_NAME}}  {'Sector':<{W_SECTOR}}  "
        f"{'Country':<{W_CTY}}  {'Headlines':>{W_HL}}  {'Chunks':>{W_CH}}"
    )
    div = "─" * len(header)

    print(f"\n{'═'*70}")
    print(f"  PepsiCo Co-mention Peers  (after sector filtering)")
    print(f"  Window : {START_DATE} → {END_DATE}  |  Pool: {FETCH_N} entities  |  Shown: top {len(companies)}")
    print(f"{'═'*70}")
    print(f"\n{header}")
    print(div)

    for rank, ent in enumerate(companies, start=1):
        name   = ent.get("name",    "")[:W_NAME]
        sector = ent.get("sector",  "")[:W_SECTOR]
        cty    = ent.get("country", "")
        hl     = ent.get("total_headlines_count", 0)
        ch     = ent.get("total_chunks_count",    0)
        print(
            f"{rank:>{W_RANK}}  {name:<{W_NAME}}  {sector:<{W_SECTOR}}  "
            f"{cty:<{W_CTY}}  {hl:>{W_HL},}  {ch:>{W_CH},}"
        )

    print(div)
    total_hl = sum(e.get("total_headlines_count", 0) for e in companies)
    total_ch = sum(e.get("total_chunks_count",    0) for e in companies)
    print(
        f"{'Total':>{W_RANK}}  {'':>{W_NAME}}  {'':>{W_SECTOR}}  {'':>{W_CTY}}  "
        f"{total_hl:>{W_HL},}  {total_ch:>{W_CH},}"
    )

    print(f"\n  Filtered out ({len(dropped)} entities  — media, fincos, gov't, retail/distribution):")
    for ent in sorted(dropped, key=lambda e: e.get("total_headlines_count", 0), reverse=True)[:15]:
        print(f"    {ent.get('name',''):<34}  sector: {ent.get('sector','')}")
    if len(dropped) > 15:
        print(f"    … and {len(dropped)-15} more")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"PepsiCo Co-mention Network Graph")
    print(f"Window : {START_DATE} → {END_DATE}  (90-day lookback)")
    print(f"{'─'*60}\n")

    # Step 1 — resolve PepsiCo
    pepsi_id, pepsi_name = resolve_pepsi_id()

    # Step 2 — fetch co-mentioned companies
    print(f"\nFetching co-mentions (limit={FETCH_N})...")
    all_companies = fetch_comentions(pepsi_id)
    print(f"  {len(all_companies)} companies before filtering")

    if not all_companies:
        sys.exit("No co-mentioned companies found. Try a longer lookback or larger FETCH_N.")

    # Step 3 — resolve entity metadata (name, sector, country)
    print("\nResolving entity metadata...")
    entity_ids = [e["id"] for e in all_companies]
    entity_info = resolve_entities(entity_ids)
    # Merge metadata into company records
    for ent in all_companies:
        info = entity_info.get(ent["id"], {})
        ent.update(info)

    # Step 4 — apply sector filters
    kept, dropped = apply_filters(all_companies, entity_info)
    kept = kept[:TOP_N]
    print(f"  After filtering: {len(kept)} companies kept, {len(dropped)} filtered out")

    # Step 5 — print table
    print_table(kept, dropped)

    # Step 6 — export CSV
    print("\nExporting underlying data...")
    export_csv(kept, dropped, pepsi_name)

    # Step 7 — build network graph
    print("Rendering network graph...")
    build_graph(kept, pepsi_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
