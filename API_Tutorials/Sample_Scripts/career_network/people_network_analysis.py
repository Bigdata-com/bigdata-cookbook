"""
Career Path Network Analysis — Bigdata.com APIs
================================================
Starting from any person name, this script:

  Step 1 — Entity ID resolution via co-mentions hack
            POST /v1/search/co-mentions/entities with the person name as text
            and entity_details=True.  The first entry in the "people" category
            whose name matches is the person — giving us rp_entity_id for free,
            with zero extra API calls.

  Step 2 — Full-history co-mentions  (2010 → today)
            Use the resolved entity_id in the co-mentions filter to pull every
            company ever mentioned alongside this person.

  Step 3 — Yearly time-slices
            Repeat co-mentions for each calendar year to score
            person ↔ company association strength per year.
            High score in consecutive years ≈ active role at that firm.

  Step 4 — Career path output
            Print a trajectory table and render a heatmap + network graph.

Focus: PERSON → COMPANY relationships only  (no P→P, no orgs)

Usage:
    export BIGDATA_API_KEY=your_key_here
    python people_network_analysis.py                        # default: Dario Amodei
    python people_network_analysis.py "Ray Dalio"
    python people_network_analysis.py "Ray Dalio" 2005 2024  # custom year range

Requirements:
    pip install requests networkx matplotlib pandas
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional

# ── Configuration ────────────────────────────────────────────────────────────

API_KEY  = os.environ.get("BIGDATA_API_KEY", "")
BASE_URL = "https://api.bigdata.com"
HEADERS  = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

DEFAULT_YEAR_START   = 2010
CO_MENTION_LIMIT     = 50   # companies per time window  (max 1000)
CO_MENTION_LIMIT_ALL = 100  # for the full-history pull

# ── HTTP helper ───────────────────────────────────────────────────────────────

def _post(path: str, body: dict) -> dict:
    resp = requests.post(f"{BASE_URL}{path}", headers=HEADERS,
                         json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ── Step 1: Resolve rp_entity_id via co-mentions + entity_details hack ───────

def resolve_person_entity_id(person_name: str) -> Optional[str]:
    """
    Call the co-mentions endpoint with the person's name as the text query
    and entity_details=True.  Because every article about "Dario Amodei"
    also tags him as a PEOPLE entity, he surfaces at the top of the people
    category in his own co-mentions result — giving us his rp_entity_id
    with a single API call and no separate KG lookup.

    Match priority:
      1. People entry whose name exactly matches (case-insensitive)
      2. People entry whose name contains the query as a substring
      3. First people entry (highest co-mention volume)

    Falls back to None if no people are returned; the rest of the pipeline
    continues with text-based queries in that case.
    """
    print(f"\n[Step 1] Resolving entity ID for '{person_name}' "
          f"via co-mentions + entity_details …")

    body = {
        "query": {
            "text": person_name,
            "filters": {},
            "limit": 1000,
            "entity_details": True,        # ← inline entity metadata, no extra KG call
        }
    }
    try:
        data = _post("/v1/search/co-mentions/entities", body)
    except requests.HTTPError as exc:
        print(f"  [!] Co-mentions call failed: {exc}")
        return None

    people = data.get("results", {}).get("people", [])
    if not people:
        print(f"  [!] No people returned — will use text-based co-mentions")
        return None

    query_lower = person_name.lower()

    # Priority 1: exact name match
    for entry in people:
        ename = _entity_name_from_entry(entry)
        if ename.lower() == query_lower:
            print(f"  [✓] Exact match  → {ename}  (id={entry['id']})")
            return entry["id"]

    # Priority 2: name contains the query
    for entry in people:
        ename = _entity_name_from_entry(entry)
        if query_lower in ename.lower():
            print(f"  [~] Partial match → {ename}  (id={entry['id']})")
            return entry["id"]

    # Priority 3: highest-volume person (first entry)
    fallback = people[0]
    fname = _entity_name_from_entry(fallback)
    print(f"  [~] Fallback (top person) → {fname}  (id={fallback['id']})")
    return fallback["id"]


def _entity_name_from_entry(entry: dict) -> str:
    """
    Extract a display name from a co-mentions entity entry.
    When entity_details=True the API embeds the name directly; we also
    check the nested 'entity_details' sub-object as a fallback.
    """
    return (
        entry.get("name")
        or entry.get("entity_details", {}).get("name")
        or entry["id"]
    )


def _batch_resolve_entities(ids: list[str]) -> dict[str, dict]:
    """Resolve up to 100 entity IDs in one call → {id: entity_dict}."""
    if not ids:
        return {}
    results = {}
    for i in range(0, len(ids), 100):
        data = _post("/v1/knowledge-graph/entities/id",
                     {"values": ids[i : i + 100]})
        results.update(data.get("results", {}))
    return results

# ── Step 2: Full-history co-mentions (PERSON → COMPANY) ──────────────────────

def get_company_co_mentions(
    person_name: str,
    entity_id: Optional[str],
    year_start: int,
    year_end: int,
    limit: int = CO_MENTION_LIMIT_ALL,
) -> list[dict]:
    """
    Retrieve companies co-mentioned with the person for the given date range.

    Uses entity_id filter when available (precise entity-tagged match).
    Falls back to text query when entity_id is None.
    entity_details=True returns company names inline — no extra KG batch call.

    Returns list of {id, name, total_chunks_count, total_headlines_count}.
    """
    ts_start = f"{year_start}-01-01T00:00:00Z"
    ts_end   = f"{year_end}-12-31T23:59:59Z"

    if entity_id:
        # ── Precise: filter by resolved entity ID ──────────────────────────
        body = {
            "query": {
                "text": "",
                "filters": {
                    "timestamp": {"start": ts_start, "end": ts_end},
                    "entity": {
                        "any_of": [entity_id],
                        "all_of": [],
                        "none_of": [],
                    },
                },
                "limit": limit,
                "entity_details": True,    # ← inline names, skip KG roundtrip
            }
        }
    else:
        # ── Fallback: text-based (less precise) ───────────────────────────
        body = {
            "query": {
                "text": person_name,
                "filters": {
                    "timestamp": {"start": ts_start, "end": ts_end},
                },
                "limit": limit,
                "entity_details": True,
            }
        }

    data = _post("/v1/search/co-mentions/entities", body)
    return data.get("results", {}).get("companies", [])

# ── Step 3: Yearly time-slices ────────────────────────────────────────────────

def build_career_timeline(
    person_name: str,
    entity_id: Optional[str],
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """
    For each calendar year in [year_start, year_end], call co-mentions and
    record each company's chunk count.

    Returns a DataFrame:
        rows    = companies
        columns = years
        values  = total_chunks_count  (0 = not co-mentioned that year)
    """
    current_year = datetime.now(timezone.utc).year
    year_end = min(year_end, current_year)

    print(f"\n[Step 3] Building yearly career timeline "
          f"({year_start}–{year_end}) …")

    # Accumulate: {company_id: {year: chunks}}
    data_map: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    name_map: dict[str, str] = {}  # company_id → name

    for year in range(year_start, year_end + 1):
        print(f"  [{year}] fetching co-mentions …", end=" ", flush=True)
        try:
            companies = get_company_co_mentions(
                person_name, entity_id,
                year_start=year, year_end=year,
                limit=CO_MENTION_LIMIT,
            )
        except requests.HTTPError as exc:
            print(f"ERROR: {exc}")
            continue

        print(f"{len(companies)} companies")

        for c in companies:
            cid    = c["id"]
            chunks = c.get("total_chunks_count", 0)
            # entity_details=True → name is embedded directly in the entry
            cname  = _entity_name_from_entry(c)
            data_map[cid][year] += chunks
            name_map[cid] = cname

        time.sleep(0.2)   # be polite to the API

    if not data_map:
        print("  [!] No data collected — check entity resolution and date range.")
        return pd.DataFrame()

    years = list(range(year_start, year_end + 1))
    rows  = {
        name_map.get(cid, cid): [year_data.get(yr, 0) for yr in years]
        for cid, year_data in data_map.items()
    }
    df = pd.DataFrame(rows, index=years).T   # companies × years
    df.index.name   = "Company"
    df.columns.name = "Year"

    # Sort by total mentions descending
    df["_total"] = df.sum(axis=1)
    df = df.sort_values("_total", ascending=False).drop(columns=["_total"])
    return df

# ── Step 4a: Career path table ────────────────────────────────────────────────

def print_career_path(person_name: str, df: pd.DataFrame, top_n: int = 15):
    """
    Print a human-readable career path table.
    For each company, show: first_year seen, last_year seen, total chunks,
    and a sparkline of activity.
    """
    print(f"\n{'='*65}")
    print(f"CAREER PATH — {person_name}")
    print(f"{'='*65}")
    print(f"{'Company':<35} {'First':>6} {'Last':>6} {'Total':>7}  Activity")
    print(f"{'-'*65}")

    for company, row in df.head(top_n).iterrows():
        active = row[row > 0]
        if active.empty:
            continue
        first = int(active.index.min())
        last  = int(active.index.max())
        total = int(row.sum())
        # Micro sparkline using block chars
        spark = _sparkline(row.values)
        print(f"{str(company):<35} {first:>6} {last:>6} {total:>7}  {spark}")

    print(f"{'='*65}")
    print(f"Coverage: {df.shape[1]} years × {df.shape[0]} companies")


def _sparkline(values, width: int = 20) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    mx = max(values) if max(values) > 0 else 1
    return "".join(blocks[min(int(v / mx * 8), 8)] for v in values)

# ── Step 4b: Heatmap visualization ───────────────────────────────────────────

def visualize_heatmap(person_name: str, df: pd.DataFrame,
                      top_n: int = 20, path: str = "career_heatmap.png"):
    """
    Render a companies × years heatmap showing career activity intensity.
    Darker = more co-mentions that year.
    """
    if df.empty:
        print("[!] No data to visualize")
        return

    plot_df = df.head(top_n)
    fig, ax = plt.subplots(figsize=(max(12, df.shape[1] * 0.6), max(6, top_n * 0.45)))

    im = ax.imshow(plot_df.values, aspect="auto", cmap="Blues",
                   norm=mcolors.PowerNorm(gamma=0.4))   # compress dynamic range

    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index, fontsize=9)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Company", fontsize=11)
    ax.set_title(f"Career Timeline — {person_name}\n"
                 f"(co-mention intensity per year, Bigdata.com)", fontsize=13,
                 fontweight="bold")

    plt.colorbar(im, ax=ax, label="Co-mention chunk count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"[+] Heatmap saved → {path}")
    plt.show()

# ── Step 4c: Network graph ────────────────────────────────────────────────────

def visualize_network(person_name: str, df: pd.DataFrame,
                      top_n: int = 20, path: str = "career_network.png"):
    """
    Bipartite graph: person node at center, company nodes around it.
    Both company node size AND edge width are proportional to total
    co-mention chunk volume — the more documents, the bigger / thicker.
    The seed person node is anchored at 1.5× the largest company node.
    """
    if df.empty:
        return

    G = nx.Graph()
    seed_id = f"PERSON:{person_name}"
    G.add_node(seed_id, node_type="person", label=person_name, total=0)

    plot_df = df.head(top_n)
    for company, row in plot_df.iterrows():
        total = int(row.sum())
        if total == 0:
            continue
        cid = f"COMPANY:{company}"
        G.add_node(cid, node_type="company", label=str(company), total=total)
        G.add_edge(seed_id, cid, weight=total)

    if G.number_of_nodes() <= 1:
        print("[!] No company nodes to draw")
        return

    # ── Size scaling: map chunk totals → pixel area ────────────────────────
    company_totals = [
        G.nodes[n]["total"]
        for n in G.nodes()
        if G.nodes[n]["node_type"] == "company"
    ]
    max_total  = max(company_totals) if company_totals else 1
    SIZE_MIN   = 300
    SIZE_MAX   = 4000

    def _node_size(total: int) -> float:
        """Linear scale: 0 → SIZE_MIN, max_total → SIZE_MAX."""
        return SIZE_MIN + (total / max_total) * (SIZE_MAX - SIZE_MIN)

    # Seed person node = 1.5× the biggest company node so it reads clearly
    G.nodes[seed_id]["total"] = int(max_total * 1.5)

    # ── Edge width: proportional to chunk volume ───────────────────────────
    max_edge   = max((G[u][v]["weight"] for u, v in G.edges()), default=1)
    edge_widths = [
        max(0.5, (G[u][v]["weight"] / max_edge) * 8.0)
        for u, v in G.edges()
    ]

    pos = nx.spring_layout(G, seed=42, k=3.0)

    node_colors, node_sizes, labels = [], [], {}
    for n in G.nodes():
        ntype = G.nodes[n].get("node_type", "")
        lbl   = G.nodes[n].get("label", n)
        labels[n] = lbl[:28]
        if ntype == "person":
            node_colors.append("#2C7BB6")
        else:
            node_colors.append("#F5A623")
        node_sizes.append(_node_size(G.nodes[n].get("total", 0)))

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_title(
        f"Professional Network — {person_name}\n"
        f"(PERSON → COMPANY  |  node size & edge width ∝ co-mention volume  |  Bigdata.com)",
        fontsize=13, fontweight="bold",
    )

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7.5, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths,
                           alpha=0.55, edge_color="#888888", ax=ax)

    legend = [
        mpatches.Patch(color="#2C7BB6", label="Target person"),
        mpatches.Patch(color="#F5A623", label="Company  (size & edge ∝ chunk volume)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"[+] Network graph saved → {path}")
    plt.show()

# ── Step 4d: JSON export ──────────────────────────────────────────────────────

def export_json(person_name: str, df: pd.DataFrame, entity_id: Optional[str],
                path: str = "career_path.json"):
    """Export structured career data for downstream dashboards / pipelines."""
    records = []
    for company, row in df.iterrows():
        active = row[row > 0]
        if active.empty:
            continue
        records.append({
            "company":        str(company),
            "first_year":     int(active.index.min()),
            "last_year":      int(active.index.max()),
            "total_chunks":   int(row.sum()),
            "active_years":   int((row > 0).sum()),
            "yearly_chunks":  {str(yr): int(v) for yr, v in row.items() if v > 0},
        })

    output = {
        "person":      person_name,
        "entity_id":   entity_id,
        "generated":   datetime.now(timezone.utc).isoformat(),
        "source":      "Bigdata.com co-mentions API",
        "companies":   records,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[+] JSON exported → {path}")
    return output

# ── Orchestrator ──────────────────────────────────────────────────────────────

def analyze_career_path(
    person_name: str,
    year_start:  int = DEFAULT_YEAR_START,
    year_end:    int = datetime.now(timezone.utc).year,
    top_n:       int = 20,
    output_prefix: str = "",
):
    """
    Full pipeline for a single person.

    Parameters
    ----------
    person_name   : Full name, e.g. "Dario Amodei" or "Ray Dalio"
    year_start    : First year to include  (default 2010)
    year_end      : Last year to include   (default: current year)
    top_n         : Top N companies to show in outputs
    output_prefix : Prefix for output file names (defaults to person slug)
    """
    if not output_prefix:
        output_prefix = person_name.lower().replace(" ", "_")

    print("=" * 65)
    print(f"BIGDATA.COM — PEOPLE-CENTRIC CAREER PATH ANALYSIS")
    print(f"Person  : {person_name}")
    print(f"Window  : {year_start} – {year_end}")
    print(f"Source  : {BASE_URL}")
    print("=" * 65)

    # ── Step 1: Resolve entity ID ─────────────────────────────────────
    entity_id = resolve_person_entity_id(person_name)

    # ── Step 2: Full-history co-mentions ──────────────────────────────
    print(f"\n[Step 2] Full-history company co-mentions ({year_start}–{year_end}) …")
    all_companies = get_company_co_mentions(
        person_name, entity_id,
        year_start=year_start, year_end=year_end,
        limit=CO_MENTION_LIMIT_ALL,
    )
    print(f"  → {len(all_companies)} unique companies found across full history")

    # ── Step 3: Yearly slices ─────────────────────────────────────────
    df = build_career_timeline(person_name, entity_id, year_start, year_end)

    if df.empty:
        print("\n[!] No timeline data — exiting.")
        return

    # ── Step 4: Outputs ───────────────────────────────────────────────
    print_career_path(person_name, df, top_n=top_n)

    visualize_heatmap(
        person_name, df, top_n=top_n,
        path=f"{output_prefix}_heatmap.png",
    )
    visualize_network(
        person_name, df, top_n=top_n,
        path=f"{output_prefix}_network.png",
    )
    export_json(
        person_name, df, entity_id,
        path=f"{output_prefix}_career.json",
    )

    return df

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="People-centric career path analysis via Bigdata.com APIs"
    )
    parser.add_argument(
        "person",
        nargs="?",
        default="Dario Amodei",
        help='Full name, e.g. "Ray Dalio"  (default: Dario Amodei)',
    )
    parser.add_argument(
        "year_start",
        nargs="?",
        type=int,
        default=DEFAULT_YEAR_START,
        help=f"Start year  (default: {DEFAULT_YEAR_START})",
    )
    parser.add_argument(
        "year_end",
        nargs="?",
        type=int,
        default=datetime.now(timezone.utc).year,
        help="End year  (default: current year)",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Top N companies to include in outputs  (default: 20)",
    )
    args = parser.parse_args()

    if not API_KEY:
        sys.exit(
            "ERROR: BIGDATA_API_KEY environment variable is not set.\n"
            "  export BIGDATA_API_KEY=your_key_here"
        )

    analyze_career_path(
        person_name  = args.person,
        year_start   = args.year_start,
        year_end     = args.year_end,
        top_n        = args.top,
    )


if __name__ == "__main__":
    main()
