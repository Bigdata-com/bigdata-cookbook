# Requirements: pip install requests matplotlib

"""
Iran Volume Spike Detector + Co-mentions Analysis
--------------------------------------------------
1. Fetches daily volume for "Iran" over the last 3 months via Bigdata.com Volume API.
2. Detects spike days (chunks > mean + 2 * std).
3. Identifies the largest contiguous spike window.
4. Queries Co-mentions API for that spike period.
5. Resolves entity IDs to names via Knowledge Graph.
6. Prints a ranked table and plots the volume with spike highlights.

Usage:
    export BIGDATA_API_KEY="your-key-here"
    python iran_volume_spikes.py
"""

import os
import statistics
from datetime import datetime, timedelta, timezone

import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Auth ──────────────────────────────────────────────────────────────────────
API_KEY = os.environ["BIGDATA_API_KEY"]
BASE_URL = "https://api.bigdata.com"
HEADERS = {"X-API-KEY": API_KEY, "Content-Type": "application/json"}

# ── Date range: last 3 months ─────────────────────────────────────────────────
END_DT = datetime.now(timezone.utc)
START_DT = END_DT - timedelta(days=90)
START = START_DT.strftime("%Y-%m-%dT00:00:00Z")
END   = END_DT.strftime("%Y-%m-%dT23:59:59Z")

SPIKE_THRESHOLD_STDS = 2.0  # days above mean + N*std are considered spikes
CO_MENTION_LIMIT = 500       # fetch large pool, rank client-side


# ── Step 1: Volume ─────────────────────────────────────────────────────────────
def fetch_volume() -> list[dict]:
    """Return daily volume records for Iran-related coverage."""
    print(f"[1/4] Fetching volume for Iran  ({START[:10]} → {END[:10]}) ...")
    body = {
        "query": {
            # Semantically rich sentence for better embedding matching
            "text": "Iran geopolitical events news and developments",
            "auto_enrich_filters": True,
            "filters": {
                "timestamp": {"start": START, "end": END},
            },
        }
    }
    resp = requests.post(f"{BASE_URL}/v1/search/volume", headers=HEADERS, json=body)
    resp.raise_for_status()
    volume = resp.json().get("results", {}).get("volume", [])
    print(f"    → {len(volume)} daily data points returned.")
    return volume


# ── Step 2: Spike detection ────────────────────────────────────────────────────
def detect_spikes(volume: list[dict]) -> tuple[list[dict], list[dict], float, float]:
    """
    Flag spike days and return the largest contiguous spike window.
    Returns: (all_days, spike_days, mean_chunks, std_chunks)
    """
    print(f"[2/4] Detecting spikes (threshold: mean + {SPIKE_THRESHOLD_STDS}σ) ...")

    # Normalise field name — API may return 'date' or 'day'
    days = []
    for rec in volume:
        d = rec.get("date") or rec.get("day", "")
        days.append({
            "date": d,
            "documents": rec.get("documents", 0),
            "chunks": rec.get("chunks", 0),
            "sentiment": rec.get("sentiment", 0.0),
        })

    if not days:
        raise ValueError("No volume data returned — check API key or date range.")

    days.sort(key=lambda x: x["date"])
    chunk_counts = [d["chunks"] for d in days]
    mean_c  = statistics.mean(chunk_counts)
    std_c   = statistics.stdev(chunk_counts) if len(chunk_counts) > 1 else 0
    threshold = mean_c + SPIKE_THRESHOLD_STDS * std_c

    for d in days:
        d["is_spike"] = d["chunks"] > threshold

    spike_days = [d for d in days if d["is_spike"]]
    print(f"    → mean={mean_c:.0f}  std={std_c:.0f}  threshold={threshold:.0f}  "
          f"spike days={len(spike_days)}")
    return days, spike_days, mean_c, std_c


def largest_spike_window(days: list[dict]) -> tuple[str, str]:
    """Return (start_date, end_date) of the largest contiguous spike run."""
    best_run, current_run = [], []
    for d in days:
        if d["is_spike"]:
            current_run.append(d)
        else:
            if len(current_run) > len(best_run):
                best_run = current_run
            current_run = []
    if len(current_run) > len(best_run):
        best_run = current_run

    if best_run:
        return best_run[0]["date"], best_run[-1]["date"]

    # Fallback: single highest-volume day
    peak = max(days, key=lambda d: d["chunks"])
    return peak["date"], peak["date"]


# ── Step 3: Co-mentions ────────────────────────────────────────────────────────
def fetch_comentions(spike_start: str, spike_end: str) -> dict:
    """Return raw co-mention results for the spike window."""
    print(f"[3/4] Fetching co-mentions  ({spike_start} → {spike_end}) ...")
    body = {
        "query": {
            "text": "Iran geopolitical events news and developments",
            "auto_enrich_filters": True,
            "filters": {
                "timestamp": {
                    "start": f"{spike_start}T00:00:00Z",
                    "end":   f"{spike_end}T23:59:59Z",
                },
            },
        },
        "limit": CO_MENTION_LIMIT,
    }
    resp = requests.post(
        f"{BASE_URL}/v1/search/co-mentions/entities", headers=HEADERS, json=body
    )
    resp.raise_for_status()
    results = resp.json().get("results", {})
    total = sum(len(v) for v in results.values())
    print(f"    → {total} co-mentioned entities returned across all categories.")
    return results


# ── Step 4: Resolve entity IDs to names ───────────────────────────────────────
def resolve_ids(entity_ids: list[str]) -> dict[str, str]:
    """Batch-resolve up to 100 IDs per request. Returns {id: name}."""
    names: dict[str, str] = {}
    for i in range(0, len(entity_ids), 100):
        batch = entity_ids[i : i + 100]
        resp = requests.post(
            f"{BASE_URL}/v1/knowledge-graph/entities/id",
            headers=HEADERS,
            json={"values": batch},
        )
        resp.raise_for_status()
        for eid, info in resp.json().get("results", {}).items():
            names[eid] = info.get("name", eid)
    return names


# ── Step 5: Print ranked table ────────────────────────────────────────────────
def print_ranked(category: str, entities: list[dict], names: dict[str, str], top_n: int = 20):
    """Print top-N entities for a category, ranked by chunks then headlines."""
    ranked = sorted(
        entities,
        key=lambda e: (e.get("total_chunks_count", 0), e.get("total_headlines_count", 0)),
        reverse=True,
    )[:top_n]
    print(f"\n  Top {top_n} {category.upper()}:")
    print(f"  {'Rank':<5} {'Name':<35} {'Chunks':>8} {'Headlines':>10}")
    print(f"  {'-'*5} {'-'*35} {'-'*8} {'-'*10}")
    for rank, e in enumerate(ranked, 1):
        name   = names.get(e["id"], e["id"])[:34]
        chunks = e.get("total_chunks_count", 0) or 0
        heads  = e.get("total_headlines_count", 0) or 0
        print(f"  {rank:<5} {name:<35} {chunks:>8,} {heads:>10,}")


# ── Step 6: Plot ───────────────────────────────────────────────────────────────
def plot_volume(days: list[dict], mean_c: float, std_c: float,
                spike_start: str, spike_end: str, outfile: str = "iran_volume.png"):
    """Save a volume chart with spike threshold and window highlighted."""
    dates  = [d["date"] for d in days]
    chunks = [d["chunks"] for d in days]
    spikes = [d["is_spike"] for d in days]
    threshold = mean_c + SPIKE_THRESHOLD_STDS * std_c

    xs = range(len(dates))
    colors = ["#e74c3c" if s else "#2980b9" for s in spikes]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(xs, chunks, color=colors, width=0.8)
    ax.axhline(threshold, color="#e74c3c", linestyle="--", linewidth=1.2,
               label=f"Spike threshold (mean+{SPIKE_THRESHOLD_STDS}σ = {threshold:,.0f})")
    ax.axhline(mean_c, color="#7f8c8d", linestyle=":", linewidth=1,
               label=f"Mean ({mean_c:,.0f})")

    # Shade the selected spike window
    try:
        si = dates.index(spike_start)
        ei = dates.index(spike_end)
        ax.axvspan(si - 0.5, ei + 0.5, color="#f39c12", alpha=0.15,
                   label=f"Selected window: {spike_start} → {spike_end}")
    except ValueError:
        pass

    # X-axis: show every ~7th label
    tick_step = max(1, len(dates) // 15)
    ax.set_xticks(list(xs)[::tick_step])
    ax.set_xticklabels(dates[::tick_step], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Chunks per day")
    ax.set_title("Iran — Daily Coverage Volume (last 3 months)\nRed bars = spike days")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"\n  Chart saved → {outfile}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Iran Volume Spike Detector + Co-mentions  (Bigdata.com API)")
    print("=" * 65)

    # 1. Volume
    volume = fetch_volume()

    # 2. Spike detection
    days, spike_days, mean_c, std_c = detect_spikes(volume)
    spike_start, spike_end = largest_spike_window(days)
    print(f"    → Largest spike window: {spike_start} → {spike_end}")

    if not spike_days:
        print("\n  No spikes detected. Using peak day for co-mentions.")

    # 3. Co-mentions in spike window
    comentions = fetch_comentions(spike_start, spike_end)

    # 4. Collect all IDs and resolve names
    print("[4/4] Resolving entity names ...")
    all_ids: list[str] = []
    for category_entities in comentions.values():
        all_ids.extend(e["id"] for e in category_entities)
    names = resolve_ids(all_ids) if all_ids else {}
    print(f"    → {len(names)} names resolved.")

    # 5. Print ranked results
    print(f"\n{'='*65}")
    print(f"  Co-mentions with IRAN  |  Spike window: {spike_start} → {spike_end}")
    print(f"{'='*65}")

    priority_categories = ["places", "companies", "people", "organizations", "concepts", "products"]
    for cat in priority_categories:
        entities = comentions.get(cat, [])
        if entities:
            print_ranked(cat, entities, names, top_n=15)

    # 6. Plot
    plot_volume(days, mean_c, std_c, spike_start, spike_end)


if __name__ == "__main__":
    main()
